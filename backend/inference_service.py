import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

class FraudInferenceService:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.model_path = os.path.join(self.models_dir, "champion_fraud_model.joblib")
        self.scaler_path = os.path.join(self.models_dir, "robust_scaler.joblib")
        self.metadata_path = os.path.join(self.models_dir, "model_metadata.json")
        self.presets_path = os.path.join(self.models_dir, "sample_presets.json")
        
        self.model = None
        self.scaler = None
        self.metadata = None
        self.presets = None
        self.load_artifacts()

    def load_artifacts(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        if os.path.exists(self.presets_path):
            with open(self.presets_path, "r") as f:
                self.presets = json.load(f)

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = (np.sin(delta_phi / 2.0) ** 2 +
             np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def preprocess_single(self, data: dict):
        # 1. Geographic distance
        dist = self.haversine_distance(
            data['lat'], data['long'], data['merch_lat'], data['merch_long']
        )
        
        # 2. Time signals
        trans_dt = pd.to_datetime(data.get('trans_date_trans_time', '2026-11-13 12:00:00'))
        hour = trans_dt.hour
        day_of_week = trans_dt.dayofweek
        is_night = 1 if (hour >= 22 or hour <= 4) else 0

        # 3. Age
        dob_dt = pd.to_datetime(data.get('dob', '1985-01-01'))
        age = (trans_dt - dob_dt).days // 365
        if age < 18 or age > 100:
            age = 40  # fallback default

        # 4. Gender
        gender_enc = 1 if str(data.get('gender', 'F')).upper() == 'M' else 0

        # 5. Log population
        city_pop = max(float(data.get('city_pop', 1000.0)), 1.0)
        log_city_pop = np.log1p(city_pop)

        amt = float(data.get('amt', 0.0))

        # Build numerical DataFrame with matching column names
        num_cols = self.metadata['num_features']
        num_df = pd.DataFrame([[amt, dist, age, log_city_pop, hour, day_of_week]], columns=num_cols)
        scaled_num = self.scaler.transform(num_df)

        # Categories
        cat_selected = str(data.get('category', '')).lower()
        cat_dict = {}
        for cat in self.metadata['categories']:
            cat_dict[f"cat_{cat}"] = 1 if cat == cat_selected else 0

        # Full feature DataFrame
        full_row = {}
        for idx, col in enumerate(num_cols):
            full_row[col] = scaled_num[0, idx]
        full_row['is_night'] = is_night
        full_row['gender_enc'] = gender_enc
        full_row.update(cat_dict)

        feature_cols = self.metadata['feature_cols']
        X_df = pd.DataFrame([full_row])[feature_cols]

        computed_info = {
            "distance_km": round(float(dist), 2),
            "hour": int(hour),
            "age": int(age),
            "is_night": bool(is_night),
            "category": cat_selected
        }

        return X_df, computed_info

    def predict_single(self, data: dict):
        start_time = time.perf_counter()
        X_df, computed_info = self.preprocess_single(data)
        
        prob = float(self.model.predict_proba(X_df)[0, 1])
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        threshold = self.metadata.get('optimal_threshold', 0.35)
        is_fraud = bool(prob >= threshold)

        # Risk classification
        if prob >= 0.65:
            risk_level = "HIGH"
            action = "DECLINE / IMMEDIATE CARD LOCK & SMS VERIFICATION REQUIRED"
        elif prob >= 0.25:
            risk_level = "MEDIUM"
            action = "SUSPICIOUS / STEP-UP 2FA AUTHENTICATION CHALLENGE"
        else:
            risk_level = "LOW"
            action = "APPROVED / TRANSACTION AUTHORIZED INSTANTLY"

        return {
            "is_fraud": is_fraud,
            "fraud_probability": round(prob, 4),
            "risk_score_pct": round(prob * 100, 2),
            "risk_level": risk_level,
            "action_recommendation": action,
            "decision_threshold": threshold,
            "latency_ms": latency_ms,
            "features_computed": computed_info
        }

    def predict_batch_csv(self, file_content: bytes):
        import io
        df = pd.read_csv(io.BytesIO(file_content))
        
        # Calculate features vectorized
        df_clean = df.copy()
        df_clean['distance_km'] = self.haversine_distance(
            df_clean['lat'], df_clean['long'], df_clean['merch_lat'], df_clean['merch_long']
        )
        trans_dt = pd.to_datetime(df_clean['trans_date_trans_time'])
        df_clean['hour'] = trans_dt.dt.hour
        df_clean['day_of_week'] = trans_dt.dt.dayofweek
        df_clean['is_night'] = ((df_clean['hour'] >= 22) | (df_clean['hour'] <= 4)).astype(int)
        
        dob_dt = pd.to_datetime(df_clean['dob'])
        df_clean['age'] = (trans_dt - dob_dt).dt.days // 365
        df_clean['gender_enc'] = (df_clean['gender'] == 'M').astype(int)
        df_clean['log_city_pop'] = np.log1p(df_clean['city_pop'])

        for cat in self.metadata['categories']:
            col_name = f"cat_{cat}"
            df_clean[col_name] = (df_clean['category'] == cat).astype(int)

        feature_cols = self.metadata['feature_cols']
        num_cols = self.metadata['num_features']

        X_df = df_clean[feature_cols].copy()
        X_df[num_cols] = self.scaler.transform(X_df[num_cols])

        probs = self.model.predict_proba(X_df)[:, 1]
        threshold = self.metadata.get('optimal_threshold', 0.35)
        is_fraud_preds = (probs >= threshold).astype(int)

        df['fraud_probability'] = np.round(probs, 4)
        df['risk_score_pct'] = np.round(probs * 100, 2)
        df['prediction'] = np.where(is_fraud_preds == 1, 'FLAGGED FRAUD', 'LEGIT')

        flagged_count = int(is_fraud_preds.sum())
        total_count = len(df)
        approved_count = total_count - flagged_count
        fraud_rate = round((flagged_count / max(total_count, 1)) * 100, 2)

        flagged_sample = df[df['prediction'] == 'FLAGGED FRAUD'].head(15).to_dict(orient='records')
        clean_sample = []
        for row in flagged_sample:
            clean_sample.append({
                'amt': float(row.get('amt', 0)),
                'category': str(row.get('category', '')),
                'risk_score_pct': float(row.get('risk_score_pct', 0)),
                'prediction': str(row.get('prediction', ''))
            })

        return {
            "total_processed": total_count,
            "flagged_fraud_count": flagged_count,
            "approved_count": approved_count,
            "fraud_rate_pct": fraud_rate,
            "sample_flagged_rows": clean_sample,
            "full_audited_csv": df.to_csv(index=False)
        }
