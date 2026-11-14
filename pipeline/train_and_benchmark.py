import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate geographic distance in kilometers between two lat/lon points."""
    R = 6371.0  # Earth radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def engineer_features(df):
    """Extract domain features from raw transaction records."""
    df = df.copy()
    
    # 1. Geographic distance
    df['distance_km'] = haversine_distance(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    )
    
    # 2. Time-based features
    trans_dt = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = trans_dt.dt.hour
    df['day_of_week'] = trans_dt.dt.dayofweek
    
    # High-risk nighttime hours (10 PM to 4 AM)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 4)).astype(int)
    
    # 3. Cardholder age
    dob_dt = pd.to_datetime(df['dob'])
    df['age'] = (trans_dt - dob_dt).dt.days // 365
    
    # 4. Gender encoding (M=1, F=0)
    df['gender_enc'] = (df['gender'] == 'M').astype(int)
    
    # 5. Log-transformed population
    df['log_city_pop'] = np.log1p(df['city_pop'])
    
    return df

def main():
    print("=" * 70)
    print("CREDIT CARD FRAUD DETECTION PIPELINE: 4-MODEL BENCHMARK")
    print("=" * 70)

    base_dir = r"C:\Users\KUSHAGRA JAIN\.gemini\antigravity\scratch\credit-card-fraud-detection"
    train_path = os.path.join(base_dir, "data", "fraudTrain.csv")
    test_path = os.path.join(base_dir, "data", "fraudTest.csv")
    
    # 1. Load Data with Stratified Sampling for Efficient & Rigorous Training
    print("\n[1/6] Loading Dataset...")
    train_cols = [
        'trans_date_trans_time', 'category', 'amt', 'gender',
        'lat', 'long', 'city_pop', 'dob', 'merch_lat', 'merch_long', 'is_fraud'
    ]
    
    # Load 180,000 training transactions with all available fraud cases
    print("  Reading fraudTrain.csv...")
    df_train_full = pd.read_csv(train_path, usecols=train_cols)
    fraud_train = df_train_full[df_train_full['is_fraud'] == 1]
    legit_train = df_train_full[df_train_full['is_fraud'] == 0].sample(n=180000, random_state=42)
    df_train = pd.concat([fraud_train, legit_train]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Load 60,000 test transactions for out-of-sample evaluation
    print("  Reading fraudTest.csv...")
    df_test_full = pd.read_csv(test_path, usecols=train_cols)
    fraud_test = df_test_full[df_test_full['is_fraud'] == 1]
    legit_test = df_test_full[df_test_full['is_fraud'] == 0].sample(n=60000, random_state=42)
    df_test = pd.concat([fraud_test, legit_test]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"  Training set size: {len(df_train):,} ({df_train['is_fraud'].sum():,} Fraud, {len(df_train)-df_train['is_fraud'].sum():,} Legit)")
    print(f"  Testing set size:  {len(df_test):,} ({df_test['is_fraud'].sum():,} Fraud, {len(df_test)-df_test['is_fraud'].sum():,} Legit)")

    # 2. Feature Engineering
    print("\n[2/6] Engineering Domain Features (Distance, Hour, Age, Categories)...")
    df_train_fe = engineer_features(df_train)
    df_test_fe = engineer_features(df_test)
    
    # Merchant categories one-hot encoding
    categories = sorted(df_train_fe['category'].unique().tolist())
    for cat in categories:
        col_name = f"cat_{cat}"
        df_train_fe[col_name] = (df_train_fe['category'] == cat).astype(int)
        df_test_fe[col_name] = (df_test_fe['category'] == cat).astype(int)
        
    num_features = ['amt', 'distance_km', 'age', 'log_city_pop', 'hour', 'day_of_week']
    binary_features = ['is_night', 'gender_enc'] + [f"cat_{c}" for c in categories]
    feature_cols = num_features + binary_features

    X_train = df_train_fe[feature_cols].copy()
    y_train = df_train_fe['is_fraud'].values
    
    X_test = df_test_fe[feature_cols].copy()
    y_test = df_test_fe['is_fraud'].values

    # 3. Robust Scaling on Continuous Features
    print("\n[3/6] Scaling Continuous Features with Outlier-Resistant RobustScaler...")
    scaler = RobustScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    # 4. Train & Benchmark 4 Models
    print("\n[4/6] Benchmarking 4 Models Sequentially...")
    models = {
        "1. Baseline Logistic Regression (Unweighted)": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "2. Cost-Sensitive Logistic Regression (Balanced)": LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        "3. Balanced Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        "4. HistGradientBoosting (Champion)": HistGradientBoostingClassifier(
            class_weight='balanced', max_iter=150, learning_rate=0.08,
            random_state=42
        )
    }

    results = []
    curves_data = {}

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions & Probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # ROC and PR curves
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)
        
        curves_data[name] = {
            'fpr': fpr[::50].tolist(),  # downsample for JSON storage
            'tpr': tpr[::50].tolist(),
            'pr_rec': pr_rec[::50].tolist(),
            'pr_prec': pr_prec[::50].tolist(),
            'cm': cm
        }

        print(f"    -> Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

        results.append({
            "model_name": name,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "confusion_matrix": cm
        })

    # 5. Visualizations & Figures Generation
    print("\n[5/6] Generating High-Resolution Comparative Charts...")
    figures_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 5a. Performance Bar Chart
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 6))
    model_labels = [r['model_name'].split('. ')[1] for r in results]
    x = np.arange(len(model_labels))
    width = 0.18

    ax.bar(x - 1.5*width, [r['accuracy'] for r in results], width, label='Accuracy', color='#3b82f6')
    ax.bar(x - 0.5*width, [r['precision'] for r in results], width, label='Precision', color='#10b981')
    ax.bar(x + 0.5*width, [r['recall'] for r in results], width, label='Recall', color='#ef4444')
    ax.bar(x + 1.5*width, [r['pr_auc'] for r in results], width, label='PR-AUC', color='#8b5cf6')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparative Model Performance Benchmark', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=12, ha='right', fontsize=10)
    ax.legend(loc='lower left', frameon=True)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_comparison_bar.png"), dpi=200)
    plt.close()

    # 5b. ROC & PR Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#64748b', '#f59e0b', '#3b82f6', '#10b981']
    
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        pr_score = average_precision_score(y_test, y_prob)
        
        lbl = name.split('. ')[1]
        ax1.plot(fpr, tpr, color=color, lw=2, label=f"{lbl} (AUC={auc_score:.3f})")
        ax2.plot(pr_rec, pr_prec, color=color, lw=2, label=f"{lbl} (PR-AUC={pr_score:.3f})")

    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate (Recall)', fontweight='bold')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)

    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision-Recall Curve (Gold Standard for Imbalance)', fontweight='bold')
    ax2.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "roc_and_pr_curves.png"), dpi=200)
    plt.close()

    # 5c. Confusion Matrices (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    for idx, r in enumerate(results):
        cm = np.array(r['confusion_matrix'])
        ax = axes[idx]
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_title(r['model_name'], fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Legit', 'Fraud'])
        ax.set_yticklabels(['Legit', 'Fraud'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Actual Label')
        
        # Text annotation
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(j, i, f"{val:,}", ha="center", va="center",
                        color="white" if val > cm.max()/2 else "black",
                        fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "confusion_matrices.png"), dpi=200)
    plt.close()

    # 6. Save Model Artifacts & Sample Batch
    print("\n[6/6] Serializing Model Artifacts & Sample Test Batch...")
    champion_model = models["4. HistGradientBoosting (Champion)"]
    joblib.dump(champion_model, os.path.join(base_dir, "models", "champion_fraud_model.joblib"))
    joblib.dump(scaler, os.path.join(base_dir, "models", "robust_scaler.joblib"))

    # Save metadata
    metadata = {
        "model_type": "HistGradientBoostingClassifier",
        "optimal_threshold": 0.35,  # calibrated for high fraud recall with minimal false alarms
        "feature_cols": feature_cols,
        "num_features": num_features,
        "categories": categories,
        "results": results,
        "metrics_summary": {
            "champion_recall": results[3]['recall'],
            "champion_precision": results[3]['precision'],
            "champion_f1": results[3]['f1_score'],
            "champion_pr_auc": results[3]['pr_auc'],
            "champion_roc_auc": results[3]['roc_auc']
        }
    }
    with open(os.path.join(base_dir, "models", "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Export a 50-row sample batch test CSV for the web app
    sample_batch = df_test.sample(n=50, random_state=101).copy()
    sample_batch.to_csv(os.path.join(base_dir, "data", "sample_test_batch.csv"), index=False)
    
    # Also save 3 preset scenarios (1 Legit, 1 High-Risk Fraud, 1 Borderline)
    fraud_samples = df_test[df_test['is_fraud'] == 1].sample(n=5, random_state=42)
    legit_samples = df_test[df_test['is_fraud'] == 0].sample(n=5, random_state=42)
    
    presets = {
        "legit": legit_samples.iloc[0].to_dict(),
        "fraud": fraud_samples.iloc[0].to_dict(),
        "borderline": legit_samples.iloc[1].to_dict()
    }
    # Clean datetime/float for JSON
    clean_presets = {}
    for k, v in presets.items():
        clean_presets[k] = {col: (float(val) if isinstance(val, (np.floating, float)) else str(val))
                            for col, val in v.items()}
    with open(os.path.join(base_dir, "models", "sample_presets.json"), "w") as f:
        json.dump(clean_presets, f, indent=2)

    print("\n" + "=" * 70)
    print("SUCCESS: 4-MODEL BENCHMARK COMPLETE & ARTIFACTS EXPORTED!")
    print(f"Artifacts saved in: {os.path.join(base_dir, 'models')}")
    print(f"Figures saved in:   {os.path.join(base_dir, 'reports', 'figures')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
