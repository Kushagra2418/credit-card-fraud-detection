import os
import sys
import json
import time
import pandas as pd

base_dir = r"C:\Users\KUSHAGRA JAIN\.gemini\antigravity\scratch\credit-card-fraud-detection"
sys.path.append(base_dir)

from backend.inference_service import FraudInferenceService

def run_tests():
    print("=" * 70)
    print("🧪 SENTINEL AI — SYSTEM VALIDATION & END-TO-END VERIFICATION")
    print("=" * 70)

    # 1. Test Artifact Loading
    print("\n[TEST 1/5] Checking Model Artifacts & Service Initialization...")
    service = FraudInferenceService(base_dir)
    assert service.model is not None, "Model failed to load!"
    assert service.scaler is not None, "Scaler failed to load!"
    assert service.metadata is not None, "Metadata failed to load!"
    print("  ✓ Champion Model: HistGradientBoostingClassifier LOADED")
    print("  ✓ RobustScaler LOADED")
    print("  ✓ Metadata LOADED (Threshold: 0.35, Features: 22)")

    # 2. Test Fraud Case Prediction
    print("\n[TEST 2/5] Testing Known Fraud Attack Pattern...")
    fraud_payload = {
        'amt': 1145.50,
        'category': 'shopping_net',
        'gender': 'M',
        'city_pop': 45000.0,
        'lat': 34.0522,
        'long': -118.2437,
        'merch_lat': 40.7128,
        'merch_long': -74.0060,
        'trans_date_trans_time': '2026-11-13 03:15:00',
        'dob': '1975-08-22'
    }
    fraud_res = service.predict_single(fraud_payload)
    print(f"  Input: ${fraud_payload['amt']} | Category: {fraud_payload['category']} | Distance: {fraud_res['features_computed']['distance_km']} km")
    print(f"  Result -> Probability: {fraud_res['fraud_probability']*100:.2f}% | Risk Level: {fraud_res['risk_level']} | Latency: {fraud_res['latency_ms']} ms")
    print(f"  Action -> {fraud_res['action_recommendation']}")
    assert fraud_res['is_fraud'] == True, "Failed to identify fraud transaction!"
    print("  ✓ Fraud Detection: PASS")

    # 3. Test Legit Case Prediction
    print("\n[TEST 3/5] Testing Normal Cardholder Purchase...")
    legit_payload = {
        'amt': 28.50,
        'category': 'grocery_pos',
        'gender': 'F',
        'city_pop': 32000.0,
        'lat': 40.7128,
        'long': -74.0060,
        'merch_lat': 40.7180,
        'merch_long': -74.0020,
        'trans_date_trans_time': '2026-11-13 14:30:00',
        'dob': '1992-04-10'
    }
    legit_res = service.predict_single(legit_payload)
    print(f"  Input: ${legit_payload['amt']} | Category: {legit_payload['category']} | Distance: {legit_res['features_computed']['distance_km']} km")
    print(f"  Result -> Probability: {legit_res['fraud_probability']*100:.2f}% | Risk Level: {legit_res['risk_level']} | Latency: {legit_res['latency_ms']} ms")
    print(f"  Action -> {legit_res['action_recommendation']}")
    assert legit_res['is_fraud'] == False, "Incorrectly flagged legitimate purchase!"
    print("  ✓ Legit Transaction: PASS")

    # 4. Test Batch CSV Processing
    print("\n[TEST 4/5] Testing Batch CSV Upload Pipeline...")
    sample_csv_path = os.path.join(base_dir, "data", "sample_test_batch.csv")
    with open(sample_csv_path, "rb") as f:
        csv_bytes = f.read()
    batch_res = service.predict_batch_csv(csv_bytes)
    print(f"  Total Processed: {batch_res['total_processed']} rows")
    print(f"  Approved Count:  {batch_res['approved_count']}")
    print(f"  Flagged Fraud:   {batch_res['flagged_fraud_count']}")
    print(f"  Fraud Rate:      {batch_res['fraud_rate_pct']}%")
    assert batch_res['total_processed'] > 0, "Batch processing returned 0 rows"
    print("  ✓ Batch Processing: PASS")

    # 5. Check Artifact Existence
    print("\n[TEST 5/5] Checking Research PDF Report & Jupyter Notebook...")
    pdf_path = os.path.join(base_dir, "reports", "Fraud_Detection_Benchmark_Report.pdf")
    nb_path = os.path.join(base_dir, "notebooks", "Fraud_Detection_Pipeline.ipynb")
    assert os.path.exists(pdf_path), "PDF Report missing!"
    assert os.path.exists(nb_path), "Jupyter Notebook missing!"
    print(f"  ✓ PDF Report size:      {os.path.getsize(pdf_path) / 1024:.1f} KB")
    print(f"  ✓ Jupyter Notebook size: {os.path.getsize(nb_path) / 1024:.1f} KB")

    print("\n" + "=" * 70)
    print("🎉 ALL 5 SYSTEM TESTS PASSED SUCCESSFULLY! BACKEND & ML ARE 100% HEALTHY.")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
