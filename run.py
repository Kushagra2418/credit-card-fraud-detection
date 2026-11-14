import os
import sys
import uvicorn

def main():
    print("=" * 70)
    print("🛡️  SENTINEL AI — CREDIT CARD FRAUD DETECTION PLATFORM")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "champion_fraud_model.joblib")
    
    if not os.path.exists(model_path):
        print("\n[!] Pretrained model not found. Running training and benchmarking pipeline...")
        import subprocess
        train_script = os.path.join(base_dir, "pipeline", "train_and_benchmark.py")
        subprocess.run([sys.executable, train_script], check=True)
        
        pdf_script = os.path.join(base_dir, "pipeline", "generate_pdf_report.py")
        subprocess.run([sys.executable, pdf_script], check=True)

    print("\n[+] Starting FastAPI server and Web Dashboard...")
    print("    - Web Dashboard:           http://127.0.0.1:8000")
    print("    - Interactive API Docs:    http://127.0.0.1:8000/docs")
    print("    - Research PDF Report:     http://127.0.0.1:8000/api/download-report")
    print("=" * 70)
    print("Press CTRL+C to stop the server.\n")

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()