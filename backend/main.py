import os
from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.schemas import TransactionInput, PredictionResponse, BatchSummaryResponse
from backend.inference_service import FraudInferenceService

app = FastAPI(
    title="SENTINEL AI — Credit Card Fraud Detection Engine",
    description="Production-grade real-time fraud scoring API and batch processing pipeline.",
    version="1.0.0"
)

# Enable CORS for local and web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
service = FraudInferenceService(base_dir)

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "SENTINEL AI Fraud Detection Engine",
        "model_loaded": service.model is not None,
        "scaler_loaded": service.scaler is not None
    }

@app.get("/api/metrics")
def get_metrics():
    if not service.metadata:
        raise HTTPException(status_code=500, detail="Model metadata not loaded.")
    return service.metadata

@app.get("/api/sample-presets")
def get_sample_presets():
    if not service.presets:
        raise HTTPException(status_code=500, detail="Presets not loaded.")
    return service.presets

@app.post("/api/predict", response_model=PredictionResponse)
def predict_transaction(tx: TransactionInput):
    if not service.model:
        raise HTTPException(status_code=500, detail="Inference model is not initialized.")
    res = service.predict_single(tx.dict())
    return res

@app.post("/api/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV format.")
    content = await file.read()
    res = service.predict_batch_csv(content)
    return {
        "total_processed": res["total_processed"],
        "flagged_fraud_count": res["flagged_fraud_count"],
        "approved_count": res["approved_count"],
        "fraud_rate_pct": res["fraud_rate_pct"],
        "sample_flagged_rows": res["sample_flagged_rows"],
        "csv_download_ready": True
    }

@app.get("/api/download-report")
def download_pdf_report():
    pdf_path = os.path.join(base_dir, "reports", "Fraud_Detection_Benchmark_Report.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF report not found.")
    return FileResponse(
        path=pdf_path,
        filename="Fraud_Detection_Benchmark_Report.pdf",
        media_type="application/pdf"
    )

@app.get("/api/sample-batch-csv")
def download_sample_csv():
    sample_csv_path = os.path.join(base_dir, "data", "sample_test_batch.csv")
    if not os.path.exists(sample_csv_path):
        raise HTTPException(status_code=404, detail="Sample CSV not found.")
    return FileResponse(
        path=sample_csv_path,
        filename="sample_test_batch.csv",
        media_type="text/csv"
    )

# Mount frontend
frontend_dir = os.path.join(base_dir, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)