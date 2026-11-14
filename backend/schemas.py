from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class TransactionInput(BaseModel):
    amt: float = Field(..., description="Transaction amount in USD", example=284.50)
    category: str = Field(..., description="Merchant category", example="shopping_net")
    gender: str = Field(..., description="Gender (M or F)", example="M")
    city_pop: float = Field(..., description="Cardholder city population", example=35000.0)
    lat: float = Field(..., description="Cardholder latitude", example=40.7128)
    long: float = Field(..., description="Cardholder longitude", example=-74.0060)
    merch_lat: float = Field(..., description="Merchant latitude", example=40.7589)
    merch_long: float = Field(..., description="Merchant longitude", example=-73.9851)
    trans_date_trans_time: Optional[str] = Field("2026-11-13 23:30:00", description="Transaction datetime YYYY-MM-DD HH:MM:SS")
    dob: Optional[str] = Field("1988-06-15", description="Cardholder Date of Birth YYYY-MM-DD")

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_score_pct: float
    risk_level: str  # LOW, MEDIUM, HIGH
    action_recommendation: str
    decision_threshold: float
    latency_ms: float
    features_computed: Dict[str, Any]

class BatchSummaryResponse(BaseModel):
    total_processed: int
    flagged_fraud_count: int
    approved_count: int
    fraud_rate_pct: float
    sample_flagged_rows: List[Dict[str, Any]]
