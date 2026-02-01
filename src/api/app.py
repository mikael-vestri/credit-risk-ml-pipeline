"""
FastAPI application for credit risk prediction API.

This module provides a REST API for serving credit risk predictions.
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from api.serving import (
    load_model_from_disk,
    load_model_metadata,
    prepare_input_features,
    predict_default_probability,
    get_model_info
)
from api.monitoring import (
    request_metrics,
    drift_detector,
    get_metrics_response,
    update_model_loaded_status
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting loan default probability using ML models",
    version="1.0.0"
)


# Middleware for request tracking
class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        method = request.method
        path = request.url.path
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            duration = time.time() - start_time
            
            request_metrics.record_request(method, path, status_code, duration)
            
            return response
        except Exception as e:
            status_code = 500
            duration = time.time() - start_time
            request_metrics.record_request(method, path, status_code, duration)
            raise


app.add_middleware(MetricsMiddleware)

# Global variables for model and metadata
MODEL = None
MODEL_METADATA = None
MODEL_INFO = None


# Pydantic models for input validation
class LoanApplication(BaseModel):
    """Loan application input model."""
    
    # Loan details
    loan_amnt: float = Field(..., ge=0, description="Loan amount requested")
    funded_amnt: float = Field(..., ge=0, description="Total amount committed to the loan")
    funded_amnt_inv: float = Field(..., ge=0, description="Total amount committed by investors")
    int_rate: float = Field(..., ge=0, le=100, description="Interest rate on the loan")
    installment: float = Field(..., ge=0, description="Monthly payment owed")
    
    # Borrower details
    annual_inc: float = Field(..., ge=0, description="Annual income")
    dti: float = Field(..., ge=0, description="Debt-to-income ratio")
    
    # Credit history
    delinq_2yrs: int = Field(..., ge=0, description="Number of delinquencies in past 2 years")
    fico_range_low: int = Field(..., ge=300, le=850, description="Lower bound of FICO range")
    fico_range_high: int = Field(..., ge=300, le=850, description="Upper bound of FICO range")
    inq_last_6mths: int = Field(..., ge=0, description="Number of inquiries in last 6 months")
    mths_since_last_delinq: Optional[float] = Field(None, ge=0, description="Months since last delinquency")
    mths_since_last_record: Optional[float] = Field(None, ge=0, description="Months since last public record")
    open_acc: int = Field(..., ge=0, description="Number of open credit lines")
    pub_rec: int = Field(..., ge=0, description="Number of derogatory public records")
    revol_bal: float = Field(..., ge=0, description="Total credit revolving balance")
    revol_util: Optional[float] = Field(None, ge=0, le=100, description="Revolving line utilization rate")
    total_acc: int = Field(..., ge=0, description="Total number of credit accounts")
    last_fico_range_high: Optional[int] = Field(None, ge=300, le=850, description="Last reported FICO range high")
    last_fico_range_low: Optional[int] = Field(None, ge=300, le=850, description="Last reported FICO range low")
    
    # Additional features (with defaults for optional fields)
    collections_12_mths_ex_med: int = Field(0, ge=0, description="Collections in last 12 months")
    mths_since_last_major_derog: Optional[float] = Field(None, ge=0)
    policy_code: int = Field(1, description="Policy code")
    acc_now_delinq: int = Field(0, ge=0, description="Accounts currently delinquent")
    tot_coll_amt: Optional[float] = Field(None, ge=0)
    tot_cur_bal: Optional[float] = Field(None, ge=0)
    open_acc_6m: Optional[int] = Field(None, ge=0)
    open_act_il: Optional[int] = Field(None, ge=0)
    open_il_12m: Optional[int] = Field(None, ge=0)
    open_il_24m: Optional[int] = Field(None, ge=0)
    mths_since_rcnt_il: Optional[float] = Field(None, ge=0)
    total_bal_il: Optional[float] = Field(None, ge=0)
    il_util: Optional[float] = Field(None, ge=0, le=100)
    open_rv_12m: Optional[int] = Field(None, ge=0)
    open_rv_24m: Optional[int] = Field(None, ge=0)
    max_bal_bc: Optional[float] = Field(None, ge=0)
    all_util: Optional[float] = Field(None, ge=0, le=100)
    total_rev_hi_lim: Optional[float] = Field(None, ge=0)
    inq_fi: Optional[int] = Field(None, ge=0)
    total_cu_tl: Optional[int] = Field(None, ge=0)
    inq_last_12m: Optional[int] = Field(None, ge=0)
    acc_open_past_24mths: Optional[int] = Field(None, ge=0)
    avg_cur_bal: Optional[float] = Field(None, ge=0)
    bc_open_to_buy: Optional[float] = Field(None, ge=0)
    bc_util: Optional[float] = Field(None, ge=0, le=100)
    chargeoff_within_12_mths: int = Field(0, ge=0)
    delinq_amnt: Optional[float] = Field(None, ge=0)
    mo_sin_old_il_acct: Optional[float] = Field(None, ge=0)
    mo_sin_old_rev_tl_op: Optional[float] = Field(None, ge=0)
    mo_sin_rcnt_rev_tl_op: Optional[float] = Field(None, ge=0)
    mo_sin_rcnt_tl: Optional[float] = Field(None, ge=0)
    mort_acc: Optional[int] = Field(None, ge=0)
    mths_since_recent_bc: Optional[float] = Field(None, ge=0)
    mths_since_recent_bc_dlq: Optional[float] = Field(None, ge=0)
    mths_since_recent_inq: Optional[float] = Field(None, ge=0)
    mths_since_recent_revol_delinq: Optional[float] = Field(None, ge=0)
    num_accts_ever_120_pd: Optional[int] = Field(None, ge=0)
    num_actv_bc_tl: Optional[int] = Field(None, ge=0)
    num_actv_rev_tl: Optional[int] = Field(None, ge=0)
    num_bc_sats: Optional[int] = Field(None, ge=0)
    num_bc_tl: Optional[int] = Field(None, ge=0)
    num_il_tl: Optional[int] = Field(None, ge=0)
    num_op_rev_tl: Optional[int] = Field(None, ge=0)
    num_rev_accts: Optional[int] = Field(None, ge=0)
    num_rev_tl_bal_gt_0: Optional[int] = Field(None, ge=0)
    num_sats: Optional[int] = Field(None, ge=0)
    num_tl_120dpd_2m: Optional[int] = Field(None, ge=0)
    num_tl_30dpd: Optional[int] = Field(None, ge=0)
    num_tl_90g_dpd_24m: Optional[int] = Field(None, ge=0)
    num_tl_op_past_12m: Optional[int] = Field(None, ge=0)
    pct_tl_nvr_dlq: Optional[float] = Field(None, ge=0, le=100)
    percent_bc_gt_75: Optional[float] = Field(None, ge=0, le=100)
    pub_rec_bankruptcies: Optional[int] = Field(None, ge=0)
    tax_liens: Optional[int] = Field(None, ge=0)
    tot_hi_cred_lim: Optional[float] = Field(None, ge=0)
    total_bal_ex_mort: Optional[float] = Field(None, ge=0)
    total_bc_limit: Optional[float] = Field(None, ge=0)
    total_il_high_credit_limit: Optional[float] = Field(None, ge=0)
    
    # Engineered features (will be computed if not provided)
    funded_to_loan_ratio: Optional[float] = Field(None, ge=0)
    installment_to_income_ratio: Optional[float] = Field(None, ge=0)
    fico_midpoint: Optional[float] = Field(None, ge=300, le=850)
    fico_range: Optional[float] = Field(None, ge=0)
    credit_history_years: Optional[float] = Field(None, ge=0)
    issue_year: Optional[int] = Field(None, ge=2007, le=2018)
    issue_month: Optional[int] = Field(None, ge=1, le=12)
    
    @validator('fico_range_high')
    def validate_fico_range(cls, v, values):
        """Ensure fico_range_high >= fico_range_low."""
        if 'fico_range_low' in values and v < values['fico_range_low']:
            raise ValueError('fico_range_high must be >= fico_range_low')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 10000,
                "funded_amnt": 10000,
                "funded_amnt_inv": 10000,
                "int_rate": 10.5,
                "installment": 300.0,
                "annual_inc": 50000,
                "dti": 15.5,
                "delinq_2yrs": 0,
                "fico_range_low": 700,
                "fico_range_high": 704,
                "inq_last_6mths": 1,
                "open_acc": 10,
                "pub_rec": 0,
                "revol_bal": 5000,
                "revol_util": 30.0,
                "total_acc": 20
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model."""
    prediction: int = Field(..., description="Binary prediction: 0 = no default, 1 = default")
    default_probability: float = Field(..., ge=0, le=1, description="Probability of default (0-1)")
    status: str = Field(..., description="Status of the prediction")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


# Startup event: Load model
@app.on_event("startup")
async def load_model():
    """Load the model on application startup."""
    global MODEL, MODEL_METADATA, MODEL_INFO
    
    try:
        models_dir = project_root / "models"
        
        # Load best model (Random Forest based on evaluation)
        model_name = "random_forest_tuned"
        model_path = models_dir / f"{model_name}.pkl"
        metadata_path = models_dir / f"{model_name}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model: {model_name}")
        MODEL = load_model_from_disk(model_path)
        MODEL_METADATA = load_model_metadata(metadata_path)
        MODEL_INFO = get_model_info(MODEL_METADATA)
        
        logger.info(f"Model loaded successfully: {MODEL_INFO['model_name']}")
        logger.info(f"Model expects {MODEL_INFO['feature_count']} features")
        
        # Update monitoring
        update_model_loaded_status(True)
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        update_model_loaded_status(False)
        raise


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        model_name=MODEL_INFO["model_name"] if MODEL_INFO else None
    )


@app.get("/model/info", tags=["Model"])
async def get_model_information():
    """Get information about the loaded model."""
    if MODEL_INFO is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return MODEL_INFO


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format for scraping.
    """
    metrics_data, content_type = get_metrics_response()
    return Response(content=metrics_data, media_type=content_type)


@app.get("/stats", tags=["Monitoring"])
async def get_stats():
    """
    Get API statistics (request counts, error rates, latency).
    
    Returns JSON with current API statistics.
    """
    return {
        "request_stats": request_metrics.get_stats(),
        "timestamp": time.time()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(application: LoanApplication):
    """
    Predict loan default probability.
    
    This endpoint accepts loan application data and returns a prediction
    of whether the loan will default (1) or not (0), along with the
    probability of default.
    """
    if MODEL is None or MODEL_METADATA is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert Pydantic model to dictionary
        input_dict = application.dict()
        
        # Compute engineered features if not provided
        if input_dict.get('funded_to_loan_ratio') is None:
            input_dict['funded_to_loan_ratio'] = (
                input_dict['funded_amnt'] / input_dict['loan_amnt']
                if input_dict['loan_amnt'] > 0 else 0
            )
        
        if input_dict.get('installment_to_income_ratio') is None:
            monthly_income = input_dict['annual_inc'] / 12
            input_dict['installment_to_income_ratio'] = (
                input_dict['installment'] / monthly_income
                if monthly_income > 0 else 0
            )
        
        if input_dict.get('fico_midpoint') is None:
            input_dict['fico_midpoint'] = (
                (input_dict['fico_range_low'] + input_dict['fico_range_high']) / 2
            )
        
        if input_dict.get('fico_range') is None:
            input_dict['fico_range'] = (
                input_dict['fico_range_high'] - input_dict['fico_range_low']
            )
        
        # Prepare features
        feature_names = MODEL_METADATA['feature_names']
        features_df = prepare_input_features(input_dict, feature_names)
        
        # Check for data drift (optional, logs warnings)
        drift_result = drift_detector.check_drift(features_df)
        if drift_result.get("drift_detected"):
            logger.warning(f"Data drift detected: {drift_result}")
        
        # Make prediction
        result = predict_default_probability(MODEL, features_df, return_proba=True)
        
        # Record prediction metrics
        request_metrics.record_prediction(
            result["prediction"],
            result["default_probability"]
        )
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )
