# API Documentation - Credit Risk Prediction API

## Overview

The Credit Risk Prediction API provides a RESTful interface for predicting loan default probability using trained machine learning models. The API is built with FastAPI and serves the best-performing model (Random Forest) from the model evaluation phase.

## Quick Start

### Start the API Server

```bash
python scripts/run_api.py
```

The API will be available at:
- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Get model information
curl http://localhost:8000/model/info

# Make a prediction (see example below)
```

## API Endpoints

### 1. Root Endpoint

**GET** `/`

Returns basic API information.

**Response:**
```json
{
  "message": "Credit Risk Prediction API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

### 2. Health Check

**GET** `/health`

Checks if the API and model are ready to serve predictions.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "random_forest_tuned"
}
```

### 3. Model Information

**GET** `/model/info`

Returns information about the loaded model, including feature names and count.

**Response:**
```json
{
  "model_name": "random_forest_tuned",
  "feature_count": 85,
  "timestamp": "2026-01-11T23:24:21.224719",
  "feature_names": ["loan_amnt", "funded_amnt", ...]
}
```

### 4. Predict Default Probability

**POST** `/predict`

Predicts the probability of loan default based on loan application data.

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "prediction": 0,
  "default_probability": 0.15,
  "status": "success"
}
```

**Response Fields:**
- `prediction`: Binary prediction (0 = no default, 1 = default)
- `default_probability`: Probability of default (0.0 to 1.0)
- `status`: Status of the prediction ("success")

## Input Features

The API expects 85 features. The most important required features are:

### Required Features
- `loan_amnt`: Loan amount requested (≥ 0)
- `funded_amnt`: Total amount committed to the loan (≥ 0)
- `funded_amnt_inv`: Total amount committed by investors (≥ 0)
- `int_rate`: Interest rate on the loan (0-100)
- `installment`: Monthly payment owed (≥ 0)
- `annual_inc`: Annual income (≥ 0)
- `dti`: Debt-to-income ratio (≥ 0)
- `delinq_2yrs`: Number of delinquencies in past 2 years (≥ 0)
- `fico_range_low`: Lower bound of FICO range (300-850)
- `fico_range_high`: Upper bound of FICO range (300-850, must be ≥ fico_range_low)
- `inq_last_6mths`: Number of inquiries in last 6 months (≥ 0)
- `open_acc`: Number of open credit lines (≥ 0)
- `pub_rec`: Number of derogatory public records (≥ 0)
- `revol_bal`: Total credit revolving balance (≥ 0)
- `total_acc`: Total number of credit accounts (≥ 0)

### Optional Features
Many additional features are optional and will use default values if not provided. See the interactive API documentation at `/docs` for a complete list.

### Engineered Features
The following features are automatically computed if not provided:
- `funded_to_loan_ratio`: `funded_amnt / loan_amnt`
- `installment_to_income_ratio`: `installment / (annual_inc / 12)`
- `fico_midpoint`: `(fico_range_low + fico_range_high) / 2`
- `fico_range`: `fico_range_high - fico_range_low`

## Example Usage

### Python

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# Example loan application
loan_data = {
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

# Make prediction
response = requests.post(f"{BASE_URL}/predict", json=loan_data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Default Probability: {result['default_probability']:.2%}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Error Handling

The API returns appropriate HTTP status codes:

- **200 OK**: Successful prediction
- **400 Bad Request**: Invalid input data (missing features, invalid values)
- **503 Service Unavailable**: Model not loaded
- **500 Internal Server Error**: Server error during prediction

**Example Error Response:**
```json
{
  "detail": "Missing required features: ['fico_range_low', 'fico_range_high']"
}
```

## Model Information

- **Model Type**: Random Forest Classifier (tuned)
- **Performance**: ROC-AUC = 0.977 (on test set)
- **Feature Count**: 85 features
- **Input Format**: JSON with loan application data
- **Output Format**: JSON with prediction and probability

## Production Deployment Considerations

### Environment Variables
- Set `MODEL_PATH` to specify a custom model path
- Set `API_HOST` and `API_PORT` for custom host/port configuration

### Security
- Add authentication/authorization for production use
- Use HTTPS in production
- Implement rate limiting
- Add request logging and monitoring

### Scaling
- Use a production ASGI server (e.g., Gunicorn with Uvicorn workers)
- Consider containerization (Docker)
- Implement load balancing for high traffic
- Add caching for frequently requested predictions

### Monitoring
- Monitor API response times
- Track prediction distributions
- Alert on model performance degradation
- Log all predictions for audit purposes

## Next Steps

1. **Add Authentication**: Implement API keys or OAuth2
2. **Add Monitoring**: Integrate with monitoring tools (Prometheus, Grafana)
3. **Add Logging**: Structured logging for production
4. **Add Testing**: Unit and integration tests for API endpoints
5. **Add Docker**: Containerize the application for easy deployment
6. **Add CI/CD**: Automate testing and deployment
