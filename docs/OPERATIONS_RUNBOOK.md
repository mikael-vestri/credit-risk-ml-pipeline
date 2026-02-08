# Operations Runbook

This runbook provides procedures for operating and troubleshooting the Credit Risk ML Pipeline in production.

## Table of Contents

1. [System Overview](#system-overview)
2. [Health Checks](#health-checks)
3. [Monitoring](#monitoring)
4. [Common Issues & Solutions](#common-issues--solutions)
5. [Model Management](#model-management)
6. [Data Pipeline Operations](#data-pipeline-operations)
7. [Emergency Procedures](#emergency-procedures)

## System Overview

### Architecture
- **API Server**: FastAPI application serving predictions
- **Model Registry**: Local artifact registry for model versioning
- **Monitoring**: Prometheus metrics + custom stats endpoint
- **Data Pipeline**: ETL pipeline for data processing

### Key Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /model/info` - Model information
- `POST /predict` - Make predictions
- `GET /metrics` - Prometheus metrics
- `GET /stats` - API statistics

### Key Directories
- `models/` - Trained model files (.pkl)
- `artifacts/registry/` - Model registry (registry.json)
- `data/processed/` - Processed datasets
- `docs/` - Documentation and evaluation results

## Health Checks

### Basic Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "random_forest_tuned"
}
```

### Detailed Health Check
```bash
# Check model info
curl http://localhost:8000/model/info

# Check API stats
curl http://localhost:8000/stats
```

### Health Check Interpretation

| Status | Model Loaded | Meaning | Action |
|--------|-------------|---------|--------|
| healthy | true | System operational | None |
| unhealthy | false | Model not loaded | Check model files, restart API |
| unhealthy | true | API error | Check logs, restart API |

## Monitoring

### Prometheus Metrics

Access metrics endpoint:
```bash
curl http://localhost:8000/metrics
```

Key metrics to monitor:

1. **`http_requests_total`** - Total HTTP requests by endpoint and status
2. **`http_request_duration_seconds`** - Request latency (p50, p95, p99)
3. **`predictions_total`** - Total predictions by outcome
4. **`prediction_probability`** - Distribution of prediction probabilities
5. **`model_loaded`** - Whether model is loaded (1) or not (0)

### API Statistics

Get current statistics:
```bash
curl http://localhost:8000/stats
```

Response includes:
- Request counts per endpoint
- Error rates
- Average latency

### Monitoring Alerts

Set up alerts for:

1. **High Error Rate** (> 5%)
   - Check logs for error patterns
   - Verify input data format
   - Check model integrity

2. **High Latency** (p95 > 1s)
   - Check system resources (CPU, memory)
   - Review prediction complexity
   - Consider model optimization

3. **Data Drift Detected**
   - Review drift report
   - Consider retraining model
   - Validate input data sources

4. **Model Not Loaded**
   - Check model file exists
   - Verify file permissions
   - Check model file integrity

## Common Issues & Solutions

### Issue: Model Not Loading

**Symptoms:**
- Health check returns `model_loaded: false`
- API returns 503 on `/predict`

**Diagnosis:**
```bash
# Check if model file exists
ls -lh models/random_forest_tuned.pkl

# Check model metadata
cat models/random_forest_tuned_metadata.json
```

**Solutions:**
1. Verify model file exists and is readable
2. Check file permissions: `chmod 644 models/*.pkl`
3. Verify model was trained successfully
4. Check logs for loading errors

### Issue: High Error Rate

**Symptoms:**
- `/stats` shows error_rate > 5%
- Many 400/500 responses

**Diagnosis:**
```bash
# Check recent errors in logs
tail -n 100 logs/app.log | grep ERROR

# Check API stats
curl http://localhost:8000/stats
```

**Common Causes:**
1. **Invalid input data** - Check input validation
2. **Missing features** - Verify feature names match model
3. **Type mismatches** - Check data types in input
4. **Model version mismatch** - Verify model and feature engineering match

**Solutions:**
1. Review error messages in logs
2. Validate input data format
3. Check feature engineering logic
4. Verify model metadata matches current code

### Issue: Data Drift

**Symptoms:**
- Warnings in logs about data drift
- `/stats` shows drift alerts

**Diagnosis:**
```bash
# Check drift detection results (in logs)
grep "Data drift" logs/app.log
```

**Solutions:**
1. **Review drift report** - Identify which features drifted
2. **Investigate data source** - Check if input data changed
3. **Retrain model** - If drift is significant
4. **Update reference statistics** - If drift is expected

### Issue: Slow Predictions

**Symptoms:**
- High latency in `/stats`
- p95 latency > 1s

**Diagnosis:**
```bash
# Check system resources
top  # or htop

# Check prediction latency
curl -w "@curl-format.txt" -X POST http://localhost:8000/predict ...
```

**Solutions:**
1. **Check CPU usage** - May need more resources
2. **Check memory** - Model may be too large
3. **Optimize model** - Consider model compression
4. **Scale horizontally** - Add more API instances

## Model Management

### List Models in Registry

```bash
python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from registry.artifact_registry import ArtifactRegistry

registry = ArtifactRegistry(Path('artifacts/registry'))
summary = registry.get_registry_summary()
print(f'Total models: {summary[\"total_models\"]}')
print(f'Production models: {summary[\"production_models\"]}')
"
```

### Promote Model to Production (MLflow)

After retraining, new model versions are in Staging. Promote a version to Production:

```bash
# Promote version 5 to Production; optionally archive current Production
python scripts/promote_model.py --version 5 --archive-current

# With custom model name or tracking URI
python scripts/promote_model.py --version 5 --model-name credit-risk-model --tracking-uri http://mlflow:5000
```

See [docs/MLFLOW.md](MLFLOW.md) for details.

### Rollback Model

To rollback to a previous Production version:

```bash
# 1. List versions in MLflow UI or: mlflow models list --name credit-risk-model (if CLI available)

# 2. Promote the previous version to Production (current Production can be archived)
python scripts/promote_model.py --version <previous_version> --archive-current

# 3. Restart API server if serving from path (production.pkl); if serving from MLflow, it will use the new Production version
```

### Retrain Model

Run full retraining pipeline:

```bash
python scripts/retrain_model.py
```

This will:
1. Process latest data
2. Train and tune all models
3. Evaluate on test set
4. Register new models
5. Promote best model to production

**Note**: Retraining may take 30+ minutes depending on data size.

## Data Pipeline Operations

### Run Data Cleaning

```bash
# DEV mode (fast, for testing)
python scripts/clean_data.py

# FULL mode (production)
# Edit scripts/clean_data.py to set mode="full"
python scripts/clean_data.py
```

### Run Feature Engineering

```bash
python scripts/engineer_features.py
```

### Validate Data Quality

```bash
python scripts/validate_dataset.py
```

## Emergency Procedures

### Complete System Failure

1. **Stop API server**
   ```bash
   # Find process
   ps aux | grep "run_api.py"
   # Kill process
   kill <PID>
   ```

2. **Check logs**
   ```bash
   tail -n 100 logs/app.log
   ```

3. **Restore from backup** (if needed)
   - Restore model files
   - Restore registry

4. **Restart API server**
   ```bash
   python scripts/run_api.py
   ```

### Model Corruption

1. **Identify corrupted model**
   ```bash
   # Check model file hash
   sha256sum models/random_forest_tuned.pkl
   ```

2. **Restore from registry**
   - Check registry for previous version
   - Copy previous model file
   - Update model path in API

3. **Or retrain model**
   ```bash
   python scripts/retrain_model.py
   ```

### Data Pipeline Failure

1. **Check data files**
   ```bash
   ls -lh data/raw/
   ls -lh data/processed/
   ```

2. **Review validation report**
   ```bash
   cat docs/validation_report.json
   ```

3. **Re-run pipeline**
   ```bash
   python scripts/clean_data.py
   python scripts/engineer_features.py
   ```

## Logging

### Log Locations
- Application logs: `logs/app.log` (if configured)
- Console output: Standard output/error

### Log Levels
- `INFO`: Normal operations
- `WARNING`: Non-critical issues (e.g., data drift)
- `ERROR`: Errors that need attention
- `CRITICAL`: System failures

### Viewing Logs
```bash
# Follow logs in real-time
tail -f logs/app.log

# Search for errors
grep ERROR logs/app.log

# Search for specific endpoint
grep "/predict" logs/app.log
```

## Performance Tuning

### API Server
- Use production WSGI server: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.app:app`
- Set appropriate worker count based on CPU cores
- Enable request timeouts

### Model Optimization
- Use model compression (quantization, pruning)
- Cache frequently used features
- Consider model serving frameworks (TensorFlow Serving, TorchServe)

## Support Contacts

- **Technical Issues**: Check logs and this runbook
- **Model Questions**: Review model evaluation results in `docs/evaluation/`
- **Data Issues**: Review validation report in `docs/validation_report.json`
