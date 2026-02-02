# Production Deployment Checklist

This checklist ensures the ML pipeline is ready for production deployment.

## Pre-Deployment Checklist

### ✅ Code Quality
- [ ] All tests pass (`pytest tests/`)
- [ ] Code linting passes (`ruff check src/ scripts/`)
- [ ] Code formatting is correct (`black --check src/ scripts/`)
- [ ] No hardcoded secrets or credentials
- [ ] Environment variables are properly configured
- [ ] API documentation is up to date

### ✅ Model Validation
- [ ] Model performance meets business requirements (ROC-AUC > 0.95)
- [ ] Model evaluated on held-out test set (temporal split)
- [ ] Model interpretability analysis completed (SHAP)
- [ ] Model registered in artifact registry
- [ ] Model metadata includes feature names and versions
- [ ] Model file integrity verified (hash check)

### ✅ Data Pipeline
- [ ] Data validation pipeline tested
- [ ] Data cleaning handles edge cases
- [ ] Feature engineering is deterministic
- [ ] Temporal split prevents data leakage
- [ ] Data quality checks in place

### ✅ API & Infrastructure
- [ ] API health check endpoint working (`/health`)
- [ ] Model loading works correctly
- [ ] Input validation is comprehensive
- [ ] Error handling is robust
- [ ] API documentation accessible (`/docs`)
- [ ] Monitoring endpoints configured (`/metrics`, `/stats`)
- [ ] Logging is properly configured

### ✅ Monitoring & Observability
- [ ] Prometheus metrics endpoint available (`/metrics`)
- [ ] Request tracking implemented
- [ ] Error rate monitoring configured
- [ ] Latency monitoring configured
- [ ] Data drift detection enabled
- [ ] Alerts configured for critical metrics

### ✅ Security
- [ ] Input validation prevents injection attacks
- [ ] Rate limiting configured (if applicable)
- [ ] Authentication/authorization in place (if required)
- [ ] Secrets management implemented
- [ ] API endpoints are properly secured

### ✅ Documentation
- [ ] README.md is up to date
- [ ] API documentation is complete
- [ ] Operations runbook is available
- [ ] Deployment procedures documented
- [ ] Rollback procedures documented

## Deployment Steps

1. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Model Deployment**
   ```bash
   # Ensure model is in models/ directory
   # Model should be: random_forest_tuned.pkl
   # Metadata should be: random_forest_tuned_metadata.json
   ```

3. **Start API Server**
   ```bash
   python scripts/run_api.py
   ```

4. **Verify Deployment**
   - Check health: `curl http://localhost:8000/health`
   - Check metrics: `curl http://localhost:8000/metrics`
   - Test prediction: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @test_request.json`

## Post-Deployment Validation

- [ ] Health check returns "healthy"
- [ ] Model info endpoint returns correct metadata
- [ ] Prediction endpoint returns valid responses
- [ ] Metrics endpoint is accessible
- [ ] Logs show no errors
- [ ] Monitoring dashboards show expected metrics

## Rollback Procedure

If deployment fails:

1. **Stop the API server**
2. **Revert to previous model version** (if model changed)
3. **Check logs** for error details
4. **Verify previous model** is still available
5. **Restart API server** with previous model

To rollback a model:
```bash
python scripts/promote_model.py --model-name random_forest --version-id <previous_version_id>
```

## Production Environment Recommendations

### Infrastructure
- Use a production-grade WSGI/ASGI server (Gunicorn + Uvicorn workers)
- Deploy behind a reverse proxy (Nginx, Traefik)
- Use container orchestration (Docker, Kubernetes)
- Implement auto-scaling based on load

### Monitoring
- Set up Prometheus + Grafana for metrics visualization
- Configure alerting for:
  - High error rates (> 5%)
  - High latency (> 1s p95)
  - Model drift detected
  - API unavailability

### Data Management
- Implement data versioning (DVC, MLflow)
- Track dataset changes
- Monitor data quality metrics
- Set up data pipeline alerts

### Model Management
- Use artifact registry for model versioning
- Implement A/B testing for new models
- Set up automated retraining pipeline
- Monitor model performance in production

## Maintenance

### Regular Tasks
- **Weekly**: Review monitoring dashboards
- **Monthly**: Check for data drift
- **Quarterly**: Retrain models with new data
- **As needed**: Update dependencies and security patches

### Model Retraining
```bash
# Run full retraining pipeline
python scripts/retrain_model.py
```

This will:
1. Process latest data
2. Train and tune models
3. Evaluate on test set
4. Register new models
5. Promote best model to production

## Support & Troubleshooting

See [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) for detailed troubleshooting procedures.
