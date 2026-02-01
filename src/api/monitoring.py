"""
Monitoring and metrics for the API.

This module provides Prometheus metrics, request tracking, and data drift detection.
"""

import time
import logging
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available. Metrics will be disabled.")

logger = logging.getLogger(__name__)

# Prometheus metrics (if available)
if PROMETHEUS_AVAILABLE:
    # Request metrics
    http_requests_total = Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status']
    )
    
    http_request_duration_seconds = Histogram(
        'http_request_duration_seconds',
        'HTTP request duration in seconds',
        ['method', 'endpoint'],
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    )
    
    # Prediction metrics
    predictions_total = Counter(
        'predictions_total',
        'Total predictions made',
        ['prediction']
    )
    
    prediction_probability = Histogram(
        'prediction_probability',
        'Distribution of prediction probabilities',
        buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    
    # Model metrics
    model_loaded = Gauge(
        'model_loaded',
        'Whether model is loaded (1) or not (0)'
    )
else:
    # Dummy metrics if prometheus_client not available
    http_requests_total = None
    http_request_duration_seconds = None
    predictions_total = None
    prediction_probability = None
    model_loaded = None


class RequestMetrics:
    """Track request metrics for monitoring."""
    
    def __init__(self):
        self.request_count = defaultdict(int)
        self.error_count = defaultdict(int)
        self.latency_sum = defaultdict(float)
        self.latency_count = defaultdict(int)
        self.last_reset = datetime.now()
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record a request."""
        key = f"{method} {endpoint}"
        self.request_count[key] += 1
        self.latency_sum[key] += duration
        self.latency_count[key] += 1
        
        if status >= 400:
            self.error_count[key] += 1
        
        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE and http_requests_total:
            http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            if http_request_duration_seconds:
                http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_prediction(self, prediction: int, probability: float):
        """Record a prediction."""
        if PROMETHEUS_AVAILABLE:
            if predictions_total:
                predictions_total.labels(prediction=str(prediction)).inc()
            if prediction_probability:
                prediction_probability.observe(probability)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = {}
        for key in self.request_count:
            count = self.request_count[key]
            errors = self.error_count.get(key, 0)
            avg_latency = (
                self.latency_sum[key] / self.latency_count[key]
                if self.latency_count[key] > 0 else 0
            )
            stats[key] = {
                "total_requests": count,
                "errors": errors,
                "error_rate": errors / count if count > 0 else 0,
                "avg_latency_ms": avg_latency * 1000,
            }
        return stats


# Global metrics instance
request_metrics = RequestMetrics()


class DataDriftDetector:
    """
    Simple data drift detection.
    
    Tracks feature distributions and alerts when input data deviates significantly
    from training data distribution.
    """
    
    def __init__(self, reference_stats: Optional[Dict[str, Any]] = None):
        """
        Initialize drift detector.
        
        Parameters
        ----------
        reference_stats : Optional[Dict[str, Any]]
            Reference statistics from training data (mean, std, min, max per feature)
        """
        self.reference_stats = reference_stats or {}
        self.drift_alerts = []
        self.drift_threshold = 3.0  # Z-score threshold
    
    def update_reference(self, reference_stats: Dict[str, Any]):
        """Update reference statistics from training data."""
        self.reference_stats = reference_stats
    
    def check_drift(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for data drift in incoming features.
        
        Parameters
        ----------
        features : pd.DataFrame
            Input features to check
            
        Returns
        -------
        Dict with drift detection results
        """
        if not self.reference_stats:
            return {
                "drift_detected": False,
                "message": "No reference statistics available"
            }
        
        drift_results = {
            "drift_detected": False,
            "features_checked": 0,
            "features_with_drift": [],
            "max_z_score": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        for col in features.columns:
            if col not in self.reference_stats:
                continue
            
            ref_mean = self.reference_stats[col].get("mean")
            ref_std = self.reference_stats[col].get("std")
            
            if ref_mean is None or ref_std is None or ref_std == 0:
                continue
            
            drift_results["features_checked"] += 1
            
            # Calculate Z-score for current value(s)
            current_values = features[col].values
            z_scores = np.abs((current_values - ref_mean) / ref_std)
            max_z = np.max(z_scores)
            
            if max_z > drift_results["max_z_score"]:
                drift_results["max_z_score"] = float(max_z)
            
            if max_z > self.drift_threshold:
                drift_results["drift_detected"] = True
                drift_results["features_with_drift"].append({
                    "feature": col,
                    "z_score": float(max_z),
                    "reference_mean": float(ref_mean),
                    "current_value": float(current_values[0]) if len(current_values) == 1 else None
                })
                logger.warning(
                    f"Data drift detected in feature {col}: "
                    f"z-score={max_z:.2f} (threshold={self.drift_threshold})"
                )
        
        return drift_results


# Global drift detector (will be initialized with training stats)
drift_detector = DataDriftDetector()


def get_metrics_response():
    """Get Prometheus metrics response."""
    if PROMETHEUS_AVAILABLE:
        return generate_latest(), CONTENT_TYPE_LATEST
    else:
        return {"error": "Prometheus metrics not available"}, "application/json"


def update_model_loaded_status(is_loaded: bool):
    """Update model loaded gauge."""
    if PROMETHEUS_AVAILABLE and model_loaded is not None:
        model_loaded.set(1 if is_loaded else 0)
