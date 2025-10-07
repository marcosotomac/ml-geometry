"""
Model monitoring for production deployments
"""

import time
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import json


class ModelMonitor:
    """
    Monitor model performance in production
    """

    def __init__(self, model_name: str = "ml-geometry", window_size: int = 1000):
        """
        Initialize model monitor

        Args:
            model_name: Name of the model
            window_size: Size of sliding window for metrics
        """
        self.model_name = model_name
        self.window_size = window_size

        # Metrics storage
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)

        # Prometheus metrics
        self.prediction_counter = Counter(
            f'{model_name}_predictions_total',
            'Total number of predictions',
            ['class']
        )

        self.latency_histogram = Histogram(
            f'{model_name}_prediction_latency_seconds',
            'Prediction latency in seconds'
        )

        self.confidence_gauge = Gauge(
            f'{model_name}_average_confidence',
            'Average prediction confidence'
        )

        self.error_counter = Counter(
            f'{model_name}_errors_total',
            'Total number of errors',
            ['error_type']
        )

        self.drift_gauge = Gauge(
            f'{model_name}_prediction_drift',
            'Prediction distribution drift score'
        )

    def log_prediction(self,
                       predicted_class: str,
                       confidence: float,
                       latency: float,
                       true_label: Optional[str] = None):
        """
        Log a prediction

        Args:
            predicted_class: Predicted class
            confidence: Prediction confidence
            latency: Inference latency
            true_label: True label if available
        """
        # Store metrics
        self.predictions.append({
            'class': predicted_class,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'true_label': true_label
        })
        self.confidences.append(confidence)
        self.latencies.append(latency)

        # Update Prometheus metrics
        self.prediction_counter.labels(class_=predicted_class).inc()
        self.latency_histogram.observe(latency)
        self.confidence_gauge.set(np.mean(list(self.confidences)))

    def log_error(self, error_type: str, error_message: str):
        """
        Log an error

        Args:
            error_type: Type of error
            error_message: Error message
        """
        self.errors.append({
            'type': error_type,
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        })
        self.error_counter.labels(error_type=error_type).inc()

    def get_metrics(self) -> Dict:
        """
        Get current metrics

        Returns:
            Dictionary of metrics
        """
        if not self.predictions:
            return {}

        # Calculate metrics
        avg_confidence = np.mean(list(self.confidences))
        avg_latency = np.mean(list(self.latencies))
        p50_latency = np.percentile(list(self.latencies), 50)
        p95_latency = np.percentile(list(self.latencies), 95)
        p99_latency = np.percentile(list(self.latencies), 99)

        # Class distribution
        class_dist = {}
        for pred in self.predictions:
            class_name = pred['class']
            class_dist[class_name] = class_dist.get(class_name, 0) + 1

        # Accuracy (if true labels available)
        accuracy = None
        if any(p['true_label'] for p in self.predictions):
            correct = sum(1 for p in self.predictions
                          if p['true_label'] and p['class'] == p['true_label'])
            total = sum(1 for p in self.predictions if p['true_label'])
            accuracy = correct / total if total > 0 else None

        return {
            'total_predictions': len(self.predictions),
            'average_confidence': float(avg_confidence),
            'average_latency_ms': float(avg_latency * 1000),
            'p50_latency_ms': float(p50_latency * 1000),
            'p95_latency_ms': float(p95_latency * 1000),
            'p99_latency_ms': float(p99_latency * 1000),
            'class_distribution': class_dist,
            'accuracy': accuracy,
            'total_errors': len(self.errors),
            'window_size': self.window_size
        }

    def detect_drift(self, baseline_distribution: Dict[str, float]) -> float:
        """
        Detect prediction drift using KL divergence

        Args:
            baseline_distribution: Baseline class distribution

        Returns:
            Drift score
        """
        if not self.predictions:
            return 0.0

        # Calculate current distribution
        current_dist = {}
        for pred in self.predictions:
            class_name = pred['class']
            current_dist[class_name] = current_dist.get(class_name, 0) + 1

        # Normalize
        total = sum(current_dist.values())
        current_dist = {k: v/total for k, v in current_dist.items()}

        # Calculate KL divergence
        drift = 0.0
        for class_name in baseline_distribution:
            p = baseline_distribution.get(class_name, 1e-10)
            q = current_dist.get(class_name, 1e-10)
            drift += p * np.log(p / q)

        self.drift_gauge.set(drift)

        return drift

    def get_prometheus_metrics(self) -> bytes:
        """
        Get Prometheus-formatted metrics

        Returns:
            Prometheus metrics
        """
        return generate_latest()

    def save_report(self, output_path: str):
        """
        Save monitoring report

        Args:
            output_path: Path to save report
        """
        metrics = self.get_metrics()

        report = {
            'model_name': self.model_name,
            'generated_at': datetime.now().isoformat(),
            'metrics': metrics,
            'recent_errors': list(self.errors)[-10:]  # Last 10 errors
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)

        print(f"✅ Monitoring report saved to: {output_path}")
