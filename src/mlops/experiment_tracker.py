"""
Experiment tracking with MLflow integration
"""

import os
import mlflow
import mlflow.tensorflow
from typing import Dict, Any, Optional
import json


class ExperimentTracker:
    """
    Track experiments using MLflow
    """

    def __init__(self,
                 experiment_name: str = "ml-geometry",
                 tracking_uri: Optional[str] = None):
        """
        Initialize experiment tracker

        Args:
            experiment_name: Name of experiment
            tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif os.getenv('MLFLOW_TRACKING_URI'):
            mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

        # Set or create experiment
        mlflow.set_experiment(experiment_name)

        self.run_id = None
        self.run_name = None

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """
        Start a new MLflow run

        Args:
            run_name: Name of the run
            tags: Dictionary of tags
        """
        self.run_name = run_name
        mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = mlflow.active_run().info.run_id

        print(f"▶️  Started run: {run_name or self.run_id}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters

        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics

        Args:
            metrics: Dictionary of metrics
            step: Step number
        """
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, file_path: str):
        """
        Log an artifact

        Args:
            file_path: Path to artifact
        """
        mlflow.log_artifact(file_path)

    def log_model(self, model, model_name: str = "model", **kwargs):
        """
        Log TensorFlow/Keras model

        Args:
            model: Trained model
            model_name: Name of the model
            **kwargs: Additional arguments for mlflow.tensorflow.log_model
        """
        mlflow.tensorflow.log_model(model, model_name, **kwargs)

    def log_dict(self, dictionary: Dict, filename: str):
        """
        Log dictionary as JSON

        Args:
            dictionary: Dictionary to log
            filename: Output filename
        """
        temp_path = f"/tmp/{filename}"
        with open(temp_path, 'w') as f:
            json.dump(dictionary, f, indent=4)
        mlflow.log_artifact(temp_path)
        os.remove(temp_path)

    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()
        print(f"⏹️  Ended run: {self.run_name or self.run_id}")

    def log_training_run(self,
                         model,
                         params: Dict,
                         metrics: Dict,
                         history: Dict,
                         artifacts: Optional[Dict] = None):
        """
        Log complete training run

        Args:
            model: Trained model
            params: Training parameters
            metrics: Final metrics
            history: Training history
            artifacts: Additional artifacts to log
        """
        # Log parameters
        self.log_params(params)

        # Log final metrics
        self.log_metrics(metrics)

        # Log training history
        for epoch, epoch_metrics in enumerate(zip(*history.values())):
            epoch_dict = {k: v for k, v in zip(history.keys(), epoch_metrics)}
            self.log_metrics(epoch_dict, step=epoch)

        # Log model
        self.log_model(model)

        # Log training history as artifact
        self.log_dict(history, 'training_history.json')

        # Log additional artifacts
        if artifacts:
            for name, path in artifacts.items():
                if os.path.exists(path):
                    self.log_artifact(path)

        print("✅ Training run logged to MLflow")
