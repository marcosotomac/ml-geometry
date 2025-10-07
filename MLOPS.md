# ML Geometry - MLOps Documentation

## MLOps Architecture

This project implements a complete MLOps pipeline with:

- **Model Registry**: Version control and lifecycle management
- **Experiment Tracking**: MLflow integration for tracking experiments
- **Model Monitoring**: Prometheus metrics and performance tracking
- **CI/CD**: GitHub Actions pipeline for automated testing and deployment
- **Containerization**: Docker and Docker Compose setup
- **Orchestration**: Kubernetes deployment configurations

## Components

### 1. Model Registry

The Model Registry provides centralized version control and lifecycle management:

```python
from src.mlops.model_registry import ModelRegistry

registry = ModelRegistry()

# Register a model
registry_id = registry.register_model(
    model_path='models/saved_models/best_model.h5',
    model_name='custom_cnn',
    version='1.0.0',
    metadata={'accuracy': 0.95, 'loss': 0.15},
    stage='development'
)

# Promote to staging
registry.promote_model(registry_id, 'staging')

# Promote to production
registry.promote_model(registry_id, 'production')

# Get production model
model = registry.get_model('custom_cnn', stage='production')
```

**Lifecycle stages:**
- `development`: Initial training and experimentation
- `staging`: Testing and validation
- `production`: Live deployment
- `archived`: Deprecated models

### 2. Experiment Tracking

MLflow integration for tracking all experiments:

```python
from src.mlops.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(experiment_name='ml-geometry')

# Start a run
tracker.start_run(run_name='custom_cnn_v1', tags={'model_type': 'cnn'})

# Log parameters
tracker.log_params({
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
})

# Log metrics
tracker.log_metrics({
    'accuracy': 0.95,
    'loss': 0.15
})

# Log model
tracker.log_model(model)

# End run
tracker.end_run()
```

**MLflow UI:**
```bash
# Start MLflow server
docker-compose up mlflow

# Access at http://localhost:5000
```

### 3. Model Monitoring

Production monitoring with Prometheus metrics:

```python
from src.mlops.model_monitor import ModelMonitor

monitor = ModelMonitor(model_name='ml-geometry')

# Log predictions
monitor.log_prediction(
    predicted_class='circle',
    confidence=0.98,
    latency=0.05,
    true_label='circle'  # Optional
)

# Get metrics
metrics = monitor.get_metrics()

# Detect drift
drift_score = monitor.detect_drift(baseline_distribution)
```

**Prometheus metrics exposed:**
- `ml_geometry_predictions_total`: Total predictions by class
- `ml_geometry_prediction_latency_seconds`: Latency histogram
- `ml_geometry_average_confidence`: Average confidence gauge
- `ml_geometry_errors_total`: Error counter
- `ml_geometry_prediction_drift`: Drift score

**Grafana Dashboard:**
```bash
# Start monitoring stack
docker-compose up prometheus grafana

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin
```

## Training with MLOps

Train models with full MLOps integration:

```bash
python train_mlops.py \
  --model-type custom \
  --experiment-name ml-geometry \
  --run-name custom_cnn_experiment_1 \
  --register-model \
  --model-version 1.0.0 \
  --stage development
```

**Options:**
- `--config`: Configuration file path
- `--data-path`: Dataset directory
- `--model-type`: Architecture type
- `--experiment-name`: MLflow experiment
- `--run-name`: MLflow run name
- `--register-model`: Register in model registry
- `--model-version`: Version string
- `--stage`: Lifecycle stage

## Deployment Pipeline

### Development → Staging → Production

1. **Train and register model**:
```bash
python train_mlops.py \
  --model-type efficientnet_b0 \
  --register-model \
  --model-version 2.0.0 \
  --stage development
```

2. **Promote to staging**:
```bash
python deploy_model.py \
  --model-name efficientnet_b0 \
  --version 2.0.0 \
  --source-stage development \
  --dry-run

# If checks pass
python deploy_model.py \
  --model-name efficientnet_b0 \
  --version 2.0.0 \
  --source-stage development
```

3. **Deploy to production**:
```bash
python deploy_model.py \
  --model-name efficientnet_b0 \
  --version 2.0.0 \
  --source-stage staging
```

## Docker Deployment

### Single Container

```bash
# Build image
docker build -t ml-geometry:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_PATH=/app/models/saved_models/best_model.h5 \
  ml-geometry:latest
```

### Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
brew install kubectl

# Configure cluster
kubectl config use-context your-cluster
```

### Deploy

```bash
# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods
kubectl get services

# Get external IP
kubectl get service ml-geometry-api-service
```

### Scale

```bash
# Manual scaling
kubectl scale deployment ml-geometry-api --replicas=5

# Auto-scaling is configured via HPA (2-10 replicas)
```

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci-cd.yml`):

**On Push/PR:**
1. Test on Python 3.9, 3.10, 3.11
2. Lint with flake8
3. Format check with black
4. Type check with mypy
5. Run tests with pytest
6. Upload coverage to Codecov

**On Main Branch:**
7. Build Docker image
8. Push to Docker Hub
9. Deploy to staging
10. Deploy to production (manual approval)

**Setup:**
```bash
# Add GitHub secrets:
# - DOCKER_USERNAME
# - DOCKER_PASSWORD
```

## API with Monitoring

The updated API includes monitoring endpoints:

```bash
# Start API with monitoring
python src/api/main_mlops.py
```

**Endpoints:**
- `GET /`: API info
- `GET /health`: Health check
- `POST /predict`: Single prediction (with monitoring)
- `POST /predict/batch`: Batch predictions
- `GET /classes`: Available classes
- `GET /model/info`: Model information
- `GET /metrics`: Prometheus metrics
- `GET /metrics/summary`: Metrics summary (JSON)

**Monitor metrics:**
```bash
# Prometheus format
curl http://localhost:8000/metrics

# JSON summary
curl http://localhost:8000/metrics/summary
```

## Model Registry CLI

```bash
# List all models
python -c "from src.mlops.model_registry import ModelRegistry; \
  r = ModelRegistry(); \
  import json; \
  print(json.dumps(r.list_models(), indent=2))"

# List production models
python -c "from src.mlops.model_registry import ModelRegistry; \
  r = ModelRegistry(); \
  import json; \
  print(json.dumps(r.list_models(stage='production'), indent=2))"

# Get specific model
python -c "from src.mlops.model_registry import ModelRegistry; \
  r = ModelRegistry(); \
  import json; \
  model = r.get_model('custom_cnn', stage='production'); \
  print(json.dumps(model, indent=2))"
```

## Monitoring Best Practices

1. **Set baseline metrics** after initial deployment
2. **Monitor drift** regularly
3. **Set up alerts** for:
   - Low confidence predictions
   - High latency
   - Increased error rate
   - Distribution drift
4. **Track true labels** when available
5. **Review metrics** before promotions

## Performance Optimization

- **Horizontal scaling**: Adjust HPA settings
- **Model optimization**: Use TensorFlow Lite or ONNX
- **Caching**: Implement Redis for frequent predictions
- **Batch processing**: Use batch endpoint for multiple images
- **Load balancing**: Kubernetes handles automatically

## Security

- **API authentication**: Add JWT tokens
- **HTTPS**: Configure TLS certificates
- **Rate limiting**: Implement rate limiting
- **Model encryption**: Encrypt model artifacts
- **Secret management**: Use Kubernetes secrets

## Troubleshooting

**Model not loading:**
```bash
# Check model path
kubectl logs deployment/ml-geometry-api

# Verify volume mount
kubectl describe pod <pod-name>
```

**High latency:**
```bash
# Check metrics
curl http://your-api/metrics/summary

# Scale up
kubectl scale deployment ml-geometry-api --replicas=10
```

**Memory issues:**
```bash
# Increase limits in k8s/deployment.yaml
resources:
  limits:
    memory: "4Gi"
```
