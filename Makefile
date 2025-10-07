# MLOps Makefile for common tasks

.PHONY: help install test lint format docker-build docker-run k8s-deploy clean

help:
	@echo "ML Geometry - MLOps Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install          Install all dependencies"
	@echo "  make test            Run tests"
	@echo "  make lint            Run linters"
	@echo "  make format          Format code"
	@echo ""
	@echo "Training:"
	@echo "  make train           Train model with MLOps"
	@echo "  make train-dev       Train and register in development"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-run      Run Docker container"
	@echo "  make docker-stack    Start full MLOps stack"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make k8s-deploy      Deploy to Kubernetes"
	@echo "  make k8s-scale       Scale deployment"
	@echo ""
	@echo "Monitoring:"
	@echo "  make mlflow          Start MLflow UI"
	@echo "  make grafana         Start Grafana"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Clean temporary files"

install:
	pip install -r requirements.txt
	pip install -r requirements-mlops.txt

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black src/ tests/

train:
	python train_mlops.py \
		--model-type custom \
		--experiment-name ml-geometry \
		--register-model \
		--stage development

train-dev:
	python train_mlops.py \
		--model-type custom \
		--experiment-name ml-geometry \
		--run-name dev-$(shell date +%Y%m%d-%H%M%S) \
		--register-model \
		--model-version $(shell date +%Y.%m.%d.%H%M) \
		--stage development

docker-build:
	docker build -t ml-geometry:latest .

docker-run:
	docker run -d \
		-p 8000:8000 \
		-v $(PWD)/models:/app/models \
		-e MODEL_PATH=/app/models/saved_models/best_model.h5 \
		ml-geometry:latest

docker-stack:
	docker-compose up -d

docker-stop:
	docker-compose down

k8s-deploy:
	kubectl apply -f k8s/deployment.yaml

k8s-scale:
	kubectl scale deployment ml-geometry-api --replicas=5

k8s-status:
	kubectl get pods,svc,hpa

mlflow:
	@echo "Starting MLflow..."
	@echo "Access at http://localhost:5000"
	docker-compose up -d mlflow

grafana:
	@echo "Starting Grafana..."
	@echo "Access at http://localhost:3000"
	@echo "Default credentials: admin/admin"
	docker-compose up -d grafana prometheus

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
