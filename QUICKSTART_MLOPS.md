# 🚀 ML Geometry - Inicio Rápido MLOps

## Opción 1: Desarrollo Local

### Paso 1: Setup Inicial
```bash
# Clonar e instalar
git clone https://github.com/marcosotomac/ml-geometry.git
cd ml-geometry
pip install -r requirements.txt
pip install -r requirements-mlops.txt
```

### Paso 2: Generar Dataset
```bash
python generate_dataset.py
```

### Paso 3: Entrenar con MLOps
```bash
python train_mlops.py \
  --model-type custom \
  --register-model \
  --model-version 1.0.0 \
  --stage development
```

### Paso 4: Ver Resultados en MLflow
```bash
# Iniciar MLflow UI
docker-compose up -d mlflow

# Abrir http://localhost:5000
```

---

## Opción 2: Docker Stack Completo

### Inicio Rápido (Todo en uno)
```bash
# Clonar
git clone https://github.com/marcosotomac/ml-geometry.git
cd ml-geometry

# Generar dataset
python generate_dataset.py

# Entrenar modelo
python train_model.py

# Iniciar stack completo
docker-compose up -d

# Acceder a servicios:
# - API: http://localhost:8000/docs
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### Probar API
```bash
# Health check
curl http://localhost:8000/health

# Predicción
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"

# Métricas
curl http://localhost:8000/metrics/summary
```

---

## Opción 3: Kubernetes (Producción)

### Prerrequisitos
```bash
# Instalar kubectl
brew install kubectl  # macOS
# o descargar desde https://kubernetes.io/docs/tasks/tools/

# Configurar cluster (ejemplo con minikube)
minikube start
```

### Deploy
```bash
# Build y push imagen
docker build -t tu_usuario/ml-geometry:latest .
docker push tu_usuario/ml-geometry:latest

# Actualizar deployment.yaml con tu imagen
# Editar k8s/deployment.yaml línea 18

# Deploy
kubectl apply -f k8s/deployment.yaml

# Verificar
kubectl get pods
kubectl get svc

# Obtener URL del servicio
kubectl get svc ml-geometry-api-service
```

---

## Flujo de Trabajo MLOps

### 1. Desarrollo
```bash
# Entrenar modelo
python train_mlops.py --register-model --stage development

# Ver en MLflow
docker-compose up -d mlflow
# Abrir http://localhost:5000
```

### 2. Staging
```bash
# Promover a staging
python deploy_model.py \
  --model-name custom \
  --version 1.0.0 \
  --source-stage development
```

### 3. Producción
```bash
# Dry run (simular)
python deploy_model.py \
  --model-name custom \
  --version 1.0.0 \
  --source-stage staging \
  --dry-run

# Deploy real
python deploy_model.py \
  --model-name custom \
  --version 1.0.0 \
  --source-stage staging
```

---

## Comandos Útiles (Makefile)

```bash
make install       # Instalar todo
make test         # Ejecutar tests
make train        # Entrenar modelo
make docker-stack # Iniciar stack
make k8s-deploy   # Deploy a k8s
make clean        # Limpiar archivos
```

---

## Monitoreo

### Ver Métricas en Tiempo Real
```bash
# Iniciar Grafana
docker-compose up -d grafana prometheus

# Acceder a Grafana
# http://localhost:3000
# Usuario: admin
# Password: admin

# Configurar datasource:
# 1. Add data source → Prometheus
# 2. URL: http://prometheus:9090
# 3. Save & test
```

### Dashboards Recomendados
- **Prediction Rate**: ml_geometry_predictions_total
- **Latency**: ml_geometry_prediction_latency_seconds
- **Error Rate**: ml_geometry_errors_total
- **Drift**: ml_geometry_prediction_drift

---

## Troubleshooting

### Problema: "Model not loaded"
```bash
# Verificar que existe el modelo
ls -la models/saved_models/

# Si no existe, entrenar primero
python train_model.py
```

### Problema: "Port already in use"
```bash
# Detener servicios existentes
docker-compose down

# Limpiar todo
docker-compose down -v

# Reiniciar
docker-compose up -d
```

### Problema: "Cannot connect to MLflow"
```bash
# Verificar que MLflow está corriendo
docker-compose ps

# Ver logs
docker-compose logs mlflow

# Reiniciar solo MLflow
docker-compose restart mlflow
```

---

## Estructura del Proyecto

```
ml-geometry/
├── src/
│   ├── data/              # Dataset y loaders
│   ├── models/            # Arquitecturas y training
│   ├── evaluation/        # Evaluación y predicción
│   ├── mlops/            # 🆕 MLOps components
│   └── api/              # FastAPI server
├── tests/                # Tests
├── configs/              # Configuración
├── k8s/                  # 🆕 Kubernetes configs
├── monitoring/           # 🆕 Prometheus configs
├── .github/workflows/    # 🆕 CI/CD
├── Dockerfile           # 🆕 Container image
├── docker-compose.yml   # 🆕 Multi-service stack
├── Makefile             # 🆕 Automation
├── train_mlops.py       # 🆕 MLOps training
├── deploy_model.py      # 🆕 Deployment script
├── MLOPS.md             # 🆕 MLOps documentation
└── requirements.txt     # Dependencies
```

---

## URLs de Referencia

- **Repositorio**: https://github.com/marcosotomac/ml-geometry
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090

---

## Documentación Completa

- `README.md` - Documentación principal
- `MLOPS.md` - Guía completa de MLOps
- `MLOPS_SUMMARY.md` - Resumen de implementación
- `QUICKSTART.md` - Tutorial de inicio

---

## Soporte

Para preguntas o issues:
- GitHub Issues: https://github.com/marcosotomac/ml-geometry/issues
- Documentación: Ver archivos .md en el proyecto

---

**¡Sistema MLOps listo para producción! 🎉**
