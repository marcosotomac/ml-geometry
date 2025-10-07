# ML Geometry Detector 🔺🔴⬜

Un modelo avanzado de Machine Learning para detectar y clasificar figuras geométricas en imágenes usando Deep Learning.

## 🚀 Características

- **Generación de Dataset Sintético**: Crea automáticamente datasets de entrenamiento con figuras geométricas
- **Arquitectura CNN Personalizada**: Red neuronal convolucional con ResNet blocks y skip connections
- **Transfer Learning**: Soporte para EfficientNet, ResNet50, y MobileNetV2
- **Data Augmentation Avanzado**: Transformaciones sofisticadas para mejorar generalización
- **Pipeline de Entrenamiento Robusto**: Early stopping, learning rate scheduling, y callbacks personalizados
- **Evaluación Completa**: Matrices de confusión, curvas ROC, y visualizaciones detalladas
- **API REST**: Servidor FastAPI para predicciones en tiempo real
- **Detección Multi-Objeto**: Capacidad de detectar múltiples figuras en una imagen

## 📦 Instalación

```bash
# Clonar repositorio
git clone https://github.com/marcosotomac/ml-geometry.git
cd ml-geometry

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## 🎯 Figuras Soportadas

- Círculo
- Cuadrado
- Rectángulo
- Triángulo
- Pentágono
- Hexágono
- Octágono
- Estrella
- Rombo
- Elipse

## 🔧 Uso Rápido

### Generar Dataset
```python
from src.data.dataset_generator import GeometricShapeGenerator

generator = GeometricShapeGenerator(img_size=224, shapes_per_class=1000)
generator.generate_dataset('data/synthetic')
```

### Entrenar Modelo
```python
from src.models.train import train_model

train_model(
    data_dir='data/synthetic',
    model_type='custom_cnn',
    epochs=50,
    batch_size=32
)
```

### Hacer Predicciones
```python
from src.models.predictor import ShapePredictor

predictor = ShapePredictor('models/best_model.h5')
prediction = predictor.predict('path/to/image.jpg')
print(f"Forma detectada: {prediction['class']} (confianza: {prediction['confidence']:.2%})")
```

### Iniciar API
```bash
python src/api/main.py
```

## 📊 Arquitectura del Modelo

El modelo utiliza una arquitectura CNN personalizada con:
- Bloques ResNet con skip connections
- Batch Normalization para estabilidad
- Dropout para regularización
- Global Average Pooling
- Capas densas con activación softmax

## 📁 Estructura del Proyecto

```
ml-geometry/
├── data/
│   ├── synthetic/       # Dataset generado
│   └── real/            # Imágenes reales (opcional)
├── models/
│   ├── saved_models/    # Modelos entrenados
│   └── checkpoints/     # Checkpoints de entrenamiento
├── src/
│   ├── data/            # Generación y procesamiento de datos
│   ├── models/          # Arquitecturas y entrenamiento
│   ├── evaluation/      # Métricas y visualizaciones
│   └── api/             # API REST
├── notebooks/           # Jupyter notebooks para experimentación
├── tests/               # Tests unitarios
└── configs/             # Archivos de configuración
```

## 📈 Resultados

(Se actualizará con métricas de rendimiento)

## 🤝 Contribuir

Las contribuciones son bienvenidas! Por favor, abre un issue o pull request.

## 📄 Licencia

MIT License

## 👨‍💻 Autor

Marcos Soto Maceda
