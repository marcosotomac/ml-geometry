# ML Geometry Detector

Advanced Machine Learning system for detection and classification of geometric shapes in images using Deep Learning architectures.

## Features

- **Synthetic Dataset Generation**: Automated creation of training datasets with geometric shapes
- **Custom CNN Architecture**: Convolutional neural network with ResNet blocks and skip connections
- **Transfer Learning Support**: Integration with EfficientNet, ResNet50, and MobileNetV2 pretrained models
- **Advanced Data Augmentation**: Sophisticated transformations for improved model generalization
- **Robust Training Pipeline**: Early stopping, learning rate scheduling, and custom callbacks
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, and detailed visualizations
- **REST API**: FastAPI server for real-time predictions
- **Multi-Object Detection**: Capability to detect multiple shapes in single images

## Installation

## Installation

```bash
# Clone repository
git clone https://github.com/marcosotomac/ml-geometry.git
cd ml-geometry

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Supported Shapes

- Circle
- Square
- Rectangle
- Triangle
- Pentagon
- Hexagon
- Octagon
- Star
- Rhombus
- Ellipse

## Usage

### Dataset Generation
```python
from src.data.dataset_generator import GeometricShapeGenerator

generator = GeometricShapeGenerator(img_size=224, shapes_per_class=1000)
### Dataset Generation

```python
from src.data.dataset_generator import GeometricShapeGenerator

generator = GeometricShapeGenerator(img_size=224, shapes_per_class=1000)
generator.generate_dataset('data/synthetic')
```

### Model Training

```python
from src.models.train import train_model

train_model(
    data_dir='data/synthetic',
    model_type='custom_cnn',
    epochs=50,
    batch_size=32
)
```

### Predictions

```python
from src.models.predictor import ShapePredictor

predictor = ShapePredictor('models/best_model.h5')
prediction = predictor.predict('path/to/image.jpg')
print(f"Shape: {prediction['class']}, Confidence: {prediction['confidence']:.2%}")
```

### API Server

```bash
python src/api/main.py
```

Access interactive documentation at http://localhost:8000/docs

## Model Architecture

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
## Model Architecture

The custom CNN architecture includes:
- ResNet blocks with skip connections
- Batch Normalization layers for training stability
- Dropout regularization
- Global Average Pooling
- Dense layers with softmax activation

## Project Structure

```
ml-geometry/
├── data/
│   ├── synthetic/       # Generated dataset
│   └── real/            # Real images (optional)
├── models/
│   ├── saved_models/    # Trained models
│   └── checkpoints/     # Training checkpoints
├── src/
│   ├── data/            # Data generation and processing
│   ├── models/          # Model architectures and training
│   ├── evaluation/      # Metrics and visualizations
│   └── api/             # REST API
├── notebooks/           # Jupyter notebooks
├── tests/               # Unit tests
└── configs/             # Configuration files
```

## Performance Metrics

Expected performance with default configuration:
- Training Accuracy: ~98-99%
- Validation Accuracy: ~95-97%
- Test Accuracy: ~95-97%

Transfer learning models typically achieve 1-2% higher accuracy.

## Advanced Features

### Multi-Object Detection

```python
from src.evaluation.multi_detector import MultiShapeDetector

detector = MultiShapeDetector('models/best_model.h5')
detections = detector.detect_shapes_contours(image, confidence_threshold=0.7)
```

### Custom Training Parameters

```bash
python train_model.py \
    --model_type transfer \
    --base_model efficientnet_b0 \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --dropout_rate 0.3
```

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single image prediction
- `POST /predict/batch` - Batch prediction
- `GET /classes` - List available classes
- `GET /model/info` - Model information

## Contributing

Contributions are welcome. Please open an issue or pull request.

## License

MIT License

## Author

Marcos Soto Maceda
