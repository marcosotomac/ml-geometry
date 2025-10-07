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

### Custom CNN Architecture

The model follows a hierarchical structure with four main stages:

**Input Layer**
- Input shape: 224x224x3 (RGB images)

**Initial Convolution**
- Conv2D: 7x7 kernel, 64 filters, stride 2
- Batch Normalization + ReLU activation
- MaxPooling: 3x3, stride 2

**Stage 1** (Output: 56x56x64)
- 2x ResNet blocks with 64 filters
- Channel Attention mechanism
- Skip connections for residual learning

**Stage 2** (Output: 28x28x128)
- 3x ResNet blocks with 128 filters
- Downsampling via stride 2
- Channel Attention mechanism

**Stage 3** (Output: 14x14x256)
- 4x ResNet blocks with 256 filters
- Downsampling via stride 2
- Channel Attention mechanism

**Stage 4** (Output: 7x7x512)
- 3x ResNet blocks with 512 filters
- Downsampling via stride 2
- Channel Attention mechanism

**Classification Head**
- Global Average Pooling
- Dense layer: 512 units + ReLU + Dropout (0.3)
- Dense layer: 256 units + ReLU + Dropout (0.15)
- Output layer: 10 units + Softmax

### ResNet Block Structure

Each ResNet block contains:
- Conv2D (3x3) + BatchNorm + ReLU
- Conv2D (3x3) + BatchNorm
- Skip connection (identity or 1x1 conv for dimension matching)
- Addition + ReLU + Dropout
- L2 regularization (1e-4)

### Channel Attention (SE-Net)

- Global Average Pooling across spatial dimensions
- Dense layer with reduction (filters / 16)
- ReLU activation
- Dense layer to original filter size
- Sigmoid activation
- Element-wise multiplication with input

### Regularization Techniques

- **Dropout**: 0.3 in residual blocks, 0.15 in classification head
- **L2 Weight Decay**: 1e-4 on all convolutional and dense layers
- **Batch Normalization**: After each convolution for training stability
- **Data Augmentation**: Rotation, shifts, flips, zoom, brightness, contrast

### Transfer Learning Models

Available pretrained architectures:

**EfficientNet B0/B3**
- Compound scaling method (depth, width, resolution)
- Mobile inverted bottleneck convolutions
- Optimal efficiency-accuracy tradeoff

**ResNet50/101**
- Deep residual learning framework
- 50/101 layers with skip connections
- Proven performance on ImageNet

**MobileNetV2/V3**
- Depthwise separable convolutions
- Inverted residuals with linear bottlenecks
- Optimized for mobile and embedded devices

All transfer learning models include:
- Pretrained ImageNet weights (1000 classes)
- Custom classification head: 512 → 256 → num_classes
- Batch Normalization in classification head
- Optional fine-tuning of last N layers

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
