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

```
Input (224x224x3)
    ↓
Conv2D (7x7, 64 filters, stride=2) + BatchNorm + ReLU
    ↓
MaxPooling2D (3x3, stride=2)
    ↓
┌─────────────────────────────────────┐
│ Stage 1: ResNet Block x2 (64)      │
│  - Conv2D (3x3, 64)                │
│  - BatchNorm + ReLU                │
│  - Conv2D (3x3, 64)                │
│  - BatchNorm                       │
│  - Skip Connection                 │
│  - ReLU + Dropout                  │
│ Channel Attention (SE-Net style)   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 2: ResNet Block x3 (128)     │
│  - Downsample (stride=2)           │
│  - Conv blocks with skip           │
│ Channel Attention                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 3: ResNet Block x4 (256)     │
│  - Downsample (stride=2)           │
│  - Conv blocks with skip           │
│ Channel Attention                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 4: ResNet Block x3 (512)     │
│  - Downsample (stride=2)           │
│  - Conv blocks with skip           │
│ Channel Attention                  │
└─────────────────────────────────────┘
    ↓
Global Average Pooling
    ↓
Dense (512) + ReLU + Dropout(0.3)
    ↓
Dense (256) + ReLU + Dropout(0.15)
    ↓
Dense (num_classes) + Softmax
    ↓
Output (10 classes)
```

### Key Components

**ResNet Blocks**
- Residual connections for gradient flow
- 3x3 convolutional kernels
- Batch Normalization after each convolution
- ReLU activation
- L2 regularization (1e-4)

**Attention Mechanism**
- Channel attention (Squeeze-Excitation)
- Reduction ratio: 16
- Global average pooling for squeeze
- Two fully connected layers for excitation

**Regularization**
- Dropout: 0.3 in ResNet blocks, reduced in classification head
- L2 weight decay: 1e-4
- Batch Normalization for internal covariate shift

### Transfer Learning Models

Available pretrained architectures:
- **EfficientNet B0/B3**: Compound scaling, efficient architecture
- **ResNet50/101**: Deep residual networks
- **MobileNetV2/V3**: Lightweight, mobile-optimized

All transfer learning models use:
- Pretrained ImageNet weights
- Frozen base layers initially
- Custom classification head (512 → 256 → num_classes)
- Fine-tuning capability for last N layers

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
