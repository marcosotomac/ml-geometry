# Quick Start Guide

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/marcosotomac/ml-geometry.git
cd ml-geometry

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python generate_dataset.py --samples_per_class 1000 --output_dir data/synthetic
```

This will create a dataset with:
- 10 geometric shapes (circle, square, triangle, etc.)
- 1000 samples per class
- Train/validation/test splits

### 3. Train Model

**Option A: Custom CNN (Fast training)**
```bash
python train_model.py \
    --model_type custom_cnn \
    --epochs 50 \
    --batch_size 32 \
    --data_dir data/synthetic
```

**Option B: Transfer Learning (Better accuracy)**
```bash
python train_model.py \
    --model_type transfer \
    --base_model efficientnet_b0 \
    --epochs 30 \
    --batch_size 16 \
    --data_dir data/synthetic
```

### 4. Evaluate Model

```bash
python evaluate_model.py \
    --model_path models/saved_models/custom_cnn_best.h5 \
    --data_dir data/synthetic \
    --output_dir results
```

### 5. Make Predictions

```bash
python predict.py \
    --model_path models/saved_models/custom_cnn_best.h5 \
    --image_path path/to/your/image.jpg \
    --visualize
```

### 6. Start API Server

```bash
# Set model path
export MODEL_PATH=models/saved_models/custom_cnn_best.h5

# Start server
python src/api/main.py
```

Then open http://localhost:8000/docs for interactive API documentation.

## 📊 Expected Results

With default settings, you should achieve:
- **Training Accuracy**: ~98-99%
- **Validation Accuracy**: ~95-97%
- **Test Accuracy**: ~95-97%

Transfer learning models typically achieve 1-2% higher accuracy.

## 🎯 Next Steps

1. Experiment with different architectures
2. Try different data augmentation strategies
3. Fine-tune hyperparameters
4. Test on real-world images
5. Deploy the API to production

## 📚 Documentation

- Full documentation: See README.md
- API docs: http://localhost:8000/docs (when server is running)
- Code examples: Check `notebooks/` directory

## 🐛 Troubleshooting

**Issue: Out of memory during training**
- Reduce batch_size (try 16 or 8)
- Use lightweight model instead of custom_cnn

**Issue: Model not converging**
- Increase learning_rate to 0.01
- Try different optimizer (sgd instead of adam)
- Reduce dropout_rate to 0.2

**Issue: Poor accuracy on real images**
- Generate more diverse training data
- Add more augmentation
- Fine-tune with real images
