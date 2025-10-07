"""
Training script with MLOps integration
"""

import os
import sys
import yaml
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import GeometricShapeDataLoader
from src.models.architectures import CustomCNN, LightweightCNN
from src.models.transfer_learning import TransferLearningModel
from src.models.train import TrainingPipeline
from src.mlops.model_registry import ModelRegistry
from src.mlops.experiment_tracker import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description='Train geometric shape detection model with MLOps')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to config file')
    parser.add_argument('--data-path', type=str, default='data',
                       help='Path to dataset')
    parser.add_argument('--model-type', type=str, default='custom',
                       choices=['custom', 'lightweight', 'efficientnet_b0', 'efficientnet_b3', 
                               'resnet50', 'resnet101', 'mobilenet_v2', 'mobilenet_v3'],
                       help='Model architecture')
    parser.add_argument('--experiment-name', type=str, default='ml-geometry',
                       help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default=None,
                       help='MLflow run name')
    parser.add_argument('--register-model', action='store_true',
                       help='Register model after training')
    parser.add_argument('--model-version', type=str, default=None,
                       help='Model version for registry')
    parser.add_argument('--stage', type=str, default='development',
                       choices=['development', 'staging', 'production'],
                       help='Model lifecycle stage')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("🚀 ML GEOMETRY - TRAINING WITH MLOPS")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Model Type: {args.model_type}")
    print(f"   Experiment: {args.experiment_name}")
    print(f"   Stage: {args.stage}")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(experiment_name=args.experiment_name)
    
    # Start MLflow run
    run_name = args.run_name or f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tracker.start_run(
        run_name=run_name,
        tags={
            'model_type': args.model_type,
            'stage': args.stage,
            'framework': 'tensorflow'
        }
    )
    
    # Log configuration
    tracker.log_params({
        'model_type': args.model_type,
        'image_size': config['data']['image_size'],
        'batch_size': config['training']['batch_size'],
        'epochs': config['training']['epochs'],
        'learning_rate': config['training']['learning_rate'],
        'optimizer': config['training']['optimizer']
    })
    
    # Load data
    print("\n📊 Loading dataset...")
    data_loader = GeometricShapeDataLoader(
        data_dir=args.data_path,
        image_size=tuple(config['data']['image_size']),
        batch_size=config['training']['batch_size']
    )
    train_ds, val_ds, test_ds = data_loader.load_datasets()
    
    print(f"✅ Dataset loaded:")
    print(f"   Classes: {len(data_loader.class_names)}")
    print(f"   Training samples: ~{config['data']['num_samples'] * 0.7:.0f}")
    print(f"   Validation samples: ~{config['data']['num_samples'] * 0.15:.0f}")
    print(f"   Test samples: ~{config['data']['num_samples'] * 0.15:.0f}")
    
    # Create model
    print(f"\n🏗️  Building {args.model_type} model...")
    
    if args.model_type == 'custom':
        model = CustomCNN(
            input_shape=tuple(config['data']['image_size']) + (3,),
            num_classes=len(data_loader.class_names)
        ).build()
    elif args.model_type == 'lightweight':
        model = LightweightCNN(
            input_shape=tuple(config['data']['image_size']) + (3,),
            num_classes=len(data_loader.class_names)
        ).build()
    else:
        transfer_model = TransferLearningModel(
            base_model_name=args.model_type,
            input_shape=tuple(config['data']['image_size']) + (3,),
            num_classes=len(data_loader.class_names)
        )
        model = transfer_model.build_model()
    
    print(f"✅ Model built with {model.count_params():,} parameters")
    
    # Train model
    print("\n🎯 Training model...")
    pipeline = TrainingPipeline(
        model=model,
        config=config['training']
    )
    
    history = pipeline.train(
        train_dataset=train_ds,
        val_dataset=val_ds
    )
    
    # Evaluate on test set
    print("\n📈 Evaluating on test set...")
    test_results = model.evaluate(test_ds)
    test_metrics = {
        'test_loss': float(test_results[0]),
        'test_accuracy': float(test_results[1])
    }
    
    print(f"\n✅ Test Results:")
    print(f"   Loss: {test_metrics['test_loss']:.4f}")
    print(f"   Accuracy: {test_metrics['test_accuracy']:.4f}")
    
    # Log to MLflow
    tracker.log_metrics(test_metrics)
    tracker.log_training_run(
        model=model,
        params={
            'model_type': args.model_type,
            'num_parameters': int(model.count_params())
        },
        metrics=test_metrics,
        history=history.history,
        artifacts={
            'model_plot': 'models/model_plot.png'
        }
    )
    
    # Save model
    model_path = os.path.join('models', 'saved_models', 'best_model.h5')
    print(f"\n💾 Saving model to {model_path}...")
    
    # Register model if requested
    if args.register_model:
        print("\n📦 Registering model...")
        registry = ModelRegistry()
        
        # Generate version if not provided
        version = args.model_version or datetime.now().strftime('%Y.%m.%d.%H%M')
        
        registry_id = registry.register_model(
            model_path=model_path,
            model_name=args.model_type,
            version=version,
            metadata={
                'accuracy': test_metrics['test_accuracy'],
                'loss': test_metrics['test_loss'],
                'num_parameters': int(model.count_params()),
                'mlflow_run_id': tracker.run_id,
                'training_date': datetime.now().isoformat()
            },
            stage=args.stage
        )
        
        print(f"✅ Model registered with ID: {registry_id}")
    
    # End MLflow run
    tracker.end_run()
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
