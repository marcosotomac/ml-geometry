"""
Main training script
Run this to train a model on the synthetic dataset
"""

import os
import argparse
from src.models.train import train_model


def main():
    parser = argparse.ArgumentParser(description='Train geometric shape detection model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/synthetic',
                       help='Directory containing the dataset')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='custom_cnn',
                       choices=['custom_cnn', 'lightweight', 'transfer'],
                       help='Type of model to train')
    parser.add_argument('--base_model', type=str, default='efficientnet_b0',
                       help='Base model for transfer learning')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'rmsprop'],
                       help='Optimizer')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    
    # Other arguments
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Train model
    print("\n" + "="*70)
    print("🚀 GEOMETRIC SHAPE DETECTION - TRAINING")
    print("="*70)
    
    model, history = train_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        base_model=args.base_model if args.model_type == 'transfer' else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        patience=args.patience,
        use_augmentation=not args.no_augmentation,
        dropout_rate=args.dropout_rate
    )
    
    print("\n✨ Training completed successfully!")


if __name__ == "__main__":
    main()
