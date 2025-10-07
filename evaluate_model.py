"""
Evaluate trained model
Run this to evaluate a trained model on test data
"""

import argparse
import os
from tensorflow import keras
from src.data.data_loader import DataLoader
from src.evaluation.evaluator import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.h5)')
    parser.add_argument('--data_dir', type=str, default='data/synthetic',
                       help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("📊 MODEL EVALUATION")
    print("="*70)
    
    # Load model
    print(f"\n🔮 Loading model from: {args.model_path}")
    model = keras.models.load_model(args.model_path)
    
    # Load data
    print(f"📊 Loading test data from: {args.data_dir}")
    data_loader = DataLoader(args.data_dir, batch_size=args.batch_size)
    _, _, test_gen = data_loader.create_data_generators(augment_train=False)
    
    # Get class names
    class_names = data_loader.get_class_names()
    
    # Create evaluator
    evaluator = ModelEvaluator(model, class_names, output_dir=args.output_dir)
    
    # Evaluate
    results = evaluator.evaluate(test_gen)
    
    print("\n✨ Evaluation completed!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
