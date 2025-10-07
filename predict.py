"""
Make predictions on new images
Run this to predict shapes in your own images
"""

import argparse
import os
from src.evaluation.predictor import ShapePredictor


def main():
    parser = argparse.ArgumentParser(
        description='Predict geometric shapes in images')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model (.h5)')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to model config JSON')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to image file or directory')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization of predictions')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🔮 SHAPE PREDICTION")
    print("="*70)

    # Initialize predictor
    predictor = ShapePredictor(args.model_path, args.config_path)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Predict single image or directory
    if os.path.isfile(args.image_path):
        # Single image
        print(f"\n📷 Predicting: {args.image_path}")

        if args.visualize:
            save_path = os.path.join(args.output_dir,
                                     f"prediction_{os.path.basename(args.image_path)}")
            result = predictor.predict_with_visualization(
                args.image_path, save_path)
        else:
            result = predictor.predict(
                args.image_path, return_probabilities=True)

        print(f"\n✅ Prediction: {result['class']}")
        print(f"   Confidence: {result['confidence']:.2%}")

        if 'probabilities' in result:
            print("\n📊 All Probabilities:")
            for class_name, prob in sorted(result['probabilities'].items(),
                                           key=lambda x: x[1], reverse=True):
                print(f"   {class_name}: {prob:.2%}")

    elif os.path.isdir(args.image_path):
        # Directory of images
        image_files = [f for f in os.listdir(args.image_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"\n📁 Found {len(image_files)} images")

        for img_file in image_files:
            img_path = os.path.join(args.image_path, img_file)
            result = predictor.predict(img_path)
            print(f"\n📷 {img_file}")
            print(
                f"   Prediction: {result['class']} ({result['confidence']:.2%})")

    else:
        print(f"❌ Error: {args.image_path} is not a valid file or directory")

    print("\n✨ Prediction completed!")


if __name__ == "__main__":
    main()
