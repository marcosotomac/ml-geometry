"""
Script to detect multiple shapes in an image
"""

import argparse
import os
import numpy as np
from PIL import Image
from src.evaluation.multi_detector import MultiShapeDetector


def main():
    parser = argparse.ArgumentParser(description='Detect multiple geometric shapes in image')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to classification model (.h5)')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to image file')
    parser.add_argument('--method', type=str, default='contours',
                       choices=['sliding_window', 'region_proposals', 'contours'],
                       help='Detection method')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='Minimum confidence threshold')
    parser.add_argument('--output_dir', type=str, default='detections',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔍 MULTI-SHAPE DETECTION")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector
    print(f"\n🤖 Initializing detector...")
    detector = MultiShapeDetector(args.model_path)
    
    # Load image
    print(f"📷 Loading image: {args.image_path}")
    image = np.array(Image.open(args.image_path).convert('RGB')) / 255.0
    
    # Detect shapes
    print(f"🔍 Detecting shapes using {args.method} method...")
    
    if args.method == 'sliding_window':
        detections = detector.detect_shapes_sliding_window(
            image, 
            confidence_threshold=args.confidence_threshold
        )
    elif args.method == 'region_proposals':
        detections = detector.detect_shapes_region_proposals(
            image,
            confidence_threshold=args.confidence_threshold
        )
    else:  # contours
        detections = detector.detect_shapes_contours(
            image,
            confidence_threshold=args.confidence_threshold
        )
    
    # Print results
    print(f"\n✅ Found {len(detections)} shapes:")
    for i, det in enumerate(detections, 1):
        print(f"   {i}. {det['class']} (confidence: {det['confidence']:.2%})")
        print(f"      BBox: {det['bbox']}")
    
    # Visualize
    save_path = os.path.join(args.output_dir, 
                            f"detection_{os.path.basename(args.image_path)}")
    detector.visualize_detections(image, detections, save_path)
    
    print(f"\n✨ Detection complete!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
