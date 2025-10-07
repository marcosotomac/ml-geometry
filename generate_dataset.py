"""
Generate synthetic dataset
Run this to create a dataset of geometric shapes
"""

import argparse
from src.data.dataset_generator import GeometricShapeGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic geometric shapes dataset')
    
    parser.add_argument('--output_dir', type=str, default='data/synthetic',
                       help='Output directory for dataset')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Size of generated images')
    parser.add_argument('--samples_per_class', type=int, default=1000,
                       help='Number of samples per class')
    parser.add_argument('--min_shape_size', type=int, default=50,
                       help='Minimum shape size')
    parser.add_argument('--max_shape_size', type=int, default=180,
                       help='Maximum shape size')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎨 GEOMETRIC SHAPE DATASET GENERATOR")
    print("="*70)
    
    generator = GeometricShapeGenerator(
        img_size=args.img_size,
        min_shape_size=args.min_shape_size,
        max_shape_size=args.max_shape_size,
        shapes_per_class=args.samples_per_class
    )
    
    generator.generate_dataset(args.output_dir)
    
    print("\n✨ Dataset generation completed!")


if __name__ == "__main__":
    main()
