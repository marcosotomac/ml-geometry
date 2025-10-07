"""
Advanced Geometric Shape Dataset Generator
Generates synthetic images of geometric shapes with various augmentations
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random
from typing import Tuple, List, Dict
import json
from tqdm import tqdm
import albumentations as A


class GeometricShapeGenerator:
    """
    Advanced generator for creating synthetic geometric shape datasets
    with sophisticated augmentation and variations
    """

    SHAPES = [
        'circle', 'square', 'rectangle', 'triangle',
        'pentagon', 'hexagon', 'octagon', 'star',
        'rhombus', 'ellipse'
    ]

    def __init__(self, img_size: int = 224, min_shape_size: int = 50,
                 max_shape_size: int = 180, shapes_per_class: int = 1000):
        """
        Initialize the shape generator

        Args:
            img_size: Size of output images (square)
            min_shape_size: Minimum size of shapes
            max_shape_size: Maximum size of shapes
            shapes_per_class: Number of samples per shape class
        """
        self.img_size = img_size
        self.min_shape_size = min_shape_size
        self.max_shape_size = max_shape_size
        self.shapes_per_class = shapes_per_class

        # Advanced augmentation pipeline
        self.augmentation = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
            ], p=0.4),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.2),
                A.GridDistortion(num_steps=5, distort_limit=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3),
                A.HueSaturationValue(hue_shift_limit=20,
                                     sat_shift_limit=30, val_shift_limit=20),
            ], p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        ])

    def _get_random_color(self) -> Tuple[int, int, int]:
        """Generate random RGB color"""
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    def _get_random_background(self) -> Tuple[int, int, int]:
        """Generate random background color (usually darker)"""
        return (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))

    def _draw_circle(self, draw: ImageDraw, center: Tuple[int, int],
                     radius: int, color: Tuple[int, int, int]):
        """Draw a circle"""
        x, y = center
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=color, outline=color)

    def _draw_square(self, draw: ImageDraw, center: Tuple[int, int],
                     size: int, color: Tuple[int, int, int]):
        """Draw a square"""
        x, y = center
        half_size = size // 2
        draw.rectangle([x - half_size, y - half_size, x + half_size, y + half_size],
                       fill=color, outline=color)

    def _draw_rectangle(self, draw: ImageDraw, center: Tuple[int, int],
                        width: int, height: int, color: Tuple[int, int, int]):
        """Draw a rectangle"""
        x, y = center
        draw.rectangle([x - width // 2, y - height // 2, x + width // 2, y + height // 2],
                       fill=color, outline=color)

    def _draw_triangle(self, draw: ImageDraw, center: Tuple[int, int],
                       size: int, color: Tuple[int, int, int]):
        """Draw an equilateral triangle"""
        x, y = center
        height = int(size * np.sqrt(3) / 2)
        points = [
            (x, y - 2 * height // 3),
            (x - size // 2, y + height // 3),
            (x + size // 2, y + height // 3)
        ]
        draw.polygon(points, fill=color, outline=color)

    def _draw_polygon(self, draw: ImageDraw, center: Tuple[int, int],
                      radius: int, sides: int, color: Tuple[int, int, int],
                      rotation: float = 0):
        """Draw a regular polygon"""
        x, y = center
        points = []
        for i in range(sides):
            angle = 2 * np.pi * i / sides + rotation
            px = x + radius * np.cos(angle)
            py = y + radius * np.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=color, outline=color)

    def _draw_star(self, draw: ImageDraw, center: Tuple[int, int],
                   outer_radius: int, color: Tuple[int, int, int], points: int = 5):
        """Draw a star"""
        x, y = center
        inner_radius = outer_radius // 2
        star_points = []

        for i in range(points * 2):
            angle = np.pi * i / points - np.pi / 2
            radius = outer_radius if i % 2 == 0 else inner_radius
            px = x + radius * np.cos(angle)
            py = y + radius * np.sin(angle)
            star_points.append((px, py))

        draw.polygon(star_points, fill=color, outline=color)

    def _draw_rhombus(self, draw: ImageDraw, center: Tuple[int, int],
                      width: int, height: int, color: Tuple[int, int, int]):
        """Draw a rhombus"""
        x, y = center
        points = [
            (x, y - height // 2),
            (x + width // 2, y),
            (x, y + height // 2),
            (x - width // 2, y)
        ]
        draw.polygon(points, fill=color, outline=color)

    def _draw_ellipse(self, draw: ImageDraw, center: Tuple[int, int],
                      width: int, height: int, color: Tuple[int, int, int]):
        """Draw an ellipse"""
        x, y = center
        draw.ellipse([x - width // 2, y - height // 2, x + width // 2, y + height // 2],
                     fill=color, outline=color)

    def generate_shape_image(self, shape: str, apply_augmentation: bool = True) -> np.ndarray:
        """
        Generate a single shape image

        Args:
            shape: Type of shape to generate
            apply_augmentation: Whether to apply augmentation

        Returns:
            Numpy array of the generated image
        """
        # Create base image with random background
        bg_color = self._get_random_background()
        img = Image.new('RGB', (self.img_size, self.img_size), bg_color)
        draw = ImageDraw.Draw(img)

        # Random position (with margins)
        margin = self.max_shape_size // 2 + 10
        center_x = random.randint(margin, self.img_size - margin)
        center_y = random.randint(margin, self.img_size - margin)
        center = (center_x, center_y)

        # Random size
        size = random.randint(self.min_shape_size, self.max_shape_size)

        # Random color
        color = self._get_random_color()

        # Draw the shape
        if shape == 'circle':
            self._draw_circle(draw, center, size // 2, color)
        elif shape == 'square':
            self._draw_square(draw, center, size, color)
        elif shape == 'rectangle':
            width = size
            height = int(size * random.uniform(0.5, 0.8))
            self._draw_rectangle(draw, center, width, height, color)
        elif shape == 'triangle':
            self._draw_triangle(draw, center, size, color)
        elif shape == 'pentagon':
            self._draw_polygon(draw, center, size // 2, 5, color)
        elif shape == 'hexagon':
            self._draw_polygon(draw, center, size // 2, 6, color)
        elif shape == 'octagon':
            self._draw_polygon(draw, center, size // 2, 8, color)
        elif shape == 'star':
            self._draw_star(draw, center, size // 2, color)
        elif shape == 'rhombus':
            width = size
            height = int(size * random.uniform(1.2, 1.8))
            self._draw_rhombus(draw, center, width, height, color)
        elif shape == 'ellipse':
            width = size
            height = int(size * random.uniform(0.5, 0.8))
            self._draw_ellipse(draw, center, width, height, color)

        # Add some random noise and effects
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(0.5, 1.5)))

        # Convert to numpy array
        img_array = np.array(img)

        # Apply advanced augmentation
        if apply_augmentation and random.random() < 0.7:
            augmented = self.augmentation(image=img_array)
            img_array = augmented['image']

        return img_array

    def generate_dataset(self, output_dir: str, train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        Generate complete dataset with train/val/test splits

        Args:
            output_dir: Directory to save the dataset
            train_val_test_split: Tuple of (train, val, test) split ratios
        """
        print(f"🎨 Generating geometric shapes dataset...")
        print(
            f"📊 {len(self.SHAPES)} classes, {self.shapes_per_class} samples per class")

        # Create directories
        splits = ['train', 'val', 'test']
        for split in splits:
            for shape in self.SHAPES:
                os.makedirs(os.path.join(
                    output_dir, split, shape), exist_ok=True)

        # Calculate split sizes
        train_size = int(self.shapes_per_class * train_val_test_split[0])
        val_size = int(self.shapes_per_class * train_val_test_split[1])
        test_size = self.shapes_per_class - train_size - val_size

        # Dataset metadata
        metadata = {
            'shapes': self.SHAPES,
            'total_samples': len(self.SHAPES) * self.shapes_per_class,
            'samples_per_class': self.shapes_per_class,
            'img_size': self.img_size,
            'splits': {
                'train': train_size,
                'val': val_size,
                'test': test_size
            }
        }

        # Generate images for each shape
        for shape in self.SHAPES:
            print(f"\n🔹 Generating {shape}s...")

            for i in tqdm(range(self.shapes_per_class)):
                # Determine split
                if i < train_size:
                    split = 'train'
                    apply_aug = True
                elif i < train_size + val_size:
                    split = 'val'
                    apply_aug = False
                else:
                    split = 'test'
                    apply_aug = False

                # Generate image
                img_array = self.generate_shape_image(
                    shape, apply_augmentation=apply_aug)

                # Save image
                img = Image.fromarray(img_array)
                filename = f"{shape}_{i:04d}.png"
                filepath = os.path.join(output_dir, split, shape, filename)
                img.save(filepath)

        # Save metadata
        metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"\n✅ Dataset generation complete!")
        print(f"📁 Saved to: {output_dir}")
        print(f"📊 Total images: {metadata['total_samples']}")
        print(f"   - Train: {train_size * len(self.SHAPES)}")
        print(f"   - Val: {val_size * len(self.SHAPES)}")
        print(f"   - Test: {test_size * len(self.SHAPES)}")


if __name__ == "__main__":
    # Example usage
    generator = GeometricShapeGenerator(
        img_size=224,
        min_shape_size=50,
        max_shape_size=180,
        shapes_per_class=1000
    )

    generator.generate_dataset('../../data/synthetic')
