"""Data loading and preprocessing utilities"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple, Dict
import json


class DataLoader:
    """
    Advanced data loader with preprocessing and augmentation
    """

    def __init__(self, data_dir: str, img_size: int = 224, batch_size: int = 32):
        """
        Initialize data loader

        Args:
            data_dir: Path to dataset directory
            img_size: Size to resize images to
            batch_size: Batch size for data loading
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size

        # Load metadata
        metadata_path = os.path.join(data_dir, 'dataset_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

    def create_data_generators(self, augment_train: bool = True) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, ...]:
        """
        Create data generators for train, validation, and test sets

        Args:
            augment_train: Whether to apply augmentation to training data

        Returns:
            Tuple of (train_gen, val_gen, test_gen)
        """
        # Training data generator with augmentation
        if augment_train:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)

        # Validation and test data generators (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)

        # Create generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        val_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'val'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        return train_generator, val_generator, test_generator

    def load_tf_dataset(self, augment_train: bool = True) -> Tuple[tf.data.Dataset, ...]:
        """
        Create TensorFlow datasets (more efficient for large datasets)

        Args:
            augment_train: Whether to apply augmentation to training data

        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        def parse_image(filename, label):
            image = tf.io.read_file(filename)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.resize(image, [self.img_size, self.img_size])
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        def augment(image, label):
            # Random flip
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            # Random rotation
            image = tf.image.rot90(image, k=tf.random.uniform(
                shape=[], minval=0, maxval=4, dtype=tf.int32))
            # Random brightness
            image = tf.image.random_brightness(image, max_delta=0.2)
            # Random contrast
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            return image, label

        # Create datasets for each split
        datasets = {}
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.data_dir, split)

            # Get all image paths and labels
            image_paths = []
            labels = []
            class_names = sorted(os.listdir(split_dir))

            for class_idx, class_name in enumerate(class_names):
                class_dir = os.path.join(split_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            image_paths.append(
                                os.path.join(class_dir, img_name))
                            labels.append(class_idx)

            # Create dataset
            ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
            ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

            # Apply augmentation to training set
            if split == 'train' and augment_train:
                ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
                ds = ds.shuffle(buffer_size=1000)

            # One-hot encode labels
            num_classes = len(class_names)
            ds = ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))

            # Batch and prefetch
            ds = ds.batch(self.batch_size)
            ds = ds.prefetch(tf.data.AUTOTUNE)

            datasets[split] = ds

        return datasets['train'], datasets['val'], datasets['test']

    def get_class_names(self) -> list:
        """Get list of class names"""
        train_dir = os.path.join(self.data_dir, 'train')
        return sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    def get_dataset_info(self) -> Dict:
        """Get information about the dataset"""
        info = {
            'class_names': self.get_class_names(),
            'num_classes': len(self.get_class_names()),
            'img_size': self.img_size,
            'batch_size': self.batch_size
        }

        if self.metadata:
            info.update(self.metadata)

        return info
