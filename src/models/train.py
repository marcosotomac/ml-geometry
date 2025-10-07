"""
Advanced training pipeline with callbacks and learning rate scheduling
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger, LearningRateScheduler
)
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Tuple
import json

from .architectures import create_model
from .transfer_learning import create_transfer_learning_model
from ..data.data_loader import DataLoader


class CustomCallback(keras.callbacks.Callback):
    """Custom callback for logging and monitoring"""

    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir
        self.epoch_logs = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_logs.append({
            'epoch': epoch + 1,
            'loss': float(logs.get('loss', 0)),
            'accuracy': float(logs.get('accuracy', 0)),
            'val_loss': float(logs.get('val_loss', 0)),
            'val_accuracy': float(logs.get('val_accuracy', 0)),
            'learning_rate': float(keras.backend.get_value(self.model.optimizer.lr))
        })

        # Print progress
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} Summary:")
        print(
            f"  Train Loss: {logs.get('loss', 0):.4f} | Train Acc: {logs.get('accuracy', 0):.4f}")
        print(
            f"  Val Loss: {logs.get('val_loss', 0):.4f} | Val Acc: {logs.get('val_accuracy', 0):.4f}")
        print(
            f"  Learning Rate: {keras.backend.get_value(self.model.optimizer.lr):.6f}")
        print(f"{'='*60}\n")

    def on_train_end(self, logs=None):
        # Save training history
        history_path = os.path.join(self.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.epoch_logs, f, indent=4)


class TrainingPipeline:
    """
    Advanced training pipeline with all the bells and whistles
    """

    def __init__(self, data_dir: str, output_dir: str = 'models/saved_models',
                 log_dir: str = 'logs'):
        """
        Initialize training pipeline

        Args:
            data_dir: Directory containing the dataset
            output_dir: Directory to save trained models
            log_dir: Directory for logs
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.log_dir = log_dir

        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize data loader
        self.data_loader = DataLoader(data_dir)
        self.dataset_info = self.data_loader.get_dataset_info()

    def create_callbacks(self, model_name: str, patience: int = 15) -> list:
        """
        Create training callbacks

        Args:
            model_name: Name for the model (used in filenames)
            patience: Patience for early stopping

        Returns:
            List of callbacks
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{timestamp}"

        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),

            # Model checkpoint
            ModelCheckpoint(
                filepath=os.path.join(self.output_dir, f'{run_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),

            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),

            # TensorBoard
            TensorBoard(
                log_dir=os.path.join(self.log_dir, 'tensorboard', run_name),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),

            # CSV Logger
            CSVLogger(
                filename=os.path.join(
                    self.log_dir, f'{run_name}_training.csv'),
                separator=',',
                append=False
            ),

            # Custom callback
            CustomCallback(log_dir=os.path.join(self.log_dir, run_name))
        ]

        return callbacks

    def compile_model(self, model: keras.Model,
                      learning_rate: float = 0.001,
                      optimizer: str = 'adam') -> keras.Model:
        """
        Compile model with optimizer and metrics

        Args:
            model: Keras model to compile
            learning_rate: Initial learning rate
            optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')

        Returns:
            Compiled model
        """
        # Choose optimizer
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=0.9, nesterov=True)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Compile model
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(
                    k=3, name='top_3_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )

        return model

    def train(self,
              model_type: str = 'custom_cnn',
              base_model: Optional[str] = None,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              optimizer: str = 'adam',
              patience: int = 15,
              use_augmentation: bool = True,
              **model_kwargs) -> Tuple[keras.Model, dict]:
        """
        Train model

        Args:
            model_type: Type of model ('custom_cnn', 'lightweight', 'transfer')
            base_model: Base model for transfer learning
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            optimizer: Optimizer name
            patience: Patience for early stopping
            use_augmentation: Whether to use data augmentation
            **model_kwargs: Additional arguments for model creation

        Returns:
            Tuple of (trained model, training history)
        """
        print(f"\n{'='*70}")
        print(f"🚀 Starting Training Pipeline")
        print(f"{'='*70}")
        print(f"Model Type: {model_type}")
        print(f"Dataset: {self.data_dir}")
        print(f"Classes: {self.dataset_info['num_classes']}")
        print(f"Batch Size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Optimizer: {optimizer}")
        print(f"{'='*70}\n")

        # Update data loader batch size
        self.data_loader.batch_size = batch_size

        # Load data
        print("📊 Loading data...")
        train_gen, val_gen, test_gen = self.data_loader.create_data_generators(
            augment_train=use_augmentation
        )

        # Create model
        print(f"\n🏗️  Building {model_type} model...")
        input_shape = (self.data_loader.img_size, self.data_loader.img_size, 3)
        num_classes = self.dataset_info['num_classes']

        if model_type == 'transfer':
            if base_model is None:
                base_model = 'efficientnet_b0'
            model = create_transfer_learning_model(
                base_model=base_model,
                input_shape=input_shape,
                num_classes=num_classes,
                **model_kwargs
            )
        else:
            model = create_model(
                architecture=model_type,
                input_shape=input_shape,
                num_classes=num_classes,
                **model_kwargs
            )

        # Print model summary
        print("\n📋 Model Architecture:")
        model.summary()

        # Compile model
        print("\n⚙️  Compiling model...")
        model = self.compile_model(model, learning_rate, optimizer)

        # Create callbacks
        model_name = f"{model_type}_{base_model}" if model_type == 'transfer' else model_type
        callbacks = self.create_callbacks(model_name, patience)

        # Train model
        print("\n🎯 Training model...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate on test set
        print("\n📈 Evaluating on test set...")
        test_results = model.evaluate(test_gen, verbose=1)
        print(f"\n✅ Test Results:")
        print(f"  Test Loss: {test_results[0]:.4f}")
        print(f"  Test Accuracy: {test_results[1]:.4f}")

        # Save final model
        final_model_path = os.path.join(
            self.output_dir, f'{model_name}_final.h5')
        model.save(final_model_path)
        print(f"\n💾 Model saved to: {final_model_path}")

        # Save training config
        config = {
            'model_type': model_type,
            'base_model': base_model,
            'num_classes': num_classes,
            'input_shape': list(input_shape),
            'epochs_trained': len(history.history['loss']),
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'test_accuracy': float(test_results[1]),
            'test_loss': float(test_results[0]),
            'class_names': self.dataset_info['class_names']
        }

        config_path = os.path.join(
            self.output_dir, f'{model_name}_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        print(f"\n{'='*70}")
        print(f"✨ Training Complete!")
        print(f"{'='*70}\n")

        return model, history.history


def train_model(data_dir: str,
                model_type: str = 'custom_cnn',
                **kwargs) -> Tuple[keras.Model, dict]:
    """
    Convenience function to train a model

    Args:
        data_dir: Directory containing the dataset
        model_type: Type of model to train
        **kwargs: Additional training arguments

    Returns:
        Tuple of (trained model, history)
    """
    pipeline = TrainingPipeline(data_dir)
    return pipeline.train(model_type=model_type, **kwargs)
