"""
Transfer learning models with pre-trained weights
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB3,
    ResNet50, ResNet101,
    MobileNetV2, MobileNetV3Large
)
from typing import Tuple, Optional


class TransferLearningModel:
    """
    Transfer learning wrapper for pre-trained models
    """
    
    AVAILABLE_MODELS = {
        'efficientnet_b0': EfficientNetB0,
        'efficientnet_b3': EfficientNetB3,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'mobilenet_v2': MobileNetV2,
        'mobilenet_v3': MobileNetV3Large,
    }
    
    @staticmethod
    def build_model(base_model_name: str = 'efficientnet_b0',
                   input_shape: Tuple[int, int, int] = (224, 224, 3),
                   num_classes: int = 10,
                   fine_tune_layers: Optional[int] = None,
                   dropout_rate: float = 0.3,
                   use_global_pooling: str = 'avg') -> keras.Model:
        """
        Build transfer learning model
        
        Args:
            base_model_name: Name of the pre-trained model
            input_shape: Input image shape
            num_classes: Number of output classes
            fine_tune_layers: Number of layers to fine-tune (None = freeze all)
            dropout_rate: Dropout rate for regularization
            use_global_pooling: Type of global pooling ('avg' or 'max')
            
        Returns:
            Compiled Keras model
        """
        if base_model_name not in TransferLearningModel.AVAILABLE_MODELS:
            raise ValueError(f"Model {base_model_name} not available. Choose from: {list(TransferLearningModel.AVAILABLE_MODELS.keys())}")
        
        # Load pre-trained base model
        base_model_class = TransferLearningModel.AVAILABLE_MODELS[base_model_name]
        base_model = base_model_class(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # If fine-tuning, unfreeze last N layers
        if fine_tune_layers is not None and fine_tune_layers > 0:
            base_model.trainable = True
            # Freeze all layers except the last fine_tune_layers
            for layer in base_model.layers[:-fine_tune_layers]:
                layer.trainable = False
        
        # Build complete model
        inputs = keras.Input(shape=input_shape)
        
        # Preprocessing (some models need specific preprocessing)
        if 'efficientnet' in base_model_name:
            x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        elif 'resnet' in base_model_name:
            x = tf.keras.applications.resnet.preprocess_input(inputs)
        elif 'mobilenet' in base_model_name:
            x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        else:
            x = inputs
        
        # Base model
        x = base_model(x, training=False)
        
        # Global pooling
        if use_global_pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        else:
            x = layers.GlobalMaxPooling2D()(x)
        
        # Classification head
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, 
                          name=f'TransferLearning_{base_model_name}')
        
        return model
    
    @staticmethod
    def unfreeze_model(model: keras.Model, fine_tune_from_layer: int = 0):
        """
        Unfreeze layers for fine-tuning
        
        Args:
            model: Keras model
            fine_tune_from_layer: Layer index to start fine-tuning from
        """
        for layer in model.layers[fine_tune_from_layer:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True


class EnsembleModel:
    """
    Ensemble of multiple models for improved accuracy
    """
    
    @staticmethod
    def build_ensemble(models_list: list, 
                      input_shape: Tuple[int, int, int] = (224, 224, 3),
                      num_classes: int = 10,
                      ensemble_method: str = 'average') -> keras.Model:
        """
        Build ensemble model from multiple base models
        
        Args:
            models_list: List of model names to ensemble
            input_shape: Input image shape
            num_classes: Number of output classes
            ensemble_method: How to combine predictions ('average', 'max', 'weighted')
            
        Returns:
            Ensemble Keras model
        """
        inputs = keras.Input(shape=input_shape)
        
        # Create all base models
        outputs = []
        for model_name in models_list:
            model = TransferLearningModel.build_model(
                base_model_name=model_name,
                input_shape=input_shape,
                num_classes=num_classes
            )
            output = model(inputs)
            outputs.append(output)
        
        # Combine predictions
        if ensemble_method == 'average':
            combined = layers.Average()(outputs)
        elif ensemble_method == 'max':
            combined = layers.Maximum()(outputs)
        elif ensemble_method == 'weighted':
            # Learn weights for each model
            weights = layers.Dense(len(outputs), activation='softmax')(inputs)
            combined = layers.Dot(axes=1)([weights, layers.Concatenate()(outputs)])
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        model = keras.Model(inputs=inputs, outputs=combined, name='EnsembleModel')
        
        return model


def create_transfer_learning_model(base_model: str = 'efficientnet_b0',
                                   input_shape: Tuple[int, int, int] = (224, 224, 3),
                                   num_classes: int = 10,
                                   **kwargs) -> keras.Model:
    """
    Factory function for transfer learning models
    
    Args:
        base_model: Name of pre-trained model
        input_shape: Input image shape
        num_classes: Number of output classes
        **kwargs: Additional arguments
        
    Returns:
        Keras model
    """
    return TransferLearningModel.build_model(
        base_model_name=base_model,
        input_shape=input_shape,
        num_classes=num_classes,
        **kwargs
    )
