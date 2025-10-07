"""
Advanced CNN architectures for geometric shape classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from typing import Tuple, Optional


class ResidualBlock(layers.Layer):
    """
    Residual block with skip connections (inspired by ResNet)
    """
    
    def __init__(self, filters: int, kernel_size: int = 3, stride: int = 1, 
                 use_batch_norm: bool = True, dropout_rate: float = 0.3, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # First conv layer
        self.conv1 = layers.Conv2D(
            filters, kernel_size, strides=stride, padding='same',
            kernel_regularizer=regularizers.l2(1e-4)
        )
        self.bn1 = layers.BatchNormalization() if use_batch_norm else None
        self.activation1 = layers.Activation('relu')
        
        # Second conv layer
        self.conv2 = layers.Conv2D(
            filters, kernel_size, strides=1, padding='same',
            kernel_regularizer=regularizers.l2(1e-4)
        )
        self.bn2 = layers.BatchNormalization() if use_batch_norm else None
        
        # Shortcut connection
        self.shortcut = None
        if stride != 1:
            self.shortcut = layers.Conv2D(
                filters, 1, strides=stride, padding='same',
                kernel_regularizer=regularizers.l2(1e-4)
            )
            self.bn_shortcut = layers.BatchNormalization() if use_batch_norm else None
        
        self.activation2 = layers.Activation('relu')
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def call(self, inputs, training=None):
        # Main path
        x = self.conv1(inputs)
        if self.bn1:
            x = self.bn1(x, training=training)
        x = self.activation1(x)
        
        x = self.conv2(x)
        if self.bn2:
            x = self.bn2(x, training=training)
        
        # Shortcut path
        shortcut = inputs
        if self.shortcut:
            shortcut = self.shortcut(inputs)
            if self.bn_shortcut:
                shortcut = self.bn_shortcut(shortcut, training=training)
        
        # Add shortcut to main path
        x = layers.add([x, shortcut])
        x = self.activation2(x)
        
        if self.dropout and training:
            x = self.dropout(x, training=training)
        
        return x


class AttentionBlock(layers.Layer):
    """
    Channel attention mechanism (inspired by SE-Net)
    """
    
    def __init__(self, filters: int, reduction_ratio: int = 16, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.reduction_ratio = reduction_ratio
        
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(filters // reduction_ratio, activation='relu')
        self.dense2 = layers.Dense(filters, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, filters))
        self.multiply = layers.Multiply()
    
    def call(self, inputs):
        # Squeeze
        x = self.global_avg_pool(inputs)
        # Excitation
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)
        # Scale
        return self.multiply([inputs, x])


class CustomCNN:
    """
    Custom CNN architecture with advanced features
    """
    
    @staticmethod
    def build_model(input_shape: Tuple[int, int, int] = (224, 224, 3),
                   num_classes: int = 10,
                   use_attention: bool = True,
                   dropout_rate: float = 0.3) -> keras.Model:
        """
        Build custom CNN model with ResNet-style blocks
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            use_attention: Whether to use attention mechanism
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=input_shape)
        
        # Initial convolution
        x = layers.Conv2D(64, 7, strides=2, padding='same', 
                         kernel_regularizer=regularizers.l2(1e-4))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks - Stage 1
        x = ResidualBlock(64, dropout_rate=dropout_rate)(x)
        x = ResidualBlock(64, dropout_rate=dropout_rate)(x)
        if use_attention:
            x = AttentionBlock(64)(x)
        
        # Residual blocks - Stage 2
        x = ResidualBlock(128, stride=2, dropout_rate=dropout_rate)(x)
        x = ResidualBlock(128, dropout_rate=dropout_rate)(x)
        x = ResidualBlock(128, dropout_rate=dropout_rate)(x)
        if use_attention:
            x = AttentionBlock(128)(x)
        
        # Residual blocks - Stage 3
        x = ResidualBlock(256, stride=2, dropout_rate=dropout_rate)(x)
        x = ResidualBlock(256, dropout_rate=dropout_rate)(x)
        x = ResidualBlock(256, dropout_rate=dropout_rate)(x)
        x = ResidualBlock(256, dropout_rate=dropout_rate)(x)
        if use_attention:
            x = AttentionBlock(256)(x)
        
        # Residual blocks - Stage 4
        x = ResidualBlock(512, stride=2, dropout_rate=dropout_rate)(x)
        x = ResidualBlock(512, dropout_rate=dropout_rate)(x)
        x = ResidualBlock(512, dropout_rate=dropout_rate)(x)
        if use_attention:
            x = AttentionBlock(512)(x)
        
        # Global pooling and classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation='relu', 
                        kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        x = layers.Dense(256, activation='relu', 
                        kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='CustomCNN')
        
        return model


class LightweightCNN:
    """
    Lightweight CNN for faster training and inference
    """
    
    @staticmethod
    def build_model(input_shape: Tuple[int, int, int] = (224, 224, 3),
                   num_classes: int = 10,
                   dropout_rate: float = 0.3) -> keras.Model:
        """
        Build lightweight CNN model
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(dropout_rate / 2),
            
            # Block 2
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(dropout_rate / 2),
            
            # Block 3
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(dropout_rate / 2),
            
            # Block 4
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Classification head
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate / 2),
            layers.Dense(num_classes, activation='softmax')
        ], name='LightweightCNN')
        
        return model


def create_model(architecture: str = 'custom_cnn',
                input_shape: Tuple[int, int, int] = (224, 224, 3),
                num_classes: int = 10,
                **kwargs) -> keras.Model:
    """
    Factory function to create models
    
    Args:
        architecture: Type of architecture ('custom_cnn', 'lightweight')
        input_shape: Input image shape
        num_classes: Number of output classes
        **kwargs: Additional arguments for specific architectures
        
    Returns:
        Keras model
    """
    if architecture == 'custom_cnn':
        return CustomCNN.build_model(input_shape, num_classes, **kwargs)
    elif architecture == 'lightweight':
        return LightweightCNN.build_model(input_shape, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
