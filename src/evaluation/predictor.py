"""
Model predictor for inference
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import json
import os
from typing import Union, List, Dict, Tuple


class ShapePredictor:
    """
    Predictor class for making inferences on geometric shapes
    """
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model (.h5 or SavedModel)
            config_path: Path to model configuration JSON (optional)
        """
        print(f"🔮 Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.class_names = self.config.get('class_names', [])
            self.input_shape = tuple(self.config.get('input_shape', [224, 224, 3]))
        else:
            # Try to infer from model
            self.config = {}
            self.input_shape = tuple(self.model.input_shape[1:])
            self.class_names = []
        
        self.img_size = self.input_shape[0]
        
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Number of classes: {len(self.class_names)}")
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for prediction
        
        Args:
            image: Image path, numpy array, or PIL Image
            
        Returns:
            Preprocessed image array
        """
        # Load image if path provided
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype('uint8'), 'RGB')
        else:
            img = image.convert('RGB')
        
        # Resize
        img = img.resize((self.img_size, self.img_size))
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image: Union[str, np.ndarray, Image.Image], 
               return_probabilities: bool = False) -> Dict:
        """
        Predict shape in image
        
        Args:
            image: Image to predict
            return_probabilities: Whether to return all class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        img_array = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        
        result = {
            'class_idx': int(predicted_class_idx),
            'confidence': confidence
        }
        
        if self.class_names:
            result['class'] = self.class_names[predicted_class_idx]
        
        if return_probabilities:
            if self.class_names:
                result['probabilities'] = {
                    self.class_names[i]: float(predictions[i])
                    for i in range(len(predictions))
                }
            else:
                result['probabilities'] = {
                    f'class_{i}': float(predictions[i])
                    for i in range(len(predictions))
                }
        
        return result
    
    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict]:
        """
        Predict multiple images
        
        Args:
            images: List of images
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        # Preprocess all images
        img_arrays = [self.preprocess_image(img) for img in images]
        batch = np.vstack(img_arrays)
        
        # Batch prediction
        predictions = self.model.predict(batch, verbose=0)
        
        # Process results
        for pred in predictions:
            predicted_class_idx = np.argmax(pred)
            confidence = float(pred[predicted_class_idx])
            
            result = {
                'class_idx': int(predicted_class_idx),
                'confidence': confidence
            }
            
            if self.class_names:
                result['class'] = self.class_names[predicted_class_idx]
            
            results.append(result)
        
        return results
    
    def predict_with_visualization(self, image_path: str, 
                                   save_path: Optional[str] = None) -> Dict:
        """
        Predict and visualize result
        
        Args:
            image_path: Path to image
            save_path: Path to save visualization (optional)
            
        Returns:
            Prediction dictionary
        """
        import matplotlib.pyplot as plt
        
        # Load and predict
        img = Image.open(image_path).convert('RGB')
        result = self.predict(image_path, return_probabilities=True)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        ax1.imshow(img)
        ax1.axis('off')
        
        predicted_class = result.get('class', f"Class {result['class_idx']}")
        confidence = result['confidence']
        ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2%}',
                     fontsize=12, fontweight='bold')
        
        # Show probability distribution
        if 'probabilities' in result:
            probs = result['probabilities']
            classes = list(probs.keys())
            values = list(probs.values())
            
            colors = ['green' if c == predicted_class else 'skyblue' for c in classes]
            
            ax2.barh(classes, values, color=colors, edgecolor='navy')
            ax2.set_xlabel('Probability', fontsize=10)
            ax2.set_title('Class Probabilities', fontsize=12, fontweight='bold')
            ax2.set_xlim([0, 1])
            ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result


from typing import Optional
