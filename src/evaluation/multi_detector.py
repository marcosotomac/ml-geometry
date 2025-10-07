"""
Advanced multi-object detection for geometric shapes
Uses sliding window and region proposal techniques
"""

import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from tensorflow import keras


class MultiShapeDetector:
    """
    Detector for multiple geometric shapes in a single image
    """
    
    def __init__(self, classifier_model_path: str, config_path: Optional[str] = None):
        """
        Initialize multi-shape detector
        
        Args:
            classifier_model_path: Path to classification model
            config_path: Path to model configuration
        """
        from ..evaluation.predictor import ShapePredictor
        
        self.classifier = ShapePredictor(classifier_model_path, config_path)
        self.img_size = self.classifier.img_size
    
    def detect_shapes_sliding_window(self, 
                                     image: np.ndarray,
                                     window_size: int = 224,
                                     stride: int = 56,
                                     confidence_threshold: float = 0.7) -> List[Dict]:
        """
        Detect shapes using sliding window approach
        
        Args:
            image: Input image (numpy array)
            window_size: Size of sliding window
            stride: Stride for sliding window
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detected shapes with bounding boxes
        """
        h, w = image.shape[:2]
        detections = []
        
        # Slide window across image
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                # Extract window
                window = image[y:y+window_size, x:x+window_size]
                
                # Predict
                result = self.classifier.predict(window)
                
                # If confident enough, add detection
                if result['confidence'] >= confidence_threshold:
                    detections.append({
                        'class': result.get('class', f"Class {result['class_idx']}"),
                        'confidence': result['confidence'],
                        'bbox': (x, y, x + window_size, y + window_size)
                    })
        
        # Apply non-maximum suppression
        detections = self._non_max_suppression(detections)
        
        return detections
    
    def detect_shapes_region_proposals(self,
                                       image: np.ndarray,
                                       confidence_threshold: float = 0.7) -> List[Dict]:
        """
        Detect shapes using selective search region proposals
        
        Args:
            image: Input image (numpy array)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detected shapes with bounding boxes
        """
        # Convert to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            img_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            img_bgr = (image * 255).astype(np.uint8)
        
        # Create selective search segmentation
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img_bgr)
        ss.switchToSelectiveSearchFast()  # Use fast mode
        
        # Get region proposals
        rects = ss.process()
        
        detections = []
        
        # Process top N proposals
        for i, rect in enumerate(rects[:500]):  # Limit to 500 proposals
            x, y, w, h = rect
            
            # Skip very small regions
            if w < 50 or h < 50:
                continue
            
            # Extract and resize region
            region = image[y:y+h, x:x+w]
            region_resized = cv2.resize(region, (self.img_size, self.img_size))
            
            # Predict
            result = self.classifier.predict(region_resized)
            
            # If confident enough, add detection
            if result['confidence'] >= confidence_threshold:
                detections.append({
                    'class': result.get('class', f"Class {result['class_idx']}"),
                    'confidence': result['confidence'],
                    'bbox': (x, y, x + w, y + h)
                })
        
        # Apply non-maximum suppression
        detections = self._non_max_suppression(detections)
        
        return detections
    
    def detect_shapes_contours(self,
                               image: np.ndarray,
                               confidence_threshold: float = 0.7) -> List[Dict]:
        """
        Detect shapes using contour detection
        
        Args:
            image: Input image (numpy array)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detected shapes with bounding boxes
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small regions
            if w < 30 or h < 30:
                continue
            
            # Add padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            # Extract and resize region
            region = image[y1:y2, x1:x2]
            region_resized = cv2.resize(region, (self.img_size, self.img_size))
            
            # Predict
            result = self.classifier.predict(region_resized)
            
            # If confident enough, add detection
            if result['confidence'] >= confidence_threshold:
                detections.append({
                    'class': result.get('class', f"Class {result['class_idx']}"),
                    'confidence': result['confidence'],
                    'bbox': (x1, y1, x2, y2),
                    'contour': contour
                })
        
        return detections
    
    def _non_max_suppression(self, 
                            detections: List[Dict], 
                            iou_threshold: float = 0.3) -> List[Dict]:
        """
        Apply non-maximum suppression to remove overlapping detections
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        while len(detections) > 0:
            # Keep highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [
                det for det in detections
                if self._calculate_iou(current['bbox'], det['bbox']) < iou_threshold
            ]
        
        return keep
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union (IoU) between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_detections(self,
                            image: np.ndarray,
                            detections: List[Dict],
                            save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detections on image
        
        Args:
            image: Input image
            detections: List of detections
            save_path: Path to save visualization (optional)
            
        Returns:
            Image with visualizations
        """
        # Create copy of image
        vis_image = image.copy()
        if vis_image.max() <= 1.0:
            vis_image = (vis_image * 255).astype(np.uint8)
        else:
            vis_image = vis_image.astype(np.uint8)
        
        # Define colors for different classes
        colors = {
            'circle': (255, 0, 0),
            'square': (0, 255, 0),
            'rectangle': (0, 0, 255),
            'triangle': (255, 255, 0),
            'pentagon': (255, 0, 255),
            'hexagon': (0, 255, 255),
            'octagon': (128, 0, 128),
            'star': (255, 128, 0),
            'rhombus': (0, 128, 255),
            'ellipse': (128, 255, 0)
        }
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # Get color
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(vis_image, 
                         (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)),
                         color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"✅ Visualization saved to: {save_path}")
        
        return vis_image


class ShapeSegmentator:
    """
    Semantic segmentation for geometric shapes
    """
    
    def __init__(self, model_path: str):
        """
        Initialize segmentator
        
        Args:
            model_path: Path to segmentation model
        """
        self.model = keras.models.load_model(model_path)
        self.img_size = self.model.input_shape[1]
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Perform semantic segmentation
        
        Args:
            image: Input image
            
        Returns:
            Segmentation mask
        """
        # Preprocess
        img_resized = cv2.resize(image, (self.img_size, self.img_size))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        mask = self.model.predict(img_batch)[0]
        
        # Resize to original size
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        return mask_resized
    
    def visualize_segmentation(self,
                              image: np.ndarray,
                              mask: np.ndarray,
                              save_path: Optional[str] = None):
        """
        Visualize segmentation results
        
        Args:
            image: Original image
            mask: Segmentation mask
            save_path: Path to save visualization (optional)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(mask, cmap='viridis')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Overlay
        overlay = image.copy()
        if overlay.max() <= 1.0:
            overlay = (overlay * 255).astype(np.uint8)
        
        mask_colored = (plt.cm.viridis(mask)[:, :, :3] * 255).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Segmentation visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
