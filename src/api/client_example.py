"""
Example script demonstrating API usage
"""

import requests
import json
from pathlib import Path


class APIClient:
    """Client for interacting with the Geometric Shape Detection API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_classes(self):
        """Get available classes"""
        response = requests.get(f"{self.base_url}/classes")
        return response.json()
    
    def predict_image(self, image_path: str, return_probabilities: bool = False):
        """
        Predict shape in image
        
        Args:
            image_path: Path to image file
            return_probabilities: Whether to return all probabilities
            
        Returns:
            Prediction results
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {'return_probabilities': return_probabilities}
            response = requests.post(
                f"{self.base_url}/predict",
                files=files,
                params=params
            )
        return response.json()
    
    def predict_batch(self, image_paths: list):
        """
        Predict shapes in multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Batch prediction results
        """
        files = [
            ('files', open(path, 'rb'))
            for path in image_paths
        ]
        
        response = requests.post(
            f"{self.base_url}/predict/batch",
            files=files
        )
        
        # Close all file handles
        for _, f in files:
            f.close()
        
        return response.json()


def main():
    """Example usage"""
    # Initialize client
    client = APIClient()
    
    # Health check
    print("\n🏥 Health Check:")
    print(json.dumps(client.health_check(), indent=2))
    
    # Get classes
    print("\n📋 Available Classes:")
    print(json.dumps(client.get_classes(), indent=2))
    
    # Example prediction (you need to provide a valid image path)
    # image_path = "path/to/your/image.jpg"
    # print("\n🔮 Prediction:")
    # result = client.predict_image(image_path, return_probabilities=True)
    # print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
