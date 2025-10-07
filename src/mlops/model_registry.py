"""
Model Registry for versioning and lifecycle management
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import hashlib


class ModelRegistry:
    """
    Centralized model registry for version control and lifecycle management
    """
    
    STAGES = ['development', 'staging', 'production', 'archived']
    
    def __init__(self, registry_path: str = 'models/registry'):
        """
        Initialize model registry
        
        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = registry_path
        self.metadata_file = os.path.join(registry_path, 'registry.json')
        
        # Create registry directory
        os.makedirs(registry_path, exist_ok=True)
        
        # Load or initialize metadata
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'models': {}, 'versions': []}
            self._save_metadata()
    
    def register_model(self, 
                      model_path: str,
                      model_name: str,
                      version: str,
                      metadata: Optional[Dict] = None,
                      stage: str = 'development') -> str:
        """
        Register a new model version
        
        Args:
            model_path: Path to model file
            model_name: Name of the model
            version: Version string (e.g., '1.0.0')
            metadata: Additional metadata
            stage: Lifecycle stage
            
        Returns:
            Registry ID
        """
        if stage not in self.STAGES:
            raise ValueError(f"Stage must be one of {self.STAGES}")
        
        # Generate registry ID
        registry_id = f"{model_name}_v{version}"
        
        # Calculate model hash
        model_hash = self._calculate_hash(model_path)
        
        # Copy model to registry
        registry_model_path = os.path.join(
            self.registry_path, 
            model_name, 
            version,
            os.path.basename(model_path)
        )
        os.makedirs(os.path.dirname(registry_model_path), exist_ok=True)
        shutil.copy2(model_path, registry_model_path)
        
        # Create model entry
        model_entry = {
            'registry_id': registry_id,
            'model_name': model_name,
            'version': version,
            'path': registry_model_path,
            'hash': model_hash,
            'stage': stage,
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Update registry
        if model_name not in self.metadata['models']:
            self.metadata['models'][model_name] = []
        
        self.metadata['models'][model_name].append(model_entry)
        self.metadata['versions'].append(registry_id)
        
        self._save_metadata()
        
        print(f"✅ Model registered: {registry_id}")
        print(f"   Stage: {stage}")
        print(f"   Path: {registry_model_path}")
        
        return registry_id
    
    def promote_model(self, registry_id: str, target_stage: str) -> bool:
        """
        Promote model to different stage
        
        Args:
            registry_id: Registry ID
            target_stage: Target stage
            
        Returns:
            Success status
        """
        if target_stage not in self.STAGES:
            raise ValueError(f"Stage must be one of {self.STAGES}")
        
        # Find model
        model_entry = self._find_model(registry_id)
        
        if not model_entry:
            print(f"❌ Model not found: {registry_id}")
            return False
        
        # Update stage
        old_stage = model_entry['stage']
        model_entry['stage'] = target_stage
        model_entry['promoted_at'] = datetime.now().isoformat()
        
        self._save_metadata()
        
        print(f"✅ Model promoted: {registry_id}")
        print(f"   {old_stage} → {target_stage}")
        
        return True
    
    def get_model(self, 
                  model_name: str, 
                  version: Optional[str] = None,
                  stage: Optional[str] = None) -> Optional[Dict]:
        """
        Get model by name, version, or stage
        
        Args:
            model_name: Name of model
            version: Specific version (optional)
            stage: Filter by stage (optional)
            
        Returns:
            Model entry or None
        """
        if model_name not in self.metadata['models']:
            return None
        
        models = self.metadata['models'][model_name]
        
        # Filter by version
        if version:
            models = [m for m in models if m['version'] == version]
        
        # Filter by stage
        if stage:
            models = [m for m in models if m['stage'] == stage]
        
        if not models:
            return None
        
        # Return latest
        return sorted(models, key=lambda x: x['registered_at'], reverse=True)[0]
    
    def list_models(self, stage: Optional[str] = None) -> List[Dict]:
        """
        List all models
        
        Args:
            stage: Filter by stage (optional)
            
        Returns:
            List of model entries
        """
        all_models = []
        
        for models in self.metadata['models'].values():
            all_models.extend(models)
        
        if stage:
            all_models = [m for m in all_models if m['stage'] == stage]
        
        return sorted(all_models, key=lambda x: x['registered_at'], reverse=True)
    
    def archive_model(self, registry_id: str) -> bool:
        """
        Archive a model
        
        Args:
            registry_id: Registry ID
            
        Returns:
            Success status
        """
        return self.promote_model(registry_id, 'archived')
    
    def delete_model(self, registry_id: str) -> bool:
        """
        Delete model from registry
        
        Args:
            registry_id: Registry ID
            
        Returns:
            Success status
        """
        model_entry = self._find_model(registry_id)
        
        if not model_entry:
            print(f"❌ Model not found: {registry_id}")
            return False
        
        # Remove from metadata
        model_name = model_entry['model_name']
        self.metadata['models'][model_name] = [
            m for m in self.metadata['models'][model_name]
            if m['registry_id'] != registry_id
        ]
        
        self.metadata['versions'].remove(registry_id)
        
        # Delete files
        model_dir = os.path.dirname(model_entry['path'])
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        
        self._save_metadata()
        
        print(f"✅ Model deleted: {registry_id}")
        
        return True
    
    def _find_model(self, registry_id: str) -> Optional[Dict]:
        """Find model by registry ID"""
        for models in self.metadata['models'].values():
            for model in models:
                if model['registry_id'] == registry_id:
                    return model
        return None
    
    def _calculate_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
