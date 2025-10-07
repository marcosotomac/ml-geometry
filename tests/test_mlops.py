"""
Unit tests for MLOps components
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
from datetime import datetime

from src.mlops.model_registry import ModelRegistry
from src.mlops.model_monitor import ModelMonitor


class TestModelRegistry:
    """Tests for ModelRegistry"""
    
    @pytest.fixture
    def registry_path(self):
        """Create temporary registry directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_model_file(self):
        """Create sample model file"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
        temp_file.write(b'dummy model data')
        temp_file.close()
        yield temp_file.name
        os.remove(temp_file.name)
    
    def test_registry_initialization(self, registry_path):
        """Test registry initialization"""
        registry = ModelRegistry(registry_path=registry_path)
        
        assert os.path.exists(registry_path)
        assert os.path.exists(registry.metadata_file)
        assert 'models' in registry.metadata
        assert 'versions' in registry.metadata
    
    def test_register_model(self, registry_path, sample_model_file):
        """Test model registration"""
        registry = ModelRegistry(registry_path=registry_path)
        
        registry_id = registry.register_model(
            model_path=sample_model_file,
            model_name='test_model',
            version='1.0.0',
            metadata={'accuracy': 0.95},
            stage='development'
        )
        
        assert registry_id == 'test_model_v1.0.0'
        assert 'test_model' in registry.metadata['models']
        assert len(registry.metadata['models']['test_model']) == 1
        
        model_entry = registry.metadata['models']['test_model'][0]
        assert model_entry['version'] == '1.0.0'
        assert model_entry['stage'] == 'development'
        assert model_entry['metadata']['accuracy'] == 0.95
    
    def test_promote_model(self, registry_path, sample_model_file):
        """Test model promotion"""
        registry = ModelRegistry(registry_path=registry_path)
        
        registry_id = registry.register_model(
            model_path=sample_model_file,
            model_name='test_model',
            version='1.0.0',
            stage='development'
        )
        
        # Promote to staging
        success = registry.promote_model(registry_id, 'staging')
        assert success
        
        model_entry = registry.get_model('test_model', version='1.0.0')
        assert model_entry['stage'] == 'staging'
    
    def test_get_model(self, registry_path, sample_model_file):
        """Test get model"""
        registry = ModelRegistry(registry_path=registry_path)
        
        registry.register_model(
            model_path=sample_model_file,
            model_name='test_model',
            version='1.0.0',
            stage='development'
        )
        
        registry.register_model(
            model_path=sample_model_file,
            model_name='test_model',
            version='2.0.0',
            stage='production'
        )
        
        # Get by version
        model_v1 = registry.get_model('test_model', version='1.0.0')
        assert model_v1['version'] == '1.0.0'
        
        # Get by stage
        prod_model = registry.get_model('test_model', stage='production')
        assert prod_model['version'] == '2.0.0'
        assert prod_model['stage'] == 'production'
    
    def test_list_models(self, registry_path, sample_model_file):
        """Test list models"""
        registry = ModelRegistry(registry_path=registry_path)
        
        registry.register_model(
            model_path=sample_model_file,
            model_name='model1',
            version='1.0.0',
            stage='development'
        )
        
        registry.register_model(
            model_path=sample_model_file,
            model_name='model2',
            version='1.0.0',
            stage='production'
        )
        
        # List all
        all_models = registry.list_models()
        assert len(all_models) == 2
        
        # List by stage
        prod_models = registry.list_models(stage='production')
        assert len(prod_models) == 1
        assert prod_models[0]['model_name'] == 'model2'
    
    def test_delete_model(self, registry_path, sample_model_file):
        """Test delete model"""
        registry = ModelRegistry(registry_path=registry_path)
        
        registry_id = registry.register_model(
            model_path=sample_model_file,
            model_name='test_model',
            version='1.0.0',
            stage='development'
        )
        
        success = registry.delete_model(registry_id)
        assert success
        
        model = registry.get_model('test_model', version='1.0.0')
        assert model is None


class TestModelMonitor:
    """Tests for ModelMonitor"""
    
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        monitor = ModelMonitor(model_name='test_model', window_size=100)
        
        assert monitor.model_name == 'test_model'
        assert monitor.window_size == 100
        assert len(monitor.predictions) == 0
    
    def test_log_prediction(self):
        """Test logging predictions"""
        monitor = ModelMonitor()
        
        monitor.log_prediction(
            predicted_class='circle',
            confidence=0.95,
            latency=0.05,
            true_label='circle'
        )
        
        assert len(monitor.predictions) == 1
        assert len(monitor.confidences) == 1
        assert len(monitor.latencies) == 1
        
        pred = monitor.predictions[0]
        assert pred['class'] == 'circle'
        assert pred['confidence'] == 0.95
        assert pred['true_label'] == 'circle'
    
    def test_log_error(self):
        """Test logging errors"""
        monitor = ModelMonitor()
        
        monitor.log_error('prediction_error', 'Test error message')
        
        assert len(monitor.errors) == 1
        error = monitor.errors[0]
        assert error['type'] == 'prediction_error'
        assert error['message'] == 'Test error message'
    
    def test_get_metrics(self):
        """Test getting metrics"""
        monitor = ModelMonitor(window_size=10)
        
        # Log some predictions
        for i in range(5):
            monitor.log_prediction(
                predicted_class='circle',
                confidence=0.9 + i * 0.01,
                latency=0.05 + i * 0.01
            )
        
        metrics = monitor.get_metrics()
        
        assert metrics['total_predictions'] == 5
        assert 'average_confidence' in metrics
        assert 'average_latency_ms' in metrics
        assert 'class_distribution' in metrics
        assert metrics['class_distribution']['circle'] == 5
    
    def test_detect_drift(self):
        """Test drift detection"""
        monitor = ModelMonitor()
        
        # Log predictions
        for _ in range(50):
            monitor.log_prediction('circle', 0.9, 0.05)
        for _ in range(30):
            monitor.log_prediction('square', 0.85, 0.05)
        for _ in range(20):
            monitor.log_prediction('triangle', 0.8, 0.05)
        
        # Baseline distribution
        baseline = {
            'circle': 0.4,
            'square': 0.3,
            'triangle': 0.3
        }
        
        drift_score = monitor.detect_drift(baseline)
        
        assert isinstance(drift_score, float)
        assert drift_score >= 0
    
    def test_window_size_limit(self):
        """Test window size limiting"""
        monitor = ModelMonitor(window_size=10)
        
        # Log more than window size
        for i in range(20):
            monitor.log_prediction(f'class_{i}', 0.9, 0.05)
        
        # Should only keep last 10
        assert len(monitor.predictions) == 10
        assert len(monitor.confidences) == 10
        assert len(monitor.latencies) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
