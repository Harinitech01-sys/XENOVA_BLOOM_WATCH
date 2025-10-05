"""
Tests for predictive modeling functionality
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictive_modeling import BloomPredictor, create_sample_training_data

class TestPredictiveModeling:
    
    def setup_method(self):
        """Setup for each test method"""
        self.predictor = BloomPredictor()
        self.sample_data = create_sample_training_data()
    
    def test_model_initialization(self):
        """Test model initialization"""
        # Test different model types
        for model_type in ['random_forest', 'gradient_boosting', 'neural_network']:
            predictor = BloomPredictor(model_type=model_type)
            assert predictor.model_type == model_type
            assert predictor.model is not None
            assert not predictor.is_trained
    
    def test_feature_creation_basic(self):
        """Test basic feature creation"""
        # Create simple test data
        dates = pd.date_range('2023-06-01', '2023-06-30', freq='D')
        ndvi_data = pd.DataFrame({
            'date': dates,
            'ndvi': np.linspace(0.3, 0.8, len(dates))  # Increasing NDVI
        })
        
        features = self.predictor.create_features_from_timeseries(ndvi_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(ndvi_data)
        assert 'current_ndvi' in features.columns
        assert 'ndvi_mean_14d' in features.columns
        assert 'month' in features.columns
        assert 'season' in features.columns
    
    def test_feature_creation_with_trend(self):
        """Test feature creation with trend calculation"""
        dates = pd.date_range('2023-01-01', '2023-01-20', freq='D')
        # Create data with clear trend
        ndvi_values = np.linspace(0.2, 0.8, len(dates))
        ndvi_data = pd.DataFrame({
            'date': dates,
            'ndvi': ndvi_values
        })
        
        features = self.predictor.create_features_from_timeseries(ndvi_data)
        
        # Check that trend is positive for increasing NDVI
        assert features['ndvi_trend_14d'].iloc[-1] > 0
        assert 'ndvi_change_rate' in features.columns
    
    def test_seasonal_encoding(self):
        """Test seasonal encoding"""
        # Test all months
        test_months = [1, 4, 7, 10]  # Winter, Spring, Summer, Autumn
        expected_seasons = [0, 1, 2, 3]
        
        for month, expected_season in zip(test_months, expected_seasons):
            season = self.predictor._get_season(month)
            assert season == expected_season
    
    def test_training_with_sample_data(self):
        """Test training with sample data"""
        # Use the create_sample_training_data function
        sample_features = create_sample_training_data()
        
        # Create features
        ndvi_data = sample_features[['date', 'ndvi']].head(100)  # Use subset for faster testing
        features = self.predictor.create_features_from_timeseries(ndvi_data)
        features['bloom_probability'] = sample_features['bloom_probability'].head(len(features))
        
        # Train model
        results = self.predictor.train(features)
        
        assert self.predictor.is_trained
        assert 'test_r2' in results
        assert 'train_r2' in results
        assert results['n_samples'] > 0
    
    def test_prediction_after_training(self):
        """Test making predictions after training"""
        # Train model first
        sample_features = create_sample_training_data()
        ndvi_data = sample_features[['date', 'ndvi']].head(50)
        features = self.predictor.create_features_from_timeseries(ndvi_data)
        features['bloom_probability'] = sample_features['bloom_probability'].head(len(features))
        
        self.predictor.train(features)
        
        # Make predictions
        test_features = features.drop(columns=['bloom_probability']).head(5)
        predictions = self.predictor.predict_bloom_probability(test_features)
        
        assert len(predictions) == 5
        assert all(isinstance(p, (int, float)) for p in predictions)
        assert all(0 <= p <= 1 for p in predictions)
    
    def test_bloom_timing_prediction(self):
        """Test bloom timing prediction"""
        # Train model first
        sample_features = create_sample_training_data()
        ndvi_data = sample_features[['date', 'ndvi']].head(30)
        features = self.predictor.create_features_from_timeseries(ndvi_data)
        features['bloom_probability'] = sample_features['bloom_probability'].head(len(features))
        
        self.predictor.train(features)
        
        # Test bloom timing prediction
        test_features = features.drop(columns=['bloom_probability']).head(1)
        timing_predictions = self.predictor.predict_bloom_timing(test_features, days_ahead=7)
        
        assert isinstance(timing_predictions, dict)
        assert len(timing_predictions) == 7
        
        # Check that all predictions have required keys
        for date_str, prediction in timing_predictions.items():
            assert 'bloom_probability' in prediction
            assert 'confidence' in prediction
            assert 0 <= prediction['bloom_probability'] <= 1
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        # Train model first
        sample_features = create_sample_training_data()
        ndvi_data = sample_features[['date', 'ndvi']].head(50)
        features = self.predictor.create_features_from_timeseries(ndvi_data)
        features['bloom_probability'] = sample_features['bloom_probability'].head(len(features))
        
        self.predictor.train(features)
        
        # Get feature importance
        importance = self.predictor.get_feature_importance()
        
        if self.predictor.model_type in ['random_forest', 'gradient_boosting']:
            assert isinstance(importance, dict)
            assert len(importance) > 0
        else:
            # Neural network doesn't have feature_importances_
            assert 'message' in importance
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        # Train model
        sample_features = create_sample_training_data()
        ndvi_data = sample_features[['date', 'ndvi']].head(30)
        features = self.predictor.create_features_from_timeseries(ndvi_data)
        features['bloom_probability'] = sample_features['bloom_probability'].head(len(features))
        
        self.predictor.train(features)
        
        # Save model
        test_file = 'test_bloom_model.pkl'
        self.predictor.save_model(test_file)
        
        # Load model in new instance
        new_predictor = BloomPredictor()
        new_predictor.load_model(test_file)
        
        assert new_predictor.is_trained
        assert new_predictor.model_type == self.predictor.model_type
        assert new_predictor.feature_names == self.predictor.feature_names
        
        # Test that loaded model can make predictions
        test_features = features.drop(columns=['bloom_probability']).head(3)
        original_predictions = self.predictor.predict_bloom_probability(test_features)
        loaded_predictions = new_predictor.predict_bloom_probability(test_features)
        
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions, decimal=5)
        
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
    
    def test_error_handling(self):
        """Test error handling scenarios"""
        # Test prediction without training
        with pytest.raises(ValueError, match="Model must be trained"):
            test_features = pd.DataFrame({'current_ndvi': [0.5]})
            self.predictor.predict_bloom_probability(test_features)
        
        # Test training with invalid data
        with pytest.raises(ValueError):
            invalid_features = pd.DataFrame({'invalid_column': [1, 2, 3]})
            self.predictor.train(invalid_features, target_column='missing_target')

if __name__ == '__main__':
    pytest.main([__file__])
