"""
Utility functions for BloomWatch Prediction Service
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from ..config import Config

logger = logging.getLogger(__name__)

class PredictionUtils:
    """Utility class for prediction-related operations"""
    
    def __init__(self):
        pass
    
    def create_sample_features(self):
        """Create sample feature matrix for demonstration"""
        np.random.seed(42)
        
        # Generate sample data
        n_samples = 1000
        data = {
            'current_ndvi': np.random.uniform(0.2, 0.9, n_samples),
            'ndvi_mean_14d': np.random.uniform(0.2, 0.8, n_samples),
            'ndvi_std_14d': np.random.uniform(0.05, 0.3, n_samples),
            'ndvi_max_14d': np.random.uniform(0.3, 0.9, n_samples),
            'ndvi_min_14d': np.random.uniform(0.1, 0.7, n_samples),
            'ndvi_trend_14d': np.random.uniform(-0.05, 0.05, n_samples),
            'ndvi_change_rate': np.random.uniform(-0.1, 0.1, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'day_of_year': np.random.randint(1, 366, n_samples),
            'season': np.random.randint(0, 4, n_samples),
            'above_bloom_threshold': np.random.randint(0, 2, n_samples),
            'rapid_increase': np.random.randint(0, 2, n_samples),
            'temperature_avg': np.random.uniform(10, 35, n_samples),
            'precipitation': np.random.uniform(0, 100, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples)
        }
        
        # Create target variable based on features
        bloom_probability = []
        for i in range(n_samples):
            prob = 0.0
            
            # NDVI contribution
            if data['current_ndvi'][i] > Config.NDVI_BLOOM_THRESHOLD:
                prob += 0.4
            
            # Trend contribution
            if data['ndvi_trend_14d'][i] > 0.01:
                prob += 0.3
            
            # Seasonal contribution
            if data['season'][i] in [1, 2]:  # Spring, Summer
                prob += 0.2
            
            # Change rate contribution
            if data['ndvi_change_rate'][i] > 0.05:
                prob += 0.1
            
            bloom_probability.append(min(prob + np.random.normal(0, 0.1), 1.0))
        
        data['bloom_probability'] = bloom_probability
        
        return pd.DataFrame(data)
    
    def prepare_prediction_features(self, input_features, region, start_date, end_date):
        """
        Prepare feature matrix for prediction
        
        Args:
            input_features (dict): Input feature values
            region (str): Region name
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            pd.DataFrame: Prepared feature matrix
        """
        try:
            # Parse dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Create date range
            date_range = pd.date_range(start_dt, end_dt, freq='D')
            
            features_list = []
            for date in date_range:
                feature_dict = {
                    # Use provided features or defaults
                    'current_ndvi': input_features.get('current_ndvi', 0.5),
                    'ndvi_mean_14d': input_features.get('ndvi_mean_14d', 0.45),
                    'ndvi_std_14d': input_features.get('ndvi_std_14d', 0.1),
                    'ndvi_max_14d': input_features.get('ndvi_max_14d', 0.6),
                    'ndvi_min_14d': input_features.get('ndvi_min_14d', 0.3),
                    'ndvi_trend_14d': input_features.get('ndvi_trend_14d', 0.01),
                    'ndvi_change_rate': input_features.get('ndvi_change_rate', 0.02),
                    
                    # Temporal features
                    'month': date.month,
                    'day_of_year': date.timetuple().tm_yday,
                    'season': self._get_season(date.month),
                    
                    # Threshold features
                    'above_bloom_threshold': 1 if input_features.get('current_ndvi', 0.5) > Config.NDVI_BLOOM_THRESHOLD else 0,
                    'rapid_increase': 1 if input_features.get('ndvi_change_rate', 0.0) > Config.TEMPORAL_CHANGE_THRESHOLD else 0,
                    
                    # Weather features
                    'temperature_avg': input_features.get('temperature_avg', 20.0),
                    'precipitation': input_features.get('precipitation', 10.0),
                    'humidity': input_features.get('humidity', 60.0)
                }
                
                features_list.append(feature_dict)
            
            return pd.DataFrame(features_list)
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {str(e)}")
            return pd.DataFrame()
    
    def _get_season(self, month):
        """Convert month to season number"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn
    
    def calculate_quick_bloom_probability(self, ndvi, region, date_str):
        """
        Calculate quick bloom probability using rule-based approach
        
        Args:
            ndvi (float): Current NDVI value
            region (str): Region name
            date_str (str): Target date
            
        Returns:
            float: Bloom probability (0-1)
        """
        try:
            date = pd.to_datetime(date_str)
            probability = 0.0
            
            # NDVI factor (0-0.5)
            if ndvi > Config.NDVI_BLOOM_THRESHOLD:
                probability += 0.4 * (ndvi - Config.NDVI_BLOOM_THRESHOLD) / (1.0 - Config.NDVI_BLOOM_THRESHOLD)
            
            # Seasonal factor (0-0.3)
            seasonal_boost = self.get_seasonal_factor(date_str, region)
            probability += 0.3 * seasonal_boost
            
            # Region factor (0-0.2)
            region_boost = self.get_region_bloom_potential(region)
            probability += 0.2 * region_boost
            
            return min(probability, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating quick bloom probability: {str(e)}")
            return 0.5  # Default probability
    
    def get_seasonal_factor(self, date_str, region):
        """
        Get seasonal bloom factor for region
        
        Args:
            date_str (str): Target date
            region (str): Region name
            
        Returns:
            float: Seasonal factor (0-1)
        """
        try:
            date = pd.to_datetime(date_str)
            month = date.month
            
            # Define bloom seasons by region
            bloom_seasons = {
                'california': [3, 4, 5, 6],  # Spring to early summer
                'amazon': [9, 10, 11, 12, 1],  # Dry to wet season transition
                'india': [3, 4, 5],  # Spring
                'europe': [4, 5, 6, 7],  # Spring to summer
                'global': [3, 4, 5, 6, 7]  # General spring-summer
            }
            
            region_seasons = bloom_seasons.get(region, bloom_seasons['global'])
            
            if month in region_seasons:
                return 1.0
            elif month in [(m-1) % 12 + 1 for m in region_seasons] or month in [(m+1) % 12 + 1 for m in region_seasons]:
                return 0.5
            else:
                return 0.1
                
        except Exception as e:
            logger.error(f"Error getting seasonal factor: {str(e)}")
            return 0.5
    
    def get_region_bloom_potential(self, region):
        """
        Get bloom potential for region
        
        Args:
            region (str): Region name
            
        Returns:
            float: Region bloom potential (0-1)
        """
        # Regional bloom potentials based on vegetation characteristics
        region_potentials = {
            'california': 0.8,  # Mediterranean climate, good for blooms
            'amazon': 0.9,      # High biodiversity, frequent blooming
            'india': 0.7,       # Monsoon-dependent, seasonal blooms
            'europe': 0.6,      # Temperate climate, moderate blooms
            'global': 0.7       # Average global potential
        }
        
        return region_potentials.get(region, 0.5)
    
    def validate_input_features(self, features):
        """
        Validate input features
        
        Args:
            features (dict): Input features
            
        Returns:
            tuple: (is_valid, error_message)
        """
        required_features = ['current_ndvi']
        
        # Check required features
        for feature in required_features:
            if feature not in features:
                return False, f"Missing required feature: {feature}"
        
        # Validate NDVI range
        if not (0.0 <= features['current_ndvi'] <= 1.0):
            return False, "NDVI value must be between 0.0 and 1.0"
        
        # Validate optional features
        if 'temperature_avg' in features:
            if not (-50 <= features['temperature_avg'] <= 60):
                return False, "Temperature must be between -50°C and 60°C"
        
        if 'precipitation' in features:
            if not (0 <= features['precipitation'] <= 1000):
                return False, "Precipitation must be between 0 and 1000mm"
        
        return True, "Valid"
    
    def format_prediction_response(self, predictions, region, model_info=None):
        """
        Format prediction response for API
        
        Args:
            predictions (dict): Raw predictions
            region (str): Region name
            model_info (dict): Model information
            
        Returns:
            dict: Formatted response
        """
        # Find peak bloom date
        peak_date = max(predictions.items(), key=lambda x: x[1]['bloom_probability'])[0]
        peak_probability = predictions[peak_date]['bloom_probability']
        
        # Calculate average probability
        avg_probability = np.mean([p['bloom_probability'] for p in predictions.values()])
        
        # Count high probability days
        high_prob_days = sum(1 for p in predictions.values() if p['bloom_probability'] > 0.7)
        
        formatted_response = {
            'summary': {
                'peak_bloom_date': peak_date,
                'peak_probability': peak_probability,
                'average_probability': avg_probability,
                'high_probability_days': high_prob_days,
                'total_days_analyzed': len(predictions)
            },
            'daily_predictions': predictions,
            'region': region,
            'analysis_metadata': {
                'model_info': model_info or {'type': 'rule_based'},
                'confidence_levels': {
                    'high': '>0.8',
                    'medium': '0.5-0.8',
                    'low': '<0.5'
                }
            }
        }
        
        return formatted_response
