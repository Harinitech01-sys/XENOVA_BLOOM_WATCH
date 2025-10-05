import ee
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Add this class to your existing predictive_modeling.py file

class UniversalBloomPredictor:
    """Universal predictor for any location - NASA Challenge Ready"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self._load_model()
    
    def _load_model(self):
        """Load universal model"""
        try:
            from config import Config
            
            self.model = joblib.load(Config.UNIVERSAL_MODEL_PATH)
            self.scaler = joblib.load(Config.UNIVERSAL_SCALER_PATH)
            
            with open(Config.UNIVERSAL_FEATURES_PATH, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            print("✅ Universal model loaded")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
    
    def predict_any_location(self, latitude, longitude, date_str):
        """Predict bloom for ANY location without species knowledge"""
        
        try:
            # This is your main NASA demo function
            # Extract NASA satellite features (implement satellite data extraction)
            # Create feature vector
            # Make prediction
            # Return NASA-formatted result
            
            result = {
                'nasa_challenge': 'Universal BloomWatch',
                'location': {'latitude': latitude, 'longitude': longitude},
                'date': date_str,
                'bloom_prediction': {
                    'will_bloom': True,  # Replace with actual prediction
                    'probability': 0.75,  # Replace with actual probability
                    'confidence': 'high'
                },
                'nasa_data_sources': [
                    'MODIS/061/MOD13Q1',
                    'LANDSAT/LC08/C02/T1_L2',
                    'ECMWF/ERA5_LAND/DAILY_AGGR'
                ]
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e)}


class BloomPredictor:
    """Advanced bloom prediction using NASA satellite data"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'latitude', 'longitude', 'day_of_year', 'month',
            'modis_ndvi', 'modis_evi', 'landsat_ndvi', 'landsat_ndwi',
            'temperature', 'precipitation', 'humidity',
            'latitude_cos', 'season_sin', 'season_cos'
        ]
        self._initialize_ee()
    
    def _initialize_ee(self):
        """Initialize Earth Engine - Disabled for development"""
        # Temporarily disable EE authentication
        logger.info("🔄 Earth Engine disabled - using mock predictions")
        return False

    def load_model(self, model_path=None, scaler_path=None):
        """Load pre-trained model and scaler"""
        
        model_path = model_path or Config.BLOOM_MODEL_PATH
        scaler_path = scaler_path or Config.SCALER_PATH
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✅ Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            logger.warning("⚠️ No trained model found. Train model first.")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def extract_satellite_features(self, latitude, longitude, date_str):
        """Extract real-time satellite features for prediction"""
        
        try:
            # Create point geometry
            point = ee.Geometry.Point([longitude, latitude])
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Date range for recent data
            start_date = date_obj - timedelta(days=30)
            end_date = date_obj + timedelta(days=5)
            
            # MODIS vegetation indices
            modis_collection = ee.ImageCollection(Config.MODIS_COLLECTION) \
                .filterBounds(point) \
                .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
                .select(['NDVI', 'EVI'])
            
            modis_data = modis_collection.median().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=250,
                maxPixels=1e9
            ).getInfo()
            
            # Landsat data
            landsat_collection = ee.ImageCollection(Config.LANDSAT_COLLECTION) \
                .filterBounds(point) \
                .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
                .filterMetadata('CLOUD_COVER', 'less_than', 30) \
                .map(self._calculate_landsat_indices)
            
            landsat_data = landsat_collection.median().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=30,
                maxPixels=1e9
            ).getInfo()
            
            # Weather data
            weather_collection = ee.ImageCollection(Config.WEATHER_COLLECTION) \
                .filterBounds(point) \
                .filterDate((date_obj - timedelta(days=7)).strftime('%Y-%m-%d'), 
                           date_obj.strftime('%Y-%m-%d')) \
                .select(['temperature_2m', 'total_precipitation', 'dewpoint_temperature_2m'])
            
            weather_data = weather_collection.mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=1000,
                maxPixels=1e9
            ).getInfo()
            
            # Process and return features
            features = self._process_satellite_data(
                modis_data, landsat_data, weather_data, 
                latitude, longitude, date_obj
            )
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting satellite features: {e}")
            return None
    
    def _calculate_landsat_indices(self, image):
        """Calculate vegetation indices from Landsat imagery"""
        
        # NDVI: (NIR - Red) / (NIR + Red)
        ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        
        # NDWI: (Green - NIR) / (Green + NIR) - water content
        ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
        
        return image.addBands([ndvi, ndwi])
    
    def _process_satellite_data(self, modis_data, landsat_data, weather_data, lat, lon, date_obj):
        """Process raw satellite data into features"""
        
        # MODIS features (scale from integer values)
        modis_ndvi = modis_data.get('NDVI', 5000) * 0.0001
        modis_evi = modis_data.get('EVI', 3000) * 0.0001
        
        # Landsat features
        landsat_ndvi = landsat_data.get('NDVI', 0.5)
        landsat_ndwi = landsat_data.get('NDWI', 0.0)
        
        # Weather features
        temperature = weather_data.get('temperature_2m', 288) - 273.15  # Kelvin to Celsius
        precipitation = weather_data.get('total_precipitation', 0) * 1000  # m to mm
        humidity = self._calculate_humidity(
            weather_data.get('temperature_2m', 288),
            weather_data.get('dewpoint_temperature_2m', 280)
        )
        
        # Temporal features
        day_of_year = date_obj.timetuple().tm_yday
        month = date_obj.month
        
        # Geographic features
        latitude_cos = np.cos(np.radians(lat))
        season_sin = np.sin(2 * np.pi * month / 12)
        season_cos = np.cos(2 * np.pi * month / 12)
        
        return {
            'latitude': lat,
            'longitude': lon,
            'day_of_year': day_of_year,
            'month': month,
            'modis_ndvi': modis_ndvi,
            'modis_evi': modis_evi,
            'landsat_ndvi': landsat_ndvi,
            'landsat_ndwi': landsat_ndwi,
            'temperature': temperature,
            'precipitation': precipitation,
            'humidity': humidity,
            'latitude_cos': latitude_cos,
            'season_sin': season_sin,
            'season_cos': season_cos
        }
    
    def _calculate_humidity(self, temp_k, dewpoint_k):
        """Calculate relative humidity from temperature and dewpoint"""
        try:
            # Simplified humidity calculation
            humidity = 100 * np.exp((17.625 * (dewpoint_k - 273.15)) / (243.04 + (dewpoint_k - 273.15))) / \
                      np.exp((17.625 * (temp_k - 273.15)) / (243.04 + (temp_k - 273.15)))
            return min(100, max(0, humidity))
        except:
            return 60.0  # Default humidity
    
    def predict_bloom(self, latitude, longitude, date_str):
        """Predict bloom probability for given location and date"""
        
        if not self.model or not self.scaler:
            return {'error': 'Model not loaded. Train model first.'}
        
        # Extract satellite features
        features = self.extract_satellite_features(latitude, longitude, date_str)
        
        if not features:
            return {'error': 'Unable to extract satellite data for this location'}
        
        try:
            # Convert to array in correct order
            feature_array = np.array([[features[name] for name in self.feature_names]])
            
            # Scale features
            feature_scaled = self.scaler.transform(feature_array)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(feature_scaled)[0]
            prediction_class = self.model.predict(feature_scaled)[0]
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                prediction_proba[1], features['modis_ndvi'], features['month']
            )
            
            return {
                'bloom_probability': float(prediction_proba[1]),
                'not_bloom_probability': float(prediction_proba[0]),
                'prediction': 'Blooming' if prediction_class == 1 else 'Not Blooming',
                'confidence': float(max(prediction_proba)),
                'recommendation': recommendation,
                'features_used': features,
                'model_version': 'BloomWatch v1.0',
                'data_sources': ['MODIS Terra', 'Landsat 8', 'ERA5 Weather']
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return {'error': f'Prediction failed: {str(e)}'}
    
    def _generate_recommendation(self, bloom_prob, ndvi, month):
        """Generate user recommendations based on prediction"""
        
        seasonal_months = {
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'autumn': [9, 10, 11],
            'winter': [12, 1, 2]
        }
        
        current_season = next(season for season, months in seasonal_months.items() 
                             if month in months)
        
        if bloom_prob > 0.8 and ndvi > 0.7:
            return {
                'status': '🌸 Peak Bloom Expected',
                'message': 'Excellent conditions for flowering! Perfect time for nature photography and botanical observation.',
                'best_time': 'Next 1-2 weeks',
                'confidence': 'Very High'
            }
        elif bloom_prob > 0.6:
            return {
                'status': '🌺 High Bloom Probability',
                'message': f'Strong blooming potential during {current_season}. Monitor conditions closely.',
                'best_time': 'Next 2-4 weeks',
                'confidence': 'High'
            }
        elif bloom_prob > 0.4:
            return {
                'status': '🌿 Moderate Bloom Chance',
                'message': 'Some blooming activity possible. Weather conditions will be crucial.',
                'best_time': f'Peak {current_season} season',
                'confidence': 'Medium'
            }
        else:
            return {
                'status': '🍃 Low Bloom Activity',
                'message': 'Limited blooming expected. Consider alternative locations or wait for seasonal changes.',
                'best_time': 'Next growing season',
                'confidence': 'Low'
            }
    
    def batch_predict(self, locations):
        """Predict bloom for multiple locations"""
        
        results = []
        for loc in locations:
            prediction = self.predict_bloom(
                loc['latitude'], 
                loc['longitude'], 
                loc.get('date', datetime.now().strftime('%Y-%m-%d'))
            )
            prediction['location'] = loc.get('name', f"{loc['latitude']}, {loc['longitude']}")
            results.append(prediction)
        
        return results
