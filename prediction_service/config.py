# config.py - Universal Bloom Detection Configuration
import ee
import os
from datetime import datetime, timedelta

class Config:
    # Flask settings
    FLASK_HOST = '127.0.0.1'
    FLASK_PORT = 5000
    FLASK_DEBUG = True
    # Add these lines to your existing config.py file
    SECRET_KEY = 'bloomwatch-secret-key-2025'
    DEBUG = True
    MODEL_VERSION = '2.1'

    
    # Your existing config variables...
    TRAINING_REGIONS = [
        {'name': 'Bay of Bengal'},
        {'name': 'Arabian Sea'},
        {'name': 'Indian Ocean'}
    ]

    """Universal Bloom Detection Configuration for NASA Challenge"""
    
    # Google Earth Engine Configuration
    GEE_PROJECT = 'bloomwatch-app'  # Your project ID
    
    # NASA Data Sources (MANDATORY for NASA challenge)
    MODIS_COLLECTION = 'MODIS/061/MOD13Q1'           # NASA MODIS Vegetation
    LANDSAT_COLLECTION = 'LANDSAT/LC08/C02/T1_L2'    # NASA Landsat 8
    WEATHER_COLLECTION = 'ECMWF/ERA5_LAND/DAILY_AGGR' # Climate data
    ELEVATION_COLLECTION = 'USGS/SRTMGL1_003'        # Elevation data
    
    # File Paths (Keep your existing structure)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, 'trained_models')
    TRAINING_DATA_DIR = os.path.join(MODELS_DIR, 'training_data')
    
    # Universal Model Files (NEW)
    UNIVERSAL_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, 'universal_bloom_model.pkl')
    UNIVERSAL_SCALER_PATH = os.path.join(TRAINED_MODELS_DIR, 'universal_scaler.pkl')
    UNIVERSAL_FEATURES_PATH = os.path.join(TRAINED_MODELS_DIR, 'universal_features.txt')
    UNIVERSAL_DATA_PATH = os.path.join(TRAINING_DATA_DIR, 'universal_training_data.csv')
    
    # Keep your existing paths
    BLOOM_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, 'bloom_predictor.pkl')
    SCALER_PATH = os.path.join(TRAINED_MODELS_DIR, 'scaler.pkl')
    FEATURES_CSV_PATH = os.path.join(TRAINING_DATA_DIR, 'features.csv')
    LABELS_CSV_PATH = os.path.join(TRAINING_DATA_DIR, 'labels.csv')
    
    # Training Parameters
    FEATURE_WINDOW_DAYS = 60
    SPATIAL_SCALE = 250
    CLOUD_COVER_THRESHOLD = 30
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Universal Feature Names (NEW - for NASA challenge)
    UNIVERSAL_FEATURE_NAMES = [
        # Spatial features
        'latitude', 'longitude', 'elevation',
        
        # Temporal features  
        'day_of_year', 'month', 'season_sin', 'season_cos',
        'days_since_spring_start', 'photoperiod',
        
        # NASA MODIS vegetation indices
        'ndvi_current', 'ndvi_mean_30d', 'ndvi_std_30d', 'ndvi_trend_30d',
        'evi_current', 'evi_mean_30d', 'evi_std_30d', 'evi_trend_30d',
        
        # NASA Landsat surface reflectance
        'landsat_ndvi', 'landsat_red', 'landsat_nir', 'landsat_swir',
        
        # Climate from ERA5
        'temperature_current', 'temperature_mean_30d', 'temperature_trend',
        'precipitation_current', 'precipitation_sum_30d',
        'humidity', 'solar_radiation_proxy',
        
        # Bloom indicators (derived)
        'vegetation_greenness_change', 'seasonal_vegetation_anomaly',
        'bloom_favorable_conditions', 'phenology_stage_indicator'
    ]
    
    @staticmethod
    def initialize_earth_engine():
        """Initialize Google Earth Engine"""
        try:
            if Config.GEE_PROJECT:
                ee.Initialize(project=Config.GEE_PROJECT)
            else:
                ee.Initialize()
            print("✅ Earth Engine initialized successfully")
            return True
        except Exception as e:
            print(f"⚠️ Earth Engine initialization failed: {e}")
            try:
                ee.Authenticate()
                ee.Initialize(project=Config.GEE_PROJECT if Config.GEE_PROJECT else None)
                print("✅ Earth Engine authenticated and initialized")
                return True
            except Exception as auth_error:
                print(f"❌ Earth Engine authentication failed: {auth_error}")
                return False
    
    @staticmethod
    def create_directories():
        """Create all necessary directories"""
        directories = [
            Config.MODELS_DIR,
            Config.TRAINED_MODELS_DIR,
            Config.TRAINING_DATA_DIR,
            os.path.join(Config.BASE_DIR, 'logs'),
            os.path.join(Config.BASE_DIR, 'temp')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("📁 Directory structure created")
        return True
    
    @staticmethod
    def calculate_photoperiod(latitude, day_of_year):
        """Calculate photoperiod (daylight hours)"""
        import math
        
        try:
            declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))
            lat_rad = math.radians(latitude)
            decl_rad = math.radians(declination)
            
            cos_hour_angle = -math.tan(lat_rad) * math.tan(decl_rad)
            cos_hour_angle = max(-1, min(1, cos_hour_angle))
            hour_angle = math.acos(cos_hour_angle)
            photoperiod = 2 * hour_angle * 12 / math.pi
            return photoperiod
        except:
            return 12.0


# Initialize when imported
if __name__ == "__main__":
    Config.initialize_earth_engine()
    Config.create_directories()
