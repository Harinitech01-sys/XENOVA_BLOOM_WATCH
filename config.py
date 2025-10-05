import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "BloomWatch Analysis Backend"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:2036@localhost:5432/bloomwatch"
    
    # NASA Earth Engine
    EARTH_ENGINE_SERVICE_ACCOUNT: Optional[str] = None
    EARTH_ENGINE_PRIVATE_KEY_PATH: Optional[str] = None
    
    # NASA APIs
    NASA_API_KEY: Optional[str] = None
    MODIS_BASE_URL: str = "https://modis.gsfc.nasa.gov/data/"
    LANDSAT_BASE_URL: str = "https://landsat.gsfc.nasa.gov/data/"
    VIIRS_BASE_URL: str = "https://www.earthdata.nasa.gov/learn/find-data/near-real-time/viirs"
    
    # Redis for caching and task queue
    REDIS_URL: str = "redis://localhost:6379"
    
    # File Storage
    DATA_STORAGE_PATH: Path = Path("./data")
    TEMP_STORAGE_PATH: Path = Path("./temp")
    OUTPUT_PATH: Path = Path("./output")
    
    # Processing Parameters
    MAX_CLOUD_COVER: float = 0.3
    BLOOM_DETECTION_THRESHOLD: float = 0.15
    NDVI_THRESHOLD: float = 0.4
    
    # Satellite Data Parameters
    MODIS_COLLECTION: str = "MODIS/006/MOD13Q1"
    LANDSAT_COLLECTION: str = "LANDSAT/LC08/C02/T1_L2"
    VIIRS_COLLECTION: str = "NOAA/VIIRS/001/VNP13A1"
    
    # Bloom Detection Models
    MODEL_CONFIDENCE_THRESHOLD: float = 0.7
    SPECTRAL_BANDS: list = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Ensure directories exist
for path in [settings.DATA_STORAGE_PATH, settings.TEMP_STORAGE_PATH, settings.OUTPUT_PATH]:
    path.mkdir(parents=True, exist_ok=True)