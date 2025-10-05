from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class BloomEvent(Base):
    """Database model for bloom events detected from satellite data"""
    __tablename__ = "bloom_events"
    
    id = Column(Integer, primary_key=True, index=True)
    location_id = Column(String(50), unique=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    bloom_start = Column(DateTime, nullable=False)
    bloom_peak = Column(DateTime, nullable=False)
    bloom_end = Column(DateTime, nullable=True)
    intensity = Column(Float, nullable=False)  # 0.0 to 1.0
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    species = Column(String(100), nullable=True)
    satellite_source = Column(String(20), nullable=False)  # MODIS, Landsat, VIIRS, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    satellite_images = relationship("SatelliteImage", back_populates="bloom_events")
    analysis_results = relationship("AnalysisResult", back_populates="bloom_event")

class SatelliteImage(Base):
    """Database model for satellite imagery metadata"""
    __tablename__ = "satellite_images"
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(String(100), unique=True, index=True)
    satellite = Column(String(20), nullable=False)  # MODIS, Landsat, VIIRS, etc.
    sensor = Column(String(20), nullable=False)
    acquisition_date = Column(DateTime, nullable=False)
    processing_date = Column(DateTime, nullable=False)
    cloud_cover = Column(Float, nullable=False)
    spatial_resolution = Column(Float, nullable=False)  # meters
    processing_level = Column(String(10), nullable=False)  # L1A, L1B, L2A, etc.
    
    # Geographic bounds
    min_latitude = Column(Float, nullable=False)
    max_latitude = Column(Float, nullable=False)
    min_longitude = Column(Float, nullable=False)
    max_longitude = Column(Float, nullable=False)
    
    # File information
    file_path = Column(String(500), nullable=True)
    file_size_mb = Column(Float, nullable=True)
    data_format = Column(String(20), nullable=False)  # GeoTIFF, NetCDF, HDF, etc.
    
    # Quality metrics
    data_quality = Column(String(20), nullable=False)  # excellent, good, fair, poor
    sun_elevation = Column(Float, nullable=True)
    sun_azimuth = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    bloom_events = relationship("BloomEvent", back_populates="satellite_images")
    vegetation_indices = relationship("VegetationIndex", back_populates="satellite_image")

class VegetationIndex(Base):
    """Database model for calculated vegetation indices"""
    __tablename__ = "vegetation_indices"
    
    id = Column(Integer, primary_key=True, index=True)
    satellite_image_id = Column(Integer, ForeignKey("satellite_images.id"))
    index_type = Column(String(10), nullable=False)  # NDVI, EVI, SAVI, ARI, etc.
    
    # Statistical values
    min_value = Column(Float, nullable=False)
    max_value = Column(Float, nullable=False)
    mean_value = Column(Float, nullable=False)
    std_deviation = Column(Float, nullable=False)
    
    # Bloom-related metrics
    bloom_threshold = Column(Float, nullable=True)
    bloom_pixel_count = Column(Integer, nullable=True)
    bloom_coverage_percent = Column(Float, nullable=True)
    
    calculation_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    satellite_image = relationship("SatelliteImage", back_populates="vegetation_indices")

class Region(Base):
    """Database model for geographic regions of interest"""
    __tablename__ = "regions"
    
    id = Column(Integer, primary_key=True, index=True)
    region_id = Column(String(50), unique=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Geographic bounds
    min_latitude = Column(Float, nullable=False)
    max_latitude = Column(Float, nullable=False)
    min_longitude = Column(Float, nullable=False)
    max_longitude = Column(Float, nullable=False)
    
    # Region characteristics
    region_type = Column(String(20), nullable=False)  # urban, rural, agricultural, forest, etc.
    area_km2 = Column(Float, nullable=True)
    elevation_m = Column(Float, nullable=True)
    climate_zone = Column(String(30), nullable=True)
    
    # Monitoring configuration
    monitoring_active = Column(Boolean, default=True)
    monitoring_frequency = Column(String(20), nullable=False)  # daily, weekly, monthly
    priority_level = Column(String(10), nullable=False)  # high, medium, low
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    analysis_results = relationship("AnalysisResult", back_populates="region")

class Species(Base):
    """Database model for plant species information"""
    __tablename__ = "species"
    
    id = Column(Integer, primary_key=True, index=True)
    scientific_name = Column(String(100), nullable=False, unique=True)
    common_name = Column(String(100), nullable=False)
    family = Column(String(50), nullable=True)
    genus = Column(String(50), nullable=True)
    
    # Bloom characteristics
    typical_bloom_duration_days = Column(Integer, nullable=True)
    peak_bloom_month = Column(String(20), nullable=True)
    bloom_color = Column(String(30), nullable=True)
    
    # Ecological information
    pollinator_species = Column(Text, nullable=True)  # JSON array of pollinators
    habitat_type = Column(String(50), nullable=True)
    conservation_status = Column(String(20), nullable=True)
    
    # Climate sensitivity
    temperature_sensitivity = Column(Float, nullable=True)  # 0.0 to 1.0
    precipitation_dependency = Column(Float, nullable=True)  # 0.0 to 1.0
    climate_change_vulnerability = Column(String(20), nullable=True)  # low, medium, high
    
    created_at = Column(DateTime, default=datetime.utcnow)

class AnalysisResult(Base):
    """Database model for analysis results and predictions"""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String(50), unique=True, index=True)
    region_id = Column(Integer, ForeignKey("regions.id"))
    bloom_event_id = Column(Integer, ForeignKey("bloom_events.id"), nullable=True)
    
    analysis_type = Column(String(30), nullable=False)  # trend, prediction, classification
    analysis_date = Column(DateTime, default=datetime.utcnow)
    
    # Results data (stored as JSON)
    results_data = Column(Text, nullable=False)  # JSON string with analysis results
    confidence_score = Column(Float, nullable=False)  # Overall confidence 0.0 to 1.0
    
    # Processing information
    algorithm_version = Column(String(10), nullable=False)
    processing_time_seconds = Column(Float, nullable=True)
    input_data_sources = Column(Text, nullable=True)  # JSON array of data sources
    
    # Status
    status = Column(String(20), default="completed")  # queued, processing, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Relationships
    region = relationship("Region", back_populates="analysis_results")
    bloom_event = relationship("BloomEvent", back_populates="analysis_results")

class ExportJob(Base):
    """Database model for data export jobs"""
    __tablename__ = "export_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(50), unique=True, index=True)
    
    # Export configuration
    data_type = Column(String(30), nullable=False)  # bloom_events, satellite_data, analysis_results
    export_format = Column(String(20), nullable=False)  # json, csv, geojson, netcdf
    
    # Request parameters (stored as JSON)
    request_parameters = Column(Text, nullable=False)  # JSON string with export parameters
    
    # Job status
    status = Column(String(20), default="queued")  # queued, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Output information
    output_file_path = Column(String(500), nullable=True)
    file_size_mb = Column(Float, nullable=True)
    download_url = Column(String(200), nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Cleanup
    expires_at = Column(DateTime, nullable=True)  # When to delete the exported file