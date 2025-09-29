from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
from datetime import datetime, date
from pydantic import BaseModel

router = APIRouter()

class SatelliteDataRequest(BaseModel):
    satellite: str
    region_bounds: List[float]  # [min_lat, min_lon, max_lat, max_lon]
    start_date: date
    end_date: date
    bands: List[str]
    cloud_cover_max: float = 0.3

class SatelliteImageMetadata(BaseModel):
    image_id: str
    satellite: str
    acquisition_date: datetime
    cloud_cover: float
    spatial_resolution: float
    spectral_bands: List[str]
    processing_level: str

@router.get("/available-satellites")
async def get_available_satellites():
    """
    Get list of available NASA satellite missions and their capabilities
    """
    satellites = {
        "MODIS": {
            "full_name": "Moderate Resolution Imaging Spectroradiometer",
            "satellites": ["Terra", "Aqua"],
            "temporal_resolution": "1-2 days",
            "spatial_resolution": "250m-1km",
            "spectral_bands": 36,
            "applications": ["vegetation_monitoring", "bloom_detection", "phenology"]
        },
        "Landsat": {
            "full_name": "Landsat 8/9 Operational Land Imager",
            "satellites": ["Landsat 8", "Landsat 9"],
            "temporal_resolution": "16 days",
            "spatial_resolution": "15-30m",
            "spectral_bands": 11,
            "applications": ["high_resolution_monitoring", "agricultural_assessment"]
        },
        "VIIRS": {
            "full_name": "Visible Infrared Imaging Radiometer Suite",
            "satellites": ["Suomi NPP", "NOAA-20", "NOAA-21"],
            "temporal_resolution": "daily",
            "spatial_resolution": "375m-750m",
            "spectral_bands": 22,
            "applications": ["global_monitoring", "near_real_time_analysis"]
        },
        "EMIT": {
            "full_name": "Earth Surface Mineral Dust Source Investigation",
            "platform": "International Space Station",
            "temporal_resolution": "variable",
            "spatial_resolution": "60m",
            "spectral_bands": 285,
            "applications": ["hyperspectral_analysis", "mineral_composition"]
        },
        "PACE": {
            "full_name": "Plankton, Aerosol, Cloud ocean Ecosystem",
            "satellites": ["PACE"],
            "temporal_resolution": "daily",
            "spatial_resolution": "1km",
            "spectral_bands": ">200",
            "applications": ["ocean_color", "atmospheric_analysis"]
        }
    }
    
    return satellites

@router.post("/query-images", response_model=List[SatelliteImageMetadata])
async def query_satellite_images(request: SatelliteDataRequest):
    """
    Query available satellite images for a specific region and time period
    """
    try:
        # Mock satellite image metadata
        mock_images = []
        
        # Generate mock images for the requested time period
        current_date = request.start_date
        image_count = 0
        
        while current_date <= request.end_date and image_count < 10:
            mock_images.append(SatelliteImageMetadata(
                image_id=f"{request.satellite}_{current_date.strftime('%Y%m%d')}_{image_count:03d}",
                satellite=request.satellite,
                acquisition_date=datetime.combine(current_date, datetime.min.time()),
                cloud_cover=0.15 if image_count % 3 == 0 else 0.05,
                spatial_resolution=250.0 if request.satellite == "MODIS" else 30.0,
                spectral_bands=request.bands,
                processing_level="L2A"
            ))
            
            # Increment date based on satellite temporal resolution
            if request.satellite == "MODIS":
                current_date = current_date.replace(day=current_date.day + 1)
            elif request.satellite == "Landsat":
                current_date = current_date.replace(day=current_date.day + 16)
            else:
                current_date = current_date.replace(day=current_date.day + 1)
            
            image_count += 1
        
        # Filter by cloud cover
        filtered_images = [
            img for img in mock_images 
            if img.cloud_cover <= request.cloud_cover_max
        ]
        
        return filtered_images
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying satellite images: {str(e)}")

@router.get("/image/{image_id}/download")
async def download_satellite_image(image_id: str):
    """
    Download a specific satellite image
    """
    try:
        # Mock download response
        download_info = {
            "image_id": image_id,
            "download_url": f"https://earthdata.nasa.gov/data/{image_id}",
            "file_size_mb": 145.7,
            "format": "GeoTIFF",
            "compression": "LZW",
            "estimated_download_time": "2-5 minutes"
        }
        
        return download_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initiating download: {str(e)}")

@router.get("/image/{image_id}/metadata")
async def get_image_metadata(image_id: str):
    """
    Get detailed metadata for a specific satellite image
    """
    try:
        # Mock detailed metadata
        metadata = {
            "image_id": image_id,
            "acquisition_info": {
                "satellite": "MODIS Terra",
                "sensor": "MODIS",
                "acquisition_date": "2024-04-15T10:30:00Z",
                "sun_elevation": 45.6,
                "sun_azimuth": 158.2
            },
            "geometric_info": {
                "spatial_resolution": 250.0,
                "pixel_size": [250, 250],
                "coordinate_system": "EPSG:4326",
                "bounds": [-74.5, 40.2, -73.5, 41.0]
            },
            "spectral_info": {
                "bands": [
                    {"band_id": "B1", "wavelength": "620-670nm", "purpose": "land/cloud boundaries"},
                    {"band_id": "B2", "wavelength": "841-876nm", "purpose": "vegetation"},
                    {"band_id": "B3", "wavelength": "459-479nm", "purpose": "soil/vegetation"}
                ]
            },
            "quality_info": {
                "cloud_cover": 0.12,
                "data_quality": "good",
                "processing_level": "L2A",
                "processing_date": "2024-04-15T14:22:00Z"
            }
        }
        
        return metadata
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metadata: {str(e)}")

@router.get("/vegetation-indices/{image_id}")
async def calculate_vegetation_indices(
    image_id: str,
    indices: List[str] = Query(["NDVI", "EVI", "SAVI", "ARI"])
):
    """
    Calculate vegetation indices from satellite imagery
    """
    try:
        # Mock vegetation index calculations
        results = {
            "image_id": image_id,
            "calculation_date": datetime.now().isoformat(),
            "indices": {}
        }
        
        for index in indices:
            if index == "NDVI":
                results["indices"]["NDVI"] = {
                    "min_value": -0.2,
                    "max_value": 0.9,
                    "mean_value": 0.45,
                    "std_deviation": 0.23,
                    "bloom_threshold": 0.3
                }
            elif index == "EVI":
                results["indices"]["EVI"] = {
                    "min_value": -0.1,
                    "max_value": 1.0,
                    "mean_value": 0.52,
                    "std_deviation": 0.28
                }
            elif index == "ARI":
                results["indices"]["ARI"] = {
                    "min_value": 0.0,
                    "max_value": 0.8,
                    "mean_value": 0.15,
                    "std_deviation": 0.12,
                    "bloom_indicator": True
                }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating indices: {str(e)}")

@router.get("/real-time-feed")
async def get_real_time_satellite_feed():
    """
    Get real-time satellite data feed for global monitoring
    """
    try:
        feed_data = {
            "last_update": datetime.now().isoformat(),
            "active_satellites": 8,
            "recent_acquisitions": [
                {
                    "satellite": "MODIS Terra",
                    "location": "North America",
                    "acquisition_time": "2024-01-27T15:30:00Z",
                    "data_quality": "excellent"
                },
                {
                    "satellite": "Landsat 9",
                    "location": "Europe",
                    "acquisition_time": "2024-01-27T11:45:00Z",
                    "data_quality": "good"
                },
                {
                    "satellite": "VIIRS",
                    "location": "Asia",
                    "acquisition_time": "2024-01-27T09:20:00Z",
                    "data_quality": "excellent"
                }
            ],
            "processing_queue": 23,
            "system_status": "operational"
        }
        
        return feed_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving real-time feed: {str(e)}")

satellite_router = router