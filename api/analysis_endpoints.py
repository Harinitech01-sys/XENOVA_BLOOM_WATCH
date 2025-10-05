from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict
from datetime import datetime, date
from pydantic import BaseModel

from bloom_detection import BloomDetectionService, BloomEvent


analysis_router = APIRouter()

# Pydantic models for request/response
class BloomEventResponse(BaseModel):
    location_id: str
    latitude: float
    longitude: float
    bloom_start: datetime
    bloom_peak: datetime
    bloom_end: Optional[datetime]
    intensity: float
    confidence: float
    species: Optional[str]
    satellite_source: str

class BloomAnalysisRequest(BaseModel):
    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float
    start_date: date
    end_date: date
    satellite_sources: List[str] = ["MODIS", "Landsat", "VIIRS"]

class TrendAnalysisResponse(BaseModel):
    region_name: str
    yearly_trends: Dict[str, float]
    seasonal_patterns: Dict[str, float]
    peak_bloom_shift: float
    confidence_score: float

@analysis_router.get("/bloom-events", response_model=List[BloomEventResponse])

async def get_bloom_events(
    min_lat: float = Query(..., ge=-90, le=90),
    max_lat: float = Query(..., ge=-90, le=90),
    min_lon: float = Query(..., ge=-180, le=180),
    max_lon: float = Query(..., ge=-180, le=180),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    min_confidence: float = Query(0.5, ge=0, le=1)
):
    """
    Retrieve bloom events for a specific geographic region and time period
    """
    try:
        # Mock data for demo - replace with actual database queries
        mock_events = [
            BloomEventResponse(
                location_id="bloom_001",
                latitude=40.7128,
                longitude=-74.0060,
                bloom_start=datetime(2024, 4, 1),
                bloom_peak=datetime(2024, 4, 15),
                bloom_end=datetime(2024, 4, 25),
                intensity=0.85,
                confidence=0.92,
                species="Cherry Blossom",
                satellite_source="MODIS"
            ),
            BloomEventResponse(
                location_id="bloom_002",
                latitude=37.7749,
                longitude=-122.4194,
                bloom_start=datetime(2024, 3, 15),
                bloom_peak=datetime(2024, 4, 1),
                bloom_end=None,
                intensity=0.73,
                confidence=0.88,
                species="Wildflowers",
                satellite_source="Landsat"
            )
        ]
        
        # Filter by confidence
        filtered_events = [
            event for event in mock_events 
            if event.confidence >= min_confidence
        ]
        
        return filtered_events
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving bloom events: {str(e)}")

@analysis_router.post("/analyze-region", response_model=List[BloomEventResponse])
async def analyze_region(request: BloomAnalysisRequest):
    """
    Perform bloom detection analysis on a specific region
    """
    try:
        # Mock satellite data
        satellite_data = {
            "MODIS": {"bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]},
            "Landsat": {"bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]},
            "VIIRS": {"bands": ["I1", "I2", "I3", "M1", "M2"]}
        }
        
        # Initialize bloom detection service
        bloom_service = BloomDetectionService()
        await bloom_service.initialize()
        
        # Detect blooms
        region_bounds = (
            request.min_latitude, 
            request.min_longitude,
            request.max_latitude,
            request.max_longitude
        )
        
        bloom_events = await bloom_service.detect_blooms_from_satellite_data(
            satellite_data, region_bounds
        )
        
        # Convert to response format
        response_events = [
            BloomEventResponse(
                location_id=event.location_id,
                latitude=event.latitude,
                longitude=event.longitude,
                bloom_start=event.bloom_start,
                bloom_peak=event.bloom_peak,
                bloom_end=event.bloom_end,
                intensity=event.intensity,
                confidence=event.confidence,
                species=event.species,
                satellite_source=event.satellite_source
            )
            for event in bloom_events
        ]
        
        return response_events
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing region: {str(e)}")

@analysis_router.get("/trends/{region_id}", response_model=TrendAnalysisResponse)
async def get_bloom_trends(
    region_id: str,
    years: int = Query(5, ge=1, le=20)
):
    """
    Get multi-year bloom trend analysis for a specific region
    """
    try:
        # Mock trend analysis data
        mock_trends = TrendAnalysisResponse(
            region_name=f"Region_{region_id}",
            yearly_trends={
                "2020": 0.65,
                "2021": 0.72,
                "2022": 0.68,
                "2023": 0.78,
                "2024": 0.82
            },
            seasonal_patterns={
                "spring": 0.85,
                "summer": 0.45,
                "autumn": 0.25,
                "winter": 0.10
            },
            peak_bloom_shift=-3.2,  # days earlier than historical average
            confidence_score=0.89
        )
        
        return mock_trends
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving trends: {str(e)}")

@analysis_router.get("/predictions/{region_id}")
async def get_bloom_predictions(
    region_id: str,
    prediction_days: int = Query(30, ge=7, le=365)
):
    """
    Get bloom timing predictions for a specific region
    """
    try:
        # Mock prediction data
        current_date = datetime.now()
        predictions = {
            "region_id": region_id,
            "prediction_date": current_date.isoformat(),
            "predicted_events": [
                {
                    "species": "Cherry Blossom",
                    "predicted_bloom_start": (current_date.replace(month=4, day=1)).isoformat(),
                    "predicted_bloom_peak": (current_date.replace(month=4, day=15)).isoformat(),
                    "confidence": 0.87,
                    "factors": ["temperature_trend", "precipitation", "daylight_hours"]
                },
                {
                    "species": "Wildflowers",
                    "predicted_bloom_start": (current_date.replace(month=3, day=20)).isoformat(),
                    "predicted_bloom_peak": (current_date.replace(month=4, day=5)).isoformat(),
                    "confidence": 0.73,
                    "factors": ["soil_moisture", "temperature_trend"]
                }
            ]
        }
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")

@analysis_router.get("/species/{species_name}")
async def get_species_analysis(
    species_name: str,
    global_analysis: bool = Query(False)
):
    """
    Get detailed analysis for a specific plant species
    """
    try:
        # Mock species analysis
        analysis = {
            "species_name": species_name,
            "global_distribution": {
                "total_bloom_sites": 1247,
                "countries": 23,
                "peak_bloom_month": "April",
                "average_bloom_duration": 18  # days
            },
            "ecological_impact": {
                "pollinator_species": ["bees", "butterflies", "hummingbirds"],
                "ecosystem_services": ["pollination", "biodiversity_support"],
                "conservation_status": "stable"
            },
            "climate_sensitivity": {
                "temperature_sensitivity": 0.72,
                "precipitation_dependency": 0.45,
                "climate_change_vulnerability": "moderate"
            }
        }
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving species analysis: {str(e)}")