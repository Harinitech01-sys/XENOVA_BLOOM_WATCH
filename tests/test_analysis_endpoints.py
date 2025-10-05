import pytest
from fastapi.testclient import TestClient
from datetime import datetime, date
import json

from app import app
from bloom_detection import BloomDetectionService

client = TestClient(app)


class TestAnalysisEndpoints:
    """Test suite for analysis API endpoints"""
    
    def test_get_bloom_events_success(self):
        """Test successful retrieval of bloom events"""
        response = client.get(
            "/api/v1/analysis/bloom-events",
            params={
                "min_lat": 40.0,
                "max_lat": 41.0,
                "min_lon": -75.0,
                "max_lon": -73.0,
                "min_confidence": 0.5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        
        # Check structure of bloom events
        for event in data:
            assert "location_id" in event
            assert "latitude" in event
            assert "longitude" in event
            assert "bloom_start" in event
            assert "bloom_peak" in event
            assert "intensity" in event
            assert "confidence" in event
            assert "species" in event
            assert "satellite_source" in event
            
            # Verify confidence filter
            assert event["confidence"] >= 0.5
    
    def test_get_bloom_events_invalid_coordinates(self):
        """Test bloom events endpoint with invalid coordinates"""
        response = client.get(
            "/api/v1/analysis/bloom-events",
            params={
                "min_lat": 100,  # Invalid latitude
                "max_lat": 41.0,
                "min_lon": -75.0,
                "max_lon": -73.0
            }
        )
        
        # Should return 422 for validation error
        assert response.status_code == 422
    
    def test_analyze_region_success(self):
        """Test successful region analysis"""
        request_data = {
            "min_latitude": 40.0,
            "max_latitude": 41.0,
            "min_longitude": -75.0,
            "max_longitude": -73.0,
            "start_date": "2024-03-01",
            "end_date": "2024-05-01",
            "satellite_sources": ["MODIS", "Landsat"]
        }
        
        response = client.post(
            "/api/v1/analysis/analyze-region",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        
        # Verify bloom event structure
        for event in data:
            assert "location_id" in event
            assert "latitude" in event
            assert "longitude" in event
            assert "bloom_start" in event
            assert "bloom_peak" in event
            
            # Verify coordinates are within requested bounds
            assert request_data["min_latitude"] <= event["latitude"] <= request_data["max_latitude"]
            assert request_data["min_longitude"] <= event["longitude"] <= request_data["max_longitude"]
    
    def test_analyze_region_invalid_date_range(self):
        """Test region analysis with invalid date range"""
        request_data = {
            "min_latitude": 40.0,
            "max_latitude": 41.0,
            "min_longitude": -75.0,
            "max_longitude": -73.0,
            "start_date": "2024-05-01",
            "end_date": "2024-03-01",  # End before start
            "satellite_sources": ["MODIS"]
        }
        
        response = client.post(
            "/api/v1/analysis/analyze-region",
            json=request_data
        )
        
        # Should handle invalid date range gracefully
        assert response.status_code in [200, 422]  # Depends on validation implementation
    
    def test_get_bloom_trends_success(self):
        """Test successful bloom trends retrieval"""
        response = client.get(
            "/api/v1/analysis/trends/region_001",
            params={"years": 5}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify trend analysis structure
        assert "region_name" in data
        assert "yearly_trends" in data
        assert "seasonal_patterns" in data
        assert "peak_bloom_shift" in data
        assert "confidence_score" in data
        
        # Verify yearly trends data
        yearly_trends = data["yearly_trends"]
        assert isinstance(yearly_trends, dict)
        assert len(yearly_trends) > 0
        
        # Verify seasonal patterns
        seasonal_patterns = data["seasonal_patterns"]
        assert isinstance(seasonal_patterns, dict)
        expected_seasons = ["spring", "summer", "autumn", "winter"]
        for season in expected_seasons:
            assert season in seasonal_patterns
    
    def test_get_bloom_predictions_success(self):
        """Test successful bloom predictions retrieval"""
        response = client.get(
            "/api/v1/analysis/predictions/region_001",
            params={"prediction_days": 30}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify prediction structure
        assert "region_id" in data
        assert "prediction_date" in data
        assert "predicted_events" in data
        
        # Verify predicted events
        predicted_events = data["predicted_events"]
        assert isinstance(predicted_events, list)
        
        for event in predicted_events:
            assert "species" in event
            assert "predicted_bloom_start" in event
            assert "predicted_bloom_peak" in event
            assert "confidence" in event
            assert "factors" in event
            
            # Verify confidence is in valid range
            assert 0 <= event["confidence"] <= 1
    
    def test_get_species_analysis_success(self):
        """Test successful species analysis retrieval"""
        response = client.get(
            "/api/v1/analysis/species/Cherry Blossom",
            params={"global_analysis": True}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify species analysis structure
        assert "species_name" in data
        assert data["species_name"] == "Cherry Blossom"
        
        assert "global_distribution" in data
        distribution = data["global_distribution"]
        assert "total_bloom_sites" in distribution
        assert "countries" in distribution
        assert "peak_bloom_month" in distribution
        assert "average_bloom_duration" in distribution
        
        assert "ecological_impact" in data
        ecological = data["ecological_impact"]
        assert "pollinator_species" in ecological
        assert "ecosystem_services" in ecological
        assert "conservation_status" in ecological
        
        assert "climate_sensitivity" in data
        climate = data["climate_sensitivity"]
        assert "temperature_sensitivity" in climate
        assert "precipitation_dependency" in climate
        assert "climate_change_vulnerability" in climate
    
    def test_get_species_analysis_encoded_name(self):
        """Test species analysis with URL-encoded species name"""
        response = client.get("/api/v1/analysis/species/Cherry%20Blossom")
        
        assert response.status_code == 200
        data = response.json()
        assert data["species_name"] == "Cherry Blossom"
    
    def test_confidence_filter_functionality(self):
        """Test that confidence filter works correctly"""
        # Test with high confidence threshold
        response_high = client.get(
            "/api/v1/analysis/bloom-events",
            params={
                "min_lat": 35.0,
                "max_lat": 45.0,
                "min_lon": -80.0,
                "max_lon": -70.0,
                "min_confidence": 0.9
            }
        )
        
        # Test with low confidence threshold
        response_low = client.get(
            "/api/v1/analysis/bloom-events",
            params={
                "min_lat": 35.0,
                "max_lat": 45.0,
                "min_lon": -80.0,
                "max_lon": -70.0,
                "min_confidence": 0.5
            }
        )
        
        assert response_high.status_code == 200
        assert response_low.status_code == 200
        
        high_conf_events = response_high.json()
        low_conf_events = response_low.json()
        
        # High confidence threshold should return fewer or equal events
        assert len(high_conf_events) <= len(low_conf_events)
        
        # All high confidence events should meet the threshold
        for event in high_conf_events:
            assert event["confidence"] >= 0.9
    
    def test_date_range_filtering(self):
        """Test date range filtering in bloom events"""
        response = client.get(
            "/api/v1/analysis/bloom-events",
            params={
                "min_lat": 35.0,
                "max_lat": 45.0,
                "min_lon": -80.0,
                "max_lon": -70.0,
                "start_date": "2024-04-01",
                "end_date": "2024-04-30"
            }
        )
        
        assert response.status_code == 200
        # Note: Mock data doesn't implement date filtering,
        # but endpoint should handle the parameters without error
    
    def test_error_handling(self):
        """Test error handling in analysis endpoints"""
        # Test with invalid region ID
        response = client.get("/api/v1/analysis/trends/invalid_region_id")
        
        # Should return data even for invalid region (mock implementation)
        assert response.status_code == 200
    
    @pytest.mark.parametrize("satellite_source", ["MODIS", "Landsat", "VIIRS"])
    def test_satellite_source_filtering(self, satellite_source):
        """Test filtering by satellite source"""
        request_data = {
            "min_latitude": 40.0,
            "max_latitude": 41.0,
            "min_longitude": -75.0,
            "max_longitude": -73.0,
            "start_date": "2024-03-01",
            "end_date": "2024-05-01",
            "satellite_sources": [satellite_source]
        }
        
        response = client.post(
            "/api/v1/analysis/analyze-region",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify only requested satellite sources are returned
        for event in data:
            assert event["satellite_source"] in request_data["satellite_sources"]


class TestAnalysisEndpointsPerformance:
    """Performance tests for analysis endpoints"""
    
    def test_bloom_events_response_time(self):
        """Test bloom events endpoint response time"""
        import time
        
        start_time = time.time()
        
        response = client.get(
            "/api/v1/analysis/bloom-events",
            params={
                "min_lat": 30.0,
                "max_lat": 50.0,
                "min_lon": -90.0,
                "max_lon": -60.0
            }
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0  # Should respond within 5 seconds
    
    def test_large_region_analysis(self):
        """Test analysis of large geographic region"""
        request_data = {
            "min_latitude": 25.0,
            "max_latitude": 50.0,
            "min_longitude": -125.0,
            "max_longitude": -60.0,  # Continental US
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "satellite_sources": ["MODIS", "Landsat", "VIIRS"]
        }
        
        response = client.post(
            "/api/v1/analysis/analyze-region",
            json=request_data
        )
        
        assert response.status_code == 200
        # Should handle large regions without timeout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])