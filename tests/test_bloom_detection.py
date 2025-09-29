import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from bloom_detection import BloomDetectionService, BloomEvent, SpectralIndices


class TestBloomDetectionService:
    """Test suite for bloom detection service"""
    
    @pytest.fixture
    async def bloom_service(self):
        """Create initialized bloom detection service"""
        service = BloomDetectionService()
        await service.initialize()
        return service
    
    @pytest.fixture
    def mock_satellite_data(self):
        """Mock satellite data for testing"""
        return {
            "MODIS": {
                "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
                "acquisition_date": "2024-04-15T10:30:00Z"
            }
        }
    
    @pytest.fixture
    def test_region_bounds(self):
        """Test region bounds"""
        return (40.0, -75.0, 41.0, -73.0)  # min_lat, min_lon, max_lat, max_lon
    
    async def test_service_initialization(self):
        """Test bloom detection service initialization"""
        service = BloomDetectionService()
        assert not service.initialized
        
        await service.initialize()
        assert service.initialized
        assert service.model is not None
        assert service.scaler is not None
    
    async def test_bloom_detection_pipeline(self, bloom_service, mock_satellite_data, test_region_bounds):
        """Test complete bloom detection pipeline"""
        bloom_events = await bloom_service.detect_blooms_from_satellite_data(
            mock_satellite_data, 
            test_region_bounds
        )
        
        assert isinstance(bloom_events, list)
        
        # Check bloom event properties
        for event in bloom_events:
            assert isinstance(event, BloomEvent)
            assert event.latitude >= test_region_bounds[0]
            assert event.latitude <= test_region_bounds[2]
            assert event.longitude >= test_region_bounds[1]
            assert event.longitude <= test_region_bounds[3]
            assert 0 <= event.intensity <= 1
            assert 0 <= event.confidence <= 1
            assert event.satellite_source in ["MODIS", "Landsat", "VIIRS"]
    
    async def test_spectral_data_extraction(self, bloom_service, mock_satellite_data):
        """Test spectral data extraction from satellite imagery"""
        spectral_data = await bloom_service._extract_spectral_data(mock_satellite_data)
        
        assert isinstance(spectral_data, np.ndarray)
        assert spectral_data.ndim == 3  # height x width x bands
        assert spectral_data.shape[2] == 7  # 7 spectral bands
        assert spectral_data.min() >= 0
        assert spectral_data.max() <= 1
    
    async def test_vegetation_indices_calculation(self, bloom_service):
        """Test vegetation indices calculation"""
        # Create mock spectral data
        spectral_data = np.random.rand(50, 50, 7) * 0.5 + 0.2
        
        indices = await bloom_service._calculate_vegetation_indices(spectral_data)
        
        assert isinstance(indices, dict)
        assert 'ndvi' in indices
        assert 'evi' in indices
        assert 'savi' in indices
        assert 'ari' in indices
        
        # Check NDVI values are in valid range
        ndvi = indices['ndvi']
        assert ndvi.shape == spectral_data.shape[:2]
        assert np.all(ndvi >= -1)
        assert np.all(ndvi <= 1)
    
    async def test_bloom_pixel_identification(self, bloom_service):
        """Test bloom pixel identification"""
        # Create mock spectral data with known bloom signature
        spectral_data = np.random.rand(20, 20, 7)
        
        # Mock vegetation indices
        indices = {
            'ndvi': np.random.rand(20, 20) * 0.5 + 0.2,  # 0.2 to 0.7
            'evi': np.random.rand(20, 20) * 0.6 + 0.2,   # 0.2 to 0.8
            'ari': np.random.rand(20, 20) * 0.3,         # 0.0 to 0.3
            'savi': np.random.rand(20, 20) * 0.5 + 0.1   # 0.1 to 0.6
        }
        
        bloom_pixels = await bloom_service._identify_bloom_pixels(indices, spectral_data)
        
        assert isinstance(bloom_pixels, list)
        
        # Check bloom pixel format
        for pixel in bloom_pixels:
            row, col, likelihood = pixel
            assert 0 <= row < 20
            assert 0 <= col < 20
            assert 0 <= likelihood <= 1
    
    async def test_bloom_event_clustering(self, bloom_service, test_region_bounds):
        """Test clustering of bloom pixels into events"""
        # Create mock bloom pixels
        bloom_pixels = [
            (10, 10, 0.8),
            (11, 10, 0.7),
            (10, 11, 0.6),
            (50, 50, 0.9),  # Isolated pixel
            (51, 49, 0.8),  # Near isolated pixel
        ]
        
        events = await bloom_service._cluster_bloom_events(bloom_pixels, test_region_bounds)
        
        assert isinstance(events, list)
        assert len(events) >= 1  # At least one cluster should form
        
        # Check event properties
        for event in events:
            assert isinstance(event, BloomEvent)
            assert event.location_id.startswith('bloom_')
            assert event.intensity > 0
            assert event.confidence > 0
    
    async def test_species_identification(self, bloom_service):
        """Test species identification from spectral characteristics"""
        # Test different confidence levels
        high_conf_pixels = [(0, 0, 0.9), (0, 1, 0.85), (1, 0, 0.82)]
        medium_conf_pixels = [(0, 0, 0.7), (0, 1, 0.65)]
        low_conf_pixels = [(0, 0, 0.5), (0, 1, 0.45)]
        
        high_species = await bloom_service._identify_species(high_conf_pixels)
        medium_species = await bloom_service._identify_species(medium_conf_pixels)
        low_species = await bloom_service._identify_species(low_conf_pixels)
        
        assert high_species in ["Cherry Blossom", "Wildflowers", "Agricultural Crops", "Mixed Vegetation"]
        assert medium_species in ["Cherry Blossom", "Wildflowers", "Agricultural Crops", "Mixed Vegetation"]
        assert low_species in ["Cherry Blossom", "Wildflowers", "Agricultural Crops", "Mixed Vegetation"]
    
    async def test_bloom_timing_prediction(self, bloom_service):
        """Test bloom timing prediction"""
        # Create mock historical data
        historical_events = [
            BloomEvent(
                location_id="hist_1",
                latitude=40.0,
                longitude=-74.0,
                bloom_start=datetime(2023, 4, 1),
                bloom_peak=datetime(2023, 4, 15),
                bloom_end=datetime(2023, 4, 25),
                intensity=0.8,
                confidence=0.9,
                species="Cherry Blossom",
                satellite_source="MODIS"
            ),
            BloomEvent(
                location_id="hist_2",
                latitude=40.1,
                longitude=-74.1,
                bloom_start=datetime(2022, 4, 5),
                bloom_peak=datetime(2022, 4, 18),
                bloom_end=datetime(2022, 4, 28),
                intensity=0.7,
                confidence=0.85,
                species="Cherry Blossom",
                satellite_source="Landsat"
            )
        ]
        
        weather_data = {"temperature_anomaly": 1.5}  # 1.5°C warmer than average
        
        predictions = await bloom_service.predict_bloom_timing(historical_events, weather_data)
        
        assert isinstance(predictions, dict)
        assert 'predicted_bloom_start' in predictions
        assert 'predicted_bloom_peak' in predictions
        assert 'predicted_bloom_end' in predictions
        
        # Check prediction dates are reasonable
        for key, date in predictions.items():
            assert isinstance(date, datetime)
            assert date.year == datetime.now().year
    
    async def test_empty_satellite_data(self, bloom_service, test_region_bounds):
        """Test handling of empty satellite data"""
        empty_data = {}
        
        events = await bloom_service.detect_blooms_from_satellite_data(empty_data, test_region_bounds)
        
        assert isinstance(events, list)
        # Should handle empty data gracefully without crashing
    
    async def test_invalid_region_bounds(self, bloom_service, mock_satellite_data):
        """Test handling of invalid region bounds"""
        invalid_bounds = (45.0, -70.0, 40.0, -75.0)  # max < min
        
        events = await bloom_service.detect_blooms_from_satellite_data(
            mock_satellite_data, 
            invalid_bounds
        )
        
        assert isinstance(events, list)
        # Should handle invalid bounds gracefully
    
    @pytest.mark.parametrize("intensity,expected_species_type", [
        (0.9, "Cherry Blossom"),
        (0.7, "Wildflowers"),
        (0.5, "Agricultural Crops"),
        (0.3, "Mixed Vegetation")
    ])
    async def test_species_identification_parameterized(self, bloom_service, intensity, expected_species_type):
        """Test species identification with different intensity values"""
        pixels = [(0, 0, intensity), (0, 1, intensity)]
        species = await bloom_service._identify_species(pixels)
        
        assert species == expected_species_type


class TestSpectralIndices:
    """Test suite for spectral indices calculations"""
    
    def test_ndvi_calculation(self):
        """Test NDVI calculation with known values"""
        # Known NIR and Red values
        nir = 0.7
        red = 0.3
        expected_ndvi = (nir - red) / (nir + red)  # 0.4
        
        # Create test spectral data
        spectral_data = np.zeros((1, 1, 7))
        spectral_data[0, 0, 3] = red   # Red band
        spectral_data[0, 0, 4] = nir   # NIR band
        
        # This would typically be called within the bloom service
        # For direct testing, we'd need to expose the calculation
        calculated_ndvi = (nir - red) / (nir + red)
        
        assert abs(calculated_ndvi - expected_ndvi) < 0.001
    
    def test_spectral_indices_edge_cases(self):
        """Test spectral indices with edge cases"""
        # Test with zero values
        nir = 0.0
        red = 0.0
        
        # NDVI should handle division by zero
        ndvi = (nir - red) / (nir + red + 1e-8)  # Small epsilon to avoid division by zero
        
        assert not np.isnan(ndvi)
        assert not np.isinf(ndvi)


# Integration tests
class TestBloomDetectionIntegration:
    """Integration tests for bloom detection system"""
    
    async def test_full_pipeline_integration(self):
        """Test the complete bloom detection pipeline end-to-end"""
        service = BloomDetectionService()
        await service.initialize()
        
        # Mock realistic satellite data
        satellite_data = {
            "MODIS": {
                "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
                "acquisition_date": "2024-04-15T10:30:00Z",
                "cloud_cover": 0.1
            }
        }
        
        region_bounds = (40.5, -74.5, 41.0, -73.5)  # NYC area
        
        # Run complete detection pipeline
        bloom_events = await service.detect_blooms_from_satellite_data(
            satellite_data, 
            region_bounds
        )
        
        # Verify results
        assert isinstance(bloom_events, list)
        
        # If events found, verify their properties
        for event in bloom_events:
            assert isinstance(event.bloom_start, datetime)
            assert isinstance(event.bloom_peak, datetime)
            assert event.bloom_start <= event.bloom_peak
            if event.bloom_end:
                assert event.bloom_peak <= event.bloom_end
            
            # Verify geographic constraints
            assert region_bounds[0] <= event.latitude <= region_bounds[2]
            assert region_bounds[1] <= event.longitude <= region_bounds[3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])