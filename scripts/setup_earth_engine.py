#!/usr/bin/env python3
"""
Setup script for Google Earth Engine authentication and initialization
NASA Space Apps Challenge - BloomWatch Application
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

import ee
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google.auth.exceptions import DefaultCredentialsError

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarthEngineSetup:
    """Setup and configuration for Google Earth Engine"""
    
    def __init__(self):
        self.ee_initialized = False
        self.credentials = None
    
    def authenticate_service_account(self, key_path: str) -> bool:
        """
        Authenticate using service account key file
        
        Args:
            key_path: Path to service account JSON key file
            
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            if not os.path.exists(key_path):
                logger.error(f"Service account key file not found: {key_path}")
                return False
            
            # Load service account credentials
            self.credentials = Credentials.from_service_account_file(
                key_path,
                scopes=['https://www.googleapis.com/auth/earthengine']
            )
            
            # Initialize Earth Engine with service account
            ee.Initialize(self.credentials)
            self.ee_initialized = True
            
            logger.info("Earth Engine authenticated with service account")
            return True
            
        except Exception as e:
            logger.error(f"Service account authentication failed: {e}")
            return False
    
    def authenticate_user_account(self) -> bool:
        """
        Authenticate using user account (interactive)
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            # Try to initialize with cached credentials
            ee.Initialize()
            self.ee_initialized = True
            logger.info("Earth Engine initialized with cached credentials")
            return True
            
        except Exception:
            try:
                # Run interactive authentication
                ee.Authenticate()
                ee.Initialize()
                self.ee_initialized = True
                logger.info("Earth Engine authenticated interactively")
                return True
                
            except Exception as e:
                logger.error(f"Interactive authentication failed: {e}")
                return False
    
    def test_earth_engine_access(self) -> bool:
        """
        Test Earth Engine access by running a simple query
        
        Returns:
            True if test successful, False otherwise
        """
        try:
            if not self.ee_initialized:
                logger.error("Earth Engine not initialized")
                return False
            
            # Test with a simple image collection query
            modis = ee.ImageCollection('MODIS/006/MOD13Q1')
            count = modis.size()
            
            logger.info(f"Earth Engine test successful - MODIS collection size: {count.getInfo()}")
            return True
            
        except Exception as e:
            logger.error(f"Earth Engine test failed: {e}")
            return False
    
    def setup_nasa_collections(self) -> dict:
        """
        Setup and verify access to NASA satellite collections
        
        Returns:
            Dictionary of collection information
        """
        collections = {}
        
        try:
            # MODIS Collections
            collections['MODIS_VEGETATION'] = {
                'collection_id': 'MODIS/006/MOD13Q1',
                'description': 'MODIS Vegetation Indices (NDVI/EVI)',
                'temporal_resolution': '16 days',
                'spatial_resolution': '250m'
            }
            
            collections['MODIS_SURFACE_REFLECTANCE'] = {
                'collection_id': 'MODIS/006/MOD09A1',
                'description': 'MODIS Surface Reflectance',
                'temporal_resolution': '8 days',
                'spatial_resolution': '500m'
            }
            
            # Landsat Collections
            collections['LANDSAT_8'] = {
                'collection_id': 'LANDSAT/LC08/C02/T1_L2',
                'description': 'Landsat 8 Surface Reflectance',
                'temporal_resolution': '16 days',
                'spatial_resolution': '30m'
            }
            
            collections['LANDSAT_9'] = {
                'collection_id': 'LANDSAT/LC09/C02/T1_L2',
                'description': 'Landsat 9 Surface Reflectance',
                'temporal_resolution': '16 days',
                'spatial_resolution': '30m'
            }
            
            # VIIRS Collections
            collections['VIIRS_VEGETATION'] = {
                'collection_id': 'NOAA/VIIRS/001/VNP13A1',
                'description': 'VIIRS Vegetation Indices',
                'temporal_resolution': '16 days',
                'spatial_resolution': '500m'
            }
            
            # Test access to each collection
            for name, info in collections.items():
                try:
                    collection = ee.ImageCollection(info['collection_id'])
                    size = collection.size().getInfo()
                    info['available'] = True
                    info['image_count'] = size
                    logger.info(f"✓ {name}: {size} images available")
                    
                except Exception as e:
                    info['available'] = False
                    info['error'] = str(e)
                    logger.warning(f"✗ {name}: Access failed - {e}")
            
            return collections
            
        except Exception as e:
            logger.error(f"Error setting up NASA collections: {e}")
            return collections
    
    def create_bloom_detection_functions(self) -> dict:
        """
        Create Earth Engine functions for bloom detection
        
        Returns:
            Dictionary of reusable Earth Engine functions
        """
        functions = {}
        
        try:
            # NDVI calculation function
            def calculate_ndvi(image):
                """Calculate NDVI from Landsat or MODIS imagery"""
                if 'SR_B4' in image.bandNames().getInfo():  # Landsat
                    nir = image.select('SR_B5')
                    red = image.select('SR_B4')
                elif 'sur_refl_b02' in image.bandNames().getInfo():  # MODIS
                    nir = image.select('sur_refl_b02')
                    red = image.select('sur_refl_b01')
                else:
                    raise ValueError("Unsupported image type for NDVI calculation")
                
                ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
                return image.addBands(ndvi)
            
            functions['calculate_ndvi'] = calculate_ndvi
            
            # Enhanced Vegetation Index (EVI) calculation
            def calculate_evi(image):
                """Calculate EVI from Landsat imagery"""
                if 'SR_B4' in image.bandNames().getInfo():  # Landsat
                    nir = image.select('SR_B5')
                    red = image.select('SR_B4')
                    blue = image.select('SR_B2')
                    
                    evi = image.expression(
                        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                        {
                            'NIR': nir,
                            'RED': red,
                            'BLUE': blue
                        }
                    ).rename('EVI')
                    
                    return image.addBands(evi)
                else:
                    raise ValueError("EVI calculation requires Landsat imagery")
            
            functions['calculate_evi'] = calculate_evi
            
            # Anthocyanin Reflectance Index (ARI) - good for bloom detection
            def calculate_ari(image):
                """Calculate ARI for bloom detection"""
                if 'SR_B3' in image.bandNames().getInfo():  # Landsat
                    green = image.select('SR_B3')
                    red = image.select('SR_B4')
                    
                    ari = green.pow(-1).subtract(red.pow(-1)).rename('ARI')
                    return image.addBands(ari)
                else:
                    raise ValueError("ARI calculation requires Landsat imagery")
            
            functions['calculate_ari'] = calculate_ari
            
            # Cloud masking function for Landsat
            def mask_clouds_landsat(image):
                """Mask clouds in Landsat imagery"""
                qa = image.select('QA_PIXEL')
                cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)  # Cloud bit
                cloud_shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)  # Cloud shadow bit
                
                mask = cloud_mask.And(cloud_shadow_mask)
                return image.updateMask(mask)
            
            functions['mask_clouds_landsat'] = mask_clouds_landsat
            
            # Bloom detection algorithm
            def detect_blooms(image):
                """
                Detect potential bloom events using spectral indices
                """
                # Calculate vegetation indices
                image = calculate_ndvi(image)
                image = calculate_evi(image)
                image = calculate_ari(image)
                
                # Bloom detection criteria
                ndvi = image.select('NDVI')
                evi = image.select('EVI')
                ari = image.select('ARI')
                
                # Multi-criteria bloom mask
                vegetation_mask = ndvi.gt(0.2).And(ndvi.lt(0.7))  # Moderate vegetation
                high_ari = ari.gt(0.1)  # High anthocyanin content
                bloom_mask = vegetation_mask.And(high_ari)
                
                bloom_intensity = ndvi.multiply(0.3).add(evi.multiply(0.3)).add(ari.multiply(0.4))
                bloom_intensity = bloom_intensity.updateMask(bloom_mask).rename('BLOOM_INTENSITY')
                
                return image.addBands(bloom_intensity)
            
            functions['detect_blooms'] = detect_blooms
            
            logger.info("Earth Engine bloom detection functions created successfully")
            
        except Exception as e:
            logger.error(f"Error creating Earth Engine functions: {e}")
        
        return functions

def main():
    """Main setup function"""
    print("BloomWatch - Earth Engine Setup")
    print("=" * 40)
    
    setup = EarthEngineSetup()
    
    # Try service account authentication first
    if settings.EARTH_ENGINE_PRIVATE_KEY_PATH and os.path.exists(settings.EARTH_ENGINE_PRIVATE_KEY_PATH):
        print("Attempting service account authentication...")
        if setup.authenticate_service_account(settings.EARTH_ENGINE_PRIVATE_KEY_PATH):
            print("✓ Service account authentication successful")
        else:
            print("✗ Service account authentication failed")
            return False
    else:
        print("Attempting user account authentication...")
        if setup.authenticate_user_account():
            print("✓ User account authentication successful")
        else:
            print("✗ User account authentication failed")
            return False
    
    # Test Earth Engine access
    print("\nTesting Earth Engine access...")
    if setup.test_earth_engine_access():
        print("✓ Earth Engine access verified")
    else:
        print("✗ Earth Engine access test failed")
        return False
    
    # Setup NASA collections
    print("\nSetting up NASA satellite collections...")
    collections = setup.setup_nasa_collections()
    
    available_collections = [name for name, info in collections.items() if info.get('available', False)]
    print(f"✓ {len(available_collections)} collections available out of {len(collections)}")
    
    # Create bloom detection functions
    print("\nCreating bloom detection functions...")
    functions = setup.create_bloom_detection_functions()
    print(f"✓ {len(functions)} Earth Engine functions created")
    
    print("\n" + "=" * 40)
    print("Earth Engine setup completed successfully!")
    print("BloomWatch backend is ready for satellite data processing.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)