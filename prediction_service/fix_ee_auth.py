# fix_ee_auth.py
import ee

def fix_earth_engine_auth():
    try:
        print("🔐 Attempting Earth Engine authentication...")
        
        # Try browser-based authentication
        ee.Authenticate(auth_mode='notebook')  # This uses browser instead of gcloud
        
        print("✅ Authentication successful!")
        
        # Test initialization
        ee.Initialize()
        print("✅ Earth Engine initialized successfully!")
        
        # Quick test
        image = ee.Image('MODIS/061/MOD13Q1/2024_04_01')
        print("✅ Test satellite data access successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

if __name__ == "__main__":
    fix_earth_engine_auth()
