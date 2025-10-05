# test_ee_fix.py
import ee

def test_different_initialization():
    """Test different Earth Engine initialization methods"""
    
    print("🔧 Testing different Earth Engine initialization methods...")
    
    # Method 1: Try basic initialization (token is already saved)
    try:
        print("\n🔄 Method 1: Basic initialization...")
        ee.Initialize()
        
        # Test with a simple image
        image = ee.Image('MODIS/061/MOD13Q1/2020_01_01')
        print("✅ Method 1 SUCCESS - Earth Engine is working!")
        return True
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Try with project specification
    try:
        print("\n🔄 Method 2: Project-based initialization...")
        ee.Initialize(project='earthengine-legacy')
        
        image = ee.Image('MODIS/061/MOD13Q1/2020_01_01')
        print("✅ Method 2 SUCCESS - Earth Engine is working!")
        return True
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Try legacy mode
    try:
        print("\n🔄 Method 3: Legacy mode...")
        ee.Initialize(opt_url='https://earthengine.googleapis.com')
        
        image = ee.Image('MODIS/061/MOD13Q1/2020_01_01')
        print("✅ Method 3 SUCCESS - Earth Engine is working!")
        return True
        
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    print("\n🤔 All methods failed - account may need approval")
    return False

if __name__ == "__main__":
    test_different_initialization()
