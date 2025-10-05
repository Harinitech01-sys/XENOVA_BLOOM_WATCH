# direct_ee_auth.py
import ee
import os

def direct_token_auth():
    """Direct token-based authentication"""
    
    try:
        print("🔐 Method 1: Direct token authentication...")
        
        # This will open browser directly for token
        ee.Authenticate()
        ee.Initialize()
        
        print("✅ Direct authentication successful!")
        return True
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        return False

def service_account_auth():
    """Try service account if available"""
    
    try:
        print("🔐 Method 2: Service account authentication...")
        
        # Initialize without explicit auth (uses default credentials if available)
        ee.Initialize()
        
        print("✅ Service account authentication successful!")
        return True
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
        return False

def test_connection():
    """Test if Earth Engine is working"""
    
    try:
        # Simple test
        image = ee.Image('MODIS/006/MOD13A1/2020_01_01')
        info = image.getInfo()
        print("✅ Earth Engine connection test successful!")
        return True
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def main():
    """Try multiple authentication methods"""
    
    print("🌍 Earth Engine Authentication Troubleshooter")
    print("=" * 50)
    
    methods = [
        ("Direct Token Auth", direct_token_auth),
        ("Service Account Auth", service_account_auth),
    ]
    
    for method_name, method_func in methods:
        print(f"\n🔄 Trying {method_name}...")
        if method_func():
            if test_connection():
                print(f"🎉 SUCCESS! {method_name} worked!")
                return True
            
    print("\n❌ All authentication methods failed")
    return False

if __name__ == "__main__":
    main()
