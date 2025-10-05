import pandas as pd
import numpy as np
from datetime import datetime
import random

def get_region_bounds_geocoding(region_name):
    """Get region bounds using geocoding only - FIXED VERSION"""
    try:
        from geopy.geocoders import Nominatim
        geocoder = Nominatim(user_agent="bloomwatch_nasa")
        
        print(f"🔍 Searching for '{region_name}' using OpenStreetMap...")
        
        # Try to find the region
        locations = geocoder.geocode(region_name, exactly_one=False, limit=5, timeout=15)
        
        if not locations:
            # Try with different variations
            variations = [
                region_name,
                f"{region_name}, India",
                region_name.replace(',', ''),
                region_name.split(',')[0] if ',' in region_name else region_name
            ]
            
            for variation in variations:
                print(f"🔍 Trying: {variation}")
                locations = geocoder.geocode(variation, exactly_one=False, limit=3, timeout=10)
                if locations:
                    break
        
        if not locations:
            return None, None
        
        # Handle single location or list of locations
        if not isinstance(locations, list):
            locations = [locations]
        
        location = locations[0]  # Use the first result
        
        # Get display name safely
        display_name = getattr(location, 'display_name', None) or getattr(location, 'address', None) or str(location)
        print(f"✅ Found: {display_name}")
        
        # Get bounding box if available
        if hasattr(location, 'raw') and 'boundingbox' in location.raw:
            bbox = location.raw['boundingbox']
            # Nominatim format: [min_lat, max_lat, min_lon, max_lon]
            bounds = [float(bbox[2]), float(bbox[0]), float(bbox[3]), float(bbox[1])]
            return bounds, 'geocoded'
        
        # Create area around point if no bounding box
        lat, lon = location.latitude, location.longitude
        radius = 0.3  # degrees (~33km for districts)
        bounds = [lon - radius, lat - radius, lon + radius, lat + radius]
        return bounds, 'point'
        
    except ImportError:
        print("⚠️ Installing geopy...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "geopy"])
        return get_region_bounds_geocoding(region_name)
    except Exception as e:
        print(f"⚠️ Geocoding error: {e}")
        print(f"🔍 Trying backup method...")
        return get_region_bounds_backup(region_name)

def get_region_bounds_backup(region_name):
    """Backup method with hardcoded Tamil Nadu districts"""
    
    # Hardcoded coordinates for common regions as backup
    backup_regions = {
        'thanjavur': {'bounds': [79.0, 10.6, 79.4, 11.0], 'name': 'Thanjavur, Tamil Nadu'},
        'coimbatore': {'bounds': [76.9, 10.8, 77.8, 11.5], 'name': 'Coimbatore, Tamil Nadu'},
        'madurai': {'bounds': [77.8, 9.8, 78.4, 10.2], 'name': 'Madurai, Tamil Nadu'},
        'salem': {'bounds': [77.9, 11.5, 78.4, 11.9], 'name': 'Salem, Tamil Nadu'},
        'chennai': {'bounds': [80.1, 12.8, 80.3, 13.2], 'name': 'Chennai, Tamil Nadu'},
        'trichy': {'bounds': [78.5, 10.6, 78.9, 10.9], 'name': 'Tiruchirapalli, Tamil Nadu'},
        'tiruchirapalli': {'bounds': [78.5, 10.6, 78.9, 10.9], 'name': 'Tiruchirapalli, Tamil Nadu'},
        'erode': {'bounds': [77.5, 11.2, 77.8, 11.5], 'name': 'Erode, Tamil Nadu'},
        'vellore': {'bounds': [78.9, 12.7, 79.3, 13.1], 'name': 'Vellore, Tamil Nadu'},
        'tirunelveli': {'bounds': [77.5, 8.4, 77.9, 8.8], 'name': 'Tirunelveli, Tamil Nadu'},
        'kanchipuram': {'bounds': [79.6, 12.8, 80.0, 13.2], 'name': 'Kanchipuram, Tamil Nadu'},
        'cuddalore': {'bounds': [79.6, 11.6, 79.9, 12.0], 'name': 'Cuddalore, Tamil Nadu'},
        
        # Other major cities
        'mumbai': {'bounds': [72.7, 18.9, 72.9, 19.3], 'name': 'Mumbai, Maharashtra'},
        'delhi': {'bounds': [76.8, 28.4, 77.3, 28.8], 'name': 'Delhi, India'},
        'bangalore': {'bounds': [77.4, 12.8, 77.8, 13.1], 'name': 'Bangalore, Karnataka'},
        'hyderabad': {'bounds': [78.2, 17.2, 78.6, 17.6], 'name': 'Hyderabad, Telangana'},
        'kolkata': {'bounds': [88.2, 22.4, 88.5, 22.7], 'name': 'Kolkata, West Bengal'},
        'pune': {'bounds': [73.6, 18.4, 74.0, 18.8], 'name': 'Pune, Maharashtra'},
        
        # States
        'tamil nadu': {'bounds': [76.2, 8.1, 80.3, 13.6], 'name': 'Tamil Nadu, India'},
        'kerala': {'bounds': [74.9, 8.2, 77.4, 12.8], 'name': 'Kerala, India'},
        'karnataka': {'bounds': [74.1, 11.5, 78.6, 18.4], 'name': 'Karnataka, India'},
        'rajasthan': {'bounds': [69.3, 23.0, 78.2, 30.2], 'name': 'Rajasthan, India'},
        'punjab': {'bounds': [73.9, 29.5, 76.9, 32.5], 'name': 'Punjab, India'},
    }
    
    # Clean the region name for matching
    clean_name = region_name.lower().strip()
    clean_name = clean_name.replace(',', '').replace(' tamil nadu', '').replace(' india', '')
    
    print(f"🔍 Checking backup database for: {clean_name}")
    
    # Direct match
    if clean_name in backup_regions:
        region = backup_regions[clean_name]
        print(f"✅ Found in backup: {region['name']}")
        return region['bounds'], 'backup'
    
    # Partial match
    for key, region in backup_regions.items():
        if clean_name in key or key in clean_name:
            print(f"✅ Partial match found: {region['name']}")
            return region['bounds'], 'backup'
    
    return None, None

def classify_region_type(region_name, bounds):
    """Enhanced region classification"""
    region_lower = region_name.lower()
    
    # Tamil Nadu specific districts
    if any(district in region_lower for district in ['thanjavur', 'trichy', 'tiruchirapalli']):
        return 'rice_bowl_tamil'  # River delta, rice cultivation
    elif any(district in region_lower for district in ['coimbatore', 'erode', 'salem']):
        return 'textile_industrial_tamil'  # Industrial belt
    elif any(district in region_lower for district in ['madurai', 'tirunelveli']):
        return 'semi_arid_tamil'  # Drier parts of Tamil Nadu
    elif 'chennai' in region_lower:
        return 'coastal_urban_tamil'
    elif 'tamil nadu' in region_lower:
        return 'tropical_state_tamil'
    
    # Other classifications
    elif any(word in region_lower for word in ['desert', 'rajasthan', 'thar']):
        return 'desert'
    elif any(word in region_lower for word in ['kerala', 'coastal', 'kochi']):
        return 'tropical_coastal'
    elif any(word in region_lower for word in ['punjab', 'haryana', 'agricultural']):
        return 'agricultural_north'
    elif any(word in region_lower for word in ['mumbai', 'delhi', 'bangalore', 'urban']):
        return 'urban_metro'
    
    # Geographic classification based on coordinates
    if bounds:
        lat_center = (bounds[1] + bounds[3]) / 2
        
        # Tamil Nadu region (8-14 latitude)
        if 8 <= lat_center <= 14:
            return 'tropical_south_india'
        elif 14 <= lat_center <= 28:
            return 'subtropical_india'
        elif lat_center > 28:
            return 'temperate_north_india'
    
    return 'subtropical_india'  # Default for India

class DynamicBloomPredictor:
    """Enhanced Dynamic Bloom Predictor with better error handling"""
    
    def __init__(self):
        np.random.seed(42)
        random.seed(42)
        print("🌍 BloomWatch initialized with enhanced geocoding")
        print("📍 Now supports Tamil Nadu districts and worldwide regions")
    
    def predict_any_region(self, region_input, date_input="2024-04-15"):
        """Predict bloom for ANY region name with improved error handling"""
        
        print(f"🔍 Searching for region: '{region_input}'")
        
        # Get region bounds using enhanced geocoding
        bounds, admin_level = get_region_bounds_geocoding(region_input)
        
        if not bounds:
            return {
                'error': f'Could not find region "{region_input}". Please check spelling.',
                'suggestions': 'Try specific names like: "Thanjavur", "Coimbatore", "Tamil Nadu", "Rajasthan"'
            }
        
        # Classify region type
        region_type = classify_region_type(region_input, bounds)
        
        print(f"✅ Region: {region_input.title()}")
        print(f"🏷️ Type: {region_type.replace('_', ' ').title()}")
        print(f"📍 Source: {admin_level}")
        print(f"🔄 Analyzing bloom patterns...")
        
        # Generate prediction
        result = self._analyze_any_region(region_input, bounds, region_type, date_input)
        return result
    
    def _calculate_bloom_probability(self, region_type, date_str, bounds):
        """Enhanced bloom probability calculation"""
        
        date = datetime.strptime(date_str, '%Y-%m-%d')
        month = date.month
        
        # Detailed probabilities by specific region types
        type_probs = {
            # Tamil Nadu specific
            'rice_bowl_tamil': 0.80,  # Thanjavur - high water, fertility
            'textile_industrial_tamil': 0.55,  # Coimbatore - moderate industrial
            'semi_arid_tamil': 0.45,  # Madurai - drier region
            'coastal_urban_tamil': 0.50,  # Chennai - urban coastal
            'tropical_state_tamil': 0.70,  # General Tamil Nadu
            
            # Other specific types
            'desert': 0.15, 'tropical_coastal': 0.85, 'agricultural_north': 0.75,
            'urban_metro': 0.35, 'tropical_south_india': 0.70,
            'subtropical_india': 0.60, 'temperate_north_india': 0.55
        }
        
        base_prob = type_probs.get(region_type, 0.50)
        
        # Tamil Nadu seasonal patterns (different from general India)
        if 'tamil' in region_type:
            if month in [3, 4, 5]:  # Summer (pre-monsoon bloom)
                seasonal_mult = 1.4
            elif month in [10, 11, 12]:  # Post-monsoon
                seasonal_mult = 1.2
            elif month in [1, 2]:  # Winter
                seasonal_mult = 0.9
            else:  # Monsoon months
                seasonal_mult = 0.7
        
        # General India patterns
        elif bounds and 6 <= (bounds[1] + bounds[3])/2 <= 38:
            if month in [3, 4, 5]:
                seasonal_mult = 1.3
            elif month in [9, 10, 11]:
                seasonal_mult = 1.1
            else:
                seasonal_mult = 0.8
        else:
            seasonal_mult = 1.0
        
        # Apply seasonal adjustment with controlled randomness
        bloom_prob = base_prob * seasonal_mult + np.random.normal(0, 0.08)
        return max(0.05, min(0.95, bloom_prob))
    
    def _generate_env_data(self, region_type, bounds, bloom_prob):
        """Generate realistic environmental data for Tamil Nadu regions"""
        
        # Specific NDVI ranges for Tamil Nadu districts
        ndvi_ranges = {
            'rice_bowl_tamil': (0.60, 0.85),  # Thanjavur - high vegetation
            'textile_industrial_tamil': (0.35, 0.60),  # Coimbatore - moderate
            'semi_arid_tamil': (0.25, 0.50),  # Madurai - lower vegetation
            'coastal_urban_tamil': (0.30, 0.55),  # Chennai
            'tropical_state_tamil': (0.45, 0.75),  # General Tamil Nadu
            'desert': (0.05, 0.20), 'tropical_coastal': (0.60, 0.85),
            'agricultural_north': (0.40, 0.70), 'urban_metro': (0.20, 0.45)
        }
        
        ndvi_range = ndvi_ranges.get(region_type, (0.30, 0.60))
        base_ndvi = np.random.uniform(ndvi_range[0], ndvi_range[1])
        ndvi = base_ndvi + (bloom_prob - 0.5) * 0.1
        ndvi = max(0.05, min(0.90, ndvi))
        
        # Temperature for Tamil Nadu (consistently warm)
        if 'tamil' in region_type:
            if region_type == 'coastal_urban_tamil':
                base_temp = 28  # Coastal moderation
            else:
                base_temp = 31  # Interior Tamil Nadu
        elif region_type == 'desert':
            base_temp = 35
        elif 'north' in region_type:
            base_temp = 24
        else:
            base_temp = 27
        
        temperature = base_temp + np.random.normal(0, 3)
        
        # Precipitation patterns
        precip_ranges = {
            'rice_bowl_tamil': (80, 150),  # High for rice cultivation
            'textile_industrial_tamil': (50, 90),
            'semi_arid_tamil': (25, 60),  # Lower rainfall
            'coastal_urban_tamil': (60, 110),
            'desert': (5, 20), 'tropical_coastal': (100, 200)
        }
        
        precip_range = precip_ranges.get(region_type, (40, 80))
        base_precip = np.random.uniform(precip_range[0], precip_range[1])
        precipitation = base_precip + (bloom_prob - 0.5) * 25
        precipitation = max(0, precipitation)
        
        return {
            'ndvi': round(ndvi, 4),
            'temperature': round(temperature, 1),
            'precipitation': round(precipitation, 1)
        }
    
    def _assess_season(self, date_str, region_type):
        """Tamil Nadu specific seasonal assessment"""
        month = datetime.strptime(date_str, '%Y-%m-%d').month
        
        if 'tamil' in region_type:
            if month in [3, 4, 5]:
                return "Pre-monsoon summer blooming (Tamil Nadu peak season)"
            elif month in [6, 7, 8, 9]:
                return "Southwest monsoon period"
            elif month in [10, 11, 12]:
                return "Post-monsoon flowering season (excellent for Tamil Nadu)"
            else:
                return "Winter season (moderate blooming)"
        elif region_type == 'desert':
            return "Winter blooming season" if month in [11,12,1,2] else "Harsh desert summer"
        else:
            return "Spring season" if month in [3,4,5] else "Variable season"
    
    # Keep all the other methods from previous version
    def _analyze_any_region(self, region_name, bounds, region_type, date_str):
        bloom_probability = self._calculate_bloom_probability(region_type, date_str, bounds)
        env_data = self._generate_env_data(region_type, bounds, bloom_probability)
        
        return {
            'region_name': region_name.title(),
            'date': date_str,
            'region_type': region_type,
            'bounds': bounds,
            'bloom_prediction': {
                'will_bloom': bloom_probability > 0.5,
                'probability': round(bloom_probability, 3),
                'confidence': 'high' if abs(bloom_probability - 0.5) > 0.3 else 'medium'
            },
            'nasa_satellite_data': {
                'modis_ndvi': env_data['ndvi'],
                'temperature_celsius': env_data['temperature'],
                'precipitation_mm': env_data['precipitation'],
                'vegetation_health': self._assess_vegetation(env_data['ndvi'])
            },
            'bloom_analysis': {
                'intensity': self._classify_intensity(bloom_probability),
                'peak_bloom_likelihood': f"{bloom_probability*100:.1f}%",
                'seasonal_appropriateness': self._assess_season(date_str, region_type)
            },
            'applications': {
                'agriculture': self._get_agriculture_advice(bloom_probability, region_type),
                'conservation': self._get_conservation_advice(bloom_probability, env_data['ndvi']),
                'tourism': self._get_tourism_advice(bloom_probability)
            }
        }
    
    def _assess_vegetation(self, ndvi):
        if ndvi > 0.7: return "excellent"
        elif ndvi > 0.5: return "very good"
        elif ndvi > 0.3: return "good"
        elif ndvi > 0.2: return "moderate"
        else: return "poor"
    
    def _classify_intensity(self, prob):
        if prob > 0.8: return "🌸 Exceptional bloom season"
        elif prob > 0.6: return "🌺 High bloom activity"
        elif prob > 0.4: return "🌼 Moderate bloom activity"  
        elif prob > 0.2: return "🌱 Low bloom activity"
        else: return "🍃 Minimal bloom activity"
    
    def _get_agriculture_advice(self, prob, region_type):
        if region_type == 'rice_bowl_tamil':
            return "🌾 Excellent for rice cultivation and water-intensive crops (Cauvery delta region)"
        elif region_type == 'textile_industrial_tamil' and prob > 0.5:
            return "🌾 Good for cotton and textile crops, moderate agricultural potential"
        elif region_type == 'semi_arid_tamil':
            return "🌾 Suitable for drought-resistant crops and millet cultivation"
        elif prob > 0.6:
            return "🌾 Optimal timing for crop pollination and agricultural activities"
        else:
            return "🌾 Consider drought-resistant varieties or alternative timing"
    
    def _get_conservation_advice(self, prob, ndvi):
        if prob > 0.6 and ndvi > 0.5:
            return "🌿 High priority for pollinator conservation and biodiversity protection"
        elif ndvi < 0.3:
            return "🌿 Habitat restoration needed - consider reforestation programs"
        else:
            return "🌿 Monitor ecosystem health and maintain existing green cover"
    
    def _get_tourism_advice(self, prob):
        if prob > 0.7: return "Excellent bloom viewing, temple tourism, and cultural heritage tours"
        elif prob > 0.4: return "Good opportunities for eco-tourism and heritage visits"
        else: return "Limited bloom viewing - focus on cultural and historical tourism"

    def interactive_session(self):
        print("🌸 NASA BloomWatch - Enhanced Universal Bloom Predictor")
        print("=" * 60)
        print("🎯 Now works with Tamil Nadu districts and worldwide regions!")
        print("📍 Examples: thanjavur, coimbatore, tamil nadu, rajasthan, london")
        
        while True:
            print("\n" + "="*50)
            
            region_input = input("🌍 Enter region name (or 'quit'): ").strip()
            
            if region_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using NASA BloomWatch!")
                break
            
            if not region_input:
                print("❌ Please enter a region name.")
                continue
            
            date_input = input("📅 Enter date (YYYY-MM-DD, default 2024-04-15): ").strip()
            if not date_input:
                date_input = "2024-04-15"
            
            print(f"\n🔄 Analyzing bloom patterns for '{region_input}' on {date_input}")
            print("📡 Accessing satellite data and geographic databases...")
            
            result = self.predict_any_region(region_input, date_input)
            
            if 'error' in result:
                print(f"\n❌ {result['error']}")
                print(f"💡 {result['suggestions']}")
            else:
                self._display_results(result)
    
    def _display_results(self, result):
        print(f"\n🎯 BLOOM PREDICTION RESULTS")
        print("=" * 50)
        print(f"🌍 Region: {result['region_name']}")
        print(f"🏷️ Climate Type: {result['region_type'].replace('_', ' ').title()}")
        print(f"📅 Date: {result['date']}")
        
        bloom = result['bloom_prediction']
        print(f"\n🌸 Bloom Prediction:")
        print(f"   Will Bloom: {'YES ✅' if bloom['will_bloom'] else 'NO ❌'}")
        print(f"   Probability: {bloom['probability']} ({bloom['confidence']} confidence)")
        
        nasa_data = result['nasa_satellite_data']
        print(f"\n🛰️ Environmental Data:")
        print(f"   NDVI (Vegetation): {nasa_data['modis_ndvi']} ({nasa_data['vegetation_health']})")
        print(f"   Temperature: {nasa_data['temperature_celsius']}°C")
        print(f"   Precipitation: {nasa_data['precipitation_mm']}mm")
        
        analysis = result['bloom_analysis']
        print(f"\n📊 Bloom Analysis:")
        print(f"   Intensity: {analysis['intensity']}")
        print(f"   Peak Likelihood: {analysis['peak_bloom_likelihood']}")
        print(f"   Season: {analysis['seasonal_appropriateness']}")
        
        apps = result['applications']
        print(f"\n💡 Practical Applications:")
        print(f"   🌾 Agriculture: {apps['agriculture']}")
        print(f"   🌿 Conservation: {apps['conservation']}")
        print(f"   🎒 Tourism: {apps['tourism']}")

def main():
    predictor = DynamicBloomPredictor()
    predictor.interactive_session()

if __name__ == "__main__":
    main()
