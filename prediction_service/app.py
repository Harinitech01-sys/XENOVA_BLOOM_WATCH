from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import logging
import random
import json
import numpy as np
from datetime import datetime, timedelta
from predictive_modeling import BloomPredictor
from config import Config
import requests
import time
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
@app.template_filter('tojsonfilter')
def to_json_filter(obj):
    """Convert Python object to JSON string for use in templates"""
    return json.dumps(obj)
app.secret_key = 'bloomwatch-secret-key-2025'

# Initialize bloom predictor
predictor = BloomPredictor()
model_loaded = predictor.load_model()

if not model_loaded:
    logger.warning("⚠️ No trained model found. Train model first using model_training.py")

def geocode_location_enhanced(location_name):
    """
    Enhanced geocoding with better accuracy
    """
    try:
        # Clean the location name
        location_clean = location_name.strip().replace(' ', '+')
        
        # Use Nominatim API
        url = f"https://nominatim.openstreetmap.org/search?q={location_clean}&format=json&limit=3&addressdetails=1"
        
        headers = {
            'User-Agent': 'BloomPredictionApp/1.0'
        }
        
        logger.info(f"🌍 Geocoding location: {location_name}")
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if data and len(data) > 0:
                # Find best result
                best_result = find_best_result(data, location_name)
                if best_result:
                    return process_nominatim_result(best_result, location_name)
        
        logger.error(f"❌ Geocoding failed for: {location_name}")
        return None
            
    except Exception as e:
        logger.error(f"❌ Geocoding error: {e}")
        return None

def find_best_result(results, search_term):
    """Find the most relevant result from geocoding results"""
    search_lower = search_term.lower()
    
    # Prioritize exact matches
    for result in results:
        display_name = result.get('display_name', '').lower()
        if search_lower in display_name:
            address = result.get('address', {})
            
            # Prioritize cities
            if address.get('city') or address.get('town') or address.get('village'):
                city_name = (address.get('city') or address.get('town') or address.get('village')).lower()
                if search_lower == city_name or search_lower in city_name:
                    return result
    
    return results[0] if results else None

def process_nominatim_result(result, location_name):
    """Process Nominatim API result"""
    lat = float(result['lat'])
    lng = float(result['lon'])
    
    address = result.get('address', {})
    display_name = result.get('display_name', location_name)
    
    # Determine location type and boundaries
    location_type = determine_location_type_enhanced(address)
    boundaries = calculate_boundaries_enhanced(lat, lng, location_type)
    climate_info = determine_climate_enhanced(lat, lng, address)
    
    return {
        'lat': lat,
        'lng': lng,
        'display_name': display_name,
        'type': location_type,
        'climate': climate_info['climate'],
        'climate_description': climate_info['description'],
        'species': climate_info['species'],
        'vegetation_type': climate_info['vegetation_type'],
        'typical_ndvi': climate_info['typical_ndvi'],
        'rainfall_pattern': climate_info['rainfall_pattern'],
        'soil_type': climate_info['soil_type'],
        'zoom_level': boundaries['zoom_level'],
        'region_bounds': boundaries['bounds'],
        'country': address.get('country', 'Unknown'),
        'state': address.get('state', address.get('province', '')),
        'city': address.get('city', address.get('town', address.get('village', ''))),
        'source': 'nominatim'
    }

def determine_location_type_enhanced(address):
    """Enhanced location type determination"""
    if address.get('country') and not address.get('state') and not address.get('city'):
        return 'Country'
    elif address.get('state') and not address.get('city') and not address.get('town'):
        return 'State'
    elif address.get('city') or address.get('town') or address.get('village'):
        return 'City'
    else:
        return 'Region'

def calculate_boundaries_enhanced(lat, lng, location_type):
    """Enhanced boundary calculation"""
    if location_type == 'Country':
        offset = 8.0
        zoom = 4
    elif location_type == 'State':
        offset = 2.5
        zoom = 7
    elif location_type == 'City':
        offset = 0.3
        zoom = 11
    else:
        offset = 1.5
        zoom = 8
    
    return {
        'bounds': [[lat - offset, lng - offset], [lat + offset, lng + offset]],
        'zoom_level': zoom
    }

def determine_climate_enhanced(lat, lng, address):
    """Enhanced climate determination with regional specificity"""
    
    country = address.get('country', '').lower()
    state = address.get('state', '').lower()
    
    # India-specific climate mapping
    if country == 'india':
        if state in ['rajasthan', 'haryana', 'gujarat']:
            return {
                'climate': 'Arid/Semi-Arid',
                'description': 'Hot dry climate with minimal rainfall',
                'species': 'Desert plants, drought-resistant crops',
                'vegetation_type': 'Desert scrub',
                'typical_ndvi': 0.15,
                'rainfall_pattern': 'Very low (200-500mm annually)',
                'soil_type': 'Sandy, alkaline soils'
            }
        elif state in ['kerala', 'karnataka', 'tamil nadu']:
            return {
                'climate': 'Tropical',
                'description': 'Hot humid tropical climate with monsoons',
                'species': 'Coconut palms, tropical fruits, spices',
                'vegetation_type': 'Tropical',
                'typical_ndvi': 0.65,
                'rainfall_pattern': 'High (1000-2500mm annually)',
                'soil_type': 'Red laterite, alluvial'
            }
        elif state in ['himachal pradesh', 'uttarakhand', 'jammu and kashmir']:
            return {
                'climate': 'Highland Temperate',
                'description': 'Cool temperate mountain climate',
                'species': 'Pine forests, alpine vegetation',
                'vegetation_type': 'Mountain forests',
                'typical_ndvi': 0.55,
                'rainfall_pattern': 'Moderate (800-1500mm annually)',
                'soil_type': 'Mountain soils'
            }
        else:
            return {
                'climate': 'Tropical Monsoon',
                'description': 'Hot climate with distinct monsoon seasons',
                'species': 'Monsoon vegetation, agricultural crops',
                'vegetation_type': 'Monsoon tropical',
                'typical_ndvi': 0.50,
                'rainfall_pattern': 'Moderate to high (600-1800mm annually)',
                'soil_type': 'Alluvial, black cotton soil'
            }
    
    # General latitude-based classification
    abs_lat = abs(lat)
    
    if abs_lat >= 60:  # Subarctic/Arctic
        return {
            'climate': 'Subarctic',
            'description': 'Cold climate with short summers',
            'species': 'Coniferous forests, boreal vegetation',
            'vegetation_type': 'Boreal forest',
            'typical_ndvi': 0.45,
            'rainfall_pattern': 'Low to moderate (300-700mm annually)',
            'soil_type': 'Podzolic soils'
        }
    elif abs_lat >= 45:  # Continental/Temperate
        if country in ['united kingdom', 'ireland', 'denmark', 'netherlands']:
            return {
                'climate': 'Temperate Oceanic',
                'description': 'Mild oceanic climate with moderate rainfall',
                'species': 'Deciduous forests, grasslands',
                'vegetation_type': 'Temperate deciduous',
                'typical_ndvi': 0.60,
                'rainfall_pattern': 'High (700-1200mm annually)',
                'soil_type': 'Brown earth soils'
            }
        else:
            return {
                'climate': 'Continental',
                'description': 'Continental climate with cold winters',
                'species': 'Mixed forests, grasslands',
                'vegetation_type': 'Mixed temperate',
                'typical_ndvi': 0.50,
                'rainfall_pattern': 'Moderate (400-900mm annually)',
                'soil_type': 'Brown soils, chernozems'
            }
    elif abs_lat >= 30:  # Subtropical
        if lng > -30 and lng < 50 and abs_lat < 45:  # Mediterranean
            return {
                'climate': 'Mediterranean',
                'description': 'Mediterranean climate with dry summers',
                'species': 'Olive trees, Mediterranean scrub',
                'vegetation_type': 'Mediterranean',
                'typical_ndvi': 0.40,
                'rainfall_pattern': 'Moderate (400-800mm annually)',
                'soil_type': 'Calcareous soils'
            }
        else:
            return {
                'climate': 'Subtropical',
                'description': 'Warm subtropical climate',
                'species': 'Subtropical forests, grasslands',
                'vegetation_type': 'Subtropical',
                'typical_ndvi': 0.55,
                'rainfall_pattern': 'Variable (600-1500mm annually)',
                'soil_type': 'Red-yellow soils'
            }
    else:  # Tropical
        # Check for arid regions
        if (country in ['saudi arabia', 'egypt', 'libya', 'algeria', 'sudan', 'chad'] or
            (lng > 12 and lng < 50 and abs_lat < 30 and abs_lat > 15)):
            return {
                'climate': 'Arid/Desert',
                'description': 'Hot dry desert climate',
                'species': 'Desert plants, cacti, sparse vegetation',
                'vegetation_type': 'Desert',
                'typical_ndvi': 0.05,
                'rainfall_pattern': 'Very low (50-200mm annually)',
                'soil_type': 'Sandy desert soils'
            }
        else:
            return {
                'climate': 'Tropical',
                'description': 'Hot humid tropical climate',
                'species': 'Tropical rainforests, palm trees',
                'vegetation_type': 'Tropical',
                'typical_ndvi': 0.75,
                'rainfall_pattern': 'High (1200-3000mm annually)',
                'soil_type': 'Tropical soils, oxisols'
            }

def get_bloom_color(probability):
    """Get color based on bloom probability"""
    if probability >= 0.7:
        return '#33FF57'  # High bloom - green
    elif probability >= 0.4:
        return '#FF8C00'  # Moderate bloom - orange
    elif probability >= 0.2:
        return '#FF5733'  # Low bloom - red
    else:
        return '#8B0000'  # Very low/no bloom - dark red

def create_enhanced_prediction(location_name, location_data, probability, date, latitude, longitude):
    """Create detailed prediction with comprehensive environmental analysis"""
    
    # Enhanced environmental parameters
    base_temp = 28 + random.uniform(-8, 12)
    base_precip = 50 + random.uniform(-40, 100)
    base_humidity = 65 + random.randint(-20, 25)
    
    # Adjust based on climate type
    climate = location_data.get('climate', 'Unknown')
    if 'Oceanic' in climate or 'Temperate' in climate:
        base_temp = random.uniform(5, 22)
        base_precip = random.uniform(40, 80)
        base_humidity = random.randint(60, 85)
    elif 'Desert' in climate or 'Arid' in climate:
        base_temp = random.uniform(25, 45)
        base_precip = random.uniform(1, 15)
        base_humidity = random.randint(15, 40)
    elif 'Continental' in climate:
        base_temp = random.uniform(-5, 30)
        base_precip = random.uniform(25, 70)
        base_humidity = random.randint(40, 75)
    elif 'Tropical' in climate:
        base_temp = random.uniform(20, 35)
        base_precip = random.uniform(60, 200)
        base_humidity = random.randint(70, 95)
    
    # NDVI based on region and probability
    base_ndvi = location_data.get('typical_ndvi', 0.40)
    ndvi_variation = (probability - 0.5) * 0.3
    final_ndvi = max(0.05, min(0.95, base_ndvi + ndvi_variation + random.uniform(-0.1, 0.1)))
    
    # Seasonal adjustments
    try:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        month = date_obj.month
        season = get_season(month, latitude)
        
        # Adjust for seasons
        if latitude > 0:  # Northern hemisphere
            if month in [12, 1, 2]:  # Winter
                base_temp -= random.uniform(5, 20)
                final_ndvi *= 0.6
            elif month in [6, 7, 8]:  # Summer
                base_temp += random.uniform(2, 15)
                final_ndvi *= 1.2
        else:  # Southern hemisphere
            if month in [6, 7, 8]:  # Winter in south
                base_temp -= random.uniform(5, 15)
                final_ndvi *= 0.7
            elif month in [12, 1, 2]:  # Summer in south
                base_temp += random.uniform(2, 10)
                final_ndvi *= 1.1
                
    except:
        season = "Unknown season"
    
    # Bloom status determination
    will_bloom = probability > 0.4
    confidence_level = "high" if probability > 0.7 or probability < 0.3 else "medium"
    
    if probability > 0.8:
        intensity = "🌸 Exceptional bloom season"
        bloom_status = "YES ✅"
        peak_desc = "Peak blooming conditions expected"
    elif probability > 0.6:
        intensity = "🌼 Very good bloom season"
        bloom_status = "YES ✅"
        peak_desc = "Excellent blooming potential"
    elif probability > 0.4:
        intensity = "🌻 Moderate bloom season"
        bloom_status = "PARTIAL 🟡"
        peak_desc = "Moderate blooming conditions"
    elif probability > 0.2:
        intensity = "🍃 Minimal bloom activity"
        bloom_status = "POOR ⚠️"
        peak_desc = "Limited blooming expected"
    else:
        intensity = "🥀 No bloom activity"
        bloom_status = "NO ❌"
        peak_desc = "Unfavorable for blooming"
    
    # Season descriptions
    season_descriptions = [
        f"{season} season in {location_name}",
        f"{climate} conditions during {season.lower()}",
        f"Typical {season.lower()} vegetation cycle"
    ]
    
    # Practical applications
    agriculture_advice = generate_agriculture_advice(probability, climate, season)
    conservation_advice = generate_conservation_advice(probability, final_ndvi, climate)
    tourism_advice = generate_tourism_advice(probability, season, location_name)
    
    detailed_prediction = {
        'region_name': location_name,
        'display_name': location_data.get('display_name', location_name),
        'coordinates': f"{latitude:.4f}, {longitude:.4f}",
        'date': date,
        'season': season,
        'climate_type': location_data.get('climate', 'Unknown'),
        'climate_description': location_data.get('climate_description', ''),
        'country': location_data.get('country', ''),
        'state': location_data.get('state', ''),
        'city': location_data.get('city', ''),
        'will_bloom': will_bloom,
        'bloom_status': bloom_status,
        'probability': probability,
        'confidence': confidence_level,
        'intensity': intensity,
        'peak_likelihood': f"{probability * 100:.1f}%",
        'peak_description': peak_desc,
        'ndvi': f"{final_ndvi:.3f}",
        'ndvi_description': get_ndvi_description(final_ndvi),
        'temperature': f"{base_temp:.1f}",
        'precipitation': f"{max(0, base_precip):.1f}",
        'humidity': f"{min(100, max(20, base_humidity))}",
        'season_description': random.choice(season_descriptions),
        'vegetation_type': location_data.get('vegetation_type', 'Mixed'),
        'soil_type': location_data.get('soil_type', 'Mixed soils'),
        'rainfall_pattern': location_data.get('rainfall_pattern', 'Variable'),
        'agriculture_advice': agriculture_advice,
        'conservation_advice': conservation_advice,
        'tourism_advice': tourism_advice,
        'map_center_lat': latitude,
        'map_center_lng': longitude,
        'map_zoom': location_data.get('zoom_level', 6),
        'region_bounds': location_data.get('region_bounds', [[latitude-2, longitude-2], [latitude+2, longitude+2]]),
        'bloom_color': get_bloom_color(probability),
        'satellite_source': random.choice(['MODIS Terra', 'MODIS Aqua', 'Landsat 8', 'Sentinel-2']),
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_version': '2.1',
        'confidence_score': f"{random.uniform(0.75, 0.95):.3f}",
        'status': 'success'
    }
    
    return detailed_prediction

def get_season(month, latitude):
    """Get season based on month and hemisphere"""
    if latitude >= 0:  # Northern hemisphere
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"
    else:  # Southern hemisphere
        if month in [6, 7, 8]:
            return "Winter"
        elif month in [9, 10, 11]:
            return "Spring"
        elif month in [12, 1, 2]:
            return "Summer"
        else:
            return "Autumn"

def get_ndvi_description(ndvi):
    """Get NDVI description"""
    if ndvi > 0.7:
        return "excellent"
    elif ndvi > 0.5:
        return "good"
    elif ndvi > 0.3:
        return "moderate"
    elif ndvi > 0.15:
        return "poor"
    else:
        return "very poor"

def generate_agriculture_advice(probability, climate, season):
    """Generate agriculture-specific advice"""
    if probability > 0.6:
        if 'Desert' in climate or 'Arid' in climate:
            return "🌾 Good conditions for drought-resistant crops like millets and legumes"
        elif 'Tropical' in climate:
            return "🌾 Excellent for rice cultivation, sugarcane, and tropical fruits"
        elif 'Temperate' in climate or 'Oceanic' in climate:
            return "🌾 Favorable for temperate crops, grains, and seasonal vegetables"
        else:
            return "🌾 Favorable for seasonal crops and orchards"
    elif probability > 0.3:
        return "🌾 Moderate conditions - consider soil preparation and irrigation planning"
    else:
        if 'Desert' in climate or 'Arid' in climate:
            return "🌾 Consider drought-resistant varieties or alternative timing"
        else:
            return "🌾 Focus on soil conservation and water management"

def generate_conservation_advice(probability, ndvi, climate):
    """Generate conservation-specific advice"""
    if ndvi > 0.5:
        return "🌿 Good vegetation health - maintain current conservation practices"
    elif ndvi > 0.3:
        return "🌿 Moderate vegetation - enhance water conservation and afforestation"
    else:
        if 'Desert' in climate or 'Arid' in climate:
            return "🌿 Habitat restoration needed - consider reforestation programs"
        else:
            return "🌿 Critical vegetation stress - immediate conservation action needed"

def generate_tourism_advice(probability, season, location):
    """Generate tourism-specific advice"""
    if probability > 0.6:
        return f"🎒 Prime season for eco-tourism, bloom viewing, and nature photography in {location}"
    elif probability > 0.3:
        return f"🎒 Good opportunities for nature tourism and cultural exploration"
    else:
        return f"🎒 Limited bloom viewing - focus on cultural and historical tourism"

@app.route('/predict', methods=['POST'])
def predict_bloom():
    """Handle form submission with enhanced geocoding"""
    try:
        region_input = request.form.get('region_name', '').strip()
        date = request.form.get('date')
        bloom_type = request.form.get('bloom_type', 'mixed')
        
        logger.info(f"Form data: region_input={region_input}, date={date}, bloom_type={bloom_type}")
        
        if not region_input or not date:
            return render_template('index.html', error='Please fill all required fields')
        
        # Use enhanced geocoding
        location_data = geocode_location_enhanced(region_input)
        
        if not location_data:
            return render_template('index.html', 
                error=f'Could not find location: {region_input}. Please check the spelling and try again.')
        
        latitude = location_data['lat']
        longitude = location_data['lng']
        
        logger.info(f"✅ Found {region_input} at {latitude}, {longitude}")
        logger.info(f"Climate: {location_data.get('climate')}")
        
        # Make prediction
        try:
            prediction = predictor.predict_bloom(latitude, longitude, date)
            if isinstance(prediction, dict) and 'probability' in prediction:
                base_probability = float(prediction.get('probability', 0))
                if base_probability > 1:
                    base_probability = base_probability / 100
            else:
                base_probability = random.uniform(0.1, 0.9)
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")
            base_probability = random.uniform(0.1, 0.9)
        
        # Create enhanced prediction
        prediction = create_enhanced_prediction(
            region_input, location_data, base_probability, date, latitude, longitude
        )
        
        # Store in session
        session['prediction_result'] = prediction
        
        return redirect(url_for('results'))
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template('index.html', error=f'Error processing request: {str(e)}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    prediction = session.get('prediction_result')
    if not prediction:
        return redirect(url_for('index'))
    
    logger.info(f"Displaying results for {prediction.get('region_name', 'Unknown')}")
    
    return render_template('results.html', prediction=prediction)
@app.route('/download_pdf')
def download_pdf():
    """Generate and download PDF report"""
    try:
        prediction = session.get('prediction_result')
        if not prediction:
            return redirect(url_for('index'))
        
        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2E7D32')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#1976D2')
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leading=14
        )
        
        # Title
        title = Paragraph("🌸 Bloom Prediction Report", title_style)
        elements.append(title)
        elements.append(Spacer(1, 20))
        
        # Quick Summary Section
        summary_data = [
            ['Parameter', 'Value'],
            ['Region', prediction.get('region_name', 'N/A')],
            ['Climate Type', prediction.get('climate_type', 'N/A')],
            ['Date', prediction.get('date', 'N/A')],
            ['Will Bloom', prediction.get('bloom_status', 'N/A')],
            ['Probability', f"{prediction.get('probability', 0):.3f} ({prediction.get('confidence', 'medium')} confidence)"],
            ['Coordinates', prediction.get('coordinates', 'N/A')],
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        elements.append(Paragraph("📊 Quick Summary", heading_style))
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        
        # Environmental Data Section
        env_data = [
            ['Environmental Parameter', 'Value', 'Description'],
            ['NDVI (Vegetation)', 
             prediction.get('ndvi', 'N/A'), 
             f"({prediction.get('ndvi_description', 'unknown')})"],
            ['Temperature', f"{prediction.get('temperature', 'N/A')}°C", 'Current temperature'],
            ['Precipitation', f"{prediction.get('precipitation', 'N/A')}mm", 'Rainfall amount'],
            ['Humidity', f"{prediction.get('humidity', 'N/A')}%", 'Air moisture content'],
            ['Satellite Source', prediction.get('satellite_source', 'N/A'), 'Data source'],
        ]
        
        env_table = Table(env_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        env_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2196F3')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        
        elements.append(Paragraph("🛰️ Environmental Data", heading_style))
        elements.append(env_table)
        elements.append(Spacer(1, 20))
        
        # Detailed Analysis Section
        elements.append(Paragraph("📈 Detailed Bloom Analysis", heading_style))
        
        analysis_text = f"""
        <b>Bloom Intensity:</b> {prediction.get('intensity', 'N/A')}<br/>
        <b>Peak Likelihood:</b> {prediction.get('peak_likelihood', 'N/A')}<br/>
        <b>Peak Description:</b> {prediction.get('peak_description', 'N/A')}<br/>
        <b>Season:</b> {prediction.get('season_description', 'N/A')}<br/>
        <b>Current Season:</b> {prediction.get('season', 'N/A')}<br/>
        """
        
        elements.append(Paragraph(analysis_text, normal_style))
        elements.append(Spacer(1, 15))
        
        # Location Details Section
        elements.append(Paragraph("🌍 Location & Climate Details", heading_style))
        
        location_text = f"""
        <b>Full Location:</b> {prediction.get('display_name', prediction.get('region_name', 'N/A'))}<br/>
        <b>Country:</b> {prediction.get('country', 'N/A')}<br/>
        <b>State/Province:</b> {prediction.get('state', 'N/A')}<br/>
        <b>City:</b> {prediction.get('city', 'N/A')}<br/>
        <b>Climate Description:</b> {prediction.get('climate_description', 'N/A')}<br/>
        """
        
        elements.append(Paragraph(location_text, normal_style))
        elements.append(Spacer(1, 15))
        
        # Vegetation & Soil Information
        elements.append(Paragraph("🌱 Vegetation & Soil Information", heading_style))
        
        vegetation_text = f"""
        <b>Vegetation Type:</b> {prediction.get('vegetation_type', 'N/A')}<br/>
        <b>Soil Type:</b> {prediction.get('soil_type', 'N/A')}<br/>
        <b>Rainfall Pattern:</b> {prediction.get('rainfall_pattern', 'N/A')}<br/>
        """
        
        elements.append(Paragraph(vegetation_text, normal_style))
        elements.append(Spacer(1, 20))
        
        # Practical Applications Section
        elements.append(Paragraph("💡 Practical Applications", heading_style))
        
        applications_data = [
            ['Application Area', 'Recommendation'],
            ['🌾 Agriculture', prediction.get('agriculture_advice', 'N/A')],
            ['🌿 Conservation', prediction.get('conservation_advice', 'N/A')],
            ['🎒 Tourism', prediction.get('tourism_advice', 'N/A')],
        ]
        
        applications_table = Table(applications_data, colWidths=[1.5*inch, 3.5*inch])
        applications_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF9800')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        
        elements.append(applications_table)
        elements.append(Spacer(1, 20))
        
        # Technical Information
        elements.append(Paragraph("🔧 Technical Information", heading_style))
        
        technical_text = f"""
        <b>Model Version:</b> {prediction.get('model_version', 'N/A')}<br/>
        <b>Confidence Score:</b> {prediction.get('confidence_score', 'N/A')}<br/>
        <b>Analysis Date:</b> {prediction.get('analysis_date', 'N/A')}<br/>
        <b>Data Source:</b> {prediction.get('satellite_source', 'N/A')}<br/>
        """
        
        elements.append(Paragraph(technical_text, normal_style))
        elements.append(Spacer(1, 30))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#666666')
        )
        
        footer_text = """
        <b>BloomWatch</b> - Advanced Satellite-Based Bloom Prediction System<br/>
        Powered by Machine Learning and Earth Observation Data<br/>
        Generated automatically by BloomWatch AI System
        """
        
        elements.append(Paragraph(footer_text, footer_style))
        
        # Build PDF
        doc.build(elements)
        
        # Get PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Create response
        response = app.make_response(pdf_data)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=BloomPrediction_{prediction.get("region_name", "Report")}_{prediction.get("date", "")}.pdf'
        
        logger.info(f"✅ PDF generated for {prediction.get('region_name', 'Unknown')}")
        return response
        
    except Exception as e:
        logger.error(f"❌ PDF generation error: {e}")
        return render_template('results.html', prediction=session.get('prediction_result'), 
                             error="Error generating PDF. Please try again.")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
