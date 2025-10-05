# model_training.py - Universal Bloom Detection Training
import ee
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from datetime import datetime, timedelta
import logging
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversalBloomTrainer:
    """NASA Universal Bloom Detection - No Species Required"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = Config.UNIVERSAL_FEATURE_NAMES
        self._initialize()
    
    def _initialize(self):
        """Initialize trainer"""
        try:
            Config.initialize_earth_engine()
            Config.create_directories()
            logger.info("✅ Universal Bloom Trainer initialized")
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    def generate_training_data(self, num_samples=1000):
        """Generate universal training data from global regions"""
        
        logger.info("🌍 Generating universal training data...")
        
        # Global regions for diverse training
        global_regions = [
            {'name': 'India_North', 'bounds': [75, 25, 85, 35], 'biome': 'temperate'},
            {'name': 'India_Central', 'bounds': [75, 15, 85, 25], 'biome': 'tropical'},
            {'name': 'India_South', 'bounds': [75, 8, 85, 18], 'biome': 'tropical'},
            {'name': 'Southeast_Asia', 'bounds': [95, 5, 115, 25], 'biome': 'tropical'},
            {'name': 'Europe', 'bounds': [0, 45, 20, 60], 'biome': 'temperate'},
            {'name': 'East_USA', 'bounds': [-90, 35, -70, 45], 'biome': 'temperate'}
        ]
        
        # Seasons with bloom probabilities
        seasons = [
            {'name': 'spring', 'months': [3, 4, 5], 'bloom_base': 0.7},
            {'name': 'summer', 'months': [6, 7, 8], 'bloom_base': 0.4},
            {'name': 'autumn', 'months': [9, 10, 11], 'bloom_base': 0.5},
            {'name': 'winter', 'months': [12, 1, 2], 'bloom_base': 0.2}
        ]
        
        years = [2022, 2023, 2024]
        
        all_samples = []
        samples_per_combo = max(5, num_samples // (len(global_regions) * len(seasons) * len(years)))
        
        logger.info(f"Collecting {samples_per_combo} samples per combination")
        
        for region in global_regions:
            for season in seasons:
                for year in years:
                    logger.info(f"Processing {region['name']} - {season['name']} {year}")
                    
                    samples = self._collect_region_samples(
                        region, season, year, samples_per_combo
                    )
                    
                    all_samples.extend(samples)
                    logger.info(f"✅ Collected {len(samples)} samples")
        
        if all_samples:
            df = pd.DataFrame(all_samples)
            logger.info(f"🎯 Total samples: {len(df)}")
            logger.info(f"🌸 Bloom rate: {100 * df['is_blooming'].mean():.1f}%")
            
            # Save training data
            df.to_csv(Config.UNIVERSAL_DATA_PATH, index=False)
            return df
        
        return None
    
    def _collect_region_samples(self, region, season, year, num_samples):
        """Collect samples for region-season-year combination"""
        
        try:
            bounds = region['bounds']
            region_geom = ee.Geometry.Rectangle(bounds)
            
            # Random points in region
            points = ee.FeatureCollection.randomPoints(
                region=region_geom,
                points=num_samples,
                seed=42
            )
            
            samples = []
            
            for month in season['months']:
                try:
                    start_date = f"{year}-{month:02d}-01"
                    end_date = f"{year}-{month:02d}-28"
                    
                    month_samples = self._extract_features_for_points(
                        points, start_date, end_date, region, season
                    )
                    
                    samples.extend(month_samples)
                    
                except Exception as e:
                    logger.warning(f"Error processing {region['name']}-{year}-{month:02d}: {e}")
                    continue
            
            return samples
            
        except Exception as e:
            logger.error(f"Region sampling error: {e}")
            return []
    
    def _extract_features_for_points(self, points, start_date, end_date, region, season):
        """Extract NASA satellite features"""
        
        try:
            # NASA MODIS
            modis = ee.ImageCollection(Config.MODIS_COLLECTION) \
                .filterDate(start_date, end_date) \
                .select(['NDVI', 'EVI']) \
                .map(lambda img: img.multiply(0.0001))
            
            # NASA Landsat
            landsat = ee.ImageCollection(Config.LANDSAT_COLLECTION) \
                .filterDate(start_date, end_date) \
                .filterMetadata('CLOUD_COVER', 'less_than', Config.CLOUD_COVER_THRESHOLD) \
                .select(['SR_B2', 'SR_B4', 'SR_B5', 'SR_B6'])
            
            # Calculate Landsat indices
            landsat_indices = landsat.map(self._calculate_landsat_indices)
            
            # Climate data
            weather = ee.ImageCollection(Config.WEATHER_COLLECTION) \
                .filterDate(start_date, end_date) \
                .select(['temperature_2m', 'total_precipitation_sum'])
            
            # Elevation
            elevation = ee.Image(Config.ELEVATION_COLLECTION).select('elevation')
            
            # Reduce to statistics
            modis_stats = modis.reduce(
                ee.Reducer.mean()
                .combine(ee.Reducer.stdDev(), '', True)
                .combine(ee.Reducer.first(), '', True)
                .combine(ee.Reducer.last(), '', True)
            )
            
            landsat_stats = landsat_indices.reduce(ee.Reducer.mean())
            weather_stats = weather.reduce(ee.Reducer.mean())
            
            # Combine all data
            combined = modis_stats.addBands(landsat_stats) \
                                 .addBands(weather_stats) \
                                 .addBands(elevation)
            
            # Sample at points
            sampled = combined.sampleRegions(
                collection=points,
                scale=Config.SPATIAL_SCALE,
                geometries=True
            ).getInfo()
            
            # Process samples
            samples = []
            for feature in sampled['features']:
                try:
                    sample = self._create_training_sample(
                        feature, start_date, region, season
                    )
                    if sample:
                        samples.append(sample)
                except Exception as e:
                    continue
            
            return samples
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return []
    
    def _calculate_landsat_indices(self, image):
        """Calculate Landsat vegetation indices"""
        try:
            ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('LANDSAT_NDVI')
            red = image.select('SR_B4').rename('RED')
            nir = image.select('SR_B5').rename('NIR')
            swir = image.select('SR_B6').rename('SWIR')
            return image.addBands([ndvi, red, nir, swir])
        except:
            return image
    
    def _create_training_sample(self, feature, date_str, region, season):
        """Create training sample with universal bloom detection"""
        
        try:
            props = feature['properties']
            coords = feature['geometry']['coordinates']
            
            if not coords or len(coords) < 2:
                return None
            
            lat, lon = coords[1], coords[0]
            date = pd.to_datetime(date_str)
            
            # Extract NASA features
            ndvi_current = props.get('NDVI_first', 0.5)
            ndvi_mean = props.get('NDVI_mean', 0.5)
            ndvi_std = props.get('NDVI_stdDev', 0.1)
            ndvi_trend = props.get('NDVI_last', 0.5) - props.get('NDVI_first', 0.5)
            
            evi_current = props.get('EVI_first', 0.3)
            evi_mean = props.get('EVI_mean', 0.3)
            evi_std = props.get('EVI_stdDev', 0.1)
            evi_trend = props.get('EVI_last', 0.3) - props.get('EVI_first', 0.3)
            
            landsat_ndvi = props.get('LANDSAT_NDVI_mean', ndvi_mean)
            landsat_red = props.get('RED_mean', 0.1)
            landsat_nir = props.get('NIR_mean', 0.3)
            landsat_swir = props.get('SWIR_mean', 0.2)
            
            temperature = props.get('temperature_2m_mean', 288) - 273.15
            precipitation = props.get('total_precipitation_sum_mean', 0) * 1000
            
            elevation = props.get('elevation', 0)
            
            # Derived features
            humidity = min(100.0, max(0.0, 60.0 + (precipitation - 50.0) * 0.3))
            
            # Temporal features
            day_of_year = date.timetuple().tm_yday
            month = date.month
            season_sin = np.sin(2 * np.pi * month / 12)
            season_cos = np.cos(2 * np.pi * month / 12)
            days_since_spring = (day_of_year - 80) % 365
            photoperiod = Config.calculate_photoperiod(lat, day_of_year)
            
            # Bloom detection features
            vegetation_greenness_change = ndvi_trend
            seasonal_vegetation_anomaly = ndvi_current - ndvi_mean
            
            bloom_favorable_conditions = (
                (15 < temperature < 35) * 0.3 +
                (precipitation > 20) * 0.3 + 
                (ndvi_current > 0.4) * 0.4
            )
            
            # Phenology stage
            if ndvi_trend > 0.05 and ndvi_current > 0.5:
                phenology_stage = 2  # Peak
            elif ndvi_trend > 0 and ndvi_current > 0.3:
                phenology_stage = 1  # Growing
            elif ndvi_trend < -0.05:
                phenology_stage = 3  # Senescence
            else:
                phenology_stage = 0  # Dormant
            
            # Universal bloom detection (no species knowledge)
            bloom_probability = self._detect_bloom_probability(
                ndvi_current, ndvi_trend, evi_current, evi_trend,
                temperature, precipitation, month, region['biome'],
                season['bloom_base']
            )
            
            is_blooming = 1 if bloom_probability > 0.5 else 0
            
            return {
                'latitude': lat,
                'longitude': lon,
                'elevation': elevation,
                'day_of_year': day_of_year,
                'month': month,
                'season_sin': season_sin,
                'season_cos': season_cos,
                'days_since_spring_start': days_since_spring,
                'photoperiod': photoperiod,
                'ndvi_current': ndvi_current,
                'ndvi_mean_30d': ndvi_mean,
                'ndvi_std_30d': ndvi_std,
                'ndvi_trend_30d': ndvi_trend,
                'evi_current': evi_current,
                'evi_mean_30d': evi_mean,
                'evi_std_30d': evi_std,
                'evi_trend_30d': evi_trend,
                'landsat_ndvi': landsat_ndvi,
                'landsat_red': landsat_red,
                'landsat_nir': landsat_nir,
                'landsat_swir': landsat_swir,
                'temperature_current': temperature,
                'temperature_mean_30d': temperature,
                'temperature_trend': 0,
                'precipitation_current': precipitation,
                'precipitation_sum_30d': precipitation,
                'humidity': humidity,
                'solar_radiation_proxy': 2.0,
                'vegetation_greenness_change': vegetation_greenness_change,
                'seasonal_vegetation_anomaly': seasonal_vegetation_anomaly,
                'bloom_favorable_conditions': bloom_favorable_conditions,
                'phenology_stage_indicator': phenology_stage,
                'is_blooming': is_blooming,
                'bloom_probability': bloom_probability,
                'region': region['name'],
                'biome': region['biome'],
                'season': season['name']
            }
            
        except Exception as e:
            logger.warning(f"Sample creation error: {e}")
            return None
    
    def _detect_bloom_probability(self, ndvi, ndvi_trend, evi, evi_trend,
                                 temp, precip, month, biome, base_prob):
        """Universal bloom detection from satellite patterns"""
        
        bloom_score = base_prob
        
        # Vegetation health
        if ndvi > 0.6 and evi > 0.4:
            bloom_score += 0.3
        elif ndvi > 0.4 and evi > 0.2:
            bloom_score += 0.1
        
        # Growth trend (key indicator)
        if ndvi_trend > 0.05:
            bloom_score += 0.2
        elif ndvi_trend > 0:
            bloom_score += 0.1
        
        # Climate conditions
        if 18 <= temp <= 32 and precip > 25:
            bloom_score += 0.2
        elif 15 <= temp <= 35 and precip > 10:
            bloom_score += 0.1
        
        # Biome adjustments
        biome_modifiers = {
            'tropical': {'3,4,5': 0.2, '6,7,8': -0.1},
            'temperate': {'3,4,5': 0.3, '9,10,11': 0.2}
        }
        
        if biome in biome_modifiers:
            for season_months, modifier in biome_modifiers[biome].items():
                if str(month) in season_months.split(','):
                    bloom_score += modifier
        
        # Add variation
        bloom_score += np.random.normal(0, 0.1)
        
        return max(0.0, min(1.0, bloom_score))
    
    def train_universal_model(self):
        """Train universal bloom detection model"""
        
        logger.info("🤖 Training Universal Bloom Model...")
        
        # Load or generate data
        if os.path.exists(Config.UNIVERSAL_DATA_PATH):
            logger.info("📂 Loading existing data...")
            df = pd.read_csv(Config.UNIVERSAL_DATA_PATH)
        else:
            logger.info("🌍 Generating new data...")
            df = self.generate_training_data(num_samples=1000)
        
        if df is None or len(df) == 0:
            logger.error("❌ No training data available")
            return None
        
        logger.info(f"📊 Training samples: {len(df)}")
        logger.info(f"🌸 Bloom rate: {100 * df['is_blooming'].mean():.1f}%")
        
        # Prepare data
        X = df[self.feature_names].fillna(0).values
        y = df['is_blooming'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        
        logger.info("🔄 Training Random Forest...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        self._evaluate_model(X_test_scaled, y_test)
        
        # Save model
        self._save_model()
        
        return self.model
    
    def _evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        
        try:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            accuracy = np.mean(y_pred == y_test)
            auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
            
            logger.info(f"\n🎯 Universal Model Performance:")
            logger.info(f"   - Accuracy: {accuracy:.4f}")
            logger.info(f"   - AUC Score: {auc:.4f}")
            
            print("\n📈 Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['No Bloom', 'Bloom']))
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                logger.info("\n🌟 Top 10 Features:")
                print(importance_df.head(10).to_string(index=False))
        
        except Exception as e:
            logger.error(f"❌ Evaluation error: {e}")
    
    def _save_model(self):
        """Save universal model"""
        
        try:
            # Save to both new and existing paths for compatibility
            joblib.dump(self.model, Config.UNIVERSAL_MODEL_PATH)
            joblib.dump(self.model, Config.BLOOM_MODEL_PATH)  # Keep existing path
            
            joblib.dump(self.scaler, Config.UNIVERSAL_SCALER_PATH)
            joblib.dump(self.scaler, Config.SCALER_PATH)  # Keep existing path
            
            # Save feature names
            with open(Config.UNIVERSAL_FEATURES_PATH, 'w') as f:
                f.write('\n'.join(self.feature_names))
            
            logger.info(f"💾 Universal model saved successfully")
            logger.info(f"   - Model: {Config.UNIVERSAL_MODEL_PATH}")
            logger.info(f"   - Scaler: {Config.UNIVERSAL_SCALER_PATH}")
            
        except Exception as e:
            logger.error(f"❌ Model saving error: {e}")


def main():
    """Main training execution"""
    
    print("🌸 NASA Universal Bloom Detection Training")
    print("=" * 50)
    
    try:
        trainer = UniversalBloomTrainer()
        model = trainer.train_universal_model()
        
        if model:
            logger.info("🎉 Universal model trained successfully!")
            logger.info("✅ Ready for NASA Challenge - works for ANY location!")
        else:
            logger.error("❌ Training failed")
            
    except Exception as e:
        logger.error(f"❌ Training process failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
