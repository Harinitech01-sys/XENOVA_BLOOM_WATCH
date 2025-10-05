# """
# API Endpoints for BloomWatch Prediction Service
# """
# from flask import Blueprint, request, jsonify
# import logging
# import traceback
# from datetime import datetime, timedelta
# import pandas as pd
# import numpy as np
# from prediction_utils import PredictionUtils
# from ..model_training import ModelTrainingPipeline
# from ..predictive_modeling import BloomPredictor
# from ..config import Config

# # Create Blueprint
# prediction_bp = Blueprint('prediction', __name__)
# logger = logging.getLogger(__name__)

# # Initialize utilities
# prediction_utils = PredictionUtils()

# @prediction_bp.route('/bloom-forecast', methods=['POST'])
# def bloom_forecast():
#     """
#     Predict bloom probability for given region and time period
    
#     Expected JSON payload:
#     {
#         "region": "california",
#         "start_date": "2024-06-01",
#         "end_date": "2024-06-30",
#         "features": {
#             "current_ndvi": 0.65,
#             "temperature_avg": 25.0,
#             "precipitation": 10.0
#         }
#     }
#     """
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({'error': 'No JSON data provided'}), 400
        
#         # Extract parameters
#         region = data.get('region', 'california')
#         start_date = data.get('start_date', datetime.now().strftime('%Y-%m-%d'))
#         end_date = data.get('end_date', (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'))
#         features = data.get('features', {})
        
#         logger.info(f"Bloom forecast request for {region} from {start_date} to {end_date}")
        
#         # Load trained model
#         try:
#             pipeline = ModelTrainingPipeline()
#             predictor = pipeline.load_best_model()
#         except Exception as e:
#             # If no trained model, create and train a quick one
#             logger.warning(f"No trained model found, creating new one: {e}")
#             predictor = BloomPredictor()
#             sample_data = prediction_utils.create_sample_features()
#             predictor.train(sample_data)
        
#         # Prepare features for prediction
#         prediction_features = prediction_utils.prepare_prediction_features(
#             features, region, start_date, end_date
#         )
        
#         # Make predictions
#         if len(prediction_features) > 0:
#             bloom_predictions = predictor.predict_bloom_timing(
#                 prediction_features,
#                 days_ahead=(pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
#             )
#         else:
#             bloom_predictions = {}
        
#         # Get feature importance
#         feature_importance = predictor.get_feature_importance()
        
#         response = {
#             'status': 'success',
#             'region': region,
#             'prediction_period': {
#                 'start_date': start_date,
#                 'end_date': end_date
#             },
#             'predictions': bloom_predictions,
#             'feature_importance': feature_importance,
#             'model_info': {
#                 'type': predictor.model_type,
#                 'is_trained': predictor.is_trained,
#                 'features_used': len(predictor.feature_names)
#             },
#             'timestamp': datetime.utcnow().isoformat()
#         }
        
#         logger.info(f"Bloom forecast completed for {region}")
#         return jsonify(response)
        
#     except Exception as e:
#         logger.error(f"Bloom forecast error: {str(e)}\n{traceback.format_exc()}")
#         return jsonify({
#             'status': 'error',
#             'message': str(e),
#             'timestamp': datetime.utcnow().isoformat()
#         }), 500

# @prediction_bp.route('/model-training', methods=['POST'])
# def train_model():
#     """
#     Train new prediction models
    
#     Expected JSON payload:
#     {
#         "data_source": "sample",
#         "model_types": ["random_forest", "gradient_boosting"],
#         "force_retrain": false
#     }
#     """
#     try:
#         data = request.get_json() or {}
        
#         data_source = data.get('data_source', 'sample')
#         model_types = data.get('model_types', Config.MODEL_TYPES)
#         force_retrain = data.get('force_retrain', False)
        
#         logger.info(f"Model training request: source={data_source}, models={model_types}")
        
#         # Initialize training pipeline
#         pipeline = ModelTrainingPipeline()
        
#         # Prepare training data
#         training_data = pipeline.prepare_training_data(data_source=data_source)
        
#         # Train models
#         training_results = pipeline.train_models(training_data, model_types=model_types)
        
#         # Evaluate models
#         evaluation = pipeline.evaluate_models()
        
#         # Save models
#         pipeline.save_models()
        
#         response = {
#             'status': 'success',
#             'training_completed': True,
#             'models_trained': list(training_results.keys()),
#             'best_model': evaluation['best_model'],
#             'training_results': training_results,
#             'data_info': {
#                 'samples': len(training_data),
#                 'features': len(training_data.columns) - 1,
#                 'source': data_source
#             },
#             'timestamp': datetime.utcnow().isoformat()
#         }
        
#         logger.info(f"Model training completed. Best model: {evaluation['best_model']['type']}")
#         return jsonify(response)
        
#     except Exception as e:
#         logger.error(f"Model training error: {str(e)}\n{traceback.format_exc()}")
#         return jsonify({
#             'status': 'error',
#             'message': str(e),
#             'timestamp': datetime.utcnow().isoformat()
#         }), 500

# @prediction_bp.route('/model-evaluation', methods=['GET'])
# def evaluate_models():
#     """Get evaluation metrics for all trained models"""
#     try:
#         pipeline = ModelTrainingPipeline()
        
#         # Load existing results
#         import os
#         import json
#         results_file = os.path.join(Config.MODEL_PATH, 'training_results.json')
        
#         if os.path.exists(results_file):
#             with open(results_file, 'r') as f:
#                 training_results = json.load(f)
            
#             # Find best model
#             best_model = max(training_results.items(), key=lambda x: x[1]['test_r2'])
            
#             response = {
#                 'status': 'success',
#                 'models_available': list(training_results.keys()),
#                 'best_model': {
#                     'type': best_model[0],
#                     'test_r2': best_model[1]['test_r2'],
#                     'test_mse': best_model[1]['test_mse'],
#                     'cv_r2_mean': best_model[1].get('cv_r2_mean', 'N/A')
#                 },
#                 'all_results': training_results,
#                 'timestamp': datetime.utcnow().isoformat()
#             }
#         else:
#             response = {
#                 'status': 'success',
#                 'message': 'No trained models found',
#                 'models_available': [],
#                 'timestamp': datetime.utcnow().isoformat()
#             }
        
#         return jsonify(response)
        
#     except Exception as e:
#         logger.error(f"Model evaluation error: {str(e)}")
#         return jsonify({
#             'status': 'error',
#             'message': str(e),
#             'timestamp': datetime.utcnow().isoformat()
#         }), 500

# @prediction_bp.route('/quick-predict', methods=['POST'])
# def quick_predict():
#     """
#     Quick bloom prediction with minimal parameters
    
#     Expected JSON payload:
#     {
#         "ndvi": 0.65,
#         "region": "california",
#         "date": "2024-06-15"
#     }
#     """
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({'error': 'No JSON data provided'}), 400
        
#         ndvi = data.get('ndvi', 0.5)
#         region = data.get('region', 'california')
#         target_date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
#         # Simple rule-based prediction for quick response
#         bloom_probability = prediction_utils.calculate_quick_bloom_probability(ndvi, region, target_date)
        
#         # Determine confidence level
#         if bloom_probability > 0.8:
#             confidence = 'high'
#         elif bloom_probability > 0.5:
#             confidence = 'medium'
#         else:
#             confidence = 'low'
        
#         response = {
#             'status': 'success',
#             'bloom_probability': float(bloom_probability),
#             'confidence': confidence,
#             'factors': {
#                 'ndvi_threshold_met': ndvi > Config.NDVI_BLOOM_THRESHOLD,
#                 'seasonal_factor': prediction_utils.get_seasonal_factor(target_date, region),
#                 'region_suitability': prediction_utils.get_region_bloom_potential(region)
#             },
#             'prediction_date': target_date,
#             'region': region,
#             'timestamp': datetime.utcnow().isoformat()
#         }
        
#         return jsonify(response)
        
#     except Exception as e:
#         logger.error(f"Quick prediction error: {str(e)}")
#         return jsonify({
#             'status': 'error',
#             'message': str(e),
#             'timestamp': datetime.utcnow().isoformat()
#         }), 500

# @prediction_bp.route('/health', methods=['GET'])
# def health_check():
#     """Health check for prediction service"""
#     try:
#         # Check if models are available
#         model_status = "available" if os.path.exists(Config.MODEL_PATH) else "not_available"
        
#         response = {
#             'status': 'healthy',
#             'service': 'BloomWatch Prediction Service',
#             'models_status': model_status,
#             'supported_regions': list(Config.SUPPORTED_REGIONS.keys()),
#             'timestamp': datetime.utcnow().isoformat()
#         }
        
#         return jsonify(response)
        
#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'message': str(e),
#             'timestamp': datetime.utcnow().isoformat()
#         }), 500

from flask import Blueprint, request, jsonify
from predictive_modeling import BloomPredictor
import logging

logger = logging.getLogger(__name__)

# Create Blueprint for API endpoints
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Initialize predictor
predictor = BloomPredictor()
predictor.load_model()

@api_bp.route('/predict', methods=['POST'])
def api_predict():
    """API endpoint for single location prediction"""
    try:
        data = request.get_json()
        
        result = predictor.predict_bloom(
            data['latitude'],
            data['longitude'],
            data.get('date', '2024-04-15')
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/batch', methods=['POST'])
def api_batch_predict():
    """API endpoint for batch predictions"""
    try:
        data = request.get_json()
        locations = data.get('locations', [])
        
        results = predictor.batch_predict(locations)
        
        return jsonify({
            'success': True,
            'data': {
                'predictions': results,
                'count': len(results)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    # Add this to your existing prediction_endpoints.py

@bp.route('/nasa_universal_predict', methods=['POST'])
def nasa_universal_prediction():
    """
    NASA Challenge Endpoint: Universal bloom prediction
    No species required - just location and date
    """
    try:
        data = request.get_json()
        
        required_fields = ['latitude', 'longitude', 'date']
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': 'Missing required fields',
                'required': required_fields,
                'nasa_challenge': 'Universal BloomWatch'
            }), 400
        
        # Use your universal predictor
        from predictive_modeling import UniversalBloomPredictor
        
        predictor = UniversalBloomPredictor()
        result = predictor.predict_any_location(
            data['latitude'],
            data['longitude'], 
            data['date']
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'nasa_challenge': 'Universal BloomWatch'
        }), 500


