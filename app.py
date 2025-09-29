from flask import Flask, request, jsonify
from data_acquisition import fetch_satellite_data
from ndvi_processing import calculate_ndvi

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def get_data():
    region = request.args.get('region', 'India')
    data = fetch_satellite_data(region)
    ndvi = calculate_ndvi(data)
    return jsonify({"region": region, "ndvi": ndvi})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
