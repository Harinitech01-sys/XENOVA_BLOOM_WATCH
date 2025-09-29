from ndvi_processing import calculate_ndvi
import json

with open('../regions.json') as f:
    regions = json.load(f)
for region in regions.keys():
    data = {"NIR":[0.8,0.7], "RED":[0.4,0.3]} # Replace with real data
    print(calculate_ndvi(data))
