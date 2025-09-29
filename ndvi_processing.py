import numpy as np

def calculate_ndvi(data):
    nir = np.array(data['NIR'])
    red = np.array(data['RED'])
    ndvi = (nir - red) / (nir + red)
    return ndvi.tolist()
