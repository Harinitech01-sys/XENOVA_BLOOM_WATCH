from plotly_charts import ndvi_timeseries

def generate_dashboard_chart(data):
    months = data.get('months', ['Jan','Feb','Mar','Apr'])
    ndvi = data.get('ndvi', [0.3, 0.4, 0.7, 0.6])
    return ndvi_timeseries(months, ndvi)
