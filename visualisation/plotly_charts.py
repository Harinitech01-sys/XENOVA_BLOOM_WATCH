import plotly.graph_objects as go

def ndvi_timeseries(months, ndvi_values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=ndvi_values, mode='lines+markers', name='NDVI'))
    fig.update_layout(title="NDVI Over Time", xaxis_title="Month", yaxis_title="NDVI")
    return fig.to_json()
