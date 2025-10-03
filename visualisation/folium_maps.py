import folium

def make_basic_map(center=[20, 79], zoom_start=3):
    m = folium.Map(location=center, zoom_start=zoom_start)
    folium.Marker([20, 79], popup="Center").add_to(m)
    return m._repr_html_()
