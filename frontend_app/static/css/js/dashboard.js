document.addEventListener('DOMContentLoaded', function () {
    // Initialize Leaflet Map
    var map = L.map('map').setView([20, 79], 3);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 18}).addTo(map);

    // Fetch summary/dashboard data from Flask API and render chart
    fetch("/api/dashboard_data")
        .then(res => res.json())
        .then(data => {
            // Example: update Plotly chart with NDVI values by month
            let months = data.months || ['Jan','Feb','Mar','Apr'];
            let ndvi = data.ndvi || [0.3, 0.5, 0.7, 0.6];
            let chartDiv = document.getElementById('chart-div');
            let trace = { x: months, y: ndvi, type: 'scatter', mode: 'lines+markers', name: 'NDVI' };
            let layout = { title: 'Monthly Bloom NDVI', xaxis: {title:'Month'}, yaxis: {title:'NDVI'} };
            Plotly.newPlot(chartDiv, [trace], layout);
        })
        .catch(() => {
            // Draw example static chart on error
            let chartDiv = document.getElementById('chart-div');
            let trace = { x: ['Jan', 'Feb', 'Mar', 'Apr'], y: [0.3, 0.5, 0.7, 0.6], type: 'scatter', mode: 'lines+markers' };
            Plotly.newPlot(chartDiv, [trace], { title: "Demo NDVI Chart" });
        });
});
