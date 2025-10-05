document.addEventListener('DOMContentLoaded', function() {
    console.log('🛰️ BloomWatch Region Predictor Initialized');
    
    const form = document.getElementById('region-prediction-form');
    const resultDiv = document.getElementById('result');
    
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const regionName = document.getElementById('region-select').value.trim();
            const date = document.getElementById('prediction-date').value;
            
            if (!regionName) {
                showError('⚠️ Please select a region');
                return;
            }
            
            // Show loading
            const button = form.querySelector('button');
            const originalText = button.textContent;
            button.textContent = '🔄 Analyzing...';
            button.disabled = true;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        region_name: regionName,
                        date: date
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showResult(data);
                } else {
                    showError(data.error || 'Prediction failed');
                }
                
            } catch (error) {
                showError('Error: ' + error.message);
            } finally {
                button.textContent = originalText;
                button.disabled = false;
            }
        });
    }
    
    function showResult(data) {
        const probability = data.probability || 0;
        const riskLevel = probability >= 70 ? '🔴 High Risk' : 
                         probability >= 40 ? '🟡 Medium Risk' : '🟢 Low Risk';
        
        resultDiv.innerHTML = `
            <h3>🎯 Prediction Results</h3>
            <p><strong>🗺️ Region:</strong> ${data.region_name}</p>
            <p><strong>📍 Coordinates:</strong> ${data.coordinates}</p>
            <p><strong>🌊 Bloom Probability:</strong> ${probability}%</p>
            <p><strong>🔬 Species:</strong> ${data.species || 'Mixed'}</p>
            <p><strong>📅 Date:</strong> ${data.date}</p>
            <p><strong>⚡ Confidence:</strong> ${data.confidence || 'N/A'}%</p>
            <p><strong>🚨 Risk Level:</strong> ${riskLevel}</p>
        `;
        
        resultDiv.style.display = 'block';
        resultDiv.className = 'result-section';
    }
    
    function showError(error) {
        resultDiv.innerHTML = `
            <h3>❌ Error</h3>
            <p style="color: red;">${error}</p>
        `;
        resultDiv.style.display = 'block';
        resultDiv.className = 'result-section error';
    }
});
