from flask import Flask, render_template, jsonify
from api_client import get_dashboard_data

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    # (Optional) You could supply data here for the dashboard
    return render_template("dashboard.html")

@app.route("/api/dashboard_data")
def dashboard_data():
    # This proxies data from backend (or simulation)
    data = get_dashboard_data()
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
