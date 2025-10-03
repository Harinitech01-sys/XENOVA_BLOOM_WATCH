import requests
from config import BACKEND_API_URL

def get_dashboard_data():
    try:
        resp = requests.get(f"{BACKEND_API_URL}/api/bloom-dashboard")
        if resp.status_code == 200:
            return resp.json()
        return {"error": "Failed to fetch dashboard data"}
    except Exception as e:
        return {"error": str(e)}
