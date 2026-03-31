# Dashboard.py
import requests
from fastapi import APIRouter
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

router = APIRouter()

# API Keys from .env
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


# Configuration
FB_HEADERS = {"X-Auth-Token": FOOTBALL_DATA_KEY}
FB_BASE_URL = "https://api.football-data.org/v4/competitions/PL"
NEWS_URL = "https://newsapi.org/v2/everything"

def get_fixtures_and_results():
    """Fetches both upcoming fixtures and last completed results."""
    try:
        # 1. Get Next Matchweek number
        res = requests.get(f"{FB_BASE_URL}/matches", headers=FB_HEADERS, params={"status": "SCHEDULED"})
        next_data = res.json().get('matches', [])
        next_mw = next_data[0]['matchday'] if next_data else None

        # 2. Get Previous Matchweek number
        res = requests.get(f"{FB_BASE_URL}/matches", headers=FB_HEADERS, params={"status": "FINISHED"})
        prev_data = res.json().get('matches', [])
        prev_mw = max(m['matchday'] for m in prev_data) if prev_data else None

        # 3. Fetch full data for both weeks
        fixtures = []
        if next_mw:
            res = requests.get(f"{FB_BASE_URL}/matches", headers=FB_HEADERS, params={"matchday": next_mw})
            fixtures = res.json().get('matches', [])

        results = []
        if prev_mw:
            res = requests.get(f"{FB_BASE_URL}/matches", headers=FB_HEADERS, params={"matchday": prev_mw})
            results = res.json().get('matches', [])

        return {
            "next_matchweek": next_mw,
            "fixtures": fixtures,
            "prev_matchweek": prev_mw,
            "results": results
        }
    except Exception as e:
        return {"error": f"Match data error: {str(e)}"}

def get_standings():
    try:
        res = requests.get(f"{FB_BASE_URL}/standings", headers=FB_HEADERS)
        data = res.json()
        return data['standings'][0]['table']
    except Exception:
        return []

def get_news():
    try:
        params = {
            'q': '"Premier League" OR "EPL"',
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': NEWS_API_KEY
        }
        res = requests.get(NEWS_URL, params=params)
        return res.json().get('articles', [])[:10]
    except Exception:
        return []

@router.get("/dashboard")
async def get_full_dashboard():
    match_info = get_fixtures_and_results()
    
    return {
        "standings": get_standings(),
        "fixtures": match_info.get("fixtures"),
        "results": match_info.get("results"),
        "news": get_news(),
        "metadata": {
            "next_mw": match_info.get("next_matchweek"),
            "prev_mw": match_info.get("prev_matchweek")
        }
    }