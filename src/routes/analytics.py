# analytics.py
import pandas as pd
from sqlalchemy import create_engine
from fastapi import APIRouter, HTTPException
from src.utils.setting import get_settings
from src.utils.logger import logging

# Initialize the router
router = APIRouter()

def fetch_supabase_analytics_tables():
    """
    Fetches all analytical tables and converts them to 
    JSON-friendly dictionaries.
    """
    settings = get_settings()
    db_url = settings.database_url
    
    # Correct the database protocol for SQLAlchemy compatibility
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    elif db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)

    tables_to_fetch = [
        "top_players_goals", "top_players_assists", "top_players_shots",
        "top_players_key_passes", "top_players_yellow_cards", "top_players_red_cards",
        "top_teams_goals", "top_teams_shots", "top_teams_yellow_cards",
        "top_teams_red_cards", "top_teams_created_chances"
    ]

    all_data = {}
    engine = create_engine(db_url)

    try:
        logging.info(f"Connecting to Supabase to fetch {len(tables_to_fetch)} tables...")
        with engine.connect() as connection:
            for table in tables_to_fetch:
                try:
                    df = pd.read_sql(f"SELECT * FROM {table}", con=connection)
                    # Convert DataFrame to list of dicts for the JSON API response
                    all_data[table] = df.to_dict(orient='records')
                    logging.info(f"Successfully fetched table: {table}")
                except Exception as e:
                    logging.warning(f"Could not fetch table {table}: {e}")
                    all_data[table] = [] # Return empty list if table fails
                    
        return all_data

    except Exception as e:
        logging.error(f"Critical error connecting to Supabase: {e}")
        return None
    finally:
        engine.dispose()

@router.get("/")
async def get_pl_analytics():
    """
    Route: GET /api/analytics/
    Returns all deep-dive stats from Supabase.
    """
    stats = fetch_supabase_analytics_tables()
    
    if stats is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    return {
        "status": "success",
        "data": stats
    }