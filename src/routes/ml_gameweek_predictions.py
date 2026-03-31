import pandas as pd
from sqlalchemy import create_engine
from fastapi import APIRouter, HTTPException
from src.utils.setting import get_settings
from src.utils.logger import logging

router = APIRouter()

def get_db_engine():
    """Helper to create a SQLAlchemy engine with protocol correction."""
    settings = get_settings()
    db_url = settings.database_url
    
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    elif db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        
    return create_engine(db_url)

def fetch_gameweek_predictions():
    """Fetches predictions and cleans them for JSON serialization."""
    engine = get_db_engine()
    try:
        logging.info("Fetching ML predictions from Supabase...")
        with engine.connect() as connection:
            df = pd.read_sql("SELECT * FROM gameweek_predictions ORDER BY match_date ASC", con=connection)
            
            if df.empty:
                return []

            # CRITICAL FOR UI: Convert date objects to strings so JSON doesn't crash
            if 'match_date' in df.columns:
                df['match_date'] = df['match_date'].astype(str)
            
            return df.to_dict(orient='records')
    except Exception as e:
        logging.error(f"Error fetching predictions: {e}")
        return None
    finally:
        engine.dispose()

def fetch_feature_importance():
    """Fetches the top 8 important features from Supabase."""
    engine = get_db_engine()
    try:
        logging.info("Fetching model feature importance from Supabase...")
        with engine.connect() as connection:
            df = pd.read_sql("SELECT * FROM feature_importance", con=connection)
            
            if df.empty:
                return []

            # Convert timestamp to string for JSON compatibility
            if 'created_at' in df.columns:
                df['created_at'] = df['created_at'].astype(str)
            
            return df.to_dict(orient='records')
    except Exception as e:
        logging.error(f"Error fetching feature importance: {e}")
        return []
    finally:
        engine.dispose()

@router.get("/")
async def get_predictions():
    """
    Route: GET /api/predictions/
    Returns ML-based match outcomes, win probabilities, and feature importance.
    """
    predictions = fetch_gameweek_predictions()
    importance  = fetch_feature_importance()
    
    if predictions is None:
        raise HTTPException(status_code=500, detail="Failed to fetch predictions from database")
    
    return {
        "status": "success",
        "count": len(predictions),
        "predictions": predictions,
        "feature_importance": importance
    }