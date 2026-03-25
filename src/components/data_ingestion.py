"""
Data Ingestion Module for EPL Match Prediction

This module contains the DataIngestion class which retrieves Premier League
match data from Supabase/PostgreSQL database and returns it as a pandas DataFrame.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
from sqlalchemy import select, MetaData, Table
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
from typing import Optional
from src.utils.logger import logging
from src.utils.setting import get_settings
from src.database.connection import AsyncSessionLocal, engine
from config.constants import TABLE_NAME


# ============================================================
# DATA INGESTION CLASS
# ============================================================

class DataIngestion:
    """
    Handles fetching EPL match data from Supabase PostgreSQL database.
    
    This class provides methods to retrieve Premier League match statistics
    from the database and return them as pandas DataFrames for further
    processing and analysis.
    
    Attributes:
        table_name (str): Name of the table in database containing EPL data
        settings (Settings): Application settings including database configuration
    """

    def __init__(self, table_name: str = TABLE_NAME):
        """
        Initialize DataIngestion with table name.
        
        Args:
            table_name (str): Name of the database table. Defaults to TABLE_NAME from config.
        """
        self.table_name = table_name
        self.settings = get_settings()
        logging.info(f"Initialized DataIngestion for table: {self.table_name}")

    async def fetch_all_data_async(self) -> pd.DataFrame:
        """
        Asynchronously fetch all EPL match data from database.
        
        Returns:
            pd.DataFrame: DataFrame containing all EPL match data with columns:
                - date
                - home_team
                - away_team
                - home_goals
                - away_goals
                - home_xg
                - away_xg
                - home_np_xg
                - away_np_xg
                - home_ppda
                - away_ppda
                - home_deep_completions
                - away_deep_completions
                - home_points
                - away_points
        
        Raises:
            Exception: If database connection fails or query returns no data
        """
        try:
            logging.info(f"Fetching all data from table '{self.table_name}'...")
            
            async with AsyncSessionLocal() as session:
                # Reflect table metadata from database
                metadata = MetaData()
                await session.run_sync(lambda s: metadata.reflect(bind=s.connection()))
                
                # Get the table object
                if self.table_name not in metadata.tables:
                    error_msg = f"Table '{self.table_name}' not found in database"
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                
                table = metadata.tables[self.table_name]
                
                # Create and execute select query
                query = select(table)
                result = await session.execute(query)
                
                # Fetch all rows
                rows = result.fetchall()
                
                if not rows:
                    logging.warning(f"No data found in table '{self.table_name}'")
                    return pd.DataFrame()
                
                # Convert rows to DataFrame
                df = pd.DataFrame([dict(row._mapping) for row in rows])
                
                logging.info(
                    f"Successfully fetched {len(df)} rows from '{self.table_name}' "
                    f"with {len(df.columns)} columns"
                )
                logging.debug(f"Columns: {list(df.columns)}")
                
                return df
                
        except ValueError as ve:
            logging.error(f"Validation error during data ingestion: {str(ve)}")
            raise
        except Exception as e:
            error_msg = f"Error fetching data from database: {type(e).__name__}: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg) from e

    def fetch_all_data(self) -> pd.DataFrame:
        """
        Synchronously fetch all EPL match data from database.
        
        This is a wrapper around the async method for use in synchronous contexts.
        
        Returns:
            pd.DataFrame: DataFrame containing all EPL match data
        
        Raises:
            Exception: If database connection fails or query returns no data
        """
        try:
            logging.info("Starting synchronous data ingestion...")
            df = asyncio.run(self.fetch_all_data_async())
            return df
        except Exception as e:
            error_msg = f"Error in synchronous data ingestion: {type(e).__name__}: {str(e)}"
            logging.error(error_msg)
            raise

    async def fetch_data_by_season_async(self, season: int) -> pd.DataFrame:
        """
        Asynchronously fetch EPL data for a specific season.
        
        Args:
            season (int): Season year (e.g., 2023 for 2023-24 season)
        
        Returns:
            pd.DataFrame: DataFrame containing matches from the specified season
        
        Raises:
            Exception: If database connection fails or no data found for season
        """
        try:
            logging.info(f"Fetching data for season {season} from table '{self.table_name}'...")
            
            async with AsyncSessionLocal() as session:
                metadata = MetaData()
                await session.run_sync(lambda s: metadata.reflect(bind=s.connection()))
                
                if self.table_name not in metadata.tables:
                    error_msg = f"Table '{self.table_name}' not found in database"
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                
                table = metadata.tables[self.table_name]
                
                # Extract year from date and filter by season
                # Assuming date column exists and season runs from Aug to May
                query = select(table)
                result = await session.execute(query)
                rows = result.fetchall()
                
                if not rows:
                    logging.warning(f"No data found for season {season}")
                    return pd.DataFrame()
                
                df = pd.DataFrame([dict(row._mapping) for row in rows])
                
                # Filter by season (assuming 'date' column exists)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    season_df = df[
                        ((df['date'].dt.year == season) & (df['date'].dt.month >= 8)) |
                        ((df['date'].dt.year == season + 1) & (df['date'].dt.month < 8))
                    ]
                    
                    logging.info(f"Found {len(season_df)} matches for season {season}")
                    return season_df
                else:
                    logging.warning("'date' column not found, returning all data")
                    return df
                
        except ValueError as ve:
            logging.error(f"Validation error: {str(ve)}")
            raise
        except Exception as e:
            error_msg = f"Error fetching season data: {type(e).__name__}: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg) from e

    def fetch_data_by_season(self, season: int) -> pd.DataFrame:
        """
        Synchronously fetch EPL data for a specific season.
        
        Args:
            season (int): Season year (e.g., 2023 for 2023-24 season)
        
        Returns:
            pd.DataFrame: DataFrame containing matches from the specified season
        
        Raises:
            Exception: If database connection fails or no data found for season
        """
        try:
            logging.info(f"Starting synchronous season data ingestion for {season}...")
            df = asyncio.run(self.fetch_data_by_season_async(season))
            return df
        except Exception as e:
            error_msg = f"Error in synchronous season data ingestion: {type(e).__name__}: {str(e)}"
            logging.error(error_msg)
            raise

    async def fetch_data_by_team_async(self, team_name: str) -> pd.DataFrame:
        """
        Asynchronously fetch EPL data for a specific team.
        
        Args:
            team_name (str): Name of the team (home or away)
        
        Returns:
            pd.DataFrame: DataFrame containing all matches for the specified team
        
        Raises:
            Exception: If database connection fails or no data found for team
        """
        try:
            logging.info(f"Fetching data for team '{team_name}' from table '{self.table_name}'...")
            
            async with AsyncSessionLocal() as session:
                metadata = MetaData()
                await session.run_sync(lambda s: metadata.reflect(bind=s.connection()))
                
                if self.table_name not in metadata.tables:
                    error_msg = f"Table '{self.table_name}' not found in database"
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                
                table = metadata.tables[self.table_name]
                query = select(table)
                result = await session.execute(query)
                rows = result.fetchall()
                
                if not rows:
                    logging.warning(f"No data found for team '{team_name}'")
                    return pd.DataFrame()
                
                df = pd.DataFrame([dict(row._mapping) for row in rows])
                
                # Filter by team (home or away)
                team_df = df[
                    (df.get('home_team', '') == team_name) | 
                    (df.get('away_team', '') == team_name)
                ]
                
                logging.info(f"Found {len(team_df)} matches for team '{team_name}'")
                return team_df
                
        except ValueError as ve:
            logging.error(f"Validation error: {str(ve)}")
            raise
        except Exception as e:
            error_msg = f"Error fetching team data: {type(e).__name__}: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg) from e

    def fetch_data_by_team(self, team_name: str) -> pd.DataFrame:
        """
        Synchronously fetch EPL data for a specific team.
        
        Args:
            team_name (str): Name of the team (home or away)
        
        Returns:
            pd.DataFrame: DataFrame containing all matches for the specified team
        
        Raises:
            Exception: If database connection fails or no data found for team
        """
        try:
            logging.info(f"Starting synchronous team data ingestion for {team_name}...")
            df = asyncio.run(self.fetch_data_by_team_async(team_name))
            return df
        except Exception as e:
            error_msg = f"Error in synchronous team data ingestion: {type(e).__name__}: {str(e)}"
            logging.error(error_msg)
            raise

    async def get_table_info_async(self) -> dict:
        """
        Asynchronously retrieve metadata information about the data table.
        
        Returns:
            dict: Dictionary containing:
                - total_rows: Total number of rows in table
                - columns: List of column names
                - dtypes: Dictionary of column data types
                - date_range: Tuple of (min_date, max_date) if date column exists
        
        Raises:
            Exception: If database connection fails
        """
        try:
            logging.info(f"Retrieving table info for '{self.table_name}'...")
            
            async with AsyncSessionLocal() as session:
                metadata = MetaData()
                await session.run_sync(lambda s: metadata.reflect(bind=s.connection()))
                
                if self.table_name not in metadata.tables:
                    error_msg = f"Table '{self.table_name}' not found in database"
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                
                table = metadata.tables[self.table_name]
                query = select(table)
                result = await session.execute(query)
                rows = result.fetchall()
                
                if not rows:
                    return {
                        "total_rows": 0,
                        "columns": [],
                        "dtypes": {},
                        "date_range": None
                    }
                
                df = pd.DataFrame([dict(row._mapping) for row in rows])
                
                info = {
                    "total_rows": len(df),
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.to_dict(),
                }
                
                # Add date range if date column exists
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    info["date_range"] = (
                        df['date'].min().strftime('%Y-%m-%d'),
                        df['date'].max().strftime('%Y-%m-%d')
                    )
                
                logging.info(f"Table '{self.table_name}' info retrieved successfully")
                return info
                
        except ValueError as ve:
            logging.error(f"Validation error: {str(ve)}")
            raise
        except Exception as e:
            error_msg = f"Error retrieving table info: {type(e).__name__}: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg) from e

    def get_table_info(self) -> dict:
        """
        Synchronously retrieve metadata information about the data table.
        
        Returns:
            dict: Dictionary containing table metadata
        
        Raises:
            Exception: If database connection fails
        """
        try:
            logging.info("Starting synchronous table info retrieval...")
            info = asyncio.run(self.get_table_info_async())
            return info
        except Exception as e:
            error_msg = f"Error in synchronous table info retrieval: {type(e).__name__}: {str(e)}"
            logging.error(error_msg)
            raise
