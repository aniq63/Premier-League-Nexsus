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

    async def _fetch_all_as_df_async(self, operation_name: str) -> pd.DataFrame:
        """
        Generic helper to fetch all rows from the table as a DataFrame.
        """
        try:
            logging.debug(f"[{operation_name}] Opening database session...")
            async with AsyncSessionLocal() as session:
                metadata = MetaData()
                await session.run_sync(lambda s: metadata.reflect(bind=s.connection()))
                
                if self.table_name not in metadata.tables:
                    error_msg = f"Table '{self.table_name}' not found in database"
                    logging.error(f"[{operation_name}] {error_msg}")
                    raise ValueError(error_msg)
                
                table = metadata.tables[self.table_name]
                query = select(table)
                result = await session.execute(query)
                rows = result.fetchall()
                
                if not rows:
                    logging.warning(f"[{operation_name}] No data found in table '{self.table_name}'")
                    return pd.DataFrame()
                
                df = pd.DataFrame([dict(row._mapping) for row in rows])
                logging.info(f"[{operation_name}] Fetched {len(df)} rows from '{self.table_name}'")
                return df
                
        except Exception as e:
            error_msg = f"Error in '{operation_name}' during database fetch: {type(e).__name__}: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg) from e

    def _run_sync_wrapper(self, coro, operation_name: str):
        """
        Generic wrapper to run async methods synchronously.
        """
        try:
            logging.info(f"Starting synchronous operation: {operation_name}")
            return asyncio.run(coro)
        except Exception as e:
            error_msg = f"Error in synchronous '{operation_name}': {type(e).__name__}: {str(e)}"
            logging.error(error_msg)
            raise

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
        return await self._fetch_all_as_df_async("fetch_all_data")

    def fetch_all_data(self) -> pd.DataFrame:
        """
        Synchronously fetch all EPL match data from database.
        
        This is a wrapper around the async method for use in synchronous contexts.
        
        Returns:
            pd.DataFrame: DataFrame containing all EPL match data
        
        Raises:
            Exception: If database connection fails or query returns no data
        """
        return self._run_sync_wrapper(self.fetch_all_data_async(), "fetch_all_data")

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
        df = await self._fetch_all_as_df_async(f"fetch_data_by_season({season})")
        
        if df.empty:
            return df
            
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
        return self._run_sync_wrapper(self.fetch_data_by_season_async(season), f"fetch_data_by_season({season})")

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
        df = await self._fetch_all_as_df_async(f"fetch_data_by_team({team_name})")
        
        if df.empty:
            return df
            
        # Filter by team (home or away)
        team_df = df[
            (df.get('home_team', '') == team_name) | 
            (df.get('away_team', '') == team_name)
        ]
        
        logging.info(f"Found {len(team_df)} matches for team '{team_name}'")
        return team_df

    def fetch_data_by_team(self, team_name: str) -> pd.DataFrame:
        """
        Synchronously fetch EPL data for a specific team.
        
        Args:
            team_name (str): Name of the team
            
        Returns:
            pd.DataFrame: DataFrame containing all matches for the specified team
        """
        return self._run_sync_wrapper(self.fetch_data_by_team_async(team_name), f"fetch_data_by_team({team_name})")

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
        df = await self._fetch_all_as_df_async("get_table_info")
        
        if df.empty:
            return {
                "total_rows": 0,
                "columns": [],
                "dtypes": {},
                "date_range": None
            }
        
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

    def get_table_info(self) -> dict:
        """
        Synchronously retrieve metadata information about the data table.
        
        Returns:
            dict: Dictionary containing table metadata
        
        Raises:
            Exception: If database connection fails
        """
        return self._run_sync_wrapper(self.get_table_info_async(), "get_table_info")
