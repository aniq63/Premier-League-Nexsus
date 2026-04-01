"""
Configuration and Constants for EPL Match Prediction Project

This file centralizes all constant values and configuration settings
used throughout the project to ensure consistency and ease of updates.
"""

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# Table name for storing Premier League match data
TABLE_NAME = "epl_matches"

# ============================================================================
# ETL PIPELINE CONFIGURATION
# ============================================================================

# Required columns for transformed data before loading to database
ETL_REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "home_np_xg",
    "away_np_xg",
    "home_ppda",
    "away_ppda",
    "home_deep_completions",
    "away_deep_completions",
    "home_points",
    "away_points",
]

# Columns to drop during data transformation
ETL_COLUMNS_TO_DROP = [
    "league_id",
    "season_id",
    "game_id",
    "home_team_id",
    "away_team_id",
    "home_team_code",
    "away_team_code",
    "home_expected_points",
    "away_expected_points",
    "home_np_xg_difference",
    "away_np_xg_difference",
]

# Base season for continuous data extraction
ETL_BASE_SEASON = 2023

# ============================================================================
# DATA SOURCE CONFIGURATION
# ============================================================================

# Football data source (using soccerdata library)
DATA_SOURCE_LEAGUE = "ENG-Premier League"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Maximum log file size in bytes (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024

# Number of backup log files to keep
LOG_BACKUP_COUNT = 5

# Log file name
LOG_FILE_NAME = "app.log"


# ============================================================
# Constants used in training
# ============================================================

MODEL_NAME      = "AdaBoostClassifier"
RESULT_CLASSES  = ["Win", "Draw", "Lose"]
N_ITER          = 50
CV_FOLDS        = 5
RANDOM_STATE    = 42
SCORING         = "f1_macro"


INPUT_FEATURES = [
    "home_goals_avg_last5","away_goals_avg_last5","home_goals_conceded_avg_last5",
    "away_goals_conceded_avg_last5","home_xg_avg_last5","away_xg_avg_last5",
    "home_ppda_avg_last5","away_ppda_avg_last5","home_deep_completions_avg_last5",
    "away_deep_completions_avg_last5","home_points_last5","away_points_last5",
    "home_team_home_wins_last5","home_team_home_draws_last5","home_team_home_losses_last5",
    "away_team_away_wins_last5","away_team_away_draws_last5","away_team_away_losses_last5",
    "points_diff_last5","goal_diff_avg5","xg_diff_avg5","x_defense_diff",
    "ppda_diff_avg5","deep_comp_diff_avg5","venue_wins_diff","home_venue_advantage",
    "home_advantage"
]


EXPERIMENT_NAME ="EPL_Match_Prediction"
# ============================================================
# Splitting Configuration
# ============================================================

TEST_SIZE_WEEKS = 1