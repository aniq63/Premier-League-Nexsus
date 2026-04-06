<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS_S3-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white" />
  <img src="https://img.shields.io/badge/Supabase-3FCF8E?style=for-the-badge&logo=supabase&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" />
  <a href="https://hub.docker.com/r/aniqramzan/epl-match-prediction"><img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" /></a>
</p>

<h1 align="center">⚽ EPL Nexus — Premier League Match Prediction Platform</h1>

<p align="center">
  <b>An end-to-end MLOps platform that predicts English Premier League match outcomes using engineered features, automated pipelines, and a production-grade API — fully autonomous and self-updating.</b>
</p>

<p align="center">
  <a href="https://epl-match-prediction-teal.vercel.app/"><img src="https://img.shields.io/badge/🔴_LIVE_DEMO-Visit_App-F2055C?style=for-the-badge" /></a>
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Feature Engineering](#-feature-engineering)
- [ML Pipeline](#-ml-pipeline)
- [CI/CD Automation](#-cicd-automation)
- [API Endpoints](#-api-endpoints)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [License](#-license)

---

## 🧠 Overview

**EPL Nexus** is a full-stack machine learning platform that predicts the outcome of English Premier League matches (**Win / Draw / Lose** from the home team's perspective) using historical performance data sourced from [Understat](https://understat.com/) via the [`soccerdata`](https://github.com/probberechts/soccerdata) library.

The platform goes far beyond a simple model — it implements a complete **MLOps lifecycle**:

1. **Automated ETL** extracts and loads multi-season match data into a Supabase data warehouse
2. **27 engineered features** capture rolling form, venue advantage, xG differentials, and pressing intensity
3. **AdaBoostClassifier** is tuned via RandomizedSearchCV with TimeSeriesSplit to prevent data leakage
4. **MLflow on DagsHub** tracks experiments, manages model versions, and promotes through Staging → Production
5. **AWS S3** hosts the production model for low-latency inference
6. **FastAPI** serves predictions, live standings, analytics, and news through a REST API
7. **GitHub Actions** orchestrates everything autonomously on a cron schedule

---

## 🔴 Live Demo

👉 **[https://epl-match-prediction-teal.vercel.app/](https://epl-match-prediction-teal.vercel.app/)**

The frontend is a futuristic, neon-themed dashboard featuring:

| Tab | What You'll See |
|---|---|
| **Dashboard** | Live EPL standings, recent results, upcoming fixtures, and Premier League news |
| **Analytics** | Deep-dive player & team stats — top scorers, assists, key passes, shots, cards |
| **ML Predictions** | Upcoming gameweek predictions with Win/Draw/Lose probabilities and confidence scores, plus a full technical breakdown of the ML pipeline architecture |
| **PL History** | Historical champions since 1992 with managers, points, and title counts |

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        GITHUB ACTIONS (CI/CD)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  ┌────────────┐  │
│  │  ETL Pipeline │  │  ML Pipeline │  │ Prediction│  │   Stats    │  │
│  │  (Tue & Fri)  │  │ (Conditional)│  │ (Every 3d)│  │ (Every 3d) │  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  └─────┬──────┘  │
└─────────┼──────────────────┼────────────────┼──────────────┼─────────┘
          │                  │                │              │
          ▼                  ▼                ▼              ▼
┌─────────────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────┐
│   Understat      │  │  MLflow /    │  │  AWS S3   │  │ Understat│
│   (soccerdata)   │  │  DagsHub     │  │  (Model)  │  │ (Stats)  │
└────────┬────────┘  └──────┬───────┘  └─────┬─────┘  └────┬─────┘
         │                  │                │              │
         ▼                  ▼                ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SUPABASE (PostgreSQL)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐    │
│  │ epl_matches   │  │ predictions  │  │ top_players_*           │    │
│  │ (raw data)    │  │ (gameweek)   │  │ top_teams_* (analytics) │    │
│  └──────────────┘  └──────────────┘  └─────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────────┐    │
│  │  /dashboard  │  │  /analytics  │  │  /predictions            │    │
│  │  (Standings, │  │  (Player &   │  │  (ML Predictions,        │    │
│  │   Fixtures,  │  │   Team Stats)│  │   Feature Importances)   │    │
│  │   News)      │  │              │  │                          │    │
│  └─────────────┘  └──────────────┘  └──────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Vercel)                                 │
│        Neon-themed dashboard with glassmorphism UI                   │
│        Dashboard · Analytics · ML Predictions · PL History          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🤖 Machine Learning
- **AdaBoostClassifier** with DecisionTree base estimator and balanced class weights
- **RandomizedSearchCV** (50 iterations) with **TimeSeriesSplit** cross-validation — prevents temporal data leakage
- **27 engineered features** including rolling xG, PPDA, deep completions, venue form, and differential metrics
- Optimized for **F1-Macro** to handle class imbalance across Win/Draw/Lose outcomes

### 🔁 MLOps & Automation
- **MLflow on DagsHub** for experiment tracking, metric logging, confusion matrix artifacts, and model registry
- Automated **Staging → Production promotion** — new models are only promoted if they beat the current production model
- Production models serialized to **AWS S3** for fast inference
- **4 GitHub Actions workflows** running on cron schedules for fully autonomous operation

### 📊 Data & Analytics
- **ETL pipeline** extracting multi-season data from Understat via `soccerdata`
- **Supabase (PostgreSQL)** as the central data warehouse
- **11 analytics tables** refreshed automatically: top players (goals, assists, shots, key passes, cards) and top teams (goals, shots, cards, created chances)
- Live EPL data from **Football-Data.org** and **NewsAPI**

### 🌐 API & Frontend
- **FastAPI REST API** with structured endpoints for dashboard, analytics, and ML predictions
- **Neon-themed frontend** with glassmorphism, micro-animations, and responsive design
- Deployed on **Vercel** with live data updates

---

## 🛠 Tech Stack

| Layer | Technologies |
|---|---|
| **Language** | Python 3.10 |
| **ML / Data Science** | Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Plotly |
| **Model** | AdaBoostClassifier (DecisionTreeClassifier base), RandomizedSearchCV, TimeSeriesSplit |
| **MLOps** | MLflow, DagsHub, Model Registry with Staging/Production stages |
| **Data Source** | Understat (via `soccerdata`), ESPN, Football-Data.org, NewsAPI |
| **Data Warehouse** | Supabase (PostgreSQL), SQLAlchemy, asyncpg |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Cloud Storage** | AWS S3 (Boto3) — production model hosting |
| **CI/CD** | GitHub Actions (4 workflows, cron-scheduled) |
| **Frontend** | Vanilla HTML/CSS/JS, Chart.js |
| **Deployment** | Vercel (Frontend), Render / Railway (Backend) |

---

## 📁 Project Structure

```
EPL_Nexus/
│
├── .github/workflows/              # CI/CD automation
│   ├── etl_ml_pipeline.yml         # Scheduled ETL + conditional ML training
│   ├── prediction_pipeline.yml     # Gameweek prediction generation (every 3 days)
│   ├── stats_analyzer.yml          # Player & team analytics refresh (every 3 days)
│   └── pipeline_health.yml         # Weekly data quality monitoring
│
├── config/
│   └── constants.py                # Centralized configuration & hyperparameters
│
├── frontend/
│   └── index.html                  # Neon-themed dashboard (deployed on Vercel)
│
├── src/
│   ├── etl/                        # Extract-Transform-Load pipeline
│   │   ├── data_extraction.py      # Understat data extraction via soccerdata
│   │   ├── data_transformation.py  # Schema normalization & cleaning
│   │   └── data_load.py            # Load to Supabase (PostgreSQL)
│   │
│   ├── feature_engineering/
│   │   └── feature_enginnering.py  # 27-feature pipeline (rolling, venue form, diffs)
│   │
│   ├── components/                 # Core ML components
│   │   ├── data_ingestion.py       # Fetch data from Supabase warehouse
│   │   ├── data_transformation.py  # ML-specific data prep
│   │   ├── model_training.py       # AdaBoost + RandomizedSearchCV + TimeSeriesSplit
│   │   ├── model_evaluation.py     # Metrics, confusion matrix, MLflow logging
│   │   └── model_registry_and_deploy.py  # Staging → Production + S3 upload
│   │
│   ├── pipelines/                  # Pipeline orchestrators
│   │   ├── etl_pipeline.py         # End-to-end ETL orchestration
│   │   └── ml_pipeline.py          # End-to-end ML training orchestration
│   │
│   ├── services/                   # Business logic services
│   │   ├── prediction_pipeline.py  # Gameweek inference (ESPN fixtures → predictions)
│   │   └── premier_league_stats_analyzer.py  # Player & team analytics
│   │
│   ├── routes/                     # FastAPI API routes
│   │   ├── dashboard.py            # Standings, fixtures, results, news
│   │   ├── analytics.py            # Player & team deep-dive stats
│   │   └── ml_gameweek_predictions.py  # ML prediction endpoints
│   │
│   ├── database/
│   │   └── connection.py           # Async SQLAlchemy + Supabase connection
│   │
│   └── utils/                      # Shared utilities
│       ├── logger/                 # Structured logging
│       ├── exception/              # Custom exception handling
│       ├── setting.py              # Pydantic settings management
│       └── data_split.py           # TimeSeriesSplit data splitter
│
├── main.py                         # FastAPI application entry point
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
└── pyproject.toml                  # Project metadata
```

---

## 🧪 Feature Engineering

The model uses **27 carefully engineered features** derived from raw match data:

### Rolling Averages (Last 5 Matches)
| Feature | Description |
|---|---|
| `home/away_goals_avg_last5` | Average goals scored |
| `home/away_goals_conceded_avg_last5` | Average goals conceded |
| `home/away_xg_avg_last5` | Average Expected Goals (xG) |
| `home/away_ppda_avg_last5` | Average Passes Per Defensive Action (pressing intensity) |
| `home/away_deep_completions_avg_last5` | Average passes completed within 20m of goal |
| `home/away_points_last5` | Total points accumulated |

### Venue-Specific Form (Last 5 Home / Away Matches)
| Feature | Description |
|---|---|
| `home_team_home_wins/draws/losses_last5` | Win/Draw/Loss count at home |
| `away_team_away_wins/draws/losses_last5` | Win/Draw/Loss count away |

### Differential Features
| Feature | Description |
|---|---|
| `points_diff_last5` | Home points − Away points |
| `goal_diff_avg5` | Goal scoring difference |
| `xg_diff_avg5` | Expected Goals difference |
| `x_defense_diff` | Defensive quality difference |
| `ppda_diff_avg5` | Pressing intensity difference |
| `deep_comp_diff_avg5` | Deep completion difference |
| `venue_wins_diff` | Home venue wins − Away venue wins |
| `home_venue_advantage` | Home win rate at venue |
| `home_advantage` | Binary home advantage indicator |

---

## 🤖 ML Pipeline

```mermaid
graph LR
    A[Data Ingestion<br/>from Supabase] --> B[Data Transformation<br/>& Cleaning]
    B --> C[Feature Engineering<br/>27 Features]
    C --> D[TimeSeriesSplit<br/>Train/Test Split]
    D --> E[RandomizedSearchCV<br/>50 iterations, 5-fold]
    E --> F[AdaBoostClassifier<br/>DecisionTree Base]
    F --> G[Model Evaluation<br/>Accuracy, F1, Precision]
    G --> H[MLflow Logging<br/>DagsHub Tracking]
    H --> I{Staging vs<br/>Production}
    I -->|Better| J[Promote to<br/>Production]
    I -->|Worse| K[Keep Current<br/>Production]
    J --> L[Upload to<br/>AWS S3]
```

### Model Details

| Parameter | Value |
|---|---|
| **Algorithm** | AdaBoostClassifier |
| **Base Estimator** | DecisionTreeClassifier (balanced class weights) |
| **Hyperparameter Search** | RandomizedSearchCV (50 iterations) |
| **Cross-Validation** | TimeSeriesSplit (5 folds) |
| **Scoring Metric** | F1-Macro |
| **Classes** | Win, Draw, Lose (home team perspective) |
| **Features** | 27 |
| **Test Split** | Last 6 gameweeks |

---

## ⚙ CI/CD Automation

The platform runs **4 fully automated GitHub Actions workflows**:

| Workflow | Schedule | Purpose |
|---|---|---|
| **ETL + ML Pipeline** | Tue & Fri at 2 AM UTC | Extract new match data → conditionally retrain model (≥5 new samples) |
| **Prediction Pipeline** | Every 3 days | Fetch upcoming fixtures from ESPN → generate predictions → save to Supabase |
| **Stats Analyzer** | Every 3 days | Refresh 11 player & team analytics tables in Supabase |
| **Pipeline Health Monitor** | Every Monday at 9 AM UTC | Data quality checks, freshness validation, weekly health report |

### Smart Conditional Training
The ETL pipeline tracks the data count between runs. The ML training job only triggers when **≥5 new match samples** are detected — saving compute resources and avoiding unnecessary model churn.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Welcome message & API info |
| `GET` | `/health` | Health check & uptime |
| `GET` | `/api/dashboard` | Live standings, fixtures, results, news |
| `GET` | `/api/analytics/` | Player & team deep-dive statistics |
| `GET` | `/api/predictions/` | Current gameweek ML predictions |
| `GET` | `/api/predictions/feature-importance` | Top model feature importances |

Full interactive docs available at `/docs` (Swagger UI).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Supabase account (PostgreSQL database)
- AWS account (S3 bucket for model storage)
- DagsHub account (MLflow experiment tracking)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/aniqramzan5758/EPL_Match_Prediction.git
cd EPL_Match_Prediction

# 2. Create virtual environment
python -m venv myenv
source myenv/bin/activate     # Linux/Mac
myenv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables (see section below)
cp .env.example .env
# Edit .env with your credentials

# 5. Run the ETL pipeline (first time — populates the database)
python -m src.pipelines.etl_pipeline

# 6. Run the ML pipeline (trains and deploys the model)
python -m src.pipelines.ml_pipeline

# 7. Start the API server
python main.py
# or
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000` and docs at `http://localhost:8000/docs`.

---

## 🐳 Docker

You can easily run the pre-built application directly from Docker Hub:

```bash
# Pull the latest image
docker pull aniqramzan/epl-match-prediction:latest

# Run the container (requires your .env file)
docker run -p 8000:8000 --env-file .env aniqramzan/epl-match-prediction:latest
```

The API will be accessible at `http://localhost:8000`.

---

## 🔐 Environment Variables

Create a `.env` file in the project root with the following:

```env
# ── Database (Supabase) ──
DATABASE_URL=postgresql+asyncpg://user:password@host:port/dbname

# ── AWS S3 (Model Storage) ──
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name

# ── MLflow / DagsHub (Experiment Tracking) ──
MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token

# ── External APIs ──
FOOTBALL_DATA_KEY=your_football_data_api_key
NEWS_API_KEY=your_newsapi_key

# ── App ──
DEBUG=False
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Built with ❤️ by <a href="https://github.com/aniqramzan5758">Aniq Ramzan</a></b>
</p>

<p align="center">
  <a href="https://epl-match-prediction-teal.vercel.app/">🔴 Live Demo</a> •
  <a href="https://dagshub.com/aniqramzan5758/EPL_Match_Prediction">📊 MLflow Experiments</a> •
  <a href="https://github.com/aniqramzan5758/EPL_Match_Prediction/actions">⚙️ CI/CD Pipelines</a>
</p>
