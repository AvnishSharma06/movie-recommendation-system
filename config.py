"""
INT315 - Cluster Computing Project
Configuration Module for Movie Recommendation System Using Apache Spark MLlib
"""

import os
import sys
from pathlib import Path

# Explicitly ensure Spark worker processes use the current Python interpreter
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATASET_DIR", BASE_DIR / "data"))
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
REPORT_DIR = BASE_DIR / "report"

# Ensure runtime directories exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, SCREENSHOTS_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset File Paths
RATINGS_FILE = DATA_DIR / "ratings.csv"
MOVIES_FILE = DATA_DIR / "movies.csv"
TAGS_FILE = DATA_DIR / "tags.csv"
LINKS_FILE = DATA_DIR / "links.csv"

# Model Paths
ALS_MODEL_PATH = MODELS_DIR / "als_movie_recommender_model"
MODEL_METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"
CACHE_STATS_PATH = OUTPUTS_DIR / "dataset_analytics.json"

# Apache Spark Configuration
SPARK_APP_NAME = "INT315_MovieRecommendationSystem"
# Default to local multi-threaded mode (all available CPU cores)
SPARK_MASTER = os.environ.get("SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "2g")
SPARK_EXECUTOR_MEMORY = os.environ.get("SPARK_EXECUTOR_MEMORY", "2g")
SPARK_SHUFFLE_PARTITIONS = os.environ.get("SPARK_SHUFFLE_PARTITIONS", "4")

# Spark MLlib ALS Hyperparameters
ALS_CONFIG = {
    "userCol": "userId",
    "itemCol": "movieId",
    "ratingCol": "rating",
    "rank": int(os.environ.get("ALS_RANK", 10)),
    "maxIter": int(os.environ.get("ALS_MAX_ITER", 10)),
    "regParam": float(os.environ.get("ALS_REG_PARAM", 0.1)),
    "coldStartStrategy": "drop",  # Drop NaN evaluations during validation
    "nonnegative": True,          # Constrain latent factors to non-negative values
    "seed": 42
}

# Train/Test Split
TRAIN_TEST_SPLIT = [0.8, 0.2]
SPLIT_SEED = 42

# Recommendation Engine Defaults
TOP_N_RECOMMENDATIONS = 10
POPULAR_MIN_RATINGS = 30  # Threshold for reliable highest-rated movies

# Web Server Configuration
WEB_HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
WEB_PORT = int(os.environ.get("FLASK_PORT", 5000))
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1")

# Windows Environment Setup (winutils / HADOOP_HOME)
if os.name == "nt":
    # If HADOOP_HOME is not set, fallback to C:\Spark or SPARK_HOME where winutils.exe exists
    if "HADOOP_HOME" not in os.environ:
        if os.path.exists(r"C:\Spark\bin\winutils.exe"):
            os.environ["HADOOP_HOME"] = r"C:\Spark"
        elif "SPARK_HOME" in os.environ and os.path.exists(os.path.join(os.environ["SPARK_HOME"], "bin", "winutils.exe")):
            os.environ["HADOOP_HOME"] = os.environ["SPARK_HOME"]
