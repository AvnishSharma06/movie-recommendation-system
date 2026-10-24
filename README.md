# Movie Recommendation System Using Big Data and Apache Spark MLlib

**Course:** INT315 – Cluster Computing  
**Author:** Big Data & Machine Learning Engineering  
**Technology Stack:** Apache Spark 3.5, PySpark, Spark MLlib (ALS), Spark SQL, Spark RDDs, Flask, Python 3.10, Bootstrap 5, Plotly.js

---

## 1. Project Overview

This project is a moderate-level, production-grade **Big Data Movie Recommendation System** developed for **INT315 – Cluster Computing**. It processes the MovieLens dataset (100,000+ ratings, 9,700+ movies) using **Apache Spark's distributed computing engine**.

The system features:
1. **Collaborative Filtering Recommendation Engine:** Matrix Factorization using Spark MLlib Alternating Least Squares (ALS) with reproducible train/test splits, evaluation via RMSE and MAE, unseen candidate filtering, and cold-start fallback handling.
2. **Big Data Analytics via Spark SQL:** Distributed aggregations, window functions, and genre popularity analytics.
3. **Core Cluster Computing Demonstration:** Low-level RDD transformations (`map`, `filter`, `flatMap`, `reduceByKey`, `sortBy`), RDD Lineage DAG (`toDebugString()`), Catalyst Query Optimizer execution plan inspection (Parsed, Analyzed, Optimized, Physical Plans), and an empirical benchmark comparing primitive RDD vs Catalyst DataFrame speed.
4. **Cinematic Dark Web Application:** Netflix/HBO Max-inspired streaming interface built with Flask, Bootstrap 5, FontAwesome, and interactive Plotly visualizations.

---

## 2. Project Directory Structure

```text
movie-recommendation-system/
│
├── app.py                     # Flask web server, API routes, and Plotly visualization integrations
├── config.py                  # Central configuration (Spark master, memory, paths, ALS hyperparameters)
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive setup, execution, and troubleshooting manual
│
├── data/                      # Local MovieLens CSV datasets
│   ├── ratings.csv            # 100,836 ratings by 610 users
│   ├── movies.csv             # 9,742 movie records with genres and release years
│   ├── tags.csv               # 3,683 user-submitted community tags
│   └── links.csv              # External IMDb and TMDb identifiers
│
├── src/                       # Core Big Data engineering and ML modules
│   ├── __init__.py            # Package initializer
│   ├── spark_session.py       # Thread-safe singleton SparkSession with Windows winutils auto-detection
│   ├── data_loader.py         # Schema-enforced distributed ingestion of CSV files
│   ├── preprocessing.py       # Deduplication, rating validation [0.5, 5.0], and relational join
│   ├── spark_analytics.py     # Spark SQL analytical queries and Catalyst query execution plan capture
│   ├── rdd_demo.py            # RDD MapReduce word count, key-value average ratings, and DAG lineage
│   ├── train_model.py         # Standalone ALS training, test set evaluation (RMSE/MAE), and serialization
│   └── recommender.py         # Web recommendation service, cold-start fallback, and catalog search
│
├── templates/                 # Jinja2 HTML5 frontend templates
│   ├── base.html              # Cinematic base navigation layout and styling dependencies
│   ├── index.html             # Home dashboard with Big Data metric cards and featured movies
│   ├── recommendations.html   # Personalized recommendations page with user selector & demo presets
│   ├── movies.html            # Movie catalog with title search, genre filters, and pagination
│   ├── movie_details.html     # Deep-dive movie details with Plotly rating frequency breakdown
│   ├── analytics.html         # Distributed analytics dashboard with 3 interactive Plotly charts
│   ├── cluster_demo.html      # Dedicated INT315 Cluster Computing demonstration page
│   └── 404.html               # Resource not found error page
│
├── static/                    # Frontend styling and scripts
│   ├── css/
│   │   └── style.css          # Dark cinematic theme, glassmorphic panels, glowing hover effects
│   └── js/
│       └── main.js            # Micro-interactions, loading spinners, Plotly auto-resizing
│
├── models/                    # Serialized machine learning models
│   └── als_movie_recommender_model/  # Saved Spark MLlib ALSModel artifact
│
├── outputs/                   # Computed analytical results and evaluation metrics
│   └── model_metrics.json     # Measured training duration, RMSE, MAE, and dataset stats
│
├── screenshots/               # Directory for demo and viva documentation captures
├── tests/                     # Unit and integration test suite
│   └── test_recommendation_system.py # 12 comprehensive automated test cases
└── report/                    # Academic project documentation
    └── INT315_PROJECT_REPORT.md      # Complete 12-section college report with viva Q&A
```

---

## 3. Technology Stack & Cluster Architecture

| Component | Technology | Role & Educational Context |
| :--- | :--- | :--- |
| **Distributed Engine** | Apache Spark 3.5 / PySpark | In-memory distributed computing, RDD transformations, and DAG execution |
| **ML Engine** | Spark MLlib ALS | Matrix factorization collaborative filtering: $R \approx U \times V^T$ |
| **Relational Analytics** | Spark SQL & Catalyst Optimizer | Distributed joins, group-by aggregations, window ranking, and physical plans |
| **Low-Level Abstraction** | Spark RDDs | Demonstrates classic MapReduce paradigm (`flatMap`, `reduceByKey`, `lineage`) |
| **Web Server** | Flask 3.1 | Asynchronous prediction serving, caching, and template rendering |
| **Interactive Charts** | Plotly.js | Visualizing genre popularity, star distributions, and review trends |
| **Frontend UI** | HTML5, CSS3, Bootstrap 5 | Cinematic streaming user interface with dark glassmorphic styling |

---

## 4. Setup and Installation Instructions (Windows & VS Code)

### Prerequisites

1. **Python 3.8 to 3.10** installed and added to `PATH`.
2. **Java 8 (JDK 1.8) or Java 11 (64-bit)** installed with `JAVA_HOME` configured.
3. **Apache Spark 3.5.x** installed (e.g. at `C:\Spark`) with `SPARK_HOME` set and `%SPARK_HOME%\bin` in `PATH`.
4. **Hadoop Windows Native Binaries (`winutils.exe` and `hadoop.dll`)** placed in `%SPARK_HOME%\bin` (or `%HADOOP_HOME%\bin`).

### Step-by-Step Execution

#### Step 1: Open Terminal in Project Directory
Open PowerShell or VS Code Terminal and navigate to the project directory:
```powershell
cd "C:\Users\Lenovo\.gemini\antigravity\scratch\movie-recommendation-system"
```

#### Step 2: Install Python Dependencies
```powershell
python -m pip install -r requirements.txt
```

#### Step 3: Train and Evaluate the ALS Model
Run the standalone model training script:
```powershell
python src\train_model.py
```
*Expected Output:*
```text
[*] Loading raw datasets...
[*] Cleaning and preprocessing datasets...
[*] Total verified ratings available for training: 100836
[*] Splitting ratings into 80% Train / 20% Test...
[*] Training records: 80853 | Test records: 19983
[*] Training ALS model on distributed executors...
[+] Model training completed in ~7 seconds.
[+] Evaluation Results:
    - Root Mean Squared Error (RMSE): 0.8666
    - Mean Absolute Error (MAE):     0.6694
[+] Evaluation metrics saved to: outputs\model_metrics.json
[+] Model successfully saved to disk.
```

#### Step 4: Run the Automated Test Suite
Verify that all 12 system components function properly:
```powershell
python -m unittest -v tests/test_recommendation_system.py
```
*Expected Output:*
```text
Ran 12 tests in ~110s
OK
```

#### Step 5: Start the Web Application
```powershell
python app.py
```
Open your web browser and visit:
```text
http://127.0.0.1:5000
```
To inspect active Spark stages, DAGs, and executor metrics, visit the Spark Web UI:
```text
http://127.0.0.1:4040
```

---

## 5. Web Application Features

1. **Dashboard (`/`):** Real-time dataset statistics (100,836 ratings, 9,724 movies, 610 users, 3.50 global rating), popular movies, highest-rated titles, and Spark cluster status.
2. **Personalized Recommendations (`/recommendations`):** 
   - Enter any user ID (1 to 610) or click quick preset buttons.
   - Click **Cold-Start User (9999)** to see graceful fallback handling for unseen users.
   - Compares past rated movies against newly predicted recommendations.
3. **Movie Catalog & Search (`/movies`):** Instant search by title with genre filter badges and responsive pagination.
4. **Movie Deep-Dive (`/movie/<id>`):** Community tags, IMDb/TMDb links, and an interactive Plotly star distribution histogram.
5. **Big Data Analytics (`/analytics`):** Interactive Plotly charts for genre popularity, rating distribution, and rating volume over time (1996–present).
6. **Cluster Demonstration (`/cluster-demo`):** 
   - Tag word frequency tokenization using RDD `flatMap` and `reduceByKey`.
   - Movie average rating computation using key-value RDDs.
   - RDD Lineage Graph inspection (`toDebugString()`).
   - Catalyst Query Optimizer execution plan inspection.
   - Live speed comparison: Primitive RDD vs Catalyst DataFrame.

---

## 6. Troubleshooting Common Errors

1. **`UnsatisfiedLinkError: NativeIO$Windows.access0`:**
   - Cause: Missing `hadoop.dll` on Windows.
   - Fix: Ensure `hadoop.dll` is present in `C:\Spark\bin` and `C:\Spark\bin` is in system `PATH`.
2. **`Python was not found... Microsoft Store alias`:**
   - Cause: Windows App Execution Aliases taking precedence over standard Python.
   - Fix: `spark_session.py` and `config.py` explicitly set `spark.pyspark.python` to `sys.executable`. You can also disable the alias in Windows: *Settings > Apps > Advanced app settings > App execution aliases*.
3. **`OutOfMemoryError (Java Heap Space)`:**
   - Cause: Insufficient driver memory allocated for large matrix factorizations.
   - Fix: Driver memory is configured to `2g` in `config.py`. Adjust `SPARK_DRIVER_MEMORY` if running on machines with limited RAM.
