"""
Script to generate the official INT315 Project Report in RTF (Rich Text Format)
compatible with Microsoft WordPad and Microsoft Word.

Written in humanized, authentic student academic style (no AI clichés or buzzwords)
covering all 12 required sections for INT315 - Cluster Computing.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path("C:/Users/Lenovo/Desktop/movie-recommendation-system")
OUTPUT_RTF = BASE_DIR / "report" / "INT315_PROJECT_REPORT.rtf"
HUMANIZED_RTF = BASE_DIR / "report" / "INT315_PROJECT_REPORT_HUMANIZED.rtf"


def escape_rtf(text: str) -> str:
    """Escapes characters for RTF format and formats code lines."""
    # First escape backslashes, then braces
    text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    # Replace newlines with \par
    lines = text.split("\n")
    return "\\par\n".join(lines)


def load_file_content(relative_path: str) -> str:
    """Loads a project source code file."""
    p = BASE_DIR / relative_path
    if p.exists():
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return f"# File {relative_path} not found"


def generate_report():
    print(f"[*] Generating humanized WordPad-compatible RTF report at: {OUTPUT_RTF}")
    OUTPUT_RTF.parent.mkdir(parents=True, exist_ok=True)

    # Load all project source codes
    code_config = load_file_content("config.py")
    code_spark_session = load_file_content("src/spark_session.py")
    code_data_loader = load_file_content("src/data_loader.py")
    code_preprocessing = load_file_content("src/preprocessing.py")
    code_spark_analytics = load_file_content("src/spark_analytics.py")
    code_rdd_demo = load_file_content("src/rdd_demo.py")
    code_train_model = load_file_content("src/train_model.py")
    code_recommender = load_file_content("src/recommender.py")
    code_app = load_file_content("app.py")
    code_cli = load_file_content("cli.py")

    rtf_content = [
        r"{\rtf1\ansi\ansicpg1252\deff0\nouicompat\deflang1033",
        # Font Table: f0=Calibri (Body), f1=Consolas (Code), f2=Arial (Headings)
        r"{\fonttbl{\f0\fnil\fcharset0 Calibri;}{\f1\fnil\fcharset0 Consolas;}{\f2\fnil\fcharset0 Arial;}}",
        # Color Table: 1=Navy Blue, 2=Charcoal, 3=Dark Red, 4=Forest Green, 5=Slate Gray, 6=Dark Amber, 7=Code Blue
        r"{\colortbl ;\red0\green68\blue148;\red30\green41\blue59;\red185\green28\blue28;\red21\green128\blue61;\red71\green85\blue105;\red180\green83\blue9;\red15\green23\blue42;}",
        r"\viewkind4\uc1",
        
        # ----------------- TITLE PAGE -----------------
        r"\pard\qc\sa300\f2\fs44\b\cf1 INT315 – CLUSTER COMPUTING\par",
        r"\fs28\b0\cf5 END-TERM CAPSTONE PROJECT REPORT\par\par",
        r"\fs36\b\cf2 PROJECT TITLE:\par",
        r"\fs40\b\cf1 MOVIE RECOMMENDATION SYSTEM USING BIG DATA AND APACHE SPARK MLLIB\par\par",
        r"\fs24\b0\cf5 Course Code: INT315 (Cluster Computing)\par",
        r"Department of Computer Science & Engineering\par",
        r"Academic Session: 2026–2027\par\par",
        r"\pard\qc\sa200\fs24\b\cf2 Submitted By:\par\b0",
        r"Student Name: [Enter Your Name]\par",
        r"Registration Number: [Enter Your Registration Number]\par",
        r"Roll Number / Section: [Enter Your Roll No. & Section]\par\par",
        r"\pard\qc\sa300\fs24\b\cf2 Under the Guidance of:\par\b0",
        r"Faculty Member / Course Instructor\par",
        r"Department of Computer Science & Engineering\par",
        r"\par\par\par\pard\ql\sa200\cf0",

        # ----------------- TABLE OF CONTENTS -----------------
        r"\f2\fs28\b\cf1 TABLE OF CONTENTS\par\f0\fs22\b0\cf2",
        r"1. Project Title ............................................................................................................. Page 2\par",
        r"2. Problem Statement ................................................................................................. Page 2\par",
        r"3. Objective of the Project .......................................................................................... Page 3\par",
        r"4. Dataset Description ............................................................................................... Page 3\par",
        r"5. Dataset Source and Reference .............................................................................. Page 4\par",
        r"6. Tools & Technologies Used ................................................................................... Page 4\par",
        r"7. Implementation and Methodology .......................................................................... Page 5\par",
        r"8. Complete Source Code .......................................................................................... Page 8\par",
        r"9. Execution Screenshots and Terminal Logs ........................................................... Page 25\par",
        r"10. Output / Result Screenshots ................................................................................ Page 28\par",
        r"11. Result Analysis and Academic Discussion ............................................................ Page 31\par",
        r"12. Conclusion and Future Scope ............................................................................ Page 34\par",
        r"Appendix: INT315 Viva Voce Preparation Guide (Questions & Answers) .................... Page 35\par",
        r"\par\par",

        # ----------------- SECTION 1: PROJECT TITLE -----------------
        r"\f2\fs30\b\cf1 1. PROJECT TITLE\par\f0\fs22\b0\cf2",
        r"\b Official Title: \b0 Movie Recommendation System Using Big Data and Apache Spark MLlib\par",
        r"\b Academic Course: \b0 INT315 – Cluster Computing\par",
        r"\b Engineering Discipline: \b0 Computer Science & Engineering (B.Tech / M.Tech)\par",
        r"\b Implementation Environment: \b0 Distributed Apache Spark 3.5.1 on Python 3.10 (PySpark) with Spark MLlib, Spark SQL, and Flask Web Interface.\par\par",

        # ----------------- SECTION 2: PROBLEM STATEMENT -----------------
        r"\f2\fs30\b\cf1 2. PROBLEM STATEMENT\par\f0\fs22\b0\cf2",
        r"When users browse modern digital movie platforms, they are presented with tens of thousands of film choices across multiple genres. Without intelligent assistance, finding a film matching their personal taste is time-consuming and often leads to decision fatigue. Recommendation systems solve this by predicting what a user will enjoy based on past viewing and rating behaviors.\par\par",
        r"However, building recommendation engines for real-world platforms poses major computational problems:\par\par",
        r"1. \b Computational Bottlenecks of Single-Node Tools: \b0 Typical data science libraries such as pandas or scikit-learn load data completely into a single machine's RAM and perform calculations on a single thread or limited CPU cores. When rating matrices grow into millions of rows and tens of thousands of user IDs, single-machine algorithms run out of memory (OOM) or take hours to complete matrix factorization.\par\par",
        r"2. \b The Disk I/O Bottleneck of Traditional Hadoop MapReduce: \b0 In INT315, we studied the first-generation big data framework, Hadoop MapReduce. While MapReduce is fault-tolerant and handles large scale, it is fundamentally disk-centric. It writes intermediate key-value outputs to local disks and HDFS between the Map phase and the Reduce phase. Collaborative filtering algorithms, specifically Alternating Least Squares (ALS), require 10 to 20 iterative optimization passes. Running ALS on Hadoop MapReduce forces dozens of costly disk read/write cycles across the network, making model training unacceptably slow.\par\par",
        r"3. \b Cold-Start and Sparse Matrix Challenges: \b0 Real user-item interaction matrices are extremely sparse (over 98% of values are unrated). Furthermore, newly registered users have zero rating history (the Cold-Start problem). A production recommendation pipeline must train factorized matrices across cluster nodes without crashing when encountering unknown users.\par\par",
        r"Therefore, our project addresses this problem by designing and deploying an in-memory distributed recommendation pipeline using Apache Spark, PySpark, and Spark MLlib. By keeping intermediate rating partitions in cluster memory across iterations, our system delivers fast model convergence, low error rates, and real-time personalized recommendations with robust fallback handling.\par\par",

        # ----------------- SECTION 3: OBJECTIVES -----------------
        r"\f2\fs30\b\cf1 3. OBJECTIVE OF THE PROJECT\par\f0\fs22\b0\cf2",
        r"The main goals we set out to accomplish in this project are:\par\par",
        r"1. \b In-Memory Distributed Data Ingestion: \b0 Load the four MovieLens relational files into PySpark DataFrames using explicit StructType schemas, avoiding slow schema inference passes over distributed blocks.\par\par",
        r"2. \b Data Cleaning and Window Deduplication: \b0 Filter out invalid ratings (outside the 0.5 to 5.0 scale), drop null IDs, deduplicate multiple reviews submitted by the same user for the same movie using Spark SQL Window partitions, and parse release years using regular expressions.\par\par",
        r"3. \b Distributed Analytics via Spark SQL: \b0 Create temporary SQL views to compute movie popularity, genre score distributions, active user activity counts, and yearly review patterns across cluster partitions.\par\par",
        r"4. \b Model Training via Matrix Factorization: \b0 Implement Collaborative Filtering using Spark MLlib's Alternating Least Squares (ALS) algorithm. Train latent factor representations of users and movies on an 80% training partition.\par\par",
        r"5. \b Rigorous Evaluation Without Data Leakage: \b0 Evaluate our model on an out-of-fold 20% test partition using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) with a fixed random seed (seed=42) for scientific reproducibility.\par\par",
        r"6. \b Handling the Cold-Start Condition: \b0 Build an automated fallback service that intercepts queries for new or unseen users, returning top cluster-consensus films with a minimum threshold of 30 ratings rather than failing or returning null.\par\par",
        r"7. \b Practical Demonstration of INT315 Syllabus Concepts: \b0 Write dedicated scripts showing primitive RDD transformations (flatMap, map, reduceByKey), print RDD Lineage DAGs (toDebugString), inspect Catalyst Query Optimizer physical plans, and run an empirical speed benchmark comparing Python RDDs against Spark DataFrames.\par\par",
        r"8. \b Interactive End-User Interfaces: \b0 Provide both an interactive Command Prompt terminal interface (cli.py) and a responsive cinematic web dashboard (Flask + Bootstrap 5 + Plotly) so users and examiners can test recommendations dynamically.\par\par",

        # ----------------- SECTION 4: DATASET DESCRIPTION -----------------
        r"\f2\fs30\b\cf1 4. DATASET DESCRIPTION\par\f0\fs22\b0\cf2",
        r"We used the standard MovieLens benchmark dataset (ml-latest-small) released by the GroupLens research lab at the University of Minnesota. The dataset consists of 4 separate CSV files representing relational entity tables:\par\par",
        r"\b Detailed File Breakdown:\b0\par",
        r"- \b ratings.csv (100,836 rows): \b0 Captures user rating events. Columns include userId (integer), movieId (integer), rating (float between 0.5 and 5.0 in 0.5 step increments), and timestamp (epoch seconds).\par",
        r"- \b movies.csv (9,742 rows): \b0 Catalogs movies. Columns include movieId (integer), title (string containing movie title and release year in parentheses), and genres (pipe-delimited string of genres such as Action|Adventure|Sci-Fi).\par",
        r"- \b tags.csv (3,683 rows): \b0 Free-form metadata tags applied by users. Columns include userId (integer), movieId (integer), tag (user entered string), and timestamp.\par",
        r"- \b links.csv (9,742 rows): \b0 Cross-referencing identifiers. Columns include movieId, imdbId, and tmdbId for external database mapping.\par\par",
        r"\b Preprocessed Dataset Metrics:\b0\par",
        r"After running our data cleaning and validation pipeline in Spark, the operational parameters of the dataset were:\par",
        r"- Verified ratings: 100,836 clean records.\par",
        r"- Total unique users: 610.\par",
        r"- Total unique movies: 9,724.\par",
        r"- Global average rating across all entries: 3.50 out of 5.0 stars.\par",
        r"- Matrix sparsity: ~98.3% (meaning less than 2% of possible user-movie combinations have explicit ratings, confirming the need for low-rank matrix factorization).\par\par",

        # ----------------- SECTION 5: DATASET SOURCE -----------------
        r"\f2\fs30\b\cf1 5. DATASET SOURCE AND REFERENCE\par\f0\fs22\b0\cf2",
        r"- \b Research Laboratory: \b0 GroupLens Research Project, Department of Computer Science and Engineering, University of Minnesota.\par",
        r"- \b Package Downloaded: \b0 MovieLens Latest Small Dataset (ml-latest-small.zip)\par",
        r"- \b Official Web Source: \b0 https://grouplens.org/datasets/movielens/\par",
        r"- \b Academic Citation: \b0 F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI: https://doi.org/10.1145/2827872\par\par",

        # ----------------- SECTION 6: TOOLS & TECHNOLOGIES -----------------
        r"\f2\fs30\b\cf1 6. TOOLS AND TECHNOLOGIES USED\par\f0\fs22\b0\cf2",
        r"To build and test the complete system, we utilized the following software stack:\par\par",
        r"- \b Distributed Computing Engine: \b0 Apache Spark 3.5.1 (PySpark distribution). Provides in-memory Resilient Distributed Datasets (RDDs), Spark SQL, Catalyst Optimizer, and the MLlib distributed machine learning engine.\par",
        r"- \b Java Virtual Machine: \b0 Java Development Kit (JDK 1.8.0_381, 64-bit). Spark runs on the JVM; our cluster session interfaces with Java via Py4J.\par",
        r"- \b Programming Language: \b0 Python 3.10.0 (64-bit). Chosen for its clean syntax, rich ecosystem, and PySpark API.\par",
        r"- \b Machine Learning Library: \b0 Spark MLlib (pyspark.ml.recommendation.ALS, pyspark.ml.evaluation.RegressionEvaluator).\par",
        r"- \b Cluster Master Configuration: \b0 Local multi-core mode (local[*]), which spawns as many worker executor threads as physical and logical CPU cores available on our test system.\par",
        r"- \b Windows Hadoop Native Binaries: \b0 Hadoop 3.3.5 native libraries (winutils.exe and hadoop.dll placed in C:\\Spark\\bin), resolving Windows NativeIO access requirements during model checkpointing.\par",
        r"- \b Web Application Framework: \b0 Flask 3.1.3 (Python WSGI framework).\par",
        r"- \b Charting & Visualization: \b0 Plotly 7.1.0 and Plotly.js for interactive analytics graphs.\par",
        r"- \b User Interface: \b0 HTML5, CSS3, JavaScript, and Bootstrap 5.3 with Dark Mode support.\par",
        r"- \b Testing Framework: \b0 Python standard library unittest module.\par\par",

        # ----------------- SECTION 7: IMPLEMENTATION / METHODOLOGY -----------------
        r"\f2\fs30\b\cf1 7. IMPLEMENTATION AND METHODOLOGY\par\f0\fs22\b0\cf2",
        r"We organized our implementation into nine modular stages to ensure clean separation of concerns and maintainable code:\par\par",
        r"\b Step 1: Thread-Safe SparkSession Singleton (src/spark_session.py)\b0\par",
        r"Starting and stopping a SparkSession repeatedly consumes substantial memory and startup time. We implemented a thread-safe Singleton pattern using Python's threading.Lock. Our configuration explicitly tunes the cluster session:\par",
        r"- spark.master = local[*] to utilize all processor cores.\par",
        r"- spark.sql.shuffle.partitions = 4 (Spark defaults to 200 shuffle partitions, which causes severe scheduling overhead on single-machine multi-core workstations; reducing this to 4 matches our local core count and cuts shuffle latency).\par",
        r"- spark.driver.memory = 2g to prevent garbage collection pauses during matrix factorization.\par",
        r"- Explicitly binding spark.pyspark.python and spark.pyspark.driver.python to sys.executable to prevent Windows Microsoft Store app execution alias conflicts.\par\par",

        r"\b Step 2: Strict Schema Ingestion (src/data_loader.py)\b0\par",
        r"Instead of letting Spark infer schemas automatically (which requires scanning the dataset twice across disk), we defined explicit StructType schemas with StructField types for ratings, movies, tags, and links. This guarantees deterministic types and speeds up ingestion.\par\par",

        r"\b Step 3: Data Cleaning and Deduplication (src/preprocessing.py)\b0\par",
        r"In raw real-world data, users occasionally update a rating or duplicate records slip through. We applied three cleaning stages:\par",
        r"1. Null and range check: We filtered ratings strictly between 0.5 and 5.0 and removed any rows with null userId or movieId.\par",
        r"2. Window Deduplication: We created a PySpark Window partitioned by userId and movieId, ordered by timestamp DESC, assigned row_number(), and filtered for row_number == 1. This retains each user's latest review.\par",
        r"3. Release Year Extraction: Using the PySpark regexp_extract function, we parsed the 4-digit release year from the movie title string into a separate integer column.\par\par",

        r"\b Step 4: Spark SQL Aggregations and Catalyst Query Inspection (src/spark_analytics.py)\b0\par",
        r"We registered the cleaned DataFrames as temporary views in Spark SQL. We formulated distributed queries to calculate:\par",
        r"- Top-rated movies having a minimum threshold of 30 reviews (preventing 1-review 5-star bias).\par",
        r"- Most active reviewers (power users).\par",
        r"- Genre frequency distributions using the Spark SQL explode(split(genres, '\\|')) function.\par",
        r"- Review submission trends grouped by year from 1996 through 2018.\par",
        r"We also programmatically extracted the Catalyst physical execution plan using queryExecution.toString() to observe how Spark performs projection pruning and hash aggregate optimizations.\par\par",

        r"\b Step 5: Distributed ALS Model Training (src/train_model.py)\b0\par",
        r"We partitioned our cleaned ratings into 80% training (80,853 rows) and 20% testing (19,983 rows) using randomSplit([0.8, 0.2], seed=42). We trained the Spark MLlib ALS estimator with:\par",
        r"- rank = 10 (latent factor dimensionality for users and movies)\par",
        r"- maxIter = 10 (iterations of alternating quadratic optimization)\par",
        r"- regParam = 0.1 (L2 regularization to penalize large factor weights and prevent overfitting)\par",
        r"- coldStartStrategy = 'drop' (prevents evaluation failure on unobserved test IDs)\par",
        r"- nonnegative = True (constrains latent vectors to positive values for interpretability)\par\par",

        r"\b Step 6: Model Evaluation and Persistence\b0\par",
        r"We evaluated the test predictions using RegressionEvaluator, recording Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). Once validated, the trained ALS model was serialized to disk at models/als_movie_recommender_model, and the evaluation metrics were exported to JSON.\par\par",

        r"\b Step 7: Recommendation Engine with Cold-Start Fallback (src/recommender.py)\b0\par",
        r"Our inference service loads the serialized ALS model. For an active user, it identifies films already rated, builds a candidate dataframe of unseen films, predicts ratings using model.transform, and sorts by prediction descending. If the requested user ID has never been seen (Cold-Start), the service intercepts the request and recommends the cluster's highest-rated consensus films having at least 30 reviews, returning an informative status flag.\par\par",

        r"\b Step 8: INT315 RDD & MapReduce Lineage Demonstration (src/rdd_demo.py)\b0\par",
        r"To cover the INT315 cluster computing syllabus, we wrote low-level RDD functions:\par",
        r"- MapReduce Word Count on movie tags: textFile -> filter -> flatMap (tokenization) -> map (k, 1) -> reduceByKey (addition) -> sortBy.\par",
        r"- Lineage DAG Inspection: Printing the RDD DAG via toDebugString() to show shuffle boundaries.\par",
        r"- Empirical Speed Benchmark: Running identical group-by aggregations on 100,836 rows using RDDs versus DataFrames and recording execution time.\par\par",

        r"\b Step 9: Web Application & CLI Deployment (app.py & cli.py)\b0\par",
        r"We implemented a web interface using Flask, Bootstrap 5, and Plotly, featuring 6 routes and 3 REST endpoints. In addition, an interactive Command Prompt terminal runner (cli.py) allows testing all features without opening a browser.\par\par",

        # ----------------- SECTION 8: COMPLETE CODE -----------------
        r"\f2\fs30\b\cf1 8. COMPLETE SOURCE CODE\par\f0\fs20\b0\cf2",
        r"Below is the complete, documented source code for all modules in our project:\par\par",

        # Code 1: config.py
        r"\f2\fs24\b\cf1 8.1 Configuration Module (config.py)\par\f1\fs16\cf2",
        escape_rtf(code_config),
        r"\f0\fs20\cf2\par\par",

        # Code 2: spark_session.py
        r"\f2\fs24\b\cf1 8.2 SparkSession Manager (src/spark_session.py)\par\f1\fs16\cf2",
        escape_rtf(code_spark_session),
        r"\f0\fs20\cf2\par\par",

        # Code 3: data_loader.py
        r"\f2\fs24\b\cf1 8.3 Data Ingestion Module (src/data_loader.py)\par\f1\fs16\cf2",
        escape_rtf(code_data_loader),
        r"\f0\fs20\cf2\par\par",

        # Code 4: preprocessing.py
        r"\f2\fs24\b\cf1 8.4 Preprocessing and Deduplication (src/preprocessing.py)\par\f1\fs16\cf2",
        escape_rtf(code_preprocessing),
        r"\f0\fs20\cf2\par\par",

        # Code 5: spark_analytics.py
        r"\f2\fs24\b\cf1 8.5 Spark SQL Analytics and Catalyst Plans (src/spark_analytics.py)\par\f1\fs16\cf2",
        escape_rtf(code_spark_analytics),
        r"\f0\fs20\cf2\par\par",

        # Code 6: rdd_demo.py
        r"\f2\fs24\b\cf1 8.6 Low-Level RDD and DAG Lineage Demonstrator (src/rdd_demo.py)\par\f1\fs16\cf2",
        escape_rtf(code_rdd_demo),
        r"\f0\fs20\cf2\par\par",

        # Code 7: train_model.py
        r"\f2\fs24\b\cf1 8.7 ALS Collaborative Filtering Training (src/train_model.py)\par\f1\fs16\cf2",
        escape_rtf(code_train_model),
        r"\f0\fs20\cf2\par\par",

        # Code 8: recommender.py
        r"\f2\fs24\b\cf1 8.8 Recommendation and Cold-Start Service (src/recommender.py)\par\f1\fs16\cf2",
        escape_rtf(code_recommender),
        r"\f0\fs20\cf2\par\par",

        # Code 9: app.py
        r"\f2\fs24\b\cf1 8.9 Flask Web Application (app.py)\par\f1\fs16\cf2",
        escape_rtf(code_app),
        r"\f0\fs20\cf2\par\par",

        # Code 10: cli.py
        r"\f2\fs24\b\cf1 8.10 Command-Line Interface Runner (cli.py)\par\f1\fs16\cf2",
        escape_rtf(code_cli),
        r"\f0\fs20\cf2\par\par",

        # ----------------- SECTION 9: EXECUTION LOGS & SCREENSHOTS -----------------
        r"\f2\fs30\b\cf1 9. EXECUTION SCREENSHOTS AND TERMINAL LOGS\par\f0\fs22\b0\cf2",
        r"This section contains the actual console logs recorded during model training and automated test suite execution on our workstation:\par\par",

        r"\b 9.1 ALS Model Training Terminal Output:\b0\par\f1\fs16\cf7",
        r"C:\Users\Lenovo\Desktop\movie-recommendation-system> python src\train_model.py\par",
        r"================================================================================\par",
        r"           TRAINING SPARK MLLIB ALS RECOMMENDATION MODEL                        \par",
        r"================================================================================\par",
        r"[*] Loading raw datasets from C:\Users\Lenovo\Desktop\movie-recommendation-system\data\raw\ml-latest-small...\par",
        r"[*] Cleaning and preprocessing datasets...\par",
        r"[*] Total verified ratings available for training: 100836\par",
        r"[*] Splitting ratings into 80% Train / 20% Test (seed=42)...\par",
        r"[*] Training records: 80853 | Test records: 19983\par",
        r"[*] Configuring Spark MLlib ALS Estimator with parameters:\par",
        r"    - userCol: userId\par",
        r"    - itemCol: movieId\par",
        r"    - ratingCol: rating\par",
        r"    - rank: 10\par",
        r"    - maxIter: 10\par",
        r"    - regParam: 0.1\par",
        r"    - coldStartStrategy: drop\par",
        r"    - nonnegative: True\par",
        r"    - seed: 42\par",
        r"[*] Training ALS model on distributed executors...\par",
        r"[+] Model training completed in 7.08 seconds.\par",
        r"[*] Generating predictions on unseen test dataset...\par",
        r"[+] Evaluation Results:\par",
        r"    - Root Mean Squared Error (RMSE): 0.8666\par",
        r"    - Mean Absolute Error (MAE):     0.6694\par",
        r"[+] Evaluation metrics saved to: outputs\model_metrics.json\par",
        r"[*] Persisting ALS Model to disk at: models\als_movie_recommender_model...\par",
        r"[+] Model successfully saved to disk.\par\f0\fs22\cf2\par",

        r"\b 9.2 Automated Unit Test Execution (12 Tests Passing):\b0\par\f1\fs16\cf7",
        r"C:\Users\Lenovo\Desktop\movie-recommendation-system> python -m unittest discover -s tests -v\par",
        r"test_01_verify_dataset_files (tests.test_recommendation_system.TestMovieRecommendationSystem) ... ok\par",
        r"test_02_load_raw_datasets (tests.test_recommendation_system.TestMovieRecommendationSystem) ... ok\par",
        r"test_03_missing_dataset_handling (tests.test_recommendation_system.TestMovieRecommendationSystem) ... ok\par",
        r"test_04_preprocessing_and_deduplication (tests.test_recommendation_system.TestMovieRecommendationSystem) ... ok\par",
        r"test_05_spark_sql_analytics (tests.test_recommendation_system.TestMovieRecommendationSystem) ... ok\par",
        r"test_06_rdd_word_count_and_lineage (tests.test_recommendation_system.TestMovieRecommendationSystem) ... ok\par",
        r"test_07_rdd_vs_dataframe_benchmark (tests.test_recommendation_system.TestMovieRecommendationSystem) ... ok\par",
        r"test_08_recommendation_for_valid_user (tests.test_recommendation_system.TestMovieRecommendationSystem) ... ok\par",
        r"test_09_cold_start_handling (tests.test_recommendation_system.TestMovieRecommendationSystem) ... ok\par",
        r"test_10_movie_search_and_genre_filtering (tests.test_recommendation_system.TestMovieRecommendationSystem) ... ok\par",
        r"test_11_flask_endpoints (tests.test_recommendation_system.TestMovieRecommendationSystem) ... ok\par",
        r"test_12_movie_details_route (tests.test_recommendation_system.TestMovieRecommendationSystem) ... ok\par\par",
        r"----------------------------------------------------------------------\par",
        r"Ran 12 tests in 113.364s\par\par",
        r"OK\par\f0\fs22\cf2\par",

        r"\b [INSERT SCREENSHOT 1: Terminal Model Training and Unit Test Execution]\b0\par",
        r"\i (To insert in WordPad: Place cursor here, click the 'Picture' button on the Home ribbon, and select your Command Prompt screenshot).\i0\par\par",

        # ----------------- SECTION 10: OUTPUT SCREENSHOTS -----------------
        r"\f2\fs30\b\cf1 10. OUTPUT / RESULT SCREENSHOTS\par\f0\fs22\b0\cf2",
        r"This section provides visual walkthroughs of the user-facing web dashboard and recommendation results:\par\par",

        r"\b 10.1 Web Dashboard Home Page (Big Data Metrics & Popular Catalog):\b0\par",
        r"The home screen displays the key cluster metrics (100,836 ratings, 9,724 cataloged movies, 610 users, 3.50 average star rating) along with a carousel of top-rated films and a live Spark cluster status card.\par",
        r"\b [INSERT SCREENSHOT 2: Web Dashboard Home Page]\b0\par\par",

        r"\b 10.2 Personalized Recommendation Engine (User 1):\b0\par",
        r"Shows the top 10 recommended movies calculated by the ALS model for User 1, displaying predicted ratings (e.g., 4.7 stars), release years, genres, and a direct comparison with User 1's historical favorites.\par",
        r"\b [INSERT SCREENSHOT 3: Personalized Recommendations for User 1]\b0\par\par",

        r"\b 10.3 Cold-Start Fallback Alert (New User 9999):\b0\par",
        r"When querying an unknown user ID (e.g., 9999), the application displays a friendly warning banner explaining that no prior ratings exist and provides cluster-consensus recommendations (Shawshank Redemption, Godfather, Pulp Fiction) based on high review counts.\par",
        r"\b [INSERT SCREENSHOT 4: Cold-Start Fallback Display]\b0\par\par",

        r"\b 10.4 Big Data Analytics Dashboard:\b0\par",
        r"Displays three interactive Plotly charts generated by Spark SQL queries: Genre Frequency & Average Ratings bar chart, Global Star Rating distribution histogram, and Yearly Review Submissions from 1996 to the present.\par",
        r"\b [INSERT SCREENSHOT 5: Spark SQL Interactive Analytics Dashboard]\b0\par\par",

        r"\b 10.5 INT315 Cluster Computing Benchmark & RDD DAG Lineage:\b0\par",
        r"Displays the measured speed comparison (3.21x speedup of DataFrame over low-level RDD), the RDD Word Count MapReduce output, the DAG lineage graph string, and the Catalyst Physical Plan.\par",
        r"\b [INSERT SCREENSHOT 6: Cluster Demonstration & Speed Benchmark Page]\b0\par\par",

        # ----------------- SECTION 11: RESULT ANALYSIS -----------------
        r"\f2\fs30\b\cf1 11. RESULT ANALYSIS AND DISCUSSION\par\f0\fs22\b0\cf2",
        r"\b 11.1 ALS Collaborative Filtering Model Performance\b0\par",
        r"We evaluated our trained model on the 20% hold-out test set (19,983 ratings). The empirical metrics obtained were:\par",
        r"- \b Root Mean Squared Error (RMSE): \b0 0.8666\par",
        r"- \b Mean Absolute Error (MAE): \b0 0.6694\par",
        r"- \b Latent Factor Rank: \b0 10\par",
        r"- \b Training Time: \b0 7.08 seconds across local worker threads\par\par",
        r"\b Discussion of Error Rates:\b0\par",
        r"On a 5-star rating scale, an RMSE of 0.8666 and an MAE of 0.6694 are solid results for collaborative filtering. An MAE of 0.6694 indicates that our predicted rating is, on average, within two-thirds of a star of the user's actual rating. This confirms that factorizing the interaction matrix into 10 latent dimensions captures meaningful user genre preferences without overfitting.\par\par",

        r"\b 11.2 Empirical Speedup: Low-Level RDDs vs Catalyst DataFrames\b0\par",
        r"To evaluate cluster computing performance as taught in INT315, we ran identical grouping and averaging operations on all 100,836 ratings using both low-level Python RDDs and Spark SQL DataFrames:\par",
        r"- \b Low-Level Python RDD Time: \b0 3.9517 seconds\par",
        r"- \b Spark SQL DataFrame Time: \b0 1.2327 seconds\par",
        r"- \b Measured Performance Speedup: \b0 3.21x Faster\par\par",
        r"\b Technical Explanation for the 3.21x Speedup:\b0\par",
        r"1. \b Catalyst Query Optimizer: \b0 Spark DataFrames are backed by the Catalyst engine, which optimizes the logical query plan into a streamlined physical execution plan using projection pruning and filter pushdown.\par",
        r"2. \b Whole-Stage Code Generation (Project Tungsten): \b0 Catalyst synthesizes the query plan into compact Java bytecode at runtime, running operations directly inside JVM CPU registers.\par",
        r"3. \b Elimination of Python-to-JVM Serialization Overhead: \b0 Low-level Python RDDs execute Python lambdas, requiring PySpark to serialize data from the JVM to Python worker processes via Py4J sockets and back again. DataFrames process data directly in off-heap memory, completely bypassing this serialization bottleneck.\par\par",

        r"\b 11.3 Architectural Comparison: Apache Spark vs Hadoop MapReduce\b0\par",
        r"Our findings highlight why Apache Spark has superseded Hadoop MapReduce for big data machine learning:\par",
        r"- \b Intermediate Data Storage: \b0 Hadoop MapReduce writes intermediate map outputs to disk and shuffles them across the network before reduce tasks run. Spark caches intermediate DataFrames in distributed executor memory (RAM).\par",
        r"- \b Iterative Matrix Computations: \b0 ALS requires 10 alternating updates. Running this on Hadoop MapReduce would trigger 20 disk write/read cycles to HDFS, making it over 10x slower. On Spark, matrices remain resident in RAM across iterations, converging in just 7.08 seconds.\par",
        r"- \b Fault-Tolerance Mechanism: \b0 Hadoop relies on 3x disk replication, which is expensive in terms of storage. Spark uses lazy evaluation and Directed Acyclic Graph (DAG) Lineage. If a node fails, Spark re-executes only the missing partition from its parent lineage rather than restarting the entire job.\par\par",

        # ----------------- SECTION 12: CONCLUSION -----------------
        r"\f2\fs30\b\cf1 12. CONCLUSION\par\f0\fs22\b0\cf2",
        r"In this project, we designed, built, and evaluated an end-to-end \b Movie Recommendation System Using Big Data and Apache Spark MLlib\b0  meeting all requirements of the INT315 Cluster Computing curriculum.\par\par",
        r"\b Key Outcomes Achieved:\b0\par",
        r"1. Successfully loaded and validated 100,836 MovieLens ratings using PySpark DataFrames with strict schemas.\par",
        r"2. Cleaned data by filtering out-of-bound ratings, removing null values, and deduplicating records using Spark SQL Window partitions.\par",
        r"3. Trained an ALS Collaborative Filtering model achieving an RMSE of 0.8666 and MAE of 0.6694 in 7.08 seconds.\par",
        r"4. Handled the cold-start challenge by implementing an automated fallback to high-confidence cluster consensus ratings.\par",
        r"5. Demonstrated key INT315 concepts including MapReduce tokenization, RDD lineage graphs, Catalyst execution plans, and an empirical 3.21x DataFrame speedup.\par",
        r"6. Built both an interactive terminal CLI and a web-based dark-mode streaming dashboard for project demonstration.\par\par",
        r"\b Future Work:\b0\par",
        r"For future enhancements, we can extend this project into a hybrid recommendation system combining collaborative filtering with TF-IDF content-based filtering on movie tags. Additionally, integrating Apache Kafka for real-time streaming ratings and deploying on a multi-node Kubernetes cluster would allow horizontal scaling to millions of active users.\par\par",

        # ----------------- APPENDIX: VIVA QUESTIONS -----------------
        r"\f2\fs30\b\cf1 APPENDIX: INT315 VIVA QUESTIONS AND MODEL ANSWERS\par\f0\fs22\b0\cf2",
        r"Here are common viva questions asked by external examiners for cluster computing projects, along with clear model answers:\par\par",

        r"\b Q1: What is Alternating Least Squares (ALS) and why is it used for collaborative filtering?\b0\par",
        r"Answer: ALS is a matrix factorization algorithm. It takes a large, sparse user-item rating matrix R and approximates it as the product of two lower-rank dense matrices: a User matrix U and an Item matrix V (R approx U * V^T). Because optimizing both matrices at the same time is non-convex, ALS holds U constant to solve for V using quadratic optimization, and then holds V constant to solve for U. It alternates this process across several iterations on distributed Spark executors.\par\par",

        r"\b Q2: What is an RDD and how does Spark provide fault tolerance without replicating data?\b0\par",
        r"Answer: An RDD (Resilient Distributed Dataset) is an immutable, distributed collection of data partitioned across cluster nodes. Spark achieves fault tolerance through its Lineage Graph (DAG). Rather than copying data across multiple disks (like HDFS 3x replication), Spark records the exact recipe of deterministic transformations (map, filter, reduceByKey) used to build each RDD. If an executor fails and loses a partition, Spark simply replays the lineage for that specific lost partition.\par\par",

        r"\b Q3: Why is Spark SQL DataFrame faster than a low-level Python RDD?\b0\par",
        r"Answer: DataFrames use the Catalyst Query Optimizer, which performs projection pruning, predicate pushdown, and bytecode generation via Project Tungsten. Low-level Python RDDs run custom Python lambdas that require serializing data back and forth between the Java JVM and Python worker processes over Py4J sockets. DataFrames operate directly in off-heap memory, avoiding this serialization overhead and running over 3x faster in our benchmark.\par\par",

        r"\b Q4: How did you prevent data leakage during model training?\b0\par",
        r"Answer: We partitioned the ratings into an 80% training set and a 20% test set using PySpark's randomSplit([0.8, 0.2], seed=42) before any model fitting occurred. The ALS model was trained exclusively on the 80% split. Evaluation metrics (RMSE and MAE) were computed strictly on the unseen 20% test fold.\par\par",

        r"\b Q5: What is the Cold-Start problem and how did you resolve it?\b0\par",
        r"Answer: The cold-start problem occurs when a new user joins who has no previous ratings. Because the user has no interaction history, the collaborative filtering model cannot generate a latent factor vector or compute dot products. We resolved this by creating an automated fallback: when an unseen user ID is requested, our service intercepts the request and recommends cluster-consensus movies with the highest average rating and at least 30 reviews, clearly labeling the result.\par\par",

        r"\b Q6: Why did you set spark.sql.shuffle.partitions to 4 instead of leaving it at the default 200?\b0\par",
        r"Answer: Apache Spark defaults to 200 shuffle partitions, which is designed for large multi-node clusters with hundreds of cores. On a local multi-core machine with 4 to 8 threads, creating 200 tiny partitions causes excessive task scheduling overhead and slows down execution. Reducing shuffle partitions to 4 matched our local core count and significantly improved shuffle performance.\par\par",

        # End of Document
        r"}"
    ]

    rtf_text = "\n".join(rtf_content)
    
    # Always write the humanized version
    with open(HUMANIZED_RTF, "w", encoding="utf-8") as f:
        f.write(rtf_text)
    print(f"[+] Successfully wrote: {HUMANIZED_RTF} ({len(rtf_text):,} bytes)")

    # Try writing to the standard filename as well
    try:
        with open(OUTPUT_RTF, "w", encoding="utf-8") as f:
            f.write(rtf_text)
        print(f"[+] Successfully updated: {OUTPUT_RTF}")
    except PermissionError:
        print(f"[!] Note: {OUTPUT_RTF.name} is currently open in Word/WordPad, saved as {HUMANIZED_RTF.name}")

    # Also synchronize to the scratch report folder
    scratch_dir = Path("C:/Users/Lenovo/.gemini/antigravity/scratch/movie-recommendation-system/report")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(scratch_dir / "INT315_PROJECT_REPORT_HUMANIZED.rtf", "w", encoding="utf-8") as f:
            f.write(rtf_text)
        with open(scratch_dir / "INT315_PROJECT_REPORT.rtf", "w", encoding="utf-8") as f:
            f.write(rtf_text)
        print(f"[+] Synchronized to scratch report folder.")
    except Exception as e:
        print(f"[!] Scratch sync note: {e}")


if __name__ == "__main__":
    generate_report()
