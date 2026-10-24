"""
INT315 - Cluster Computing Project
Command-Line Interface (CLI) Runner
Allows direct execution and demonstration of Big Data operations in Windows Command Prompt (CMD).
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure project root is in system path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.spark_session import get_spark_session, stop_spark_session
from src.train_model import train_and_evaluate_als
from src.recommender import MovieRecommenderService
from src.spark_analytics import SparkMovieAnalytics
from src.rdd_demo import RDDDemonstrator


def print_banner():
    banner = r"""
========================================================================
   CINESPARK: Movie Recommendation System Using Apache Spark MLlib
   Course: INT315 – Cluster Computing (Terminal Demonstration Mode)
========================================================================
    """
    print(banner)


def run_recommendations_cli(user_id: int = 1, top_n: int = 10):
    """Generates and prints personalized recommendations in CMD."""
    print(f"\n[*] Initializing Recommendation Service for User ID: {user_id}...")
    spark = get_spark_session()
    recommender = MovieRecommenderService(spark)
    rec_data = recommender.recommend_for_user(user_id=user_id, top_n=top_n)

    print("\n" + "=" * 75)
    if rec_data["is_cold_start"]:
        print(f" [!] COLD-START DETECTED: {rec_data['status_message']}")
        print("     Falling back to cluster-wide consensus top-rated movies.")
    else:
        print(f" [+] ACTIVE PROFILE FOUND: User {user_id} has {rec_data['user_rated_count']} historical reviews.")
        print(f"     Personalized predictions generated using Spark ALS Matrix Factorization:")
    print("=" * 75)

    print(f"\n{'Rank':<6}{'Predicted':<12}{'Avg':<8}{'Reviews':<10}{'Title':<35}")
    print("-" * 75)
    for idx, movie in enumerate(rec_data["recommendations"], start=1):
        pred = f"{movie.get('predicted_rating', movie.get('avg_rating')):.2f} ★"
        avg = f"{movie.get('avg_rating', 0):.2f} ★"
        reviews = str(movie.get("rating_count", 0))
        title = movie["title"][:33]
        print(f"#{idx:<5}{pred:<12}{avg:<8}{reviews:<10}{title:<35}")
    print("-" * 75 + "\n")


def run_analytics_cli():
    """Runs and prints Spark SQL analytics in CMD."""
    print("\n[*] Initializing Spark SQL Analytics Engine...")
    spark = get_spark_session()
    recommender = MovieRecommenderService(spark)
    analytics = SparkMovieAnalytics(spark, {
        "movies": recommender.movies_df,
        "ratings": recommender.ratings_df,
        "tags": recommender.tags_df,
        "links": recommender.links_df,
        "enriched_ratings": recommender.enriched_df,
    })

    stats = analytics.get_overview_statistics()
    print("\n" + "=" * 60)
    print("               BIG DATA DATASET STATISTICS")
    print("=" * 60)
    print(f" Total Movies Cataloged:  {stats['total_movies']:,}")
    print(f" Total Ratings Processed: {stats['total_ratings']:,}")
    print(f" Total Distinct Users:    {stats['total_users']:,}")
    print(f" Total Tags Submitted:    {stats['total_tags']:,}")
    print(f" Global Average Rating:   {stats['global_average_rating']} ★")
    print("=" * 60)

    print("\n[*] Top 5 Most Popular Movies (Spark SQL GROUP BY & COUNT):")
    popular = analytics.get_most_popular_movies(limit=5)
    print(f"{'Title':<38}{'Reviews':<10}{'Avg Rating':<10}")
    print("-" * 60)
    for m in popular:
        print(f"{m['title'][:36]:<38}{m['rating_count']:<10}{m['avg_rating']:.2f} ★")

    print("\n[*] Top 5 Highest-Rated Movies (Minimum 30 Reviews Threshold):")
    top_rated = analytics.get_highest_rated_movies(min_ratings=30, limit=5)
    print(f"{'Title':<38}{'Reviews':<10}{'Avg Rating':<10}")
    print("-" * 60)
    for m in top_rated:
        print(f"{m['title'][:36]:<38}{m['rating_count']:<10}{m['avg_rating']:.2f} ★")
    print("-" * 60 + "\n")


def run_rdd_demo_cli():
    """Demonstrates low-level RDD primitives and prints results in CMD."""
    print("\n[*] Initializing INT315 RDD Demonstrator...")
    spark = get_spark_session()
    rdd_demo = RDDDemonstrator(spark)

    print("\n" + "=" * 70)
    print(" 1. RDD TAG WORD COUNT (MapReduce Paradigm)")
    print("    flatMap -> map -> reduceByKey -> sortBy -> take(10)")
    print("=" * 70)
    tag_result = rdd_demo.demonstrate_tag_word_count()
    print(f"{'Word / Token':<25}{'Distributed Frequency':<15}")
    print("-" * 45)
    for item in tag_result["results"][:10]:
        print(f"{item['word']:<25}{item['frequency']:<15}")

    print("\n" + "=" * 70)
    print(" 2. EMPIRICAL BENCHMARK: Low-Level RDD vs Catalyst DataFrame")
    print("=" * 70)
    bench = rdd_demo.compare_rdd_vs_dataframe_performance()
    print(f" Primitive RDD Execution Time:       {bench['rdd_benchmark']['execution_time_seconds']} seconds")
    print(f" Catalyst DataFrame Execution Time:   {bench['dataframe_benchmark']['execution_time_seconds']} seconds")
    print(f" Measured Catalyst Speedup Factor:   {bench['speedup_factor']}x Faster")
    print("\n Theoretical Insight:")
    print(f" {bench['theoretical_insight']}")
    print("=" * 70 + "\n")


def interactive_menu():
    """Interactive text menu for CMD."""
    while True:
        print_banner()
        print("Select an operation to run in Command Prompt:\n")
        print(" [1] Get Personalized Recommendations (Enter User ID)")
        print(" [2] Test Cold-Start Fallback (New User ID 9999)")
        print(" [3] Run Spark SQL Big Data Analytics")
        print(" [4] Run INT315 RDD Demonstrations & Benchmark")
        print(" [5] Train / Retrain Spark MLlib ALS Model")
        print(" [6] Run Full 12-Test Automated Unit Test Suite")
        print(" [7] Launch Web Application (Starts Flask Server on :5000)")
        print(" [8] Exit\n")

        choice = input("Enter choice [1-8]: ").strip()

        if choice == "1":
            uid = input("Enter User ID (1 to 610) [default: 1]: ").strip()
            user_id = int(uid) if uid.isdigit() else 1
            run_recommendations_cli(user_id=user_id)
            input("Press Enter to continue...")
        elif choice == "2":
            run_recommendations_cli(user_id=9999)
            input("Press Enter to continue...")
        elif choice == "3":
            run_analytics_cli()
            input("Press Enter to continue...")
        elif choice == "4":
            run_rdd_demo_cli()
            input("Press Enter to continue...")
        elif choice == "5":
            print("\n[*] Starting ALS Model Training Pipeline...")
            train_and_evaluate_als()
            input("Press Enter to continue...")
        elif choice == "6":
            print("\n[*] Running Automated Test Suite...")
            os.system(f'"{sys.executable}" -m unittest discover -s tests -v')
            input("Press Enter to continue...")
        elif choice == "7":
            print("\n[*] Starting Web Server on http://127.0.0.1:5000 (Press Ctrl+C to stop)...")
            os.system(f'"{sys.executable}" app.py')
            break
        elif choice == "8":
            print("\nExiting. Thank you!")
            break
        else:
            print("\n[!] Invalid choice. Please enter a number between 1 and 8.")


def main():
    parser = argparse.ArgumentParser(description="INT315 Movie Recommendation System CLI Runner")
    parser.add_argument("--recommend", type=int, help="Generate recommendations for a specific User ID (e.g. --recommend 1)")
    parser.add_argument("--analytics", action="store_true", help="Print Big Data Spark SQL analytics")
    parser.add_argument("--rdd", action="store_true", help="Run RDD MapReduce demonstration and speed benchmark")
    parser.add_argument("--train", action="store_true", help="Train and evaluate Spark MLlib ALS model")

    args = parser.parse_args()

    if args.recommend:
        print_banner()
        run_recommendations_cli(user_id=args.recommend)
    elif args.analytics:
        print_banner()
        run_analytics_cli()
    elif args.rdd:
        print_banner()
        run_rdd_demo_cli()
    elif args.train:
        print_banner()
        train_and_evaluate_als()
    else:
        # If run without flags, open interactive menu
        interactive_menu()


if __name__ == "__main__":
    main()
