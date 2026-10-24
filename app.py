"""
INT315 - Cluster Computing Project
Flask Application: Movie Recommendation System Using Big Data & Apache Spark MLlib
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for
import plotly
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in system path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.spark_session import get_spark_session, stop_spark_session
from src.recommender import MovieRecommenderService
from src.spark_analytics import SparkMovieAnalytics
from src.rdd_demo import RDDDemonstrator

app = Flask(__name__)
app.config["SECRET_KEY"] = "int315-spark-cluster-computing-secret-key"

# Global lazy service holder
recommender_service = None
analytics_service = None
rdd_service = None


def get_services():
    """Initializes and returns shared Spark services."""
    global recommender_service, analytics_service, rdd_service
    if recommender_service is None:
        spark = get_spark_session()
        recommender_service = MovieRecommenderService(spark)
        analytics_service = SparkMovieAnalytics(spark, {
            "movies": recommender_service.movies_df,
            "ratings": recommender_service.ratings_df,
            "tags": recommender_service.tags_df,
            "links": recommender_service.links_df,
            "enriched_ratings": recommender_service.enriched_df,
        })
        rdd_service = RDDDemonstrator(spark)
    return recommender_service, analytics_service, rdd_service


# ---------------------------------------------------------------------------
# Web Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Home Dashboard with dataset metrics and top movie highlights."""
    service, analytics, _ = get_services()
    overview = analytics.get_overview_statistics()
    popular_movies = service.get_popular_movies(limit=8)
    top_rated_movies = service.get_top_rated_movies(min_ratings=config.POPULAR_MIN_RATINGS, limit=8)

    # Load model metrics if available
    metrics = None
    if config.MODEL_METRICS_PATH.exists():
        try:
            with open(config.MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

    cluster_info = {
        "app_name": config.SPARK_APP_NAME,
        "master": config.SPARK_MASTER,
        "driver_memory": config.SPARK_DRIVER_MEMORY,
        "shuffle_partitions": config.SPARK_SHUFFLE_PARTITIONS,
        "spark_version": service.spark.version,
    }

    return render_template(
        "index.html",
        overview=overview,
        popular_movies=popular_movies,
        top_rated_movies=top_rated_movies,
        metrics=metrics,
        cluster_info=cluster_info
    )


@app.route("/recommendations")
def recommendations():
    """Personalized Recommendation Engine page."""
    service, _, _ = get_services()
    user_id_param = request.args.get("user_id", default=1, type=int)

    rec_data = service.recommend_for_user(user_id=user_id_param, top_n=config.TOP_N_RECOMMENDATIONS)
    user_history = service.get_user_rating_history(user_id=user_id_param, limit=8) if not rec_data["is_cold_start"] else []

    # Representative sample user presets for the viva presentation
    sample_users = [1, 2, 50, 100, 300, 610]

    return render_template(
        "recommendations.html",
        user_id=user_id_param,
        rec_data=rec_data,
        user_history=user_history,
        sample_users=sample_users
    )


@app.route("/movies")
def movies():
    """Movie catalog search and genre browsing."""
    service, _, _ = get_services()
    query = request.args.get("q", default="", type=str)
    genre = request.args.get("genre", default="All", type=str)
    page = request.args.get("page", default=1, type=int)

    search_result = service.search_movies(query=query, genre=genre, page=page, page_size=20)

    return render_template(
        "movies.html",
        search_result=search_result,
        genres=["All"] + service.genre_list,
        current_genre=genre,
        current_query=query
    )


@app.route("/movie/<int:movie_id>")
def movie_details(movie_id: int):
    """Deep-dive movie details view with rating distribution chart."""
    service, _, _ = get_services()
    details = service.get_movie_details(movie_id)
    if not details:
        return render_template("404.html", message=f"Movie ID {movie_id} was not found."), 404

    # Build Plotly Rating Distribution Chart
    dist = details.get("rating_distribution", [])
    ratings_x = [d["rating"] for d in dist]
    counts_y = [d["count"] for d in dist]

    fig = go.Figure(data=[
        go.Bar(
            x=ratings_x,
            y=counts_y,
            marker_color="#3b82f6",
            marker_line_color="#1d4ed8",
            marker_line_width=1.5,
            opacity=0.9
        )
    ])
    fig.update_layout(
        title="Rating Distribution (Stars vs Reviews)",
        xaxis_title="Rating (Stars)",
        yaxis_title="Count of Reviews",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#94a3b8"),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(tickmode="linear", tick0=0.5, dtick=0.5, gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("movie_details.html", movie=details, chart_json=chart_json)


@app.route("/analytics")
def analytics():
    """Interactive Big Data Analytics dashboard with Plotly charts."""
    _, analytics_service, _ = get_services()

    # 1. Genre Analytics
    genre_data = analytics_service.get_genre_analytics()
    top_genres = genre_data[:12]
    fig_genre = px.bar(
        top_genres,
        x="genre",
        y="total_ratings",
        color="avg_rating",
        color_continuous_scale="Viridis",
        title="Top Genres by Rating Volume & Average Score",
        labels={"genre": "Movie Genre", "total_ratings": "Total Ratings", "avg_rating": "Avg Score"}
    )
    fig_genre.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#94a3b8"),
        margin=dict(l=40, r=20, t=60, b=50),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    genre_chart_json = json.dumps(fig_genre, cls=plotly.utils.PlotlyJSONEncoder)

    # 2. Rating Distribution
    dist_data = analytics_service.get_rating_distribution()
    fig_dist = px.bar(
        dist_data,
        x="rating",
        y="count",
        title="Global Rating Distribution across All Movies",
        labels={"rating": "Star Rating (0.5 - 5.0)", "count": "Frequency"}
    )
    fig_dist.update_traces(marker_color="#f59e0b")
    fig_dist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#94a3b8"),
        margin=dict(l=40, r=20, t=60, b=50),
        xaxis=dict(tickmode="linear", tick0=0.5, dtick=0.5, gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    dist_chart_json = json.dumps(fig_dist, cls=plotly.utils.PlotlyJSONEncoder)

    # 3. Rating Trends Over Time
    trend_data = analytics_service.get_rating_trends_over_time()
    fig_trend = px.line(
        trend_data,
        x="review_year",
        y="ratings_count",
        markers=True,
        title="Cluster Review Volume Over Time (1996 - Present)",
        labels={"review_year": "Year of Review", "ratings_count": "Total Ratings Logged"}
    )
    fig_trend.update_traces(line_color="#ec4899", line_width=3)
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#94a3b8"),
        margin=dict(l=40, r=20, t=60, b=50),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    trend_chart_json = json.dumps(fig_trend, cls=plotly.utils.PlotlyJSONEncoder)

    # 4. Most Popular Movies Table & Active Users
    most_popular = analytics_service.get_most_popular_movies(limit=10)
    active_users = analytics_service.get_most_active_users(limit=10)

    return render_template(
        "analytics.html",
        genre_chart=genre_chart_json,
        dist_chart=dist_chart_json,
        trend_chart=trend_chart_json,
        most_popular=most_popular,
        active_users=active_users
    )


@app.route("/cluster-demo")
def cluster_demo():
    """INT315 Cluster Computing RDD, DAG Lineage, and Catalyst Optimizer Demonstration."""
    _, analytics_service, rdd_service = get_services()

    # RDD Word count demo
    tag_rdd_demo = rdd_service.demonstrate_tag_word_count()

    # RDD key-value average ratings demo
    rdd_ratings_demo = rdd_service.demonstrate_rdd_average_ratings()

    # Performance benchmark: RDD vs Catalyst DataFrame
    perf_benchmark = rdd_service.compare_rdd_vs_dataframe_performance()

    # Catalyst Optimizer Execution Plan Inspection
    execution_plans = analytics_service.inspect_execution_plans()

    return render_template(
        "cluster_demo.html",
        tag_demo=tag_rdd_demo,
        ratings_demo=rdd_ratings_demo,
        perf_benchmark=perf_benchmark,
        execution_plans=execution_plans
    )


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/recommend/<int:user_id>")
def api_recommend(user_id: int):
    """API endpoint for asynchronous recommendation retrieval."""
    top_n = request.args.get("n", default=10, type=int)
    service, _, _ = get_services()
    result = service.recommend_for_user(user_id=user_id, top_n=top_n)
    return jsonify(result)


@app.route("/api/search")
def api_search():
    """API endpoint for movie search."""
    q = request.args.get("q", default="", type=str)
    genre = request.args.get("genre", default=None, type=str)
    service, _, _ = get_services()
    results = service.search_movies(query=q, genre=genre, limit=10)
    return jsonify(results)


@app.route("/api/stats")
def api_stats():
    """API endpoint returning dataset summary stats."""
    _, analytics_service, _ = get_services()
    return jsonify(analytics_service.get_overview_statistics())


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html", message="The requested resource was not found."), 404


if __name__ == "__main__":
    print(f"[*] Starting INT315 Spark Movie Recommendation Web Service on http://{config.WEB_HOST}:{config.WEB_PORT}")
    app.run(host=config.WEB_HOST, port=config.WEB_PORT, debug=config.DEBUG_MODE)
