"""
warehouse/pipelines/gold.py
----------------------------
HEQP Gold Layer

Business-level aggregations and metrics for analytics and reporting.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ---------------------------------------------------------------------------
# Gold 1 — Daily Robot Statistics
# ---------------------------------------------------------------------------

@dp.table(
    name="gold_daily_robot_stats",
    comment="Pass rate, avg score and episode volume per robot per day",
)
def gold_daily_robot_stats():
    """
    Daily performance metrics by robot and task type.
    
    Aggregates:
    - Episode counts by routing decision
    - Composite score statistics (avg, min, max, percentiles)
    - Certification rate percentage
    """
    silver = spark.read.table("silver_episodes")
    return (
        silver
        .withColumn("report_date", F.to_date("ingested_at"))
        .groupBy("report_date", "robot_id", "task_type")
        .agg(
            F.count("*").alias("total_episodes"),
            F.round(F.avg("composite_score"), 2).alias("avg_composite_score"),
            F.round(F.min("composite_score"), 2).alias("min_score"),
            F.round(F.max("composite_score"), 2).alias("max_score"),
            F.round(F.expr("percentile(composite_score, 0.5)"), 2).alias("p50_score"),
            F.round(F.expr("percentile(composite_score, 0.95)"), 2).alias("p95_score"),
            F.sum(F.when(F.col("routing_decision") == "CERTIFIED",  1).otherwise(0)).alias("certified_count"),
            F.sum(F.when(F.col("routing_decision") == "BORDERLINE", 1).otherwise(0)).alias("borderline_count"),
            F.sum(F.when(F.col("routing_decision") == "REJECTED",   1).otherwise(0)).alias("rejected_count"),
            F.round(
                100.0 * F.sum(F.when(F.col("routing_decision") == "CERTIFIED", 1).otherwise(0))
                / F.count("*"), 2
            ).alias("certified_rate_pct"),
        )
        .orderBy("report_date", "robot_id", "task_type")
    )


# ---------------------------------------------------------------------------
# Gold 2 — Failure Summary
# ---------------------------------------------------------------------------

@dp.table(
    name="gold_failure_summary",
    comment="Failure flag frequency by task type and routing decision.",
)
def gold_failure_summary():
    """
    Analysis of failure patterns across episodes.
    
    Explodes failure_flags array and ranks failures by frequency.
    Shows which failures are most common and their impact on scores.
    """
    silver = spark.read.table("silver_episodes")
    exploded = (
        silver
        .filter(F.col("failure_flags").isNotNull() & (F.size("failure_flags") > 0))
        .withColumn("failure_flag", F.explode("failure_flags"))
        .withColumn("failure_flag", F.trim("failure_flag"))
        .withColumn("report_date", F.to_date("ingested_at"))
    )
    return (
        exploded
        .groupBy("report_date", "task_type", "routing_decision", "failure_flag")
        .agg(
            F.countDistinct("episode_id").alias("affected_episodes"),
            F.round(F.avg("composite_score"), 2).alias("avg_score_when_flagged"),
            F.round(F.min("composite_score"), 2).alias("min_score_when_flagged"),
        )
        .withColumn(
            "failure_rank",
            F.row_number().over(
                Window
                .partitionBy("report_date", "task_type")
                .orderBy(F.desc("affected_episodes"))
            )
        )
        .orderBy("report_date", "task_type", "failure_rank")
    )


# ---------------------------------------------------------------------------
# Gold 3 — SLA Compliance
# ---------------------------------------------------------------------------

@dp.table(
    name="gold_sla_compliance",
    comment="Daily CERTIFIED rate with rolling 7-day trend and SLA status.",
)
def gold_sla_compliance():
    """
    SLA monitoring dashboard table.
    
    Tracks:
    - Daily certification rates
    - 7-day rolling averages
    - SLA status (MET ≥70%, AT_RISK 60-70%, BREACHED <60%)
    """
    silver = spark.read.table("silver_episodes")
    daily = (
        silver
        .withColumn("report_date", F.to_date("ingested_at"))
        .groupBy("report_date")
        .agg(
            F.count("*").alias("total_episodes"),
            F.sum(F.when(F.col("routing_decision") == "CERTIFIED",  1).otherwise(0)).alias("certified"),
            F.sum(F.when(F.col("routing_decision") == "BORDERLINE", 1).otherwise(0)).alias("borderline"),
            F.sum(F.when(F.col("routing_decision") == "REJECTED",   1).otherwise(0)).alias("rejected"),
            F.round(F.avg("composite_score"), 2).alias("avg_composite_score"),
            F.round(
                100.0 * F.sum(F.when(F.col("routing_decision") == "CERTIFIED", 1).otherwise(0))
                / F.count("*"), 2
            ).alias("certified_rate_pct"),
        )
    )
    window_7d = Window.orderBy(F.col("report_date").cast("long")).rowsBetween(-6, 0)
    return (
        daily
        .withColumn("rolling_7d_certified_rate_pct",
            F.round(F.avg("certified_rate_pct").over(window_7d), 2))
        .withColumn("rolling_7d_avg_episodes",
            F.round(F.avg("total_episodes").over(window_7d), 0))
        .withColumn("sla_status",
            F.when(F.col("certified_rate_pct") >= 70, "MET")
             .when(F.col("certified_rate_pct") >= 60, "AT_RISK")
             .otherwise("BREACHED")
        )
        .orderBy(F.desc("report_date"))
    )
