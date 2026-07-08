import json, gzip
import dlt

from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, BooleanType, IntegerType, BinaryType, ArrayType
)
from pyspark.sql.functions import udf, col, current_timestamp
from pyspark.sql import functions as F
from pyspark.sql.window import Window


BRONZE_PATH = "/Volumes/main/heqp/bronze"

SILVER_SCHEMA = StructType([
    StructField("episode_id",           StringType(),  True),
    StructField("robot_id",             StringType(),  True),
    StructField("operator_id",          StringType(),  True),
    StructField("task_type",            StringType(),  True),
    StructField("status",               StringType(),  True),
    StructField("duration_ms",          FloatType(),   True),
    StructField("frame_count",          IntegerType(), True),
    StructField("injected_failure",     StringType(),  True),
    StructField("sensor_completeness",  FloatType(),   True),
    StructField("temporal_coherence",   FloatType(),   True),
    StructField("motion_smoothness",    FloatType(),   True),
    StructField("task_completion",      FloatType(),   True),
    StructField("trajectory_validity",  FloatType(),   True),
    StructField("composite_score",      FloatType(),   True),
    StructField("routing_decision",     StringType(),  False),
    StructField("failure_flags", ArrayType(StringType()), True),
    StructField("scoring_latency_ms",   FloatType(),   True),
])

@udf(returnType=SILVER_SCHEMA)
def extract_payload(payload):
    if not payload:
        return None
    try:
        ep = json.loads(gzip.decompress(bytes(payload)).decode("utf-8"))
    except Exception:
        return None
    score = ep.get("_score", {})
    if not score:
        return None
    return {
        "episode_id":          ep.get("episode_id"),
        "robot_id":            ep.get("robot_id"),
        "operator_id":         ep.get("operator_id"),
        "task_type":           ep.get("task_type"),
        "status":              ep.get("status"),
        "duration_ms":         float(ep.get("duration_ms", 0.0)),
        "frame_count":         int(score.get("frame_count", 0)),
        "injected_failure":    ep.get("injected_failure", "none"),
        "sensor_completeness": float(score.get("score_sensor_completeness", 0.0)),
        "temporal_coherence":  float(score.get("score_temporal_coherence", 0.0)),
        "motion_smoothness":   float(score.get("score_motion_smoothness", 0.0)),
        "task_completion":     float(score.get("score_task_completion", 0.0)),
        "trajectory_validity": float(score.get("score_trajectory_validity", 0.0)),
        "composite_score":     float(score.get("composite_score", 0.0)),
        "routing_decision":    score.get("routing_decision", "REJECTED"),
        "failure_flags":       score.get("failure_flags", []),
        "scoring_latency_ms":  float(score.get("scoring_latency_ms", 0.0)),
    }

@dlt.view
def bronze_raw():
    return spark.read.format("delta").load(BRONZE_PATH)


@dlt.table(
    name="silver",
    comment="Structured episodes extracted from Bronze compressed payloads.",
    partition_cols=["routing_decision"],
)
@dlt.expect_or_drop("valid_episode_id", "episode_id IS NOT NULL")
@dlt.expect_or_drop("valid_routing",    "routing_decision IN ('CERTIFIED', 'BORDERLINE', 'REJECTED')")
@dlt.expect("valid_score",              "composite_score BETWEEN 0 AND 100")
def silver():
    bronze = dlt.read("bronze_raw")
    return (
        bronze
        .withColumn("parsed", extract_payload(col("payload").cast(BinaryType())))
        .filter(col("parsed").isNotNull())
        .select(
            col("parsed.*"),
            col("ingested_at"),
            current_timestamp().cast(StringType()).alias("scored_at"),
        )
    )

@dlt.table(
    name="gold_daily_robot_stats",
    comment="Pass rate, avg score and episode volume per robot per day",
)
def gold_daily_robot_stats():
    silver = dlt.read("silver")
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
# Gold 2 — Failure summary
# ---------------------------------------------------------------------------
@dlt.table(
    name="gold_failure_summary",
    comment="Failure flag frequency by task type and routing decision.",
)
def gold_failure_summary():
    silver = dlt.read("silver")
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
# Gold 3 — SLA compliance
# ---------------------------------------------------------------------------
@dlt.table(
    name="gold_sla_compliance",
    comment="Daily CERTIFIED rate with rolling 7-day trend and SLA status.",
)
def gold_sla_compliance():
    silver = dlt.read("silver")
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