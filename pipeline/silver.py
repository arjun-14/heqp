"""
warehouse/pipelines/silver.py
------------------------------
HEQP Silver Layer

Parses compressed JSON payloads from Bronze and validates episode data.
"""

import json
import gzip
import dlt

from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType, BinaryType, ArrayType
)
from pyspark.sql.functions import udf, col, current_timestamp


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

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
    StructField("failure_flags",        ArrayType(StringType()), True),
    StructField("scoring_latency_ms",   FloatType(),   True),
])


# ---------------------------------------------------------------------------
# Payload extraction UDF
# ---------------------------------------------------------------------------

@udf(returnType=SILVER_SCHEMA)
def extract_payload(payload):
    """
    Decompress and parse gzipped JSON payload from Bronze.
    
    Returns structured episode data with quality scores and routing decision.
    """
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


# ---------------------------------------------------------------------------
# Silver streaming table
# ---------------------------------------------------------------------------

@dlt.table(
    name="silver_episodes",
    comment="Structured episodes extracted from Bronze compressed payloads.",
    partition_cols=["routing_decision"],
)
@dlt.expect_or_drop("valid_episode_id", "episode_id IS NOT NULL")
@dlt.expect_or_drop("valid_routing",    "routing_decision IN ('CERTIFIED', 'BORDERLINE', 'REJECTED')")
@dlt.expect("valid_score",              "composite_score BETWEEN 0 AND 100")
def silver_episodes():
    """
    Parse and validate episode data from Bronze Event Hubs stream.
    
    Transforms:
    - Decompress gzipped JSON payloads
    - Extract episode metadata and quality scores
    - Add processing timestamp
    - Validate routing decisions and scores
    
    Returns streaming DataFrame partitioned by routing_decision.
    """
    bronze = dlt.read_stream("bronze_episodes")
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
