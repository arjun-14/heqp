from pyspark.sql.functions import col, udf, current_timestamp
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, BooleanType, IntegerType, BinaryType
)
import json
import gzip

BRONZE_PATH = "/Volumes/main/heqp/bronze"
SILVER_PATH = "/Volumes/main/heqp/silver"

# ---------------------------------------------------------------------------
# Silver schema — flat structured columns extracted from raw Bronze payload
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
    # Score dimensions
    StructField("sensor_completeness",  FloatType(),   True),
    StructField("temporal_coherence",   FloatType(),   True),
    StructField("motion_smoothness",    FloatType(),   True),
    StructField("task_completion",      FloatType(),   True),
    StructField("trajectory_validity",  FloatType(),   True),
    StructField("composite_score",      FloatType(),   True),
    # Routing
    StructField("routing_decision",     StringType(),  False),
    StructField("failure_flags",        StringType(),  True),
    StructField("scoring_latency_ms",   FloatType(),   True),
])

# ---------------------------------------------------------------------------
# UDF — parse Bronze payload, extract top-level fields + _score
# ---------------------------------------------------------------------------

@udf(returnType=SILVER_SCHEMA)
def extract_payload(payload):
    if not payload:
        return None
    try:
        ep = json.loads(gzip.decompress(payload).decode("utf-8"))
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
        "failure_flags":       ",".join(score.get("failure_flags", [])),
        "scoring_latency_ms":  float(score.get("scoring_latency_ms", 0.0)),
    }


# ---------------------------------------------------------------------------
# Read Bronze, extract, write Silver
# ---------------------------------------------------------------------------

bronze_df = spark.read.format("delta").load(BRONZE_PATH)
print(f"Bronze rows: {bronze_df.count()}")

silver_df = (
    bronze_df
    .withColumn("parsed", extract_payload(col("payload").cast(BinaryType())))
    .filter(col("parsed").isNotNull())
    .select(
        col("parsed.*"),
        col("ingested_at"),
        current_timestamp().cast(StringType()).alias("scored_at"),
    )
)


silver_df.write.format("delta").mode("append").partitionBy("routing_decision").save(SILVER_PATH)
print(f"Silver rows: {silver_df.count()}")
silver_df.groupBy("routing_decision").count().orderBy("routing_decision").show()

