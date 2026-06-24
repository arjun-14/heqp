"""
warehouse/bronze.py
-------------------
HEQP Bronze Layer — Spark Structured Streaming

Reads raw teleoperation episodes from Azure Event Hubs (Kafka-compatible API)
and lands them as-is into a Delta Lake Bronze table.

No transformation happens here. Bronze = exact copy of what was ingested,
with metadata columns added for lineage and debugging.

Setup:
    1. Store Event Hubs connection string in Databricks Secrets:
       databricks secrets create-scope --scope event-hubs
       databricks secrets put --scope event-hubs --key connection-string
    
    2. Create Unity Catalog volumes:
       CREATE CATALOG IF NOT EXISTS main;
       CREATE SCHEMA IF NOT EXISTS main.heqp;
       CREATE VOLUME IF NOT EXISTS main.heqp.bronze;
       CREATE VOLUME IF NOT EXISTS main.heqp.checkpoints;
"""

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp
from pyspark.sql.types import StringType, StructType, StructField

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Unity Catalog volume paths for Databricks
BRONZE_PATH = os.environ.get("HEQP_BRONZE_PATH", "/Volumes/main/heqp/bronze")
CHECKPOINT_PATH = os.environ.get("HEQP_CHECKPOINT_BRONZE", "/Volumes/main/heqp/checkpoints/bronze")

# ---------------------------------------------------------------------------
# Event Hubs config (Kafka-compatible)
# ---------------------------------------------------------------------------

def get_eventhubs_kafka_config(spark: SparkSession) -> dict:
    """
    Build Spark Kafka source options for Azure Event Hubs.
    Retrieves connection string from Databricks Secrets.

    Uses kafkashaded package (Databricks Runtime shades the Kafka client).
    
    Args:
        spark: SparkSession for accessing dbutils.secrets
    
    Returns:
        Dictionary of Kafka connection options
        
    Raises:
        Exception: If connection string cannot be retrieved from secrets
    """
    # Get connection string from Databricks Secrets
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)
    conn_str = dbutils.secrets.get(scope="event-hubs", key="connection-string")
    print("[Event Hubs] Using connection string from Databricks Secrets")

    # Extract namespace from connection string
    # Format: Endpoint=sb://<namespace>.servicebus.windows.net/;...
    namespace = conn_str.split("Endpoint=sb://")[1].split(".servicebus.windows.net")[0]
    bootstrap_server = f"{namespace}.servicebus.windows.net:9093"

    # Databricks Runtime uses kafkashaded package
    login_module = "kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule"

    # Encode connection string for SASL JAAS config
    sasl_config = (
        f"{login_module} required "
        f'username="$ConnectionString" password="{conn_str}";'
    )

    return {
        "kafka.bootstrap.servers": bootstrap_server,
        "kafka.security.protocol": "SASL_SSL",
        "kafka.sasl.mechanism": "PLAIN",
        "kafka.sasl.jaas.config": sasl_config,
        "kafka.request.timeout.ms": "60000",
        "kafka.session.timeout.ms": "30000",
        "subscribe": "heqp-episodes",
        "startingOffsets": "latest",
        "failOnDataLoss": "false",
    }


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------

def get_spark() -> SparkSession:
    """
    Get the active Databricks-managed SparkSession.
    Delta Lake is built into Databricks Runtime.
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(
            "No active SparkSession found. "
            "This script must be run in a Databricks environment."
        )
    print("[Spark] Using Databricks-managed SparkSession")
    return spark


# ---------------------------------------------------------------------------
# Bronze schema
#
# We read everything as raw strings from Kafka and store the full
# episode JSON as a single `payload` column. This preserves the original
# message exactly — Bronze is append-only, never modified.
# ---------------------------------------------------------------------------

BRONZE_SCHEMA = StructType([
    StructField("payload", StringType(), nullable=False),  # raw episode JSON
    StructField("ingested_at", StringType(), nullable=True),
    StructField("source", StringType(), nullable=True),
])


# ---------------------------------------------------------------------------
# Streaming job
# ---------------------------------------------------------------------------

def run_eventhubs_stream(spark: SparkSession) -> None:
    """Read from Azure Event Hubs via Kafka, write to Bronze Delta table."""
    print("[Bronze] Starting Event Hubs stream...")

    kafka_options = get_eventhubs_kafka_config(spark)

    raw_stream = (
        spark.readStream
        .format("kafka")
        .options(**kafka_options)
        .load()
    )

    # Kafka gives us: key, value (bytes), topic, partition, offset, timestamp
    # value is the episode JSON, encoded as bytes
    bronze_stream = (
        raw_stream
        .select(
            col("value").cast(StringType()).alias("payload"),
            current_timestamp().alias("ingested_at"),
            col("topic").alias("source"),
            col("partition").alias("kafka_partition"),
            col("offset").alias("kafka_offset"),
            col("timestamp").alias("kafka_timestamp"),
        )
    )

    query = (
        bronze_stream.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT_PATH)
        .option("path", BRONZE_PATH)
        .trigger(availableNow=True)
        .start()
    )

    print(f"[Bronze] Streaming to {BRONZE_PATH}")
    print("[Bronze] Stream is running. Use notebook controls to stop.")
    query.awaitTermination()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    """Main entry point for the Bronze streaming job."""
    spark = get_spark()
    run_eventhubs_stream(spark)


if __name__ == "__main__":
    main()
