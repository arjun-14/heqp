"""
warehouse/bronze.py
-------------------
HEQP Bronze Layer — Spark Declarative Pipeline (Streaming Table)

Reads raw teleoperation episodes from Azure Event Hubs (Kafka-compatible API)
and lands them as-is into a Delta Lake Bronze streaming table.

No transformation happens here. Bronze = exact copy of what was ingested,
with metadata columns added for lineage and debugging.

Setup:
    1. Store Event Hubs connection string in Databricks Secrets:
       databricks secrets create-scope --scope event-hubs
       databricks secrets put --scope event-hubs --key connection-string
    
    2. Ensure Unity Catalog destination exists (pipeline writes to main.heqp):
       CREATE CATALOG IF NOT EXISTS main;
       CREATE SCHEMA IF NOT EXISTS main.heqp;
    
    3. Enable continuous mode in pipeline settings for real-time ingestion:
       continuous: true
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.dbutils import DBUtils

# ---------------------------------------------------------------------------
# Event Hubs config (Kafka-compatible)
# ---------------------------------------------------------------------------

def get_eventhubs_kafka_config() -> dict:
    """
    Build Spark Kafka source options for Azure Event Hubs.
    Retrieves connection string from Databricks Secrets.

    Uses kafkashaded package (Databricks Runtime shades the Kafka client).
    
    Returns:
        Dictionary of Kafka connection options
        
    Raises:
        Exception: If connection string cannot be retrieved from secrets
    """
    dbutils = DBUtils(spark)
    conn_str = dbutils.secrets.get(scope="event-hubs", key="connection-string")

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
# Bronze streaming table
# ---------------------------------------------------------------------------

@dp.table(
    comment="Raw teleoperation episodes from Azure Event Hubs",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.managed": "true"
    }
)
def bronze_episodes():
    """
    Ingest raw episode JSON from Event Hubs into Bronze.
    
    Returns a streaming DataFrame with columns:
    - payload: Raw gzipped episode bytes (BinaryType)
    - ingested_at: Timestamp when row entered the pipeline
    - source: Event Hubs topic name
    - kafka_partition: Kafka partition ID
    - kafka_offset: Kafka offset within partition
    - kafka_timestamp: Record timestamp from Event Hubs
    """
    kafka_options = get_eventhubs_kafka_config()
    
    raw_stream = (
        spark.readStream
        .format("kafka")
        .options(**kafka_options)
        .load()
    )

    # Kafka gives us: key, value (bytes), topic, partition, offset, timestamp
    # Store value as-is (BinaryType) - no conversion, pure raw data
    return (
        raw_stream
        .select(
            F.col("value").alias("payload"),
            F.current_timestamp().alias("ingested_at"),
            F.col("topic").alias("source"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
            F.col("timestamp").alias("kafka_timestamp"),
        )
    )
