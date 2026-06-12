"""
ingestion/consumer.py
HEQP — Validation consumer.

Reads episodes back from Azure Event Hubs to confirm the producer
wrote correctly. Not the production consumer.

Usage:
    python -m ingestion.consumer --episodes 50 --timeout 30
"""

from __future__ import annotations

import json
import gzip
import logging
import os
import sys
import time
import argparse
from datetime import datetime, timezone

from azure.eventhub import EventHubConsumerClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("heqp.consumer")


def validate_events(
    connection_str: str,
    eventhub_name: str = "heqp-episodes",
    max_events: int = 50,
    timeout_s: int = 30,
) -> dict:
    """
    Read up to `max_events` events from all partitions.
    Returns a validation summary.
    """
    consumer = EventHubConsumerClient.from_connection_string(
        conn_str=connection_str,
        consumer_group="$Default",
        eventhub_name=eventhub_name,
    )

    received: list[dict] = []
    errors: list[str] = []
    start = time.time()

    def on_event(partition_context, event):
        if time.time() - start > timeout_s or len(received) >= max_events:
            return

        try:
            # Safely extract raw bytes from the Azure EventData object
            body_content = event.body
            raw_bytes = body_content if isinstance(body_content, bytes) else b"".join(body_content)
            
            try:
                # Attempt to decompress
                decompressed = gzip.decompress(raw_bytes)
                payload = json.loads(decompressed.decode("utf-8"))
            except gzip.BadGzipFile:
                # Fallback: if it fails to decompress, assume it's an older uncompressed message
                payload = json.loads(event.body_as_str())

            received.append(payload)

            ep_id = payload.get("episode_id", "?")
            decision = payload.get("_score", {}).get("routing_decision", "no-score")
            composite = payload.get("_score", {}).get("composite_score", "?")
            n_frames = len(payload.get("sensor_frames", []))

            logger.info(
                "✓ partition=%s  ep=%s  frames=%d  score=%.1f  decision=%s",
                partition_context.partition_id,
                ep_id,
                n_frames,
                float(composite) if composite != "?" else 0.0,
                decision,
            )
            partition_context.update_checkpoint()

        except Exception as exc:
            errors.append(str(exc))
            logger.error("Deserialisation error: %s", exc)

    def on_error(partition_context, error):
        logger.error("Consumer error (partition %s): %s",
                     partition_context.partition_id if partition_context else "?", error)

    logger.info("Consumer starting — reading up to %d events (timeout %ds)…",
                max_events, timeout_s)

    with consumer:
        consumer.receive(
            on_event=on_event,
            on_error=on_error,
            starting_position="-1",   # beginning of stream
            max_wait_time=timeout_s,
        )

    # Summarise what we got
    decisions = [r.get("_score", {}).get("routing_decision", "UNKNOWN") for r in received]
    summary = {
        "received": len(received),
        "errors": len(errors),
        "certified": decisions.count("CERTIFIED"),
        "borderline": decisions.count("BORDERLINE"),
        "rejected": decisions.count("REJECTED"),
        "schema_valid": all(
            "episode_id" in r and "sensor_frames" in r and "_score" in r
            for r in received
        ),
    }

    print("\n─── HEQP Consumer Validation ───────────────────────────────")
    print(f"  Events received     : {summary['received']}")
    print(f"  Deserialisation err : {summary['errors']}")
    print(f"  Schema valid        : {summary['schema_valid']}")
    print(f"  Certified           : {summary['certified']}")
    print(f"  Borderline          : {summary['borderline']}")
    print(f"  Rejected            : {summary['rejected']}")
    print("────────────────────────────────────────────────────────────\n")

    return summary


def main():
    import sys
    
    # Allow running from repo root so we can import config
    #sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config.azure import get_eventhubs_connection_string, EVENTHUB_NAME

    parser = argparse.ArgumentParser(description="HEQP Validation Consumer")
    parser.add_argument(
        "--connection-string",
        default=None,
    )
    parser.add_argument("--eventhub", default=EVENTHUB_NAME)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    try:
        conn_str = args.connection_string or get_eventhubs_connection_string()
    except EnvironmentError as e:
        logger.error(e)
        sys.exit(1)

    validate_events(
        connection_str=conn_str,
        eventhub_name=args.eventhub,
        max_events=args.episodes,
        timeout_s=args.timeout,
    )


if __name__ == "__main__":
    main()