"""
ingestion/producer.py
HEQP — Helix Episode Quality Platform

Kafka-compatible producer that streams scored episodes from EpisodeSimulator
into Azure Event Hubs. Uses the Kafka-compatible API (port 9093).

Target throughput: 10,000+ episodes/hour (~2.8 episodes/second).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from typing import Optional

from azure.eventhub import EventData, EventHubProducerClient
from azure.eventhub.exceptions import EventHubError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("heqp.producer")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _episode_to_dict(episode) -> dict:
    """
    Convert an Episode dataclass to a JSON-serialisable dict.
    Handles nested dataclasses, enums, and NaN sensor frames.
    """
    def _clean(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _clean(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [_clean(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if hasattr(obj, "value"):          # Enum
            return obj.value
        if isinstance(obj, float) and (obj != obj):  # NaN check
            return None
        return obj

    return _clean(episode)


def episode_to_json(episode, score_result: Optional[dict] = None) -> bytes:
    """
    Serialise episode + optional score result to UTF-8 JSON bytes.
    This is the wire format written to Event Hubs.
    """
    payload = _episode_to_dict(episode)
    if score_result is not None:
        payload["_score"] = score_result
    return json.dumps(payload, default=str).encode("utf-8")


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------

class HEQPProducer:
    """
    Wraps an Azure EventHubProducerClient and provides episode-aware
    send methods with batching, metrics, and error handling.

    Parameters
    ----------
    connection_str : str
        Azure Event Hubs connection string (primary key).
    eventhub_name : str
        Name of the Event Hub (default: "heqp-episodes").
    batch_size : int
        Max episodes per EventDataBatch before flushing (default: 50).
        Azure Event Hubs max batch size is 1 MB; 50 episodes is well under.
    """

    def __init__(
        self,
        connection_str: str,
        eventhub_name: str = "heqp-episodes",
        batch_size: int = 50,
    ):
        self.connection_str = connection_str
        self.eventhub_name = eventhub_name
        self.batch_size = batch_size

        self._client: Optional[EventHubProducerClient] = None

        # Metrics
        self._sent_total = 0
        self._failed_total = 0
        self._batch_count = 0
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "HEQPProducer":
        self._client = EventHubProducerClient.from_connection_string(
            conn_str=self.connection_str,
            eventhub_name=self.eventhub_name,
        )
        self._start_time = time.perf_counter()
        logger.info(
            "Producer connected → Event Hub '%s'", self.eventhub_name
        )
        return self

    def __exit__(self, *_):
        if self._client:
            self._client.close()
        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        eps = self._sent_total / elapsed if elapsed > 0 else 0
        logger.info(
            "Producer closed. sent=%d  failed=%d  batches=%d  rate=%.1f eps  elapsed=%.1fs",
            self._sent_total,
            self._failed_total,
            self._batch_count,
            eps,
            elapsed,
        )

    # ------------------------------------------------------------------
    # Bulk streaming
    # ------------------------------------------------------------------

    def stream_episodes(
        self,
        simulator,
        scorer,
        n_episodes: int = 1000,
        log_every: int = 100,
    ) -> dict:
        """
        Pull episodes from simulator, score them, and stream into Event Hubs.

        Parameters
        ----------
        simulator : EpisodeSimulator
            Configured simulator instance.
        scorer : EpisodeScorer
            Scoring engine instance.
        n_episodes : int
            Number of episodes to generate and send.
        log_every : int
            Log a progress line every N episodes.

        Returns
        -------
        dict
            Summary metrics dict.
        """
        if self._client is None:
            raise RuntimeError("Producer not started — use as context manager.")

        certified = borderline = rejected = 0
        latencies: list[float] = []

        current_batch_events: list[EventData] = []

        def _flush_batch(events: list[EventData]):
            """Send accumulated events as one EventDataBatch (round-robin balanced)."""
            try:
                # No partition key! Azure will round-robin the whole batch to one partition.
                batch = self._client.create_batch()
                for ev in events:
                    batch.add(ev)
                self._client.send_batch(batch)
                self._sent_total += len(events)
                self._batch_count += 1
            except EventHubError as exc:
                logger.error("Batch send failed: %s", exc)
                self._failed_total += len(events)

        stream_start = time.perf_counter()

        for i in range(1, n_episodes + 1):
            t0 = time.perf_counter()

            # 1. Generate episode
            episode = simulator.generate_episode()

            # 2. Score
            score_result = scorer.score(episode.to_json())

            # 3. Track routing
            decision = score_result.routing_decision
            if decision == "CERTIFIED":
                certified += 1
            elif decision == "BORDERLINE":
                borderline += 1
            else:
                rejected += 1

            # 4. Serialise → EventData
            payload = episode_to_json(episode, score_result.to_dict())
            event = EventData(payload)
            current_batch_events.append(event)

            # 5. Flush when batch is full
            if len(current_batch_events) >= self.batch_size:
                _flush_batch(current_batch_events)
                current_batch_events = []

            latencies.append((time.perf_counter() - t0) * 1000)

            if i % log_every == 0:
                elapsed = time.perf_counter() - stream_start
                rate = i / elapsed
                logger.info(
                    "[%d/%d] rate=%.1f eps | certified=%d borderline=%d rejected=%d",
                    i, n_episodes, rate, certified, borderline, rejected,
                )

        # Flush remainder
        if current_batch_events:
            _flush_batch(current_batch_events)

        elapsed_total = time.perf_counter() - stream_start
        throughput = n_episodes / elapsed_total

        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p99 = latencies[int(len(latencies) * 0.99)]

        summary = {
            "n_episodes": n_episodes,
            "sent": self._sent_total,
            "failed": self._failed_total,
            "certified": certified,
            "borderline": borderline,
            "rejected": rejected,
            "elapsed_s": round(elapsed_total, 2),
            "throughput_eps": round(throughput, 1),
            "throughput_eph": round(throughput * 3600, 0),
            "latency_p50_ms": round(p50, 2),
            "latency_p99_ms": round(p99, 2),
        }

        logger.info("Stream complete: %s", summary)
        return summary


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    import argparse
    import sys
    import os

    # Allow running from repo root
    #sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from simulator.simulator import EpisodeSimulator
    from scoring.scorer import EpisodeScoringEngine
    from config.azure import get_eventhubs_connection_string, EVENTHUB_NAME

    parser = argparse.ArgumentParser(description="HEQP Kafka Producer")
    parser.add_argument(
        "--connection-string",
        default=None,
        help="Azure Event Hubs connection string (overrides .env if provided)",
    )
    parser.add_argument("--eventhub", default=EVENTHUB_NAME)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    # Use CLI arg first, fallback to the config loader
    try:
        conn_str = args.connection_string or get_eventhubs_connection_string()
    except EnvironmentError as e:
        logger.error(e)
        sys.exit(1)

    simulator = EpisodeSimulator()
    scorer = EpisodeScoringEngine()

    with HEQPProducer(
        connection_str=conn_str,
        eventhub_name=args.eventhub,
        batch_size=args.batch_size,
    ) as producer:
        summary = producer.stream_episodes(
            simulator=simulator,
            scorer=scorer,
            n_episodes=args.episodes,
            log_every=args.log_every,
        )

    # Print final summary table
    print("\n─── HEQP Producer Benchmark ───────────────────────────────")
    print(f"  Episodes sent       : {summary['sent']:,}")
    print(f"  Failed              : {summary['failed']:,}")
    print(f"  Certified           : {summary['certified']:,}  ({summary['certified']/summary['n_episodes']*100:.1f}%)")
    print(f"  Borderline          : {summary['borderline']:,}  ({summary['borderline']/summary['n_episodes']*100:.1f}%)")
    print(f"  Rejected            : {summary['rejected']:,}  ({summary['rejected']/summary['n_episodes']*100:.1f}%)")
    print(f"  Elapsed             : {summary['elapsed_s']}s")
    print(f"  Throughput          : {summary['throughput_eps']} eps  ({summary['throughput_eph']:,.0f} ep/hr)")
    print(f"  Latency p50 / p99   : {summary['latency_p50_ms']}ms / {summary['latency_p99_ms']}ms")
    print("────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()