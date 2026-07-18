# HEQP — Helix Episode Quality Platform

A real-time data pipeline for scoring and analyzing teleoperation episodes from robotic systems. Episodes are streamed from Azure Event Hubs, processed through a medallion architecture on Databricks, and surfaced as business-level metrics.

## Architecture

```
Simulator → Producer → Azure Event Hubs → Bronze → Silver → Gold
                                          (Delta Lake / Databricks DLT)
```

| Layer | Description |
|---|---|
| **Simulator** | Generates synthetic teleoperation episodes with configurable failure modes |
| **Scoring** | Five-dimension quality engine; classifies episodes as Certified / Borderline / Rejected |
| **Ingestion** | Kafka-compatible producer streams scored episodes into Azure Event Hubs |
| **Bronze** | Streaming table — raw payloads landed as-is with lineage metadata |
| **Silver** | Parses and validates compressed JSON; applies schema enforcement |
| **Gold** | Business aggregations: daily robot stats, operator metrics, task-type summaries |

## Project Structure

```
heqp/
├── simulator/      # EpisodeSimulator — synthetic data generation
├── scoring/        # scorer.py — composite quality scoring engine
├── ingestion/      # producer.py / consumer.py — Event Hubs I/O
├── pipeline/       # Databricks DLT pipeline (bronze → silver → gold)
│   ├── bronze.py
│   ├── silver.py
│   └── gold.py
├── config/         # Azure credentials and environment config
└── src/            # Local test utilities
```

## Scoring

Episodes are scored across five dimensions:

| Dimension | Weight |
|---|---|
| Sensor completeness | 20% |
| Temporal coherence | 25% |
| Motion smoothness | 25% |
| Task completion | 20% |
| Operator behavior | 10% |

**Routing thresholds:**
- **Certified** — composite ≥ 85
- **Borderline** — 70 ≤ composite < 85
- **Rejected** — composite < 70

Hard overrides cap any episode at Borderline (69.9) regardless of composite score when: task completion = 0, a trajectory anomaly jump is detected, or motion smoothness < 45.

## Setup

### 1. Azure Event Hubs

Store the connection string in Databricks Secrets:

```bash
databricks secrets create-scope --scope event-hubs
databricks secrets put --scope event-hubs --key connection-string
```

### 2. Unity Catalog

```sql
CREATE CATALOG IF NOT EXISTS main;
CREATE SCHEMA IF NOT EXISTS main.heqp;
```

### 3. DLT Pipeline

Deploy `pipeline/` as a Databricks Delta Live Tables pipeline. Enable **continuous mode** for real-time ingestion.

