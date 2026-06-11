"""
HEQP Quality Scoring Engine
============================
Scores teleoperation episodes across five dimensions and routes them
to CERTIFIED, BORDERLINE, or REJECTED.

This module runs locally (no Spark dependency) so it can be used for
local testing. The production version wraps these same scoring functions
inside a Spark Structured Streaming job on Azure Databricks.

Scoring dimensions and weights:
    Sensor Completeness  25%  — fraction of non-null sensor readings
    Temporal Coherence   25%  — timestamp gap analysis
    Motion Smoothness    20%  — joint velocity RMS delta
    Task Completion      20%  — episode status and phase events
    Trajectory Validity  10%  — joint positions within task envelope

Composite score is 0-100. Routing thresholds:
    >= 85  CERTIFIED    (training-ready)
    70-84  BORDERLINE   (human review required)
    <  70  REJECTED     (excluded from training)
"""

from __future__ import annotations
import json
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

CERTIFIED_THRESHOLD  = 85.0
BORDERLINE_THRESHOLD = 70.0

DIMENSION_WEIGHTS = {
    "sensor_completeness": 0.25,
    "temporal_coherence":  0.25,
    "motion_smoothness":   0.20,
    "task_completion":     0.20,
    "trajectory_validity": 0.10,
}

SENSOR_HZ = 200
EXPECTED_INTERVAL_NS = int(1e9 / SENSOR_HZ)   # 5,000,000 ns
MAX_ALLOWED_GAP_NS   = 50_000_000              # 50ms
JITTER_THRESHOLD_RMS = 1.8                     # rad/s
COMPLETENESS_FLOOR   = 0.995                   # 99.5% non-null frames required
TRAJECTORY_TOLERANCE = 0.02                    # max fraction of out-of-envelope frames


# ── Score result ──────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    episode_id:       str
    robot_id:         str
    operator_id:      str
    task_type:        str
    frame_count:      int
    duration_ms:      float

    # Dimension scores (0-100 each)
    sensor_completeness: float = 0.0 # measures whether any data went missing
    temporal_coherence:  float = 0.0 # measures timining consistency of frames    
    motion_smoothness:   float = 0.0 # measures how smooth the joint velocities are (jerky vs smooth)
    task_completion:     float = 0.0 # measures whether the episode reached a complete phase and terminal completed status
    trajectory_validity: float = 0.0 # measures whether the joint positions are within the task envelope

    # Composite
    composite_score:    float = 0.0
    routing_decision:   str   = "PENDING"

    # Diagnostics
    failure_flags:      list[str] = field(default_factory=list)
    scoring_latency_ms: float = 0.0
    scored_at_ns:       int   = 0

    # Ground truth (from simulator injection — available for benchmarking)
    injected_failure:   str = "none"

    def to_dict(self) -> dict:
        return {
            "episode_id":           self.episode_id,
            "robot_id":             self.robot_id,
            "operator_id":          self.operator_id,
            "task_type":            self.task_type,
            "frame_count":          self.frame_count,
            "duration_ms":          self.duration_ms,
            "score_sensor_completeness": round(self.sensor_completeness, 2),
            "score_temporal_coherence":  round(self.temporal_coherence, 2),
            "score_motion_smoothness":   round(self.motion_smoothness, 2),
            "score_task_completion":     round(self.task_completion, 2),
            "score_trajectory_validity": round(self.trajectory_validity, 2),
            "composite_score":           round(self.composite_score, 2),
            "routing_decision":          self.routing_decision,
            "failure_flags":             self.failure_flags,
            "scoring_latency_ms":        round(self.scoring_latency_ms, 3),
            "scored_at_ns":              self.scored_at_ns,
            "injected_failure":          self.injected_failure,
        }


# ── Scoring engine ────────────────────────────────────────────────────────────

class EpisodeScoringEngine:
    """
    Scores a teleoperation episode across 5 quality dimensions.

    This is the core quality gate. In production this runs inside
    Spark Structured Streaming on Databricks, one episode per row.
    The logic here is pure Python to enable local testing and unit tests.
    """

    def score(self, episode_json: str) -> ScoreResult:
        """
        Score a single episode from its JSON representation.

        Args:
            episode_json: JSON string as produced by Episode.to_json()

        Returns:
            ScoreResult with all dimension scores, composite, and routing decision
        """
        t0 = time.monotonic()
        ep = json.loads(episode_json)

        frames        = ep.get("sensor_frames", [])
        phase_events  = ep.get("phase_events", [])
        status        = ep.get("status", "aborted")
        task_type     = ep.get("task_type", "pick_place")

        result = ScoreResult(
            episode_id    = ep.get("episode_id", "unknown"),
            robot_id      = ep.get("robot_id", "unknown"),
            operator_id   = ep.get("operator_id", "unknown"),
            task_type     = task_type,
            frame_count   = ep.get("frame_count", 0),
            duration_ms   = ep.get("duration_ms", 0.0),
            injected_failure = ep.get("injected_failure", "none"),
            scored_at_ns  = int(time.time() * 1e9),
        )

        if not frames:
            result.failure_flags.append("NO_FRAMES")
            result.routing_decision = "REJECTED"
            result.scoring_latency_ms = (time.monotonic() - t0) * 1000
            return result

        # ── Run all five scoring dimensions ──
        result.sensor_completeness = self._score_sensor_completeness(frames, result)
        result.temporal_coherence  = self._score_temporal_coherence(frames, result)
        result.motion_smoothness   = self._score_motion_smoothness(frames, result)
        result.task_completion     = self._score_task_completion(status, phase_events, result)
        result.trajectory_validity = self._score_trajectory_validity(frames, task_type, result)

        # ── Composite score ──
        result.composite_score = (
            result.sensor_completeness * DIMENSION_WEIGHTS["sensor_completeness"] +
            result.temporal_coherence  * DIMENSION_WEIGHTS["temporal_coherence"]  +
            result.motion_smoothness   * DIMENSION_WEIGHTS["motion_smoothness"]   +
            result.task_completion     * DIMENSION_WEIGHTS["task_completion"]      +
            result.trajectory_validity * DIMENSION_WEIGHTS["trajectory_validity"]
        )

        # ── Routing decision ──
        if result.composite_score >= CERTIFIED_THRESHOLD:
            result.routing_decision = "CERTIFIED"
        elif result.composite_score >= BORDERLINE_THRESHOLD:
            result.routing_decision = "BORDERLINE"
        else:
            result.routing_decision = "REJECTED"

        result.scoring_latency_ms = (time.monotonic() - t0) * 1000
        return result

    # ── Dimension 1: Sensor Completeness (25%) ────────────────────────────

    def _score_sensor_completeness(self, frames: list[dict], result: ScoreResult) -> float:
        """
        Fraction of non-null readings across joint_positions and joint_velocities.
        A single sensor dropout window of 5+ frames will fail this dimension.
        """
        total_readings  = 0
        null_readings   = 0

        for frame in frames:
            for field_name in ["joint_positions", "joint_velocities"]:
                values = frame.get(field_name, [])
                for v in values:
                    total_readings += 1
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        null_readings += 1

        if total_readings == 0:
            result.failure_flags.append("SENSOR_COMPLETENESS:NO_READINGS")
            return 0.0

        completeness = 1.0 - (null_readings / total_readings)

        if completeness < COMPLETENESS_FLOOR:
            result.failure_flags.append(
                f"SENSOR_COMPLETENESS:DROPOUT({null_readings}/{total_readings} null)"
            )

        # Score maps completeness 0-100 with steep penalty below floor
        if completeness >= COMPLETENESS_FLOOR:
            score = 100.0
        elif completeness >= 0.98:
            score = 80.0 + (completeness - 0.98) / (COMPLETENESS_FLOOR - 0.98) * 20
        elif completeness >= 0.90:
            score = 40.0 + (completeness - 0.90) / 0.08 * 40
        else:
            score = completeness / 0.90 * 40

        return round(max(0.0, min(100.0, score)), 2)

    # ── Dimension 2: Temporal Coherence (25%) ─────────────────────────────

    def _score_temporal_coherence(self, frames: list[dict], result: ScoreResult) -> float:
        """
        Checks timestamp continuity. Any gap > 50ms is a critical failure.
        The Helix System 1 control loop runs at 200Hz, requiring < 5ms intervals.
        A 50ms+ gap breaks temporal coherence of the episode.
        """
        timestamps = [f.get("timestamp_ns") for f in frames if f.get("timestamp_ns") is not None]

        if len(timestamps) < 2:
            result.failure_flags.append("TEMPORAL_COHERENCE:TOO_FEW_TIMESTAMPS")
            return 0.0

        gaps          = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)]
        max_gap_ns    = max(gaps)
        mean_gap_ns   = sum(gaps) / len(gaps)
        bad_gaps      = [g for g in gaps if g > MAX_ALLOWED_GAP_NS]

        if bad_gaps:
            result.failure_flags.append(
                f"TEMPORAL_COHERENCE:GAP({max_gap_ns/1e6:.1f}ms max, {len(bad_gaps)} gaps > 50ms)"
            )

        # Timing deviation from expected 5ms interval
        expected_gap  = EXPECTED_INTERVAL_NS
        deviation_pct = abs(mean_gap_ns - expected_gap) / expected_gap

        if max_gap_ns <= MAX_ALLOWED_GAP_NS and deviation_pct < 0.05:
            score = 100.0
        elif max_gap_ns <= MAX_ALLOWED_GAP_NS:
            score = max(70.0, 100.0 - deviation_pct * 200)
        else:
            # Penalize by magnitude of worst gap
            gap_ratio = min(max_gap_ns / MAX_ALLOWED_GAP_NS, 20.0)
            score = max(0.0, 60.0 - (gap_ratio - 1.0) * 10)

        return round(max(0.0, min(100.0, score)), 2)

    # ── Dimension 3: Motion Smoothness (20%) ──────────────────────────────

    def _score_motion_smoothness(self, frames: list[dict], result: ScoreResult) -> float:
        """
        RMS of joint velocity delta between consecutive frames.
        High RMS indicates motion jitter — jerky trajectories produce
        unsafe policies that inherit the jitter.
        """
        deltas = []

        for i in range(1, len(frames)):
            v_curr = frames[i].get("joint_velocities", [])
            v_prev = frames[i-1].get("joint_velocities", [])

            if not v_curr or not v_prev:
                continue

            for vc, vp in zip(v_curr, v_prev):
                if vc is None or vp is None:
                    continue
                if isinstance(vc, float) and math.isnan(vc):
                    continue
                if isinstance(vp, float) and math.isnan(vp):
                    continue
                deltas.append((vc - vp) ** 2)

        if not deltas:
            result.failure_flags.append("MOTION_SMOOTHNESS:NO_VALID_VELOCITIES")
            return 50.0  # neutral — can't assess

        rms = math.sqrt(sum(deltas) / len(deltas))

        if rms > JITTER_THRESHOLD_RMS:
            result.failure_flags.append(
                f"MOTION_SMOOTHNESS:JITTER(RMS={rms:.3f} rad/s, threshold={JITTER_THRESHOLD_RMS})"
            )

        if rms <= 0.3:
            score = 100.0
        elif rms <= JITTER_THRESHOLD_RMS:
            score = 100.0 - ((rms - 0.3) / (JITTER_THRESHOLD_RMS - 0.3)) * 40
        else:
            score = max(0.0, 60.0 - (rms - JITTER_THRESHOLD_RMS) * 20)

        return round(max(0.0, min(100.0, score)), 2)

    # ── Dimension 4: Task Completion (20%) ───────────────────────────────

    def _score_task_completion(
        self, status: str, phase_events: list[dict], result: ScoreResult
    ) -> float:
        """
        Checks whether the episode reached a COMPLETE phase event and
        whether the terminal status is COMPLETED.
        Incomplete tasks produce policies that don't know how to finish.
        """
        has_complete_phase = any(
            e.get("phase") == "complete" for e in phase_events
        )
        is_completed = status == "completed"

        if is_completed and has_complete_phase:
            # Also check operator confidence in terminal phase
            terminal_confidence = next(
                (e.get("confidence", 0) for e in reversed(phase_events)
                 if e.get("phase") == "complete"), 1.0
            )
            if terminal_confidence >= 0.9:
                return 100.0
            elif terminal_confidence >= 0.7:
                return 85.0
            else:
                result.failure_flags.append(
                    f"TASK_COMPLETION:LOW_CONFIDENCE({terminal_confidence:.2f})"
                )
                return 70.0

        elif is_completed and not has_complete_phase:
            result.failure_flags.append("TASK_COMPLETION:MISSING_COMPLETE_PHASE_EVENT")
            return 60.0

        else:
            result.failure_flags.append(
                f"TASK_COMPLETION:ABORTED(status={status}, complete_phase={has_complete_phase})"
            )
            return 0.0

    # ── Dimension 5: Trajectory Validity (10%) ───────────────────────────

    def _score_trajectory_validity(
        self, frames: list[dict], task_type: str, result: ScoreResult
    ) -> float:
        """
        Checks that joint positions stay within the task-type bounding envelope.
        Joint limits are fixed; task-type determines the expected motion range.
        Trajectory anomalies indicate the operator lost control or the robot was
        forced into an unintended configuration.
        """
        # Task-specific tighter envelope (fraction of full joint range)
        TASK_ENVELOPES = {
            "pick_place":     {"scale": 0.85, "max_position": 2.0},
            "sheet_metal":    {"scale": 0.90, "max_position": 2.5},
            "bin_sort":       {"scale": 0.80, "max_position": 1.8},
            "fastener_drive": {"scale": 0.70, "max_position": 1.6},
            "inspection":     {"scale": 0.75, "max_position": 1.9},
        }
        envelope = TASK_ENVELOPES.get(task_type, {"scale": 0.85, "max_position": 2.0})
        max_pos  = envelope["max_position"]

        total_readings   = 0
        out_of_envelope  = 0

        for frame in frames:
            positions = frame.get("joint_positions", [])
            for p in positions:
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    continue
                total_readings += 1
                if abs(p) > max_pos:
                    out_of_envelope += 1

        if total_readings == 0:
            return 50.0

        violation_rate = out_of_envelope / total_readings

        if violation_rate > TRAJECTORY_TOLERANCE:
            result.failure_flags.append(
                f"TRAJECTORY_VALIDITY:ANOMALY({violation_rate*100:.1f}% frames out of envelope)"
            )

        if violation_rate <= 0.001:
            score = 100.0
        elif violation_rate <= TRAJECTORY_TOLERANCE:
            score = 100.0 - (violation_rate / TRAJECTORY_TOLERANCE) * 20
        else:
            score = max(0.0, 80.0 - (violation_rate - TRAJECTORY_TOLERANCE) / 0.05 * 80)

        return round(max(0.0, min(100.0, score)), 2)