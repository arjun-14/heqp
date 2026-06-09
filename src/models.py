"""
HEQP Episode Data Models
========================
Defines the canonical data structures for a Figure AI teleoperation episode.

A single episode represents one human-guided robot task session.
Sensor data streams at 200Hz. Camera metadata at 30Hz. Task phase at 10Hz.

Architecture note: These models mirror the data contract between the
teleoperation rig and the streaming ingestion layer. Every field maps
to a real signal that a Figure robot would produce.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import uuid
import json
import time


# ── Enumerations ──────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    """Manufacturing task categories observed in BMW Spartanburg deployment."""
    PICK_PLACE     = "pick_place"       # Pick up part, place in target location
    SHEET_METAL    = "sheet_metal"      # Handle flat metal panels
    BIN_SORT       = "bin_sort"         # Sort parts from bin to bin
    FASTENER_DRIVE = "fastener_drive"   # Drive screws or bolts
    INSPECTION     = "inspection"       # Visual quality check task


class TaskPhase(str, Enum):
    """Sub-phases within a task episode."""
    IDLE       = "idle"
    APPROACH   = "approach"    # Robot moving toward object
    GRASP      = "grasp"       # Gripper engaging with object
    TRANSPORT  = "transport"   # Moving object to target
    PLACE      = "place"       # Releasing object at target
    RETRACT    = "retract"     # Pulling back to safe position
    COMPLETE   = "complete"    # Task finished successfully


class EpisodeStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    ABORTED     = "aborted"    # Operator stopped mid-task


class FailureMode(str, Enum):
    """Synthetic failure modes injected by the simulator."""
    NONE               = "none"
    SENSOR_DROPOUT     = "sensor_dropout"
    MOTION_JITTER      = "motion_jitter"
    TIMING_GAP         = "timing_gap"
    INCOMPLETE_TASK    = "incomplete_task"
    TRAJECTORY_ANOMALY = "trajectory_anomaly"
    COMPOUND           = "compound"


# ── Sensor frame (200 Hz) ─────────────────────────────────────────────────────

@dataclass
class SensorFrame:
    """
    One sensor reading at 200Hz.
    12 DOF: 7 arm joints + 2 wrist + 3 finger joints (Figure 02 approximation).
    """
    timestamp_ns:     int                        # Nanosecond epoch timestamp
    joint_positions:  list[Optional[float]]      # 12 DOF in radians
    joint_velocities: list[Optional[float]]      # 12 DOF in rad/s
    gripper_force:    list[Optional[float]]      # [left_N, right_N]
    ee_pose:          list[float]                # [x, y, z, qx, qy, qz, qw]
    frame_index:      int = 0                    # Sequential frame number within episode

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CameraFrame:
    """Camera metadata at 30Hz (not full image — just sync metadata)."""
    timestamp_ns: int
    frame_id:     int
    exposure_ms:  float
    sync_ok:      bool          # Whether frame is in sync with joint data
    camera_id:    str           # e.g. "head_cam", "wrist_left", "wrist_right"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PhaseEvent:
    """Task phase transitions at ~10Hz (event-driven)."""
    timestamp_ns: int
    phase:        TaskPhase
    confidence:   float         # Operator confidence signal (0.0 - 1.0)

    def to_dict(self) -> dict:
        return {
            "timestamp_ns": self.timestamp_ns,
            "phase": self.phase.value,
            "confidence": self.confidence
        }


# ── Episode (full session) ─────────────────────────────────────────────────────

@dataclass
class Episode:
    """
    A complete teleoperation episode.
    Streamed as JSON to Azure Event Hubs, partitioned by robot_id.
    """
    # Identity
    episode_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    robot_id:      str = "robot_001"
    operator_id:   str = "operator_001"
    task_type:     TaskType = TaskType.PICK_PLACE
    session_ts:    int = field(default_factory=lambda: int(time.time() * 1e9))

    # Sensor data arrays
    sensor_frames:  list[SensorFrame] = field(default_factory=list)
    camera_frames:  list[CameraFrame] = field(default_factory=list)
    phase_events:   list[PhaseEvent]  = field(default_factory=list)

    # Terminal state
    status:            EpisodeStatus = EpisodeStatus.IN_PROGRESS
    duration_ms:       float = 0.0
    task_completed:    bool  = False

    # Quality metadata (set by scoring engine, not simulator)
    injected_failure:  FailureMode = FailureMode.NONE   # Ground truth for benchmarking

    def to_json(self) -> str:
        """Serialize episode to JSON for Event Hubs message."""
        d = {
            "episode_id":       self.episode_id,
            "robot_id":         self.robot_id,
            "operator_id":      self.operator_id,
            "task_type":        self.task_type.value,
            "session_ts":       self.session_ts,
            "status":           self.status.value,
            "duration_ms":      self.duration_ms,
            "task_completed":   self.task_completed,
            "injected_failure": self.injected_failure.value,
            "frame_count":      len(self.sensor_frames),
            "sensor_frames":    [f.to_dict() for f in self.sensor_frames],
            "camera_frames":    [f.to_dict() for f in self.camera_frames],
            "phase_events":     [e.to_dict() for e in self.phase_events],
        }
        return json.dumps(d)

    @property
    def size_kb(self) -> float:
        return len(self.to_json()) / 1024


# ── Summary record (lightweight, for analytics layer) ─────────────────────────

@dataclass
class EpisodeSummary:
    """
    Lightweight episode record written to Delta Lake after scoring.
    Does NOT contain raw sensor frames — those stay in cold storage.
    """
    episode_id:         str
    robot_id:           str
    operator_id:        str
    task_type:          str
    session_ts:         int
    duration_ms:        float
    frame_count:        int
    status:             str
    injected_failure:   str

    # Quality scores (set by scoring engine)
    score_sensor_completeness: float = 0.0
    score_temporal_coherence:  float = 0.0
    score_motion_smoothness:   float = 0.0
    score_task_completion:     float = 0.0
    score_trajectory_validity: float = 0.0
    composite_score:           float = 0.0
    routing_decision:          str   = "PENDING"   # CERTIFIED / BORDERLINE / REJECTED

    def to_dict(self) -> dict:
        return asdict(self)