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
import uuid, json, time, math


class TaskType(str, Enum):
    PICK_PLACE     = "pick_place"
    SHEET_METAL    = "sheet_metal"
    BIN_SORT       = "bin_sort"
    FASTENER_DRIVE = "fastener_drive"
    INSPECTION     = "inspection"

class TaskPhase(str, Enum):
    IDLE       = "idle"
    APPROACH   = "approach"
    GRASP      = "grasp"
    TRANSPORT  = "transport"
    PLACE      = "place"
    RETRACT    = "retract"
    COMPLETE   = "complete"

class EpisodeStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    ABORTED     = "aborted"

class FailureMode(str, Enum):
    NONE               = "none"
    SENSOR_DROPOUT     = "sensor_dropout"
    MOTION_JITTER      = "motion_jitter"
    TIMING_GAP         = "timing_gap"
    INCOMPLETE_TASK    = "incomplete_task"
    TRAJECTORY_ANOMALY = "trajectory_anomaly"
    COMPOUND           = "compound"


@dataclass
class SensorFrame:
    timestamp_ns:     int
    joint_positions:  list           # float or None per element
    joint_velocities: list
    gripper_force:    list[float]
    ee_pose:          list[float]
    frame_index:      int = 0

    def to_dict(self) -> dict:
        # Serialize NaN as None so JSON stays valid
        def _clean(vals):
            return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), 6)
                    for v in vals]
        return {
            "timestamp_ns":     self.timestamp_ns,
            "joint_positions":  _clean(self.joint_positions),
            "joint_velocities": _clean(self.joint_velocities),
            "gripper_force":    [round(float(v), 4) for v in self.gripper_force],
            "ee_pose":          [round(float(v), 6) for v in self.ee_pose],
            "frame_index":      self.frame_index,
        }


@dataclass
class CameraFrame:
    timestamp_ns: int
    frame_id:     int
    exposure_ms:  float
    sync_ok:      bool
    camera_id:    str


@dataclass
class PhaseEvent:
    timestamp_ns: int
    phase:        TaskPhase
    confidence:   float


@dataclass
class Episode:
    episode_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    robot_id:      str = "robot_001"
    operator_id:   str = "operator_001"
    task_type:     TaskType = TaskType.PICK_PLACE
    session_ts:    int = field(default_factory=lambda: int(time.time() * 1e9))

    sensor_frames:  list[SensorFrame] = field(default_factory=list)
    camera_frames:  list[CameraFrame] = field(default_factory=list)
    phase_events:   list[PhaseEvent]  = field(default_factory=list)

    status:            EpisodeStatus = EpisodeStatus.IN_PROGRESS
    duration_ms:       float = 0.0
    task_completed:    bool  = False
    injected_failure:  FailureMode = FailureMode.NONE

    def to_json(self) -> str:
        return json.dumps({
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
            "camera_frames":    [{"timestamp_ns": f.timestamp_ns, "frame_id": f.frame_id,
                                  "exposure_ms": f.exposure_ms, "sync_ok": f.sync_ok,
                                  "camera_id": f.camera_id} for f in self.camera_frames],
            "phase_events":     [{"timestamp_ns": e.timestamp_ns, "phase": e.phase.value,
                                  "confidence": e.confidence} for e in self.phase_events],
        })

    @property
    def size_kb(self) -> float:
        return len(self.to_json()) / 1024