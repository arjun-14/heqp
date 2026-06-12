"""
HEQP Episode Simulator
======================
Generates realistic Figure AI teleoperation episode data at 200Hz.

Each episode simulates a human operator guiding a robot through a
manufacturing task. The simulator models:
  - Smooth, physiologically realistic joint motion trajectories
  - Gripper force profiles appropriate to each task type
  - Camera frame sync metadata
  - Task phase transitions with operator confidence signals

Failure injection is probabilistic and configurable. Ground truth
failure labels are embedded in each episode, enabling the quality
scoring engine to be benchmarked against known outcomes.

Usage:
    sim = EpisodeSimulator()
    episode = sim.generate_episode(task_type=TaskType.PICK_PLACE)
"""

from __future__ import annotations
import numpy as np
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Optional

from .models import (
    Episode, SensorFrame, CameraFrame, PhaseEvent,
    TaskType, TaskPhase, EpisodeStatus, FailureMode
)

logger = logging.getLogger(__name__)


# ── Simulator configuration ───────────────────────────────────────────────────

@dataclass
class SimulatorConfig:
    # Sampling rates
    sensor_hz:        int   = 200      # Joint + gripper data rate
    camera_hz:        int   = 30       # Camera frame metadata rate
    phase_hz:         int   = 10       # Task phase update rate

    # Task duration range (seconds)
    min_task_duration_s: float = 3.0
    max_task_duration_s: float = 12.0

    # Robot fleet
    robot_ids:    list[str] = field(default_factory=lambda: [f"robot_{i:03d}" for i in range(1, 6)])
    operator_ids: list[str] = field(default_factory=lambda: [f"operator_{i:03d}" for i in range(1, 11)])

    # Failure injection rates (probability per episode)
    failure_rates: dict[str, float] = None

    # Joint limits (radians) — 12 DOF approximate for Figure 02
    joint_limits_min: list[float] = None
    joint_limits_max: list[float] = None

    def __post_init__(self):
        if self.failure_rates is None:
            self.failure_rates = {
                FailureMode.SENSOR_DROPOUT.value:     0.03,
                FailureMode.MOTION_JITTER.value:      0.05,
                FailureMode.TIMING_GAP.value:         0.04,
                FailureMode.INCOMPLETE_TASK.value:    0.06,
                FailureMode.TRAJECTORY_ANOMALY.value: 0.02,
                FailureMode.COMPOUND.value:           0.01,
            }
        if self.joint_limits_min is None:
            self.joint_limits_min = [
                -3.14, -1.57, -3.14, -2.09, -3.14, -1.57, -3.14,  # arm 7 DOF
                -1.0,  -1.0,                                         # wrist 2 DOF
                 0.0,   0.0,   0.0                                   # fingers 3 DOF
            ]
        if self.joint_limits_max is None:
            self.joint_limits_max = [
                 3.14,  1.57,  3.14,  0.17,  3.14,  1.57,  3.14,
                 1.0,   1.0,
                 1.5,   1.5,   1.5
            ]


# ── Task motion profiles ──────────────────────────────────────────────────────

TASK_MOTION_PROFILES = {
    TaskType.PICK_PLACE: {
        "phases":           [TaskPhase.APPROACH, TaskPhase.GRASP, TaskPhase.TRANSPORT, TaskPhase.PLACE, TaskPhase.RETRACT],
        "phase_durations":  [0.20, 0.15, 0.35, 0.15, 0.15],   # fraction of total duration
        "gripper_force_N":  [0.2, 12.0, 12.0, 0.2, 0.1],      # peak force per phase
        "motion_amplitude": 0.8,
    },
    TaskType.SHEET_METAL: {
        "phases":           [TaskPhase.APPROACH, TaskPhase.GRASP, TaskPhase.TRANSPORT, TaskPhase.PLACE, TaskPhase.RETRACT],
        "phase_durations":  [0.15, 0.10, 0.45, 0.20, 0.10],
        "gripper_force_N":  [0.5, 18.0, 18.0, 5.0, 0.2],
        "motion_amplitude": 1.2,
    },
    TaskType.BIN_SORT: {
        "phases":           [TaskPhase.APPROACH, TaskPhase.GRASP, TaskPhase.TRANSPORT, TaskPhase.PLACE, TaskPhase.RETRACT],
        "phase_durations":  [0.25, 0.10, 0.30, 0.10, 0.25],
        "gripper_force_N":  [0.1, 8.0, 8.0, 0.1, 0.1],
        "motion_amplitude": 0.6,
    },
    TaskType.FASTENER_DRIVE: {
        "phases":           [TaskPhase.APPROACH, TaskPhase.GRASP, TaskPhase.PLACE, TaskPhase.RETRACT],
        "phase_durations":  [0.20, 0.15, 0.50, 0.15],
        "gripper_force_N":  [0.3, 25.0, 25.0, 0.3],
        "motion_amplitude": 0.4,
    },
    TaskType.INSPECTION: {
        "phases":           [TaskPhase.APPROACH, TaskPhase.TRANSPORT, TaskPhase.RETRACT],
        "phase_durations":  [0.30, 0.50, 0.20],
        "gripper_force_N":  [0.1, 0.5, 0.1],
        "motion_amplitude": 0.5,
    },
}


# ── Simulator ─────────────────────────────────────────────────────────────────

class EpisodeSimulator:
    """
    Generates realistic robot teleoperation episodes.

    The motion model uses sinusoidal trajectory primitives with task-specific
    amplitude and phase profiles. This is not physics-accurate but produces
    data with the statistical properties (smoothness, timing, completion
    patterns) that the quality scoring engine is calibrated against.
    """

    def __init__(self, config: Optional[SimulatorConfig] = None, seed: Optional[int] = None):
        self.config = config or SimulatorConfig()
        self.rng = np.random.default_rng(seed)
        self._episode_count = 0

    # ── Public API ────────────────────────────────────────────────────────

    def generate_episode(
        self,
        task_type:    Optional[TaskType] = None,
        robot_id:     Optional[str]      = None,
        operator_id:  Optional[str]      = None,
        force_failure: Optional[FailureMode] = None,
    ) -> Episode:
        """
        Generate one complete teleoperation episode.

        Args:
            task_type:     Override task type (random if None)
            robot_id:      Override robot ID (random if None)
            operator_id:   Override operator ID (random if None)
            force_failure: Force a specific failure mode (for testing)

        Returns:
            Episode with fully populated sensor frames and phase events
        """
        task_type   = task_type   or random.choice(list(TaskType))
        robot_id    = robot_id    or random.choice(self.config.robot_ids)
        operator_id = operator_id or random.choice(self.config.operator_ids)

        # Sample episode duration
        duration_s = self.rng.uniform(
            self.config.min_task_duration_s,
            self.config.max_task_duration_s
        )

        # Decide failure mode
        failure_mode = force_failure or self._sample_failure_mode()

        # Build base episode
        episode = Episode(
            robot_id=robot_id,
            operator_id=operator_id,
            task_type=task_type,
            injected_failure=failure_mode,
        )

        # Generate sensor frames, camera frames, phase events
        base_ts_ns = int(time.time() * 1e9)
        profile    = TASK_MOTION_PROFILES[task_type]

        episode.sensor_frames = self._generate_sensor_frames(
            duration_s, base_ts_ns, profile, failure_mode
        )
        episode.camera_frames = self._generate_camera_frames(
            duration_s, base_ts_ns, failure_mode
        )
        episode.phase_events = self._generate_phase_events(
            duration_s, base_ts_ns, profile, failure_mode
        )

        # Set terminal state
        episode.duration_ms   = duration_s * 1000
        episode.task_completed = failure_mode not in (
            FailureMode.INCOMPLETE_TASK, FailureMode.COMPOUND
        )
        episode.status = (
            EpisodeStatus.COMPLETED if episode.task_completed
            else EpisodeStatus.ABORTED
        )

        self._episode_count += 1
        return episode

    def stream_episodes(
        self,
        count:        Optional[int] = None,
        episodes_per_second: float  = 10.0,
        task_type:    Optional[TaskType] = None,
    ):
        """
        Generator that yields episodes at a configurable rate.
        Runs indefinitely if count is None.
        """
        generated = 0
        interval  = 1.0 / episodes_per_second
        while count is None or generated < count:
            t0 = time.monotonic()
            yield self.generate_episode(task_type=task_type)
            generated += 1
            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    @property
    def episodes_generated(self) -> int:
        return self._episode_count

    # ── Internal: sensor frame generation ────────────────────────────────

    def _generate_sensor_frames(
        self,
        duration_s:   float,
        base_ts_ns:   int,
        profile:      dict,
        failure_mode: FailureMode,
    ) -> list[SensorFrame]:

        n_frames    = int(duration_s * self.config.sensor_hz)
        interval_ns = int(1e9 / self.config.sensor_hz)   # 5,000,000 ns = 5ms

        # Generate smooth joint trajectories using sinusoidal primitives
        joint_positions  = self._smooth_trajectory(n_frames, profile["motion_amplitude"])
        joint_velocities = self._compute_velocities(joint_positions, interval_ns)
        gripper_force    = self._gripper_profile(n_frames, profile)

        # Build timestamp array (will be mutated by failure injection)
        timestamps = [base_ts_ns + i * interval_ns for i in range(n_frames)]

        # End-effector pose (simplified: derived from first 3 joint positions)
        ee_poses = self._compute_ee_pose(joint_positions)

        # Apply failure modes BEFORE building frames
        joint_positions, joint_velocities, timestamps = self._inject_sensor_failures(
            joint_positions, joint_velocities, gripper_force, timestamps, failure_mode
        )

        frames = []
        for i in range(n_frames):
            frames.append(SensorFrame(
                timestamp_ns     = timestamps[i],
                joint_positions  = joint_positions[i].tolist() if joint_positions[i] is not None else [None] * 12,
                joint_velocities = joint_velocities[i].tolist() if joint_velocities[i] is not None else [None] * 12,
                gripper_force    = gripper_force[i],
                ee_pose          = ee_poses[i],
                frame_index      = i,
            ))
        return frames

    def _smooth_trajectory(self, n_frames: int, amplitude: float) -> np.ndarray:
        """
        Generate smooth 12-DOF joint trajectories.
        Each joint follows a sinusoidal path with unique frequency and phase,
        bounded by joint limits. Mimics a realistic pick-and-place motion arc.
        """
        t      = np.linspace(0, 2 * np.pi, n_frames)
        n_dof  = 12
        limits_min = np.array(self.config.joint_limits_min)
        limits_max = np.array(self.config.joint_limits_max)
        mid    = (limits_min + limits_max) / 2
        rng    = (limits_max - limits_min) / 2 * amplitude * 0.4

        # Different frequency and phase offset per joint for realistic coupling
        freqs  = self.rng.uniform(0.5, 2.5, n_dof)
        phases = self.rng.uniform(0, np.pi, n_dof)

        traj = np.zeros((n_frames, n_dof))
        for j in range(n_dof):
            traj[:, j] = mid[j] + rng[j] * np.sin(freqs[j] * t + phases[j])

        # Add small operator micro-tremor (< 0.01 rad RMS)
        traj += self.rng.normal(0, 0.008, traj.shape)
        return np.clip(traj, limits_min, limits_max)

    def _compute_velocities(self, positions: np.ndarray, interval_ns: int) -> np.ndarray:
        """Finite difference velocity from position trajectory."""
        dt  = interval_ns / 1e9  # seconds
        vel = np.gradient(positions, dt, axis=0)
        return vel

    def _gripper_profile(self, n_frames: int, profile: dict) -> list[list[float]]:
        """Generate gripper force readings following task phase profile."""
        phase_durs   = profile["phase_durations"]
        phase_forces = profile["gripper_force_N"]
        n_phases     = len(phase_durs)

        # Build frame boundaries for each phase
        phase_boundaries = [0]
        for d in phase_durs:
            phase_boundaries.append(phase_boundaries[-1] + int(d * n_frames))
        phase_boundaries[-1] = n_frames

        # Create an array mapping every frame to its target force
        target_forces = np.zeros(n_frames)
        for p in range(n_phases):
            target_forces[phase_boundaries[p]:phase_boundaries[p+1]] = phase_forces[p]

        # Vectorized noise and asymmetry calculations
        noise = self.rng.normal(0, target_forces * 0.05 + 0.1)
        f_left = np.maximum(0.0, target_forces + noise)
        f_right = f_left * self.rng.uniform(0.95, 1.05, n_frames)
        
        return np.column_stack((f_left, f_right)).tolist()

    def _compute_ee_pose(self, positions: np.ndarray) -> list[list[float]]:
        """
        Simplified forward kinematics approximation.
        Not physically accurate — produces plausible [x,y,z,qx,qy,qz,qw] values.
        """
        x  = 0.3 + 0.2 * np.sin(positions[:, 0]) * np.cos(positions[:, 1])
        y  = 0.1 + 0.2 * np.sin(positions[:, 1])
        z  = 0.8 + 0.15 * np.cos(positions[:, 0])
        qx = np.sin(positions[:, 2] / 2) * 0.1
        qy = np.sin(positions[:, 3] / 2) * 0.1
        qz = np.sin(positions[:, 4] / 2) * 0.1
        qw = np.sqrt(np.maximum(0, 1 - qx**2 - qy**2 - qz**2))
        
        poses = np.column_stack((x, y, z, qx, qy, qz, qw))
        return np.round(poses, 6).tolist()

    # ── Internal: failure injection ───────────────────────────────────────

    def _inject_sensor_failures(
        self,
        positions:    np.ndarray,
        velocities:   np.ndarray,
        gripper:      list,
        timestamps:   list[int],
        failure_mode: FailureMode,
    ):
        """Apply failure modes to raw sensor arrays before frame construction."""

        positions  = positions.copy()
        velocities = velocities.copy()
        n          = len(timestamps)
        timestamps = list(timestamps)

        modes_to_apply = []
        if failure_mode == FailureMode.COMPOUND:
            # Pick two random failure modes (not NONE, not COMPOUND, not INCOMPLETE_TASK)
            pool = [FailureMode.SENSOR_DROPOUT, FailureMode.MOTION_JITTER,
                    FailureMode.TIMING_GAP, FailureMode.TRAJECTORY_ANOMALY]
            modes_to_apply = random.sample(pool, 2)
        elif failure_mode != FailureMode.NONE and failure_mode != FailureMode.INCOMPLETE_TASK:
            modes_to_apply = [failure_mode]

        for mode in modes_to_apply:
            if mode == FailureMode.SENSOR_DROPOUT:
                # NaN out 5-50 consecutive frames in a random window
                dropout_len   = random.randint(5, 50)
                dropout_start = random.randint(0, max(0, n - dropout_len))
                dropout_end   = min(dropout_start + dropout_len, n)
                positions[dropout_start:dropout_end]  = np.nan
                velocities[dropout_start:dropout_end] = np.nan

            elif mode == FailureMode.MOTION_JITTER:
                # Add high-frequency Gaussian noise to velocities
                jitter = self.rng.normal(0, 2.0, velocities.shape)
                velocities += jitter

            elif mode == FailureMode.TIMING_GAP:
                # Insert a 100ms-500ms gap in the timestamp sequence
                gap_start  = random.randint(n // 4, 3 * n // 4)
                gap_ns     = random.randint(int(100e6), int(500e6))
                for i in range(gap_start, n):
                    timestamps[i] += gap_ns

            elif mode == FailureMode.TRAJECTORY_ANOMALY:
                # Shift joint positions outside task-valid envelope
                anomaly_start = random.randint(n // 3, 2 * n // 3)
                anomaly_end   = min(anomaly_start + random.randint(20, 80), n)
                offsets = self.rng.uniform(0.8, 1.5, 12)
                for j in range(12):
                    positions[anomaly_start:anomaly_end, j] += offsets[j]

        return positions, velocities, timestamps

    # ── Internal: camera frames ───────────────────────────────────────────

    def _generate_camera_frames(
        self, duration_s: float, base_ts_ns: int, failure_mode: FailureMode
    ) -> list[CameraFrame]:

        n_frames    = int(duration_s * self.config.camera_hz)
        interval_ns = int(1e9 / self.config.camera_hz)
        cameras     = ["head_cam", "wrist_left", "wrist_right"]
        frames      = []

        for i in range(n_frames):
            ts = base_ts_ns + i * interval_ns
            for cam in cameras:
                frames.append(CameraFrame(
                    timestamp_ns = ts,
                    frame_id     = i,
                    exposure_ms  = float(self.rng.uniform(8.0, 16.0)),
                    sync_ok      = True,
                    camera_id    = cam,
                ))
        return frames

    # ── Internal: phase events ────────────────────────────────────────────

    def _generate_phase_events(
        self,
        duration_s:   float,
        base_ts_ns:   int,
        profile:      dict,
        failure_mode: FailureMode,
    ) -> list[PhaseEvent]:

        phases     = profile["phases"]
        phase_durs = profile["phase_durations"]
        events     = []
        cursor_ns  = base_ts_ns

        for phase, frac in zip(phases, phase_durs):
            # Operator confidence decreases slightly in grasp/place phases
            base_confidence = 0.95 if phase not in (TaskPhase.GRASP, TaskPhase.PLACE) else 0.82
            confidence = float(np.clip(
                self.rng.normal(base_confidence, 0.05), 0.4, 1.0
            ))
            events.append(PhaseEvent(
                timestamp_ns = cursor_ns,
                phase        = phase,
                confidence   = round(confidence, 3),
            ))
            cursor_ns += int(frac * duration_s * 1e9)

        # Add COMPLETE event unless task is incomplete
        if failure_mode not in (FailureMode.INCOMPLETE_TASK, FailureMode.COMPOUND):
            events.append(PhaseEvent(
                timestamp_ns = base_ts_ns + int(duration_s * 1e9),
                phase        = TaskPhase.COMPLETE,
                confidence   = 1.0,
            ))

        return events

    # ── Internal: failure mode sampling ──────────────────────────────────

    def _sample_failure_mode(self) -> FailureMode:
        """
        Sample a failure mode according to configured probabilities.
        Returns NONE if no failure is injected (the common case).
        """
        roll = random.random()
        cumulative = 0.0
        for mode_str, prob in self.config.failure_rates.items():
            cumulative += prob
            if roll < cumulative:
                return FailureMode(mode_str)
        return FailureMode.NONE


# ── Simulator statistics ──────────────────────────────────────────────────────

class SimulatorStats:
    """Tracks episode generation statistics for monitoring."""

    def __init__(self):
        self.total_episodes   = 0
        self.failure_counts   = {m.value: 0 for m in FailureMode}
        self.task_counts      = {t.value: 0 for t in TaskType}
        self.total_frames     = 0
        self.start_time       = time.time()

    def record(self, episode: Episode):
        self.total_episodes += 1
        self.failure_counts[episode.injected_failure.value] += 1
        self.task_counts[episode.task_type.value] += 1
        self.total_frames += len(episode.sensor_frames)

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.start_time

    @property
    def episodes_per_second(self) -> float:
        return self.total_episodes / max(self.elapsed_s, 0.001)

    def summary(self) -> dict:
        total = max(self.total_episodes, 1)
        return {
            "total_episodes":      self.total_episodes,
            "elapsed_s":           round(self.elapsed_s, 2),
            "episodes_per_second": round(self.episodes_per_second, 2),
            "total_frames":        self.total_frames,
            "failure_rate_pct": {
                k: round(v / total * 100, 2)
                for k, v in self.failure_counts.items()
            },
            "task_distribution_pct": {
                k: round(v / total * 100, 2)
                for k, v in self.task_counts.items()
            },
        }