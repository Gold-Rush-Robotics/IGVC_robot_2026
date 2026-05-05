"""Maneuver primitive controllers for IGVC task execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from igvc_task_runner.task_config import TaskProfile


class PrimitiveState(str, Enum):
    """Lifecycle state for a maneuver primitive."""

    IDLE = 'idle'
    ACTIVE = 'active'
    WAITING = 'waiting'
    DONE = 'done'
    FAILED = 'failed'


@dataclass(frozen=True)
class ManeuverCommand:
    """Controller intent emitted by one maneuver primitive."""

    maneuver: str
    controller_mode: str
    targets: tuple[str, ...] = ()
    speed_limit_mps: float | None = None
    turn_direction: str | None = None
    parking_mode: str | None = None
    stop_required: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly command mapping."""
        data: dict[str, Any] = {
            'maneuver': self.maneuver,
            'controller_mode': self.controller_mode,
            'targets': list(self.targets),
            'stop_required': self.stop_required,
            'notes': list(self.notes),
        }
        if self.speed_limit_mps is not None:
            data['speed_limit_mps'] = self.speed_limit_mps
        if self.turn_direction is not None:
            data['turn_direction'] = self.turn_direction
        if self.parking_mode is not None:
            data['parking_mode'] = self.parking_mode
        return data


@dataclass(frozen=True)
class ManeuverStatus:
    """Status emitted by one maneuver primitive tick."""

    state: PrimitiveState
    command: ManeuverCommand
    reason: str
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly status mapping."""
        return {
            'state': self.state.value,
            'reason': self.reason,
            'elapsed_sec': round(self.elapsed_sec, 3),
            'command': self.command.to_dict(),
        }


@dataclass(frozen=True)
class ManeuverContext:
    """Input context available to maneuver primitives."""

    task_id: str
    task_category: str
    perception_targets: tuple[str, ...]
    distance_m: float
    elapsed_sec: float


@dataclass(frozen=True)
class BehaviorStep:
    """One primitive step in a task behavior plan."""

    task_id: str
    task_name: str
    task_category: str
    maneuver: str
    perception_targets: tuple[str, ...]
    sequence_step: int | None = None
    sequence_name: str = ''

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly step mapping."""
        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'task_category': self.task_category,
            'maneuver': self.maneuver,
            'perception_targets': list(self.perception_targets),
            'sequence_step': self.sequence_step,
            'sequence_name': self.sequence_name,
        }


class ManeuverPrimitive:
    """Base class for one maneuver controller primitive."""

    maneuver = 'unknown'
    controller_mode = 'idle'
    default_speed_limit_mps: float | None = 0.8
    stop_required = False
    notes: tuple[str, ...] = ()

    def command(self, step: BehaviorStep) -> ManeuverCommand:
        """Build the command intent for this primitive."""
        return ManeuverCommand(
            maneuver=step.maneuver,
            controller_mode=self.controller_mode,
            targets=step.perception_targets,
            speed_limit_mps=self.default_speed_limit_mps,
            stop_required=self.stop_required,
            notes=self.notes,
        )

    def tick(
        self,
        step: BehaviorStep,
        context: ManeuverContext,
        elapsed_sec: float,
    ) -> ManeuverStatus:
        """Return the current primitive status and command intent."""
        del context
        return ManeuverStatus(
            state=PrimitiveState.ACTIVE,
            command=self.command(step),
            reason='primitive active',
            elapsed_sec=elapsed_sec,
        )


class LaneKeepPrimitive(ManeuverPrimitive):
    """Follow the currently detected lane path."""

    maneuver = 'lane_keep'
    controller_mode = 'follow_lane_path'
    default_speed_limit_mps = 1.0


class DetectLanesPrimitive(ManeuverPrimitive):
    """Hold behavior execution while lane perception is evaluated."""

    maneuver = 'detect_lanes'
    controller_mode = 'perception_check'
    default_speed_limit_mps = 0.0
    stop_required = True

    def tick(
        self,
        step: BehaviorStep,
        context: ManeuverContext,
        elapsed_sec: float,
    ) -> ManeuverStatus:
        """Return waiting status while lane perception is checked."""
        del context
        return ManeuverStatus(
            state=PrimitiveState.WAITING,
            command=self.command(step),
            reason='waiting for lane detection evaluation',
            elapsed_sec=elapsed_sec,
        )


class GoStraightPrimitive(ManeuverPrimitive):
    """Prefer straight-ahead lane path output through intersections."""

    maneuver = 'go_straight'
    controller_mode = 'intersection_straight'
    default_speed_limit_mps = 0.8


class IntersectionTurnPrimitive(ManeuverPrimitive):
    """Follow an intersection turn behavior."""

    controller_mode = 'intersection_turn'
    default_speed_limit_mps = 0.6

    direction = 'unknown'

    def command(self, step: BehaviorStep) -> ManeuverCommand:
        """Build a directional intersection command."""
        base = super().command(step)
        return ManeuverCommand(
            maneuver=base.maneuver,
            controller_mode=base.controller_mode,
            targets=base.targets,
            speed_limit_mps=base.speed_limit_mps,
            turn_direction=self.direction,
            stop_required=base.stop_required,
            notes=base.notes,
        )


class LeftTurnPrimitive(IntersectionTurnPrimitive):
    """Execute a left turn at an intersection."""

    maneuver = 'intersection_left_turn'
    direction = 'left'


class RightTurnPrimitive(IntersectionTurnPrimitive):
    """Execute a right turn at an intersection."""

    maneuver = 'intersection_right_turn'
    direction = 'right'


class LaneChangePrimitive(ManeuverPrimitive):
    """Request a lane change around an obstacle or target."""

    maneuver = 'lane_change'
    controller_mode = 'lane_change'
    default_speed_limit_mps = 0.7


class ObstacleAvoidancePrimitive(ManeuverPrimitive):
    """Bias path selection away from detected obstacles."""

    maneuver = 'obstacle_avoidance'
    controller_mode = 'avoid_obstacle'
    default_speed_limit_mps = 0.55


class StopAtSignPrimitive(ManeuverPrimitive):
    """Stop at a detected stop sign until the task can proceed."""

    maneuver = 'stop_at_sign'
    controller_mode = 'stop_at_sign'
    default_speed_limit_mps = 0.0
    stop_required = True

    def tick(
        self,
        step: BehaviorStep,
        context: ManeuverContext,
        elapsed_sec: float,
    ) -> ManeuverStatus:
        """Return stop-sign dwell status."""
        del context
        if elapsed_sec < 3.0:
            state = PrimitiveState.WAITING
            reason = 'holding stop sign dwell'
        else:
            state = PrimitiveState.ACTIVE
            reason = 'stop dwell satisfied'
        return ManeuverStatus(
            state=state,
            command=self.command(step),
            reason=reason,
            elapsed_sec=elapsed_sec,
        )


class StopOrYieldPrimitive(ManeuverPrimitive):
    """Stop or yield for pedestrian tasks."""

    maneuver = 'stop_or_yield'
    controller_mode = 'yield_to_target'
    default_speed_limit_mps = 0.0
    stop_required = True

    def tick(
        self,
        step: BehaviorStep,
        context: ManeuverContext,
        elapsed_sec: float,
    ) -> ManeuverStatus:
        """Return pedestrian yield status."""
        target_text = ', '.join(context.perception_targets) or 'target'
        return ManeuverStatus(
            state=PrimitiveState.WAITING,
            command=self.command(step),
            reason=f'waiting for {target_text} clearance',
            elapsed_sec=elapsed_sec,
        )


class ParkingPrimitive(ManeuverPrimitive):
    """Base class for parking behaviors."""

    controller_mode = 'parking'
    default_speed_limit_mps = 0.35
    parking_mode = 'unknown'

    def command(self, step: BehaviorStep) -> ManeuverCommand:
        """Build a parking command with the concrete parking mode."""
        base = super().command(step)
        return ManeuverCommand(
            maneuver=base.maneuver,
            controller_mode=base.controller_mode,
            targets=base.targets,
            speed_limit_mps=base.speed_limit_mps,
            parking_mode=self.parking_mode,
            stop_required=base.stop_required,
            notes=base.notes,
        )


class ParkingPullOutPrimitive(ParkingPrimitive):
    """Pull out from a starting parking space."""

    maneuver = 'parking_pull_out'
    parking_mode = 'pull_out'


class ParkingPullInPrimitive(ParkingPrimitive):
    """Pull into a parking space."""

    maneuver = 'parking_pull_in'
    parking_mode = 'pull_in'


class ParallelParkingPrimitive(ParkingPrimitive):
    """Perform a parallel parking maneuver."""

    maneuver = 'parallel_parking'
    parking_mode = 'parallel'


class CurveFollowingPrimitive(ManeuverPrimitive):
    """Follow curved-road lane geometry with a tighter speed cap."""

    maneuver = 'curve_following'
    controller_mode = 'follow_curved_lane'
    default_speed_limit_mps = 0.75


PRIMITIVES: dict[str, ManeuverPrimitive] = {
    primitive.maneuver: primitive
    for primitive in (
        LaneKeepPrimitive(),
        DetectLanesPrimitive(),
        GoStraightPrimitive(),
        LeftTurnPrimitive(),
        RightTurnPrimitive(),
        LaneChangePrimitive(),
        ObstacleAvoidancePrimitive(),
        StopAtSignPrimitive(),
        StopOrYieldPrimitive(),
        ParkingPullOutPrimitive(),
        ParkingPullInPrimitive(),
        ParallelParkingPrimitive(),
        CurveFollowingPrimitive(),
    )
}


@dataclass
class TaskBehaviorController:
    """Expand a task profile into maneuver primitives and tick them."""

    profile: TaskProfile
    profiles: dict[str, TaskProfile]
    primitives: dict[str, ManeuverPrimitive] = field(
        default_factory=lambda: PRIMITIVES)
    _steps: tuple[BehaviorStep, ...] = ()
    _current_index: int = 0
    _started_ns: int | None = None
    _step_started_ns: int | None = None
    _last_status: ManeuverStatus | None = None

    def __post_init__(self) -> None:
        """Build and validate the behavior plan after dataclass init."""
        self._steps = tuple(self._build_steps())
        self._validate_steps()

    @property
    def steps(self) -> tuple[BehaviorStep, ...]:
        """Return the immutable behavior plan."""
        return self._steps

    @property
    def active_step(self) -> BehaviorStep | None:
        """Return the currently active behavior step."""
        if self._current_index >= len(self._steps):
            return None
        return self._steps[self._current_index]

    @property
    def last_status(self) -> ManeuverStatus | None:
        """Return the most recent primitive status."""
        return self._last_status

    def start(self, now_ns: int) -> None:
        """Start or restart task behavior execution."""
        self._current_index = 0
        self._started_ns = now_ns
        self._step_started_ns = now_ns
        self._last_status = None

    def stop(self) -> None:
        """Forget active timing while leaving the plan intact."""
        self._started_ns = None
        self._step_started_ns = None
        self._last_status = None

    def complete_current_step(self, now_ns: int) -> bool:
        """Advance to the next behavior step for operator-led testing."""
        if self.active_step is None:
            return False
        self._current_index += 1
        self._step_started_ns = now_ns
        self._last_status = None
        return True

    def is_complete(self) -> bool:
        """Return whether all behavior steps have been completed."""
        return bool(self._steps) and self._current_index >= len(self._steps)

    def tick(self, now_ns: int, distance_m: float) -> ManeuverStatus | None:
        """Tick the active primitive and return its status."""
        step = self.active_step
        if step is None:
            return None
        if self._started_ns is None:
            self.start(now_ns)
        if self._step_started_ns is None:
            self._step_started_ns = now_ns
        elapsed_sec = (now_ns - self._step_started_ns) / 1e9
        total_elapsed_sec = (now_ns - (self._started_ns or now_ns)) / 1e9
        primitive = self.primitives[step.maneuver]
        context = ManeuverContext(
            task_id=step.task_id,
            task_category=step.task_category,
            perception_targets=step.perception_targets,
            distance_m=distance_m,
            elapsed_sec=total_elapsed_sec,
        )
        self._last_status = primitive.tick(step, context, elapsed_sec)
        return self._last_status

    def status_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly behavior status mapping."""
        step = self.active_step
        return {
            'complete': self.is_complete(),
            'current_index': self._current_index,
            'step_count': len(self._steps),
            'active_step': step.to_dict() if step is not None else None,
            'primitive_status': (
                self._last_status.to_dict()
                if self._last_status is not None else None
            ),
        }

    def _build_steps(self) -> list[BehaviorStep]:
        if self.profile.sequence:
            return self._build_sequence_steps()
        return self._steps_for_profile(self.profile)

    def _build_sequence_steps(self) -> list[BehaviorStep]:
        steps: list[BehaviorStep] = []
        for entry in self.profile.sequence:
            task_id = str(entry['task'])
            subprofile = self.profiles[task_id]
            sequence_step = int(entry.get('step', len(steps) + 1))
            sequence_name = str(entry.get('name', subprofile.display_name))
            for step in self._steps_for_profile(
                    subprofile, sequence_step, sequence_name):
                steps.append(step)
        return steps

    def _steps_for_profile(
        self,
        profile: TaskProfile,
        sequence_step: int | None = None,
        sequence_name: str = '',
    ) -> list[BehaviorStep]:
        return [
            BehaviorStep(
                task_id=profile.task_id,
                task_name=profile.display_name,
                task_category=profile.category,
                maneuver=maneuver,
                perception_targets=profile.perception_targets,
                sequence_step=sequence_step,
                sequence_name=sequence_name,
            )
            for maneuver in profile.maneuvers
        ]

    def _validate_steps(self) -> None:
        unknown = sorted({
            step.maneuver for step in self._steps
            if step.maneuver not in self.primitives
        })
        if unknown:
            raise ValueError(
                'No maneuver primitive registered for: ' + ', '.join(unknown))
