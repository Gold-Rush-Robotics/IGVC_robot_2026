"""Unit tests for IGVC maneuver behavior primitives."""

from pathlib import Path

from igvc_task_runner.behavior import (
    PRIMITIVES,
    TaskBehaviorController,
)
from igvc_task_runner.task_config import load_task_profiles


TASK_DIR = Path(__file__).resolve().parents[1] / 'config' / 'tasks'


def test_all_configured_maneuvers_have_registered_primitives():
    """Every maneuver named by task YAML should be executable."""
    profiles = load_task_profiles(TASK_DIR)
    configured = {
        maneuver
        for profile in profiles.values()
        for maneuver in profile.maneuvers
    }

    assert configured <= set(PRIMITIVES)


def test_full_course_expands_into_maneuver_steps():
    """The full course should expand Table 3 tasks into primitives."""
    profiles = load_task_profiles(TASK_DIR)
    controller = TaskBehaviorController(
        profiles['full_course_2026'], profiles)

    sequence_steps = {
        step.sequence_step for step in controller.steps
        if step.sequence_step is not None
    }

    assert len(sequence_steps) == 21
    assert controller.steps[0].maneuver == 'intersection_right_turn'
    assert controller.steps[-1].maneuver == 'parking_pull_in'


def test_turn_primitive_emits_directional_command():
    """Intersection turn primitives should expose turn direction."""
    profiles = load_task_profiles(TASK_DIR)
    controller = TaskBehaviorController(profiles['iii2_left_turn'], profiles)

    status = controller.tick(now_ns=1_000_000_000, distance_m=0.0)

    assert status is not None
    assert status.command.controller_mode == 'intersection_turn'
    assert status.command.turn_direction == 'left'


def test_stop_and_parking_primitives_emit_specific_modes():
    """Stop/yield and parking tasks should not look like lane following."""
    profiles = load_task_profiles(TASK_DIR)
    stop_controller = TaskBehaviorController(
        profiles['ii1_stop_sign_detection'], profiles)
    parking_controller = TaskBehaviorController(
        profiles['iv2_parking_pull_in'], profiles)

    stop_controller.complete_current_step(now_ns=1_000_000_000)
    stop_status = stop_controller.tick(now_ns=2_000_000_000, distance_m=0.0)
    parking_status = parking_controller.tick(
        now_ns=1_000_000_000, distance_m=0.0)

    assert stop_status is not None
    assert stop_status.command.stop_required
    assert stop_status.command.controller_mode == 'stop_at_sign'
    assert parking_status is not None
    assert parking_status.command.controller_mode == 'parking'
    assert parking_status.command.parking_mode == 'pull_in'


def test_manual_completion_advances_behavior_plan():
    """Operator-led testing should be able to advance primitives."""
    profiles = load_task_profiles(TASK_DIR)
    controller = TaskBehaviorController(
        profiles['q1_lane_keeping_go_straight'], profiles)

    assert controller.active_step is not None
    assert controller.active_step.maneuver == 'lane_keep'
    assert controller.complete_current_step(now_ns=1_000_000_000)
    assert controller.active_step is not None
    assert controller.active_step.maneuver == 'go_straight'
    assert controller.complete_current_step(now_ns=2_000_000_000)
    assert controller.is_complete()
