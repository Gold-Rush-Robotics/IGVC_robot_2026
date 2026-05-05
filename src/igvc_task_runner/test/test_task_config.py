"""Unit tests for IGVC task profile selection."""

from pathlib import Path

from igvc_task_runner.task_config import (
    load_task_profiles,
    select_task_profile,
)

import pytest


TASK_DIR = Path(__file__).resolve().parents[1] / 'config' / 'tasks'


def test_load_task_profiles_discovers_expected_profiles():
    """Task profile loading should discover the installed defaults."""
    profiles = load_task_profiles(TASK_DIR)

    assert 'q1_lane_keeping_go_straight' in profiles
    assert 'v4_obstacle_detection_lane_changing' in profiles
    assert 'full_course_2026' in profiles
    assert profiles['full_course_2026'].category == 'full_course'
    assert len(profiles['full_course_2026'].sequence) == 21


def test_selected_task_rejects_unsupported_mode():
    """Selected task mode should reject an incompatible robot mode."""
    profiles = load_task_profiles(TASK_DIR)

    with pytest.raises(ValueError):
        select_task_profile(
            profiles,
            task_mode='selected',
            selected_task='full_course_2026',
            robot_mode='airplane',
        )


def test_auto_task_selects_profile_for_mode():
    """Auto task mode should pick a profile matching the robot mode."""
    profiles = load_task_profiles(TASK_DIR)

    profile = select_task_profile(
        profiles,
        task_mode='auto',
        selected_task='',
        robot_mode='sim',
    )

    assert 'sim' in profile.supports_modes


def test_function_profile_exposes_maneuver_metadata():
    """Function tasks should expose maneuver and perception metadata."""
    profiles = load_task_profiles(TASK_DIR)
    profile = profiles['iii2_left_turn']

    assert profile.category == 'function'
    assert 'intersection_left_turn' in profile.maneuvers
    assert 'intersection' in profile.perception_targets
