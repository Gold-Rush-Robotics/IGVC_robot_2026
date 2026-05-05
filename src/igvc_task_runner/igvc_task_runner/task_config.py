"""Task profile loading and selection for the IGVC runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CompletionPolicy:
    """Policy used to decide when a selected task is complete."""

    policy_type: str = 'manual'
    distance_m: float = 0.0
    timeout_sec: float = 0.0


@dataclass(frozen=True)
class TaskProfile:
    """Static task profile loaded from YAML."""

    task_id: str
    display_name: str
    category: str
    rule_ref: str
    supports_modes: tuple[str, ...]
    required_checks: tuple[str, ...]
    maneuvers: tuple[str, ...] = ()
    perception_targets: tuple[str, ...] = ()
    sequence: tuple[dict[str, Any], ...] = ()
    max_points: int = 100
    penalty_points: int = -25
    completion: CompletionPolicy = field(default_factory=CompletionPolicy)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> 'TaskProfile':
        """Build a task profile from a YAML mapping."""
        completion = data.get('completion', {}) or {}
        return cls(
            task_id=str(data['task_id']),
            display_name=str(data.get('display_name', data['task_id'])),
            category=str(data.get('category', 'function')),
            rule_ref=str(data.get('rule_ref', '')),
            supports_modes=tuple(
                str(mode) for mode in data.get('supports_modes', [])),
            required_checks=tuple(
                str(check) for check in data.get('required_checks', [])),
            maneuvers=tuple(
                str(item) for item in data.get('maneuvers', [])),
            perception_targets=tuple(
                str(item) for item in data.get('perception_targets', [])),
            sequence=tuple(dict(step) for step in data.get('sequence', [])),
            max_points=int(data.get('max_points', 100)),
            penalty_points=int(data.get('penalty_points', -25)),
            completion=CompletionPolicy(
                policy_type=str(completion.get('type', 'manual')),
                distance_m=float(completion.get('distance_m', 0.0)),
                timeout_sec=float(completion.get('timeout_sec', 0.0)),
            ),
        )


def load_task_profiles(config_dir: str | Path) -> dict[str, TaskProfile]:
    """Load all task profiles from a directory of YAML files."""
    task_dir = Path(config_dir)
    profiles: dict[str, TaskProfile] = {}
    for path in sorted(task_dir.glob('*.yaml')):
        with path.open('r', encoding='utf-8') as stream:
            data = yaml.safe_load(stream) or {}
        profile = TaskProfile.from_mapping(data)
        profiles[profile.task_id] = profile
    return profiles


def select_task_profile(
    profiles: dict[str, TaskProfile],
    *,
    task_mode: str,
    selected_task: str,
    robot_mode: str,
) -> TaskProfile:
    """Return the selected or auto-detected task profile."""
    if not profiles:
        raise ValueError('No task profiles were loaded.')

    if task_mode == 'selected':
        try:
            profile = profiles[selected_task]
        except KeyError as exc:
            raise ValueError(
                f'Unknown selected task: {selected_task}') from exc
        if robot_mode not in profile.supports_modes:
            raise ValueError(
                f'Task {selected_task} does not support mode {robot_mode}.')
        return profile

    if task_mode != 'auto':
        raise ValueError(f'Unknown task_mode: {task_mode}')

    for profile in profiles.values():
        if robot_mode in profile.supports_modes:
            return profile
    raise ValueError(f'No task profile supports mode {robot_mode}.')
