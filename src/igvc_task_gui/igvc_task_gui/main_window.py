"""PyQt5 main window for the IGVC task operator GUI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QScrollArea,
)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_STATE_COLOURS: dict[str, str] = {
    'configuring': '#888888',
    'ready':       '#e6b800',
    'armed':       '#3399ff',
    'running':     '#33cc33',
    'recovering':  '#ff9900',
    'completed':   '#00cc88',
    'failed':      '#cc0000',
    'aborted':     '#cc0000',
    'safe_stopped': '#cc3300',
}

_BTN_STYLES: dict[str, str] = {
    'arm':               'background:#3399ff; color:white; font-weight:bold;',
    'start':             'background:#33cc33; color:white; font-weight:bold;',
    'pause':             'background:#e67300; color:white; font-weight:bold;',
    'resume':            'background:#00bfff; color:white; font-weight:bold;',
    'complete_maneuver': 'background:#8844cc; color:white; font-weight:bold;',
    'abort':             'background:#cc2200; color:white; font-weight:bold;',
    'safe_stop':         'background:#882200; color:white; font-weight:bold;',
}

# State → enabled buttons (name must match _BTN_STYLES keys)
_STATE_BUTTONS: dict[str, set[str]] = {
    'configuring':  set(),
    'ready':        {'arm'},
    'armed':        {'start', 'abort', 'safe_stop'},
    'running':      {'pause', 'complete_maneuver', 'abort', 'safe_stop'},
    'recovering':   {'abort', 'safe_stop'},
    'completed':    set(),
    'failed':       set(),
    'aborted':      set(),
    'safe_stopped': {'arm'},
}


# ---------------------------------------------------------------------------
# Task profile loader (reads YAMLs directly from share directory)
# ---------------------------------------------------------------------------

def _load_profiles(config_dir: str) -> list[dict[str, Any]]:
    """Return a list of task profile dicts sorted by display_name."""
    if not config_dir or not Path(config_dir).is_dir():
        return []
    profiles = []
    for path in sorted(Path(config_dir).glob('*.yaml')):
        try:
            with path.open('r', encoding='utf-8') as fh:
                data = yaml.safe_load(fh) or {}
            profiles.append(data)
        except Exception:  # noqa: BLE001
            pass
    return profiles


# ---------------------------------------------------------------------------
# Category filter combo helper
# ---------------------------------------------------------------------------

_CATEGORY_DISPLAY: dict[str, str] = {
    'function':    'Function Tasks',
    'full_course': 'Full Course',
    'qualifier':   'Qualifier',
}


def _category_label(cat: str) -> str:
    return _CATEGORY_DISPLAY.get(cat, cat.replace('_', ' ').title())


# ---------------------------------------------------------------------------
# Helpers: small reusable widgets
# ---------------------------------------------------------------------------

def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def _bold_label(text: str, size: int = 10) -> QLabel:
    lbl = QLabel(text)
    font = QFont()
    font.setBold(True)
    font.setPointSize(size)
    lbl.setFont(font)
    return lbl


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Operator GUI for task selection and state machine control."""

    def __init__(self, node, task_config_dir: str = '') -> None:
        super().__init__()
        self._node = node
        self._profiles = _load_profiles(task_config_dir)
        self._current_state = 'configuring'
        self._selected_task_id: str = ''

        self.setWindowTitle('IGVC Task Control')
        self.resize(1200, 800)
        self._build_ui()
        self._connect_signals()
        self._populate_task_list()
        self._update_button_states('configuring')

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(4)

        # ── Top splitter: task panel | status panel | prediction panel ──
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(self._build_task_panel())
        top_splitter.addWidget(self._build_status_panel())
        top_splitter.addWidget(self._build_prediction_panel())
        top_splitter.setSizes([320, 480, 340])
        top_splitter.setHandleWidth(4)
        root_layout.addWidget(top_splitter, stretch=2)

        root_layout.addWidget(_separator())

        # ── Camera feed ──────────────────────────────────────────────────
        root_layout.addWidget(self._build_camera_panel(), stretch=3)

        root_layout.addWidget(_separator())

        # ── Button bar ───────────────────────────────────────────────────
        root_layout.addWidget(self._build_button_bar())

        # ── Status bar ───────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage('Waiting for runner…')

    # ── Task Panel ────────────────────────────────────────────────────────

    def _build_task_panel(self) -> QGroupBox:
        box = QGroupBox('Task Selection')
        layout = QVBoxLayout(box)

        self._category_combo = QComboBox()
        self._category_combo.addItem('All Categories', userData=None)
        categories = sorted({
            p.get('category', '') for p in self._profiles if p.get('category')
        })
        for cat in categories:
            self._category_combo.addItem(_category_label(cat), userData=cat)
        self._category_combo.currentIndexChanged.connect(
            self._on_category_filter_changed)
        layout.addWidget(self._category_combo)

        self._task_list = QListWidget()
        self._task_list.setAlternatingRowColors(True)
        self._task_list.itemSelectionChanged.connect(
            self._on_task_selection_changed)
        layout.addWidget(self._task_list, stretch=1)

        self._select_btn = QPushButton('Select Task')
        self._select_btn.setEnabled(False)
        self._select_btn.setStyleSheet(
            'background:#555599; color:white; font-weight:bold;')
        self._select_btn.clicked.connect(self._on_select_task_clicked)
        layout.addWidget(self._select_btn)

        self._selected_task_label = QLabel('No task selected')
        self._selected_task_label.setWordWrap(True)
        self._selected_task_label.setStyleSheet('color:#aaaaaa;')
        layout.addWidget(self._selected_task_label)

        return box

    # ── Status Panel ──────────────────────────────────────────────────────

    def _build_status_panel(self) -> QGroupBox:
        box = QGroupBox('State Machine Status')
        layout = QVBoxLayout(box)

        # State badge
        state_row = QHBoxLayout()
        state_row.addWidget(QLabel('State:'))
        self._state_badge = QLabel('CONFIGURING')
        self._state_badge.setAlignment(Qt.AlignCenter)
        self._state_badge.setFixedHeight(28)
        self._state_badge.setStyleSheet(
            'background:#888888; color:white; font-weight:bold;'
            ' border-radius:4px; padding:0 8px;')
        state_row.addWidget(self._state_badge)
        state_row.addStretch()
        layout.addLayout(state_row)

        # YOLO override indicator
        self._override_label = QLabel('')
        self._override_label.setStyleSheet(
            'color:#ff9900; font-weight:bold;')
        layout.addWidget(self._override_label)

        layout.addWidget(_separator())

        # Task info grid
        info_grid = QGridLayout()
        info_grid.setColumnStretch(1, 1)

        self._task_id_val = QLabel('—')
        self._task_id_val.setWordWrap(True)
        info_grid.addWidget(_bold_label('Task:'), 0, 0)
        info_grid.addWidget(self._task_id_val, 0, 1)

        self._maneuver_val = QLabel('—')
        info_grid.addWidget(_bold_label('Maneuver:'), 1, 0)
        info_grid.addWidget(self._maneuver_val, 1, 1)

        self._step_val = QLabel('—')
        info_grid.addWidget(_bold_label('Step:'), 2, 0)
        info_grid.addWidget(self._step_val, 2, 1)

        self._distance_val = QLabel('0.0 m')
        info_grid.addWidget(_bold_label('Distance:'), 3, 0)
        info_grid.addWidget(self._distance_val, 3, 1)

        self._mode_val = QLabel('—')
        info_grid.addWidget(_bold_label('Mode:'), 4, 0)
        info_grid.addWidget(self._mode_val, 4, 1)

        layout.addLayout(info_grid)

        layout.addWidget(_separator())
        layout.addWidget(_bold_label('Health Checks:'))

        self._health_grid = QGridLayout()
        self._health_labels: dict[str, QLabel] = {}
        layout.addLayout(self._health_grid)

        layout.addStretch()
        return box

    # ── Prediction Panel ──────────────────────────────────────────────────

    def _build_prediction_panel(self) -> QGroupBox:
        box = QGroupBox('YOLO Task Prediction')
        layout = QVBoxLayout(box)

        self._prediction_top_label = QLabel('No prediction yet')
        self._prediction_top_label.setWordWrap(True)
        self._prediction_top_label.setStyleSheet(
            'font-weight:bold; color:#aaaaff;')
        layout.addWidget(self._prediction_top_label)

        layout.addWidget(_separator())
        layout.addWidget(QLabel('Top task candidates:'))

        self._prediction_bars: list[tuple[QLabel, QProgressBar, QLabel]] = []
        for _ in range(5):
            row = QHBoxLayout()
            name_lbl = QLabel('—')
            name_lbl.setFixedWidth(180)
            name_lbl.setWordWrap(True)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(16)
            bar.setTextVisible(False)
            conf_lbl = QLabel('0%')
            conf_lbl.setFixedWidth(36)
            row.addWidget(name_lbl)
            row.addWidget(bar)
            row.addWidget(conf_lbl)
            layout.addLayout(row)
            self._prediction_bars.append((name_lbl, bar, conf_lbl))

        layout.addWidget(_separator())
        layout.addWidget(QLabel('Detection evidence:'))

        self._evidence_label = QLabel('—')
        self._evidence_label.setWordWrap(True)
        self._evidence_label.setStyleSheet('color:#888888; font-size:9pt;')
        layout.addWidget(self._evidence_label)

        layout.addStretch()
        return box

    # ── Camera Panel ──────────────────────────────────────────────────────

    def _build_camera_panel(self) -> QGroupBox:
        box = QGroupBox('Front Camera (YOLO Overlay)')
        layout = QVBoxLayout(box)
        self._camera_label = QLabel()
        self._camera_label.setAlignment(Qt.AlignCenter)
        self._camera_label.setMinimumHeight(200)
        self._camera_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._camera_label.setStyleSheet('background:#111111;')
        self._camera_label.setText('No camera feed')
        layout.addWidget(self._camera_label)
        return box

    # ── Button Bar ────────────────────────────────────────────────────────

    def _build_button_bar(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)

        self._buttons: dict[str, QPushButton] = {}
        spec = [
            ('arm',               'ARM'),
            ('start',             'START'),
            ('pause',             'PAUSE'),
            ('resume',            'RESUME'),
            ('complete_maneuver', 'NEXT STEP'),
            ('abort',             'ABORT'),
            ('safe_stop',         'SAFE STOP'),
        ]
        for name, label in spec:
            btn = QPushButton(label)
            btn.setFixedHeight(36)
            btn.setStyleSheet(_BTN_STYLES[name])
            self._buttons[name] = btn
            layout.addWidget(btn)

        return w

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._node.status_updated.connect(self._on_status_updated)
        self._node.prediction_updated.connect(self._on_prediction_updated)
        self._node.image_updated.connect(self._on_image_updated)
        self._node.service_feedback.connect(
            lambda msg: self._status_bar.showMessage(msg, 6000))

        self._buttons['arm'].clicked.connect(self._node.call_arm)
        self._buttons['start'].clicked.connect(self._node.call_start)
        self._buttons['pause'].clicked.connect(self._node.call_pause)
        self._buttons['resume'].clicked.connect(self._node.call_resume)
        self._buttons['complete_maneuver'].clicked.connect(
            self._node.call_complete_maneuver)
        self._buttons['abort'].clicked.connect(self._node.call_abort)
        self._buttons['safe_stop'].clicked.connect(self._node.call_safe_stop)

    # ------------------------------------------------------------------
    # Task list population and filtering
    # ------------------------------------------------------------------

    def _populate_task_list(self, category_filter: str | None = None) -> None:
        self._task_list.clear()
        for profile in self._profiles:
            cat = profile.get('category', '')
            if category_filter and cat != category_filter:
                continue
            display = profile.get(
                'display_name', profile.get('task_id', ''))
            rule = profile.get('rule_ref', '')
            modes = ', '.join(profile.get('supports_modes', []))
            item_text = f'{display}\n  {rule}  [{modes}]'
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, profile.get('task_id', ''))
            # Colour-code by category
            if cat == 'full_course':
                item.setBackground(QColor('#1a1a3a'))
            elif cat == 'qualifier':
                item.setBackground(QColor('#1a2a1a'))
            self._task_list.addItem(item)

    @pyqtSlot()
    def _on_category_filter_changed(self) -> None:
        cat = self._category_combo.currentData()
        self._populate_task_list(category_filter=cat)

    @pyqtSlot()
    def _on_task_selection_changed(self) -> None:
        items = self._task_list.selectedItems()
        if items:
            self._selected_task_id = items[0].data(Qt.UserRole) or ''
            self._select_btn.setEnabled(
                bool(self._selected_task_id)
                and self._current_state in ('ready', 'safe_stopped'))
        else:
            self._select_btn.setEnabled(False)

    @pyqtSlot()
    def _on_select_task_clicked(self) -> None:
        if self._selected_task_id:
            self._node.select_task(self._selected_task_id)
            self._selected_task_label.setText(
                f'Selected: {self._selected_task_id}')

    # ------------------------------------------------------------------
    # Status update slots
    # ------------------------------------------------------------------

    @pyqtSlot(dict)
    def _on_status_updated(self, status: dict) -> None:
        state = status.get('state', 'configuring')
        self._current_state = state

        # State badge
        colour = _STATE_COLOURS.get(state, '#888888')
        self._state_badge.setText(state.upper().replace('_', ' '))
        self._state_badge.setStyleSheet(
            f'background:{colour}; color:white; font-weight:bold;'
            ' border-radius:4px; padding:0 8px;')

        # Task info
        self._task_id_val.setText(status.get('task_id', '—'))
        self._distance_val.setText(
            f"{status.get('distance_m', 0.0):.1f} m")
        self._mode_val.setText(
            f"{status.get('robot_mode', '—')} / "
            f"{status.get('task_mode', '—')}")

        # Behavior / maneuver
        behavior = status.get('behavior') or {}
        active_step = behavior.get('active_step') or {}
        step_idx = behavior.get('current_index', 0)
        step_count = behavior.get('step_count', 0)
        maneuver = active_step.get('maneuver', '—')
        self._maneuver_val.setText(maneuver)
        self._step_val.setText(
            f'{step_idx + 1} / {step_count}'
            if step_count else '—')

        # YOLO override indicator
        if status.get('yolo_override_active'):
            override_m = status.get('yolo_override_maneuver', '?')
            self._override_label.setText(
                f'⚡ YOLO override active: {override_m}')
        else:
            self._override_label.setText('')

        # Health checks
        missing: list[str] = status.get('missing_health', [])
        all_checks = set(self._health_labels.keys()) | set(
            k for k in ['odom', 'front_camera', 'lane_costmap',
                         'lane_path', 'localization_status', 'gps_fix'])
        all_checks |= set(missing)
        for check in all_checks:
            if check not in self._health_labels:
                lbl = QLabel(f'● {check}')
                count = len(self._health_labels)
                self._health_grid.addWidget(lbl, count // 2, count % 2)
                self._health_labels[check] = lbl
            ok = check not in missing
            self._health_labels[check].setStyleSheet(
                f'color:{"#33cc33" if ok else "#cc3300"};')

        # Update button enablement and Select Task button
        self._update_button_states(state)

    def _update_button_states(self, state: str) -> None:
        enabled = _STATE_BUTTONS.get(state, set())
        for name, btn in self._buttons.items():
            btn.setEnabled(name in enabled)
        # Select Task only available in READY / SAFE_STOPPED
        can_select = (
            state in ('ready', 'safe_stopped')
            and bool(self._selected_task_id))
        self._select_btn.setEnabled(can_select)

    # ------------------------------------------------------------------
    # Prediction panel update
    # ------------------------------------------------------------------

    @pyqtSlot(dict)
    def _on_prediction_updated(self, pred: dict) -> None:
        suggested = pred.get('suggested_task_id', '—')
        conf = pred.get('confidence', 0.0)
        self._prediction_top_label.setText(
            f'Suggested: {suggested}  ({conf * 100:.0f}%)')

        ranked: list[dict] = pred.get('ranked_tasks', [])
        for i, (name_lbl, bar, conf_lbl) in enumerate(self._prediction_bars):
            if i < len(ranked):
                entry = ranked[i]
                tid = entry.get('task_id', '—')
                c = entry.get('confidence', 0.0)
                # Find a nicer display name from loaded profiles
                display = next(
                    (p.get('display_name', tid)
                     for p in self._profiles
                     if p.get('task_id') == tid),
                    tid,
                )
                name_lbl.setText(display)
                bar.setValue(int(c * 100))
                conf_lbl.setText(f'{c * 100:.0f}%')
            else:
                name_lbl.setText('—')
                bar.setValue(0)
                conf_lbl.setText('0%')

        evidence: dict = pred.get('detection_evidence', {})
        if evidence:
            parts = [f'{cls}×{cnt}' for cls, cnt in evidence.items()]
            self._evidence_label.setText('  '.join(parts))
        else:
            self._evidence_label.setText('—')

    # ------------------------------------------------------------------
    # Camera feed update
    # ------------------------------------------------------------------

    @pyqtSlot(np.ndarray)
    def _on_image_updated(self, bgr: np.ndarray) -> None:
        rgb = bgr[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # Scale to fit the label while preserving aspect ratio
        lbl_w = self._camera_label.width()
        lbl_h = self._camera_label.height()
        scaled = pixmap.scaled(
            lbl_w, lbl_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._camera_label.setPixmap(scaled)
