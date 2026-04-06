"""VOX Control Center — main application window.

Shell layout: left navigation rail (NavRail) + QStackedWidget.
Pages: Dashboard · Audio · Settings · History · Diagnostics

The Settings page unifies Activation, Assistant, Voice/TTS, Permissions,
App Aliases, and Search Directories into one scrollable surface, replacing
the previous 5-tab flat navigation model with a real settings experience.

Usage:
    cc = ControlCenter(config, app_state, speaker)
    cc.restart_listener_requested.connect(vox_app.restart_listener)
    cc.rerun_validation_requested.connect(vox_app.run_validation)
    cc.show()
"""
from __future__ import annotations

import os
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

from audio_utils import (
    compute_rms,
    estimate_noise_floor,
    estimate_speech_rms,
    suggest_silence_threshold,
    signal_quality_label,
    compute_clipping_fraction,
    classify_capture_issue,
)
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QPushButton, QComboBox, QLineEdit, QCheckBox,
    QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QGridLayout,
    QFileDialog, QSizePolicy, QTextEdit, QRadioButton, QButtonGroup,
    QAbstractItemView,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QIcon, QFont

from utils.config import Config
from ui.mic_meter import MicLevelBar
from ui import theme


# ── Colour tokens (sourced from shared theme) ──────────────────────────────────
_BG      = theme.BG
_PANEL   = theme.PANEL
_CARD    = theme.CARD
_BORDER  = theme.BORDER
_BORDER2 = theme.BORDER_STRONG
_ACCENT  = theme.ACCENT
_TEXT    = theme.TEXT
_TEXT2   = theme.TEXT2
_MUTED   = theme.MUTED
_SUCCESS = theme.SUCCESS
_WARNING = theme.WARNING
_ERROR   = theme.ERROR
_INFO    = theme.INFO
_FONT    = theme.FONT

# All known executor actions with description and risk label
_ALL_ACTIONS = [
    ("open_app",         "Open an application by name or alias",           "low"),
    ("close_app",        "Close a running application by name",            "low"),
    ("set_volume",       "Set system volume (0–100)",                      "low"),
    ("mute_volume",      "Toggle system audio mute",                       "low"),
    ("play_pause_media", "Play or pause media playback",                   "low"),
    ("next_track",       "Skip to the next media track",                   "low"),
    ("prev_track",       "Go back to the previous media track",            "low"),
    ("search_file",      "Search for files in the configured directories", "low"),
    ("open_url",         "Open a URL in the default browser",              "medium"),
    ("type_text",        "Simulate keyboard typing of arbitrary text",     "medium"),
    ("take_screenshot",  "Capture a screenshot and save to Desktop",       "low"),
    ("show_time",        "Report the current time",                        "low"),
    ("show_battery",     "Report battery status and percentage",           "low"),
]

_CC_STYLE = f"""
QMainWindow, QWidget {{
    background: {_BG}; color: {_TEXT};
    font-size: 13px; font-family: {_FONT};
}}

/* ── Nav rail ── */
QWidget#nav_rail {{
    background: {_PANEL};
    border-right: 1px solid {_BORDER};
}}
QPushButton#nav_btn {{
    text-align: left;
    padding: 12px 18px 12px 22px;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 0;
    font-size: 13px;
    font-weight: 500;
    color: {_MUTED};
    background: transparent;
    letter-spacing: 0.2px;
    min-height: 42px;
}}
QPushButton#nav_btn:checked {{
    color: {_ACCENT};
    background: rgba(168,85,247,0.10);
    border-left: 3px solid {_ACCENT};
    font-weight: 600;
}}
QPushButton#nav_btn:hover:!checked {{
    background: rgba(255,255,255,0.04);
    color: {_TEXT2};
    border-left: 3px solid rgba(168,85,247,0.20);
}}

/* ── Buttons ── */
QPushButton {{
    background: rgba(168,85,247,0.10); border: 1px solid rgba(168,85,247,0.25);
    border-radius: 7px; color: {_TEXT}; padding: 7px 18px;
    font-size: 12px; font-weight: 500; letter-spacing: 0.2px;
}}
QPushButton:hover {{
    background: rgba(168,85,247,0.22); border-color: rgba(168,85,247,0.50);
    color: #f0eeff;
}}
QPushButton:pressed {{ background: rgba(168,85,247,0.35); }}
QPushButton:disabled {{
    background: rgba(255,255,255,0.03); border-color: {_BORDER};
    color: {_MUTED};
}}
QPushButton#primary {{
    background: rgba(168,85,247,0.22); border-color: rgba(168,85,247,0.55);
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background: rgba(168,85,247,0.35); border-color: rgba(168,85,247,0.75);
}}
QPushButton#danger {{
    background: rgba(248,113,113,0.08); border-color: rgba(248,113,113,0.25);
    color: {_TEXT2};
}}
QPushButton#danger:hover {{
    background: rgba(248,113,113,0.20); border-color: rgba(248,113,113,0.50);
    color: {_TEXT};
}}

/* ── Inputs ── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: rgba(255,255,255,0.04); border: 1px solid {_BORDER};
    border-radius: 6px; color: {_TEXT}; padding: 6px 10px; font-size: 12px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: rgba(168,85,247,0.55); background: rgba(168,85,247,0.06);
    outline: none;
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    border-color: {_BORDER2};
}}
QComboBox::drop-down {{ border: none; padding-right: 6px; }}
QComboBox QAbstractItemView {{
    background: {_CARD}; border: 1px solid {_BORDER2};
    selection-background-color: rgba(168,85,247,0.22);
    color: {_TEXT};
}}

/* ── Text editor (log views) ── */
QTextEdit {{
    background: rgba(0,0,0,0.25); border: 1px solid {_BORDER};
    border-radius: 8px; color: {_TEXT2}; padding: 10px 12px;
    font-size: 12px; line-height: 1.6;
    selection-background-color: rgba(168,85,247,0.28);
}}

/* ── Toggle controls ── */
QCheckBox {{ color: {_TEXT}; font-size: 12px; spacing: 9px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {_BORDER2}; border-radius: 4px;
    background: rgba(255,255,255,0.04);
}}
QCheckBox::indicator:checked {{
    background: {_ACCENT}; border-color: {_ACCENT};
}}
QCheckBox::indicator:hover {{ border-color: rgba(168,85,247,0.65); }}
QRadioButton {{ color: {_TEXT}; font-size: 12px; spacing: 9px; }}
QRadioButton::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {_BORDER2}; border-radius: 7px;
    background: rgba(255,255,255,0.04);
}}
QRadioButton::indicator:checked {{ background: {_ACCENT}; border-color: {_ACCENT}; }}
QRadioButton::indicator:hover {{ border-color: rgba(168,85,247,0.65); }}

/* ── Group boxes ── */
QGroupBox {{
    background: rgba(255,255,255,0.016);
    border: 1px solid {_BORDER}; border-radius: 10px;
    margin-top: 18px; padding: 14px 16px 14px 16px;
    color: #8a86a2; font-size: 10px; letter-spacing: 1.2px;
    font-weight: 700; text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    padding: 0 6px; left: 16px; top: 2px;
}}

/* ── Lists and tables ── */
QListWidget, QTableWidget {{
    background: rgba(0,0,0,0.20); border: 1px solid {_BORDER};
    border-radius: 8px; color: {_TEXT}; font-size: 12px;
    outline: none;
}}
QListWidget::item {{ padding: 8px 12px; border-radius: 5px; }}
QListWidget::item:selected {{
    background: rgba(168,85,247,0.20); color: {_TEXT};
}}
QListWidget::item:hover {{ background: rgba(255,255,255,0.05); }}
QTableWidget::item:selected {{ background: rgba(168,85,247,0.18); color: {_TEXT}; }}
QTableWidget {{ gridline-color: transparent; }}
QHeaderView::section {{
    background: rgba(255,255,255,0.03); color: #8a86a2;
    border: none; border-bottom: 1px solid {_BORDER};
    padding: 8px 12px; font-size: 10px;
    letter-spacing: 1.0px; font-weight: 700; text-transform: uppercase;
}}

/* ── Scrollbar ── */
QScrollBar:vertical {{
    background: transparent; width: 6px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {_BORDER2}; border-radius: 3px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {_MUTED}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 6px; margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {_BORDER2}; border-radius: 3px; min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{ background: {_MUTED}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QScrollArea {{ border: none; background: transparent; }}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"background: {_BORDER}; border: none; max-height: 1px;")
    return line


def _section(text: str) -> QLabel:
    """All-caps field label for form grids."""
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: #8a86a2; font-size: 10px; letter-spacing: 1.1px; font-weight: 700; "
        f"background: transparent;"
    )
    return lbl


def _note(text: str) -> QLabel:
    """Helper/hint text below form controls — italic, muted, wrapped."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        f"color: {_MUTED}; font-size: 11px; font-style: italic; "
        f"line-height: 1.5; background: transparent; padding-left: 2px;"
    )
    return lbl


def _apply_tag(text: str, color: str | None = None) -> QLabel:
    """Inline notice with left accent border showing when a setting takes effect."""
    c = color or _INFO
    bg = "transparent"
    h = c.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        bg = f"rgba({r},{g},{b},0.07)"
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        f"color: {c}; font-size: 11px; letter-spacing: 0.2px; "
        f"border-top: none; border-right: none; border-bottom: none; "
        f"border-left: 2px solid {c}; padding: 4px 8px 4px 10px; "
        f"border-radius: 0px 4px 4px 0px; "
        f"background: {bg};"
    )
    return lbl


def _status_chip(text: str, color: str) -> QLabel:
    """Small pill-shaped status badge with coloured border and tinted background."""
    h = color.lstrip("#")
    bg = "transparent"
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        bg = f"rgba({r},{g},{b},0.12)"
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {color}; background: {bg}; font-size: 10px; "
        f"font-weight: 600; letter-spacing: 0.4px; "
        f"border: 1px solid {color}; border-radius: 4px; padding: 1px 7px;"
    )
    return lbl


# ── Settings page structural helpers ──────────────────────────────────────────

def _settings_heading(title: str, description: str = "",
                      top_margin: int = 28) -> QWidget:
    """Large section title for the unified settings page."""
    w = QWidget()
    vb = QVBoxLayout(w)
    vb.setContentsMargins(0, top_margin, 0, 12)
    vb.setSpacing(5)
    t = QLabel(title)
    t.setStyleSheet(
        f"color: {_TEXT}; font-size: 15px; font-weight: 600; "
        f"letter-spacing: 0.2px; background: transparent;"
    )
    vb.addWidget(t)
    if description:
        d = QLabel(description)
        d.setWordWrap(True)
        d.setStyleSheet(
            f"color: {_MUTED}; font-size: 12px; background: transparent; line-height: 1.5;"
        )
        vb.addWidget(d)
    return w


def _settings_card() -> QFrame:
    """Subtle card container for a settings section's form content."""
    f = QFrame()
    f.setStyleSheet(
        "QFrame {"
        f"  background: rgba(255,255,255,0.018);"
        f"  border: 1px solid {_BORDER};"
        "  border-radius: 10px;"
        "}"
    )
    return f


def _settings_divider() -> QWidget:
    """Full-width horizontal rule between settings sections, with spacing."""
    w = QWidget()
    vb = QVBoxLayout(w)
    vb.setContentsMargins(0, 8, 0, 0)
    vb.setSpacing(0)
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {_BORDER}; border: none;")
    vb.addWidget(f)
    return w


def _card_sep() -> QFrame:
    """Subtle horizontal separator within a settings card."""
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet("background: rgba(255,255,255,0.06); border: none;")
    return f


# ── Base settings panel ────────────────────────────────────────────────────────

class _SettingsPanel(QWidget):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self._config = config

    def _show_saved(self, lbl: QLabel, ok: bool = True, msg: str = "") -> None:
        text = msg or ("✓  Saved" if ok else "✕  Error")
        color = _SUCCESS if ok else _ERROR
        lbl.setText(text)
        lbl.setStyleSheet(
            f"color: {color}; font-size: 11px; font-weight: 600; "
            f"letter-spacing: 0.2px;"
        )
        QTimer.singleShot(3000, lambda: lbl.setText(""))

    def _save_row(self) -> tuple[QHBoxLayout, QLabel, QPushButton]:
        row = QHBoxLayout()
        row.setContentsMargins(0, 6, 0, 0)
        status = QLabel("")
        status.setMinimumWidth(160)
        btn = QPushButton("Save Changes")
        btn.setObjectName("primary")
        btn.setMinimumWidth(130)
        row.addStretch()
        row.addWidget(status)
        row.addSpacing(8)
        row.addWidget(btn)
        return row, status, btn


# ── Page: Dashboard ────────────────────────────────────────────────────────────

class DashboardTab(QWidget):
    def __init__(self, app_state, config: Config, parent=None):
        super().__init__(parent)
        self._state  = app_state
        self._config = config
        self._build()
        self._connect()

    def _build(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        root = QVBoxLayout(inner)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        # ── Status hero card ──────────────────────────────────────────────────
        status_card = QFrame()
        status_card.setObjectName("status_card")
        status_card.setStyleSheet(
            "QFrame#status_card {"
            "  background: rgba(168,85,247,0.07);"
            "  border: 1px solid rgba(168,85,247,0.22);"
            "  border-radius: 12px;"
            "}"
        )
        sc_layout = QHBoxLayout(status_card)
        sc_layout.setContentsMargins(20, 16, 20, 16)
        sc_layout.setSpacing(14)

        left_col = QVBoxLayout()
        left_col.setSpacing(4)
        status_hdr = QLabel("STATUS")
        status_hdr.setStyleSheet(
            "color: #7a7490; font-size: 9px; letter-spacing: 1.6px; "
            "font-weight: 700; background: transparent; border: none;"
        )
        status_row_w = QHBoxLayout()
        status_row_w.setSpacing(7)
        self._dot_big = QLabel("●")
        self._dot_big.setStyleSheet(
            f"color: {_MUTED}; font-size: 13px; background: transparent; border: none;"
        )
        self._lbl_status = QLabel("Idle")
        self._lbl_status.setStyleSheet(
            f"color: {_TEXT}; font-size: 17px; font-weight: 700; "
            f"background: transparent; border: none;"
        )
        status_row_w.addWidget(self._dot_big)
        status_row_w.addWidget(self._lbl_status)
        status_row_w.addStretch()
        left_col.addWidget(status_hdr)
        left_col.addLayout(status_row_w)
        sc_layout.addLayout(left_col, 1)

        def _health_col(hdr_text: str):
            col = QVBoxLayout()
            col.setSpacing(4)
            hdr = QLabel(hdr_text)
            hdr.setStyleSheet(
                "color: #7a7490; font-size: 9px; letter-spacing: 1.4px; "
                "font-weight: 700; background: transparent; border: none;"
            )
            val = QLabel("–")
            val.setStyleSheet(
                f"color: {_TEXT}; font-size: 12px; font-weight: 600; "
                "background: transparent; border: none;"
            )
            col.addWidget(hdr)
            col.addWidget(val)
            return col, val

        for hdr_text in ("LLM", "TTS", "VOICE"):
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setStyleSheet(
                "background: rgba(168,85,247,0.15); border: none; max-width: 1px;"
            )
            sc_layout.addWidget(sep)
            col, val = _health_col(hdr_text)
            if hdr_text == "LLM":
                self._lbl_ollama = val
            elif hdr_text == "TTS":
                self._lbl_piper = val
            else:
                self._lbl_voice = val
            sc_layout.addLayout(col)

        root.addWidget(status_card)

        # ── Active Configuration ──────────────────────────────────────────────
        grp_cfg = QGroupBox("Active Configuration")
        g = QGridLayout(grp_cfg)
        g.setColumnMinimumWidth(0, 130)
        g.setHorizontalSpacing(16)
        g.setVerticalSpacing(8)
        g.setColumnStretch(1, 1)

        _val_ss = f"color: {_TEXT}; font-size: 12px; font-weight: 500; background: transparent;"
        self._lbl_model    = QLabel("–")
        self._lbl_language = QLabel("–")
        self._lbl_mode     = QLabel("–")
        self._lbl_mic      = QLabel("–")
        self._lbl_out      = QLabel("–")
        for w in (self._lbl_model, self._lbl_language, self._lbl_mode,
                  self._lbl_mic, self._lbl_out):
            w.setStyleSheet(_val_ss)

        for row, (label, widget) in enumerate([
            ("Model",       self._lbl_model),
            ("Language",    self._lbl_language),
            ("Activation",  self._lbl_mode),
            ("Microphone",  self._lbl_mic),
            ("Output",      self._lbl_out),
        ]):
            g.addWidget(_section(label), row, 0, Qt.AlignmentFlag.AlignTop)
            g.addWidget(widget, row, 1)

        root.addWidget(grp_cfg)

        # ── Last Interaction ──────────────────────────────────────────────────
        grp_sess = QGroupBox("Last Interaction")
        g3 = QGridLayout(grp_sess)
        g3.setColumnMinimumWidth(0, 130)
        g3.setHorizontalSpacing(16)
        g3.setVerticalSpacing(8)
        g3.setColumnStretch(1, 1)

        self._lbl_cmd    = QLabel("–")
        self._lbl_resp   = QLabel("–")
        self._lbl_action = QLabel("–")
        for w in (self._lbl_cmd, self._lbl_resp, self._lbl_action):
            w.setWordWrap(True)
            w.setStyleSheet(f"color: {_TEXT2}; font-size: 12px; background: transparent;")

        for row, (label, widget) in enumerate([
            ("Command",  self._lbl_cmd),
            ("Response", self._lbl_resp),
            ("Action",   self._lbl_action),
        ]):
            g3.addWidget(_section(label), row, 0, Qt.AlignmentFlag.AlignTop)
            g3.addWidget(widget, row, 1)

        root.addWidget(grp_sess)
        root.addStretch()

    def _connect(self) -> None:
        s = self._state
        s.status_changed.connect(self._on_status)
        s.ollama_ok_changed.connect(self._on_ollama)
        s.transcript_changed.connect(self._lbl_cmd.setText)
        s.response_changed.connect(self._lbl_resp.setText)
        s.last_action_changed.connect(self._lbl_action.setText)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        c = self._config
        project_root = _project_root()
        piper_raw = c.get("piper_path", "piper/piper/piper.exe")
        piper_ok  = os.path.exists(_resolve(project_root, piper_raw))
        voice_raw = c.get("voice_model", "")
        voice_ok  = os.path.exists(_resolve(project_root, voice_raw))

        self._lbl_piper.setText("Found" if piper_ok else "Missing")
        c_piper = _SUCCESS if piper_ok else _ERROR
        self._lbl_piper.setStyleSheet(
            f"color: {c_piper}; font-size: 12px; font-weight: 600; "
            f"background: transparent; border: none;"
        )
        self._lbl_voice.setText("Found" if voice_ok else "Missing")
        c_voice = _SUCCESS if voice_ok else _ERROR
        self._lbl_voice.setStyleSheet(
            f"color: {c_voice}; font-size: 12px; font-weight: 600; "
            f"background: transparent; border: none;"
        )

        self._lbl_model.setText(c.get("ollama_model", "–"))
        self._lbl_language.setText(c.get("language", "auto").upper())

        mode = c.get("activation_mode", "wake_word")
        if mode == "push_to_talk":
            key = c.get("push_to_talk_key", "ctrl+shift")
            self._lbl_mode.setText(f'Push-to-Talk  ({key.upper()})')
        else:
            ww = c.get("wake_word", "vox")
            self._lbl_mode.setText(f'Wake Word  ("{ww}")')

        try:
            devices   = sd.query_devices()
            mic_idx   = c.get("mic_device",    None)
            out_idx   = c.get("output_device", None)
            def_in, def_out = sd.default.device
            mic_name = devices[mic_idx]["name"] if mic_idx is not None and mic_idx < len(devices) \
                       else f'{devices[def_in]["name"]} (default)'
            out_name = devices[out_idx]["name"] if out_idx is not None and out_idx < len(devices) \
                       else f'{devices[def_out]["name"]} (default)'
        except Exception:
            mic_name = out_name = "Unknown"

        self._lbl_mic.setText(mic_name)
        self._lbl_out.setText(out_name)

        self._on_ollama(self._state.ollama_ok)
        self._on_status(self._state.status)

    @pyqtSlot(str)
    def _on_status(self, status: str) -> None:
        _map = {
            "idle":         ("Idle",          theme.MUTED,    ),
            "monitoring":   ("Monitoring",    theme.SUCCESS,  ),
            "listening":    ("Listening",     theme.LISTENING,),
            "transcribing": ("Transcribing",  theme.WARNING,  ),
            "generating":   ("Generating",    theme.ACTIVE,   ),
            "responding":   ("Responding",    theme.ACTIVE,   ),
            "speaking":     ("Speaking",      theme.ACTIVE,   ),
            "error":        ("Error",         theme.ERROR,    ),
            "cancelled":    ("Cancelled",     theme.ERROR,    ),
        }
        text, color = _map.get(status, (status.capitalize(), _TEXT))
        self._lbl_status.setText(text)
        self._lbl_status.setStyleSheet(
            f"color: {color}; font-size: 18px; font-weight: 700; "
            f"background: transparent; border: none; letter-spacing: 0.3px;"
        )
        self._dot_big.setStyleSheet(
            f"color: {color}; font-size: 15px; background: transparent; border: none;"
        )

    @pyqtSlot(bool)
    def _on_ollama(self, ok: bool) -> None:
        self._lbl_ollama.setText("Connected" if ok else "Offline")
        self._lbl_ollama.setStyleSheet(
            f"color: {_SUCCESS if ok else _ERROR}; font-size: 12px; "
            f"font-weight: 600; background: transparent; border: none;"
        )


# ── Page: Audio ────────────────────────────────────────────────────────────────

class AudioTab(_SettingsPanel):
    _mic_test_done  = pyqtSignal(float, float)
    _mic_test_error = pyqtSignal(str)
    _calib_done     = pyqtSignal(dict)
    _calib_error    = pyqtSignal(str)
    _calib_phase    = pyqtSignal(str)
    _stt_done       = pyqtSignal(str, str, str)
    _stt_error      = pyqtSignal(str)

    def __init__(self, config: Config, app_state, speaker,
                 restart_cb: Callable | None = None,
                 stt_cb: Callable | None = None,
                 parent=None):
        super().__init__(config, parent)
        self._state      = app_state
        self._speaker    = speaker
        self._restart_cb = restart_cb
        self._stt_cb     = stt_cb
        self._build()
        self._connect()

    def _build(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        root = QVBoxLayout(inner)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        # ── Device selection ───────────────────────────────────────────────────
        grp_dev = QGroupBox("Audio Devices")
        g = QGridLayout(grp_dev)
        g.setColumnMinimumWidth(0, 160)
        g.setHorizontalSpacing(16)
        g.setVerticalSpacing(10)
        g.setColumnStretch(1, 1)

        g.addWidget(_section("Input (Microphone)"), 0, 0)
        self._combo_in = QComboBox()
        g.addWidget(self._combo_in, 0, 1)

        g.addWidget(_section("Output (Speakers)"), 1, 0)
        self._combo_out = QComboBox()
        g.addWidget(self._combo_out, 1, 1)

        root.addWidget(grp_dev)
        root.addWidget(_note("Changing microphone restarts the listener immediately. "
                             "Output changes take effect on next TTS call."))

        # ── Level meter ────────────────────────────────────────────────────────
        grp_lvl = QGroupBox("Microphone Level")
        lv = QHBoxLayout(grp_lvl)
        lv.setSpacing(12)
        lv.addWidget(_section("Live Input"))
        self._mic_bar = MicLevelBar()
        lv.addWidget(self._mic_bar, 1)
        root.addWidget(grp_lvl)

        # ── Signal health card ─────────────────────────────────────────────────
        grp_health = QGroupBox("Signal Health")
        hv = QGridLayout(grp_health)
        hv.setColumnMinimumWidth(0, 160)
        hv.setHorizontalSpacing(16)
        hv.setVerticalSpacing(6)
        hv.setColumnStretch(1, 1)

        self._lbl_noise   = QLabel("–")
        self._lbl_snr     = QLabel("–")
        self._lbl_clip    = QLabel("–")
        self._lbl_quality = QLabel("–")

        for row, (label, widget) in enumerate([
            ("Noise Floor (RMS)", self._lbl_noise),
            ("SNR",               self._lbl_snr),
            ("Clipping",          self._lbl_clip),
            ("Quality",           self._lbl_quality),
        ]):
            hv.addWidget(_section(label), row, 0)
            hv.addWidget(widget, row, 1)

        root.addWidget(grp_health)

        # ── Calibration ────────────────────────────────────────────────────────
        grp_calib = QGroupBox("Microphone Calibration")
        cv = QVBoxLayout(grp_calib)
        cv.setSpacing(8)

        self._calib_status = QLabel(
            "Run calibration to measure your noise floor and get a recommended silence threshold."
        )
        self._calib_status.setWordWrap(True)
        self._calib_status.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        cv.addWidget(self._calib_status)

        calib_btns = QHBoxLayout()
        self._btn_calib = QPushButton("Calibrate  (3 s silence + 3 s speech)")
        calib_btns.addWidget(self._btn_calib)
        calib_btns.addStretch()
        cv.addLayout(calib_btns)

        self._calib_result = QLabel("")
        self._calib_result.setWordWrap(True)
        self._calib_result.setStyleSheet(f"font-size: 11px;")
        cv.addWidget(self._calib_result)

        self._btn_apply_thresh = QPushButton("Apply Suggested Threshold")
        self._btn_apply_thresh.setVisible(False)
        cv.addWidget(self._btn_apply_thresh)

        root.addWidget(grp_calib)

        # ── Device testing ─────────────────────────────────────────────────────
        grp_test = QGroupBox("Device Testing")
        tv = QVBoxLayout(grp_test)
        test_row = QHBoxLayout()
        self._btn_mic = QPushButton("Test Microphone  (3 s)")
        self._btn_tts = QPushButton("Test TTS")
        test_row.addWidget(self._btn_mic)
        test_row.addWidget(self._btn_tts)
        test_row.addStretch()
        tv.addLayout(test_row)
        self._test_lbl = QLabel("")
        self._test_lbl.setStyleSheet(f"font-size: 11px;")
        tv.addWidget(self._test_lbl)
        root.addWidget(grp_test)

        # ── STT test ───────────────────────────────────────────────────────────
        grp_stt = QGroupBox("Speech Recognition Test")
        sv = QVBoxLayout(grp_stt)
        sv.setSpacing(8)

        sv.addWidget(_note(
            "Records 5 s and runs Whisper on the audio to verify speech-to-text quality. "
            "VOX must be idle (not listening) for this test."
        ))
        stt_btns = QHBoxLayout()
        self._btn_stt = QPushButton("Run STT Test  (5 s)")
        self._btn_stt.setEnabled(self._stt_cb is not None)
        stt_btns.addWidget(self._btn_stt)
        stt_btns.addStretch()
        sv.addLayout(stt_btns)

        self._stt_result = QLabel("")
        self._stt_result.setWordWrap(True)
        self._stt_result.setStyleSheet(f"font-size: 11px;")
        sv.addWidget(self._stt_result)

        root.addWidget(grp_stt)
        root.addStretch()

        save_row, self._save_lbl, btn_save = self._save_row()
        root.addLayout(save_row)

        self._populate_devices()
        btn_save.clicked.connect(self._save)
        self._btn_mic.clicked.connect(self._test_mic)
        self._btn_tts.clicked.connect(self._test_tts)
        self._btn_calib.clicked.connect(self._run_calibration)
        self._btn_stt.clicked.connect(self._run_stt_test)
        self._btn_apply_thresh.clicked.connect(self._apply_suggested_threshold)
        self._mic_test_done.connect(self._on_mic_done)
        self._mic_test_error.connect(self._on_mic_error)
        self._calib_done.connect(self._on_calib_done)
        self._calib_error.connect(self._on_calib_error)
        self._stt_done.connect(self._on_stt_done)
        self._stt_error.connect(self._on_stt_error)

        self._suggested_threshold: float | None = None

    def _connect(self) -> None:
        self._state.mic_level_changed.connect(self._mic_bar.set_level)
        self._calib_phase.connect(self._on_calib_phase)

    @pyqtSlot(str)
    def _on_calib_phase(self, text: str) -> None:
        self._calib_status.setText(text)
        self._calib_status.setStyleSheet(f"color: {_WARNING}; font-size: 11px;")

    def _populate_devices(self) -> None:
        self._combo_in.clear()
        self._combo_out.clear()
        self._combo_in.addItem("System Default", None)
        self._combo_out.addItem("System Default", None)
        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                try:
                    host = f" [{sd.query_hostapis(d['hostapi'])['name']}]"
                except Exception:
                    host = ""
                label = f"[{i}] {d['name']}{host}"
                if d["max_input_channels"] > 0:
                    self._combo_in.addItem(label, i)
                if d["max_output_channels"] > 0:
                    self._combo_out.addItem(label, i)
        except Exception:
            pass
        self._select_combo(self._combo_in,  self._config.get("mic_device",    None))
        self._select_combo(self._combo_out, self._config.get("output_device", None))

    def _select_combo(self, combo: QComboBox, value) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._populate_devices()

    def _save(self) -> None:
        old_mic = self._config.get("mic_device", None)
        new_mic = self._combo_in.currentData()
        new_out = self._combo_out.currentData()
        self._config.set("mic_device",    new_mic)
        self._config.set("output_device", new_out)
        self._config.save()
        if self._speaker:
            self._speaker.reload_config()
        msg = "Saved."
        if new_mic != old_mic and self._restart_cb:
            self._restart_cb()
            msg = "Saved. Listener restarting…"
            self._state.add_diagnostic("info", "Microphone device changed — listener restarted.")
        self._show_saved(self._save_lbl, True, msg)

    def _test_mic(self) -> None:
        self._test_lbl.setText("Recording for 3 s…")
        self._test_lbl.setStyleSheet(f"color: {_WARNING}; font-size: 11px;")
        mic_idx = self._combo_in.currentData()

        def _record() -> None:
            try:
                data = sd.rec(int(16000 * 3), samplerate=16000, channels=1,
                              dtype="float32", device=mic_idx)
                sd.wait()
                audio = data.flatten()
                peak  = float(np.max(np.abs(audio)))
                level = min(1.0, peak / 0.30)
                self._mic_test_done.emit(peak, level)
            except Exception as exc:
                self._mic_test_error.emit(str(exc))

        threading.Thread(target=_record, daemon=True).start()

    @pyqtSlot(float, float)
    def _on_mic_done(self, peak: float, level: float) -> None:
        pct = int(level * 100)
        if level < 0.05:
            msg, c = f"Peak {peak:.3f} ({pct}%) — very quiet, check mic", _ERROR
        elif level > 0.95:
            msg, c = f"Peak {peak:.3f} ({pct}%) — may clip, lower gain", _WARNING
        else:
            msg, c = f"Peak {peak:.3f} ({pct}%) — OK", _SUCCESS
        self._test_lbl.setText(msg)
        self._test_lbl.setStyleSheet(f"color: {c}; font-size: 11px;")

    @pyqtSlot(str)
    def _on_mic_error(self, err: str) -> None:
        self._test_lbl.setText(f"Error: {err}")
        self._test_lbl.setStyleSheet(f"color: {_ERROR}; font-size: 11px;")

    def _test_tts(self) -> None:
        if self._speaker:
            self._speaker.speak("VOX is ready.")
            self._test_lbl.setText("TTS test sent.")
            self._test_lbl.setStyleSheet(f"color: {_SUCCESS}; font-size: 11px;")
        else:
            self._test_lbl.setText("Speaker unavailable.")
            self._test_lbl.setStyleSheet(f"color: {_ERROR}; font-size: 11px;")

    def _run_calibration(self) -> None:
        self._btn_calib.setEnabled(False)
        self._btn_apply_thresh.setVisible(False)
        self._suggested_threshold = None
        self._calib_result.setText("")
        self._calib_status.setText("Phase 1/2 — Stay silent for 3 seconds…")
        self._calib_status.setStyleSheet(f"color: {_WARNING}; font-size: 11px;")
        mic_idx = self._combo_in.currentData()

        def _run() -> None:
            try:
                SR = 16000
                silence_data = sd.rec(SR * 3, samplerate=SR, channels=1,
                                      dtype="float32", device=mic_idx)
                sd.wait()
                silence_audio = silence_data.flatten()
                noise_floor   = estimate_noise_floor(silence_audio)

                self._calib_phase.emit("Phase 2/2 — Speak normally for 3 seconds…")
                speech_data = sd.rec(SR * 3, samplerate=SR, channels=1,
                                     dtype="float32", device=mic_idx)
                sd.wait()
                speech_audio  = speech_data.flatten()
                speech_rms    = estimate_speech_rms(speech_audio, noise_floor)
                clip_frac     = compute_clipping_fraction(speech_audio)
                suggested     = suggest_silence_threshold(noise_floor)
                quality, expl = signal_quality_label(noise_floor, speech_rms)
                snr           = speech_rms / noise_floor if noise_floor > 0 else float("inf")

                self._calib_done.emit({
                    "noise_floor": noise_floor,
                    "speech_rms":  speech_rms,
                    "snr":         snr,
                    "clip_frac":   clip_frac,
                    "suggested":   suggested,
                    "quality":     quality,
                    "explanation": expl,
                })
            except Exception as exc:
                self._calib_error.emit(str(exc))

        threading.Thread(target=_run, daemon=True).start()

    @pyqtSlot(dict)
    def _on_calib_done(self, result: dict) -> None:
        self._btn_calib.setEnabled(True)
        self._calib_status.setText("Calibration complete.")
        self._calib_status.setStyleSheet(f"color: {_SUCCESS}; font-size: 11px;")

        nf  = result["noise_floor"]
        snr = result["snr"]
        cf  = result["clip_frac"]
        sug = result["suggested"]
        ql  = result["quality"]
        ex  = result["explanation"]

        q_color = {
            "good": _SUCCESS, "fair": _WARNING, "poor": _ERROR, "no_signal": _ERROR
        }.get(ql, _MUTED)

        sil_thresh = float(self._config.get("silence_threshold", 0.02))
        diagnosis  = classify_capture_issue(nf, result["speech_rms"], cf, sil_thresh)
        sev_color  = {
            "ok": _SUCCESS, "warning": _WARNING, "error": _ERROR
        }.get(diagnosis["severity"], _MUTED)

        snr_text = f"{snr:.1f}×" if snr != float("inf") else "∞"

        lines = [
            f"Noise floor:              {nf:.4f} RMS",
            f"Speech RMS:               {result['speech_rms']:.4f}",
            f"SNR:                      {snr_text}",
            f"Clipping:                 {cf*100:.1f}%",
            f"Suggested threshold:      {sug:.4f}",
            f"Signal quality:           {ql.upper()} — {ex}",
            "",
            f"Diagnosis:  {diagnosis['title']}",
            f"  {diagnosis['detail']}",
        ]
        self._calib_result.setText("\n".join(lines))
        self._calib_result.setStyleSheet(
            f"color: {sev_color}; font-size: 11px; line-height: 1.5;"
        )

        self._suggested_threshold = sug
        self._btn_apply_thresh.setVisible(True)

        self._lbl_noise.setText(f"{nf:.4f}")
        self._lbl_snr.setText(snr_text)
        self._lbl_clip.setText(f"{cf*100:.1f}%")
        self._lbl_clip.setStyleSheet(f"color: {_ERROR if cf > 0.01 else _SUCCESS};")
        self._lbl_quality.setText(ql.upper())
        self._lbl_quality.setStyleSheet(f"color: {q_color};")

        self._state.add_diagnostic(
            "info" if diagnosis["severity"] == "ok" else diagnosis["severity"],
            f"Calibration: noise_floor={nf:.4f}, SNR={snr_text}, "
            f"quality={ql}, suggested_threshold={sug:.4f}, "
            f"diagnosis={diagnosis['issue']}",
        )

    @pyqtSlot(str)
    def _on_calib_error(self, err: str) -> None:
        self._btn_calib.setEnabled(True)
        self._calib_status.setText(f"Calibration error: {err}")
        self._calib_status.setStyleSheet(f"color: {_ERROR}; font-size: 11px;")

    def _apply_suggested_threshold(self) -> None:
        if self._suggested_threshold is not None:
            self._config.set("silence_threshold", self._suggested_threshold)
            self._config.save()
            self._calib_result.setText(
                self._calib_result.text()
                + f"\n→ Applied silence threshold: {self._suggested_threshold:.4f}"
            )
            self._btn_apply_thresh.setEnabled(False)
            self._state.add_diagnostic(
                "info",
                f"Silence threshold set to {self._suggested_threshold:.4f} from calibration.",
            )

    def _run_stt_test(self) -> None:
        if self._stt_cb is None:
            return
        self._btn_stt.setEnabled(False)
        self._stt_result.setText("Recording for 5 s — speak now…")
        self._stt_result.setStyleSheet(f"color: {_WARNING}; font-size: 11px;")
        mic_idx = self._combo_in.currentData()
        stt_cb  = self._stt_cb

        def _run() -> None:
            try:
                SR   = 16000
                data = sd.rec(SR * 5, samplerate=SR, channels=1,
                              dtype="float32", device=mic_idx)
                sd.wait()
                audio       = data.flatten()
                transcript  = stt_cb(audio)
                nf          = estimate_noise_floor(audio)
                speech_rms  = estimate_speech_rms(audio, nf)
                quality, ex = signal_quality_label(nf, speech_rms)
                self._stt_done.emit(transcript or "(empty)", quality, ex)
            except Exception as exc:
                self._stt_error.emit(str(exc))

        threading.Thread(target=_run, daemon=True).start()

    @pyqtSlot(str, str, str)
    def _on_stt_done(self, transcript: str, quality: str, explanation: str) -> None:
        self._btn_stt.setEnabled(True)
        q_color = {
            "good": _SUCCESS, "fair": _WARNING, "poor": _ERROR, "no_signal": _ERROR
        }.get(quality, _MUTED)

        hint = ""
        if quality == "no_signal":
            hint = "\nCheck that the correct microphone is selected and not muted."
        elif quality == "poor":
            hint = "\nTry speaking louder or closer to the microphone, or run Calibration."
        elif transcript == "(empty)" and quality in ("good", "fair"):
            hint = "\nAudio is present but Whisper returned nothing — try a different language setting."

        self._stt_result.setText(
            f'Transcript: "{transcript}"\n'
            f"Quality: {quality.upper()} — {explanation}{hint}"
        )
        self._stt_result.setStyleSheet(f"color: {q_color}; font-size: 11px;")

    @pyqtSlot(str)
    def _on_stt_error(self, err: str) -> None:
        self._btn_stt.setEnabled(True)
        self._stt_result.setText(f"STT test error: {err}")
        self._stt_result.setStyleSheet(f"color: {_ERROR}; font-size: 11px;")


# ── Page: Settings (unified) ───────────────────────────────────────────────────

class SettingsPage(_SettingsPanel):
    """Unified settings page — Activation · Assistant · Voice/TTS ·
    Permissions · App Aliases · Search Directories in one scrollable surface.

    Replaces the previous five separate settings tabs with a single coherent
    configuration experience where users can see and navigate all settings
    without switching between tabs.
    """

    def __init__(self, config: Config, speaker=None, app_state=None,
                 restart_cb: Callable | None = None,
                 reload_cb:  Callable | None = None,
                 language_changed_cb: Callable | None = None,
                 parent=None):
        super().__init__(config, parent)
        self._speaker             = speaker
        self._state               = app_state
        self._restart_cb          = restart_cb
        self._reload_cb           = reload_cb
        self._language_changed_cb = language_changed_cb
        self._build()

    # ── Build ──────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        root = QVBoxLayout(inner)
        root.setContentsMargins(32, 8, 32, 32)
        root.setSpacing(0)

        self._build_activation(root)
        root.addSpacing(8)
        root.addWidget(_settings_divider())

        self._build_assistant(root)
        root.addSpacing(8)
        root.addWidget(_settings_divider())

        self._build_voice(root)
        root.addSpacing(8)
        root.addWidget(_settings_divider())

        self._build_permissions(root)
        root.addSpacing(8)
        root.addWidget(_settings_divider())

        self._build_aliases(root)
        root.addSpacing(8)
        root.addWidget(_settings_divider())

        self._build_directories(root)

        root.addStretch()
        self._load_all()

    # ── Activation section ────────────────────────────────────────────────────

    def _build_activation(self, root: QVBoxLayout) -> None:
        root.addWidget(_settings_heading(
            "Activation",
            "Configure how VOX listens for your commands.",
            top_margin=20,
        ))

        card = _settings_card()
        cv = QVBoxLayout(card)
        cv.setContentsMargins(20, 18, 20, 18)
        cv.setSpacing(14)

        mode_label = QLabel("Activation Mode")
        mode_label.setStyleSheet(
            f"color: {_TEXT}; font-size: 12px; font-weight: 600; background: transparent;"
        )
        cv.addWidget(mode_label)

        self._rb_wake = QRadioButton("Wake Word  —  always listening, activates on spoken phrase")
        self._rb_ptt  = QRadioButton("Push-to-Talk  —  manual hotkey starts and stops recording")
        self._mode_grp = QButtonGroup(self)
        self._mode_grp.addButton(self._rb_wake, 0)
        self._mode_grp.addButton(self._rb_ptt,  1)
        cv.addWidget(self._rb_wake)
        cv.addWidget(self._rb_ptt)

        cv.addWidget(_card_sep())

        form = QGridLayout()
        form.setColumnMinimumWidth(0, 220)
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(12)
        form.setColumnStretch(1, 1)

        form.addWidget(_section("Wake Word"), 0, 0)
        self._edit_ww = QLineEdit()
        self._edit_ww.setPlaceholderText("vox")
        form.addWidget(self._edit_ww, 0, 1)

        form.addWidget(_section("PTT Key Combination"), 1, 0)
        self._edit_ptt = QLineEdit()
        self._edit_ptt.setPlaceholderText("ctrl+shift")
        form.addWidget(self._edit_ptt, 1, 1)
        form.addWidget(_note("Combine keys with +, e.g.  ctrl+shift  or  alt+z"), 2, 1)

        form.addWidget(_section("Silence Threshold (RMS)"), 3, 0)
        self._spin_sil_thresh = QDoubleSpinBox()
        self._spin_sil_thresh.setRange(0.001, 0.2)
        self._spin_sil_thresh.setSingleStep(0.005)
        self._spin_sil_thresh.setDecimals(3)
        form.addWidget(self._spin_sil_thresh, 3, 1)

        form.addWidget(_section("Silence Duration (s)"), 4, 0)
        self._spin_sil_dur = QDoubleSpinBox()
        self._spin_sil_dur.setRange(0.3, 5.0)
        self._spin_sil_dur.setSingleStep(0.1)
        self._spin_sil_dur.setDecimals(1)
        form.addWidget(self._spin_sil_dur, 4, 1)

        cv.addLayout(form)
        root.addWidget(card)

        root.addSpacing(8)
        root.addWidget(_apply_tag(
            "Activation mode and PTT key restart the listener on save.", _WARNING))
        root.addWidget(_apply_tag(
            "Wake word and silence parameters apply on next capture cycle.", _INFO))
        root.addSpacing(2)

        save_row, self._save_lbl_act, btn = self._save_row()
        root.addLayout(save_row)
        btn.clicked.connect(self._save_activation)

    # ── Assistant section ─────────────────────────────────────────────────────

    def _build_assistant(self, root: QVBoxLayout) -> None:
        root.addWidget(_settings_heading(
            "Assistant",
            "Language model endpoint, response history, and transcription language.",
        ))

        card = _settings_card()
        form = QGridLayout(card)
        form.setContentsMargins(20, 18, 20, 18)
        form.setColumnMinimumWidth(0, 220)
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(12)
        form.setColumnStretch(1, 1)

        form.addWidget(_section("Ollama URL"), 0, 0)
        self._edit_url = QLineEdit()
        form.addWidget(self._edit_url, 0, 1)

        form.addWidget(_section("Model"), 1, 0)
        self._edit_model = QLineEdit()
        self._edit_model.setPlaceholderText("qwen2.5:14b")
        form.addWidget(self._edit_model, 1, 1)

        form.addWidget(_section("History Size (turns)"), 2, 0)
        self._spin_hist = QSpinBox()
        self._spin_hist.setRange(1, 100)
        form.addWidget(self._spin_hist, 2, 1)

        form.addWidget(_section("Transcription Language"), 3, 0)
        self._combo_lang = QComboBox()
        for val, label in [
            ("auto", "Auto-detect"),
            ("pt",   "Portuguese (pt)"),
            ("en",   "English (en)"),
        ]:
            self._combo_lang.addItem(label, val)
        form.addWidget(self._combo_lang, 3, 1)

        root.addWidget(card)

        root.addSpacing(8)
        root.addWidget(_apply_tag("URL and model apply on the next request.", _INFO))
        root.addWidget(_apply_tag("Language applies on the next transcription.", _INFO))
        root.addSpacing(2)

        save_row, self._save_lbl_ast, btn = self._save_row()
        root.addLayout(save_row)
        btn.clicked.connect(self._save_assistant)

    # ── Voice & TTS section ───────────────────────────────────────────────────

    def _build_voice(self, root: QVBoxLayout) -> None:
        root.addWidget(_settings_heading(
            "Voice & TTS",
            "Text-to-speech engine and voice model. Changes apply on next spoken response.",
        ))

        card = _settings_card()
        form = QGridLayout(card)
        form.setContentsMargins(20, 18, 20, 18)
        form.setColumnMinimumWidth(0, 220)
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(12)
        form.setColumnStretch(1, 1)

        form.addWidget(_section("Enable TTS"), 0, 0)
        self._chk_tts = QCheckBox("")
        form.addWidget(self._chk_tts, 0, 1)

        form.addWidget(_section("Voice Model (.onnx)"), 1, 0)
        voice_row = QHBoxLayout()
        self._edit_voice = QLineEdit()
        self._btn_browse = QPushButton("Browse…")
        self._btn_browse.setFixedWidth(90)
        voice_row.addWidget(self._edit_voice)
        voice_row.addSpacing(8)
        voice_row.addWidget(self._btn_browse)
        form.addLayout(voice_row, 1, 1)

        root.addWidget(card)

        root.addSpacing(8)
        root.addWidget(_apply_tag(
            "TTS toggle and voice model apply immediately on next speech.", _INFO))
        root.addSpacing(2)

        save_row, self._save_lbl_voice, btn = self._save_row()
        root.addLayout(save_row)
        btn.clicked.connect(self._save_voice)
        self._btn_browse.clicked.connect(self._browse_voice)

    # ── Permissions section ───────────────────────────────────────────────────

    def _build_permissions(self, root: QVBoxLayout) -> None:
        root.addWidget(_settings_heading(
            "Permissions",
            "Only enabled actions can be triggered by the assistant. "
            "Disabling an action blocks it immediately — no restart required.",
        ))

        self._actions_table = QTableWidget(0, 4)
        self._actions_table.setHorizontalHeaderLabels(["", "Action", "Description", "Risk"])
        self._actions_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed)
        self._actions_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self._actions_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self._actions_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Fixed)
        self._actions_table.setColumnWidth(0, 30)
        self._actions_table.setColumnWidth(3, 72)
        self._actions_table.verticalHeader().setVisible(False)
        self._actions_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._actions_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._actions_table.setShowGrid(False)
        self._actions_table.setMinimumHeight(320)
        root.addWidget(self._actions_table)

        btn_row = QHBoxLayout()
        btn_all  = QPushButton("Enable All")
        btn_none = QPushButton("Disable All")
        btn_def  = QPushButton("Restore Defaults")
        btn_def.setObjectName("danger")
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addWidget(btn_def)
        btn_row.addStretch()
        root.addLayout(btn_row)

        root.addSpacing(8)
        root.addWidget(_apply_tag(
            "Changes apply on the next voice command — no restart required.", _INFO))
        root.addSpacing(2)

        save_row, self._save_lbl_perm, btn_save = self._save_row()
        root.addLayout(save_row)

        btn_all.clicked.connect(lambda: self._set_all_actions(True))
        btn_none.clicked.connect(lambda: self._set_all_actions(False))
        btn_def.clicked.connect(self._restore_default_actions)
        btn_save.clicked.connect(self._save_permissions)

    # ── App Aliases section ───────────────────────────────────────────────────

    def _build_aliases(self, root: QVBoxLayout) -> None:
        root.addWidget(_settings_heading(
            "App Aliases",
            'Maps spoken names to commands or URI schemes. '
            'Example: "discord" → discord://  or  "vscode" → code',
        ))

        self._alias_table = QTableWidget(0, 2)
        self._alias_table.setHorizontalHeaderLabels(
            ["Spoken Name", "Target (command or URI)"])
        self._alias_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._alias_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._alias_table.verticalHeader().setVisible(False)
        self._alias_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._alias_table.setMinimumHeight(160)
        root.addWidget(self._alias_table)

        btn_row = QHBoxLayout()
        self._btn_alias_add = QPushButton("Add")
        self._btn_alias_del = QPushButton("Remove Selected")
        self._btn_alias_del.setObjectName("danger")
        btn_row.addWidget(self._btn_alias_add)
        btn_row.addWidget(self._btn_alias_del)
        btn_row.addStretch()
        root.addLayout(btn_row)

        save_row, self._save_lbl_alias, btn_save = self._save_row()
        root.addLayout(save_row)

        self._btn_alias_add.clicked.connect(self._alias_add_row)
        self._btn_alias_del.clicked.connect(self._alias_del_row)
        btn_save.clicked.connect(self._save_aliases)

    # ── Directories section ───────────────────────────────────────────────────

    def _build_directories(self, root: QVBoxLayout) -> None:
        root.addWidget(_settings_heading(
            "Search Directories",
            "Directories searched when you ask the assistant to find a file. "
            "~ expands to your home directory.",
        ))

        self._dirs_list = QListWidget()
        self._dirs_list.setMinimumHeight(120)
        root.addWidget(self._dirs_list)

        btn_row = QHBoxLayout()
        self._btn_dir_add = QPushButton("Add Directory…")
        self._btn_dir_del = QPushButton("Remove Selected")
        self._btn_dir_del.setObjectName("danger")
        self._btn_dir_def = QPushButton("Restore Defaults")
        btn_row.addWidget(self._btn_dir_add)
        btn_row.addWidget(self._btn_dir_del)
        btn_row.addWidget(self._btn_dir_def)
        btn_row.addStretch()
        root.addLayout(btn_row)

        save_row, self._save_lbl_dirs, btn_save = self._save_row()
        root.addLayout(save_row)

        self._btn_dir_add.clicked.connect(self._dirs_add)
        self._btn_dir_del.clicked.connect(self._dirs_del)
        self._btn_dir_def.clicked.connect(self._dirs_restore_defaults)
        btn_save.clicked.connect(self._save_directories)

    # ── Load / Save ────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        c = self._config
        # Activation
        mode = c.get("activation_mode", "wake_word")
        self._rb_wake.setChecked(mode != "push_to_talk")
        self._rb_ptt.setChecked(mode == "push_to_talk")
        self._edit_ww.setText(str(c.get("wake_word", "vox")))
        self._edit_ptt.setText(str(c.get("push_to_talk_key", "ctrl+shift")))
        self._spin_sil_thresh.setValue(float(c.get("silence_threshold", 0.01)))
        self._spin_sil_dur.setValue(float(c.get("silence_duration", 1.5)))
        # Assistant
        self._edit_url.setText(c.get("ollama_url", "http://localhost:11434"))
        self._edit_model.setText(c.get("ollama_model", "qwen2.5:14b"))
        self._spin_hist.setValue(int(c.get("max_history", 20)))
        lang = c.get("language", "auto")
        for i in range(self._combo_lang.count()):
            if self._combo_lang.itemData(i) == lang:
                self._combo_lang.setCurrentIndex(i)
                break
        # Voice
        self._chk_tts.setChecked(bool(c.get("tts_enabled", True)))
        self._edit_voice.setText(c.get("voice_model", ""))
        # Tables
        self._load_actions()
        self._load_aliases()
        self._load_dirs()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._load_all()

    def _save_activation(self) -> None:
        c = self._config
        prev_mode = c.get("activation_mode", "wake_word")
        prev_ptt  = c.get("push_to_talk_key", "ctrl+shift")
        mode = "push_to_talk" if self._rb_ptt.isChecked() else "wake_word"
        c.set("activation_mode",   mode)
        c.set("wake_word",         self._edit_ww.text().strip() or "vox")
        c.set("push_to_talk_key",  self._edit_ptt.text().strip() or "ctrl+shift")
        c.set("silence_threshold", round(self._spin_sil_thresh.value(), 3))
        c.set("silence_duration",  round(self._spin_sil_dur.value(), 1))
        c.save()
        new_ptt = c.get("push_to_talk_key")
        needs_restart = (mode != prev_mode or new_ptt != prev_ptt)
        if needs_restart and self._restart_cb:
            self._restart_cb()
            self._show_saved(self._save_lbl_act, ok=True, msg="Saved — listener restarted.")
        else:
            self._show_saved(self._save_lbl_act)

    def _save_assistant(self) -> None:
        c = self._config
        prev_lang = c.get("language", "auto")
        c.set("ollama_url",   self._edit_url.text().strip())
        c.set("ollama_model", self._edit_model.text().strip())
        c.set("max_history",  self._spin_hist.value())
        c.set("language",     self._combo_lang.currentData())
        c.save()
        new_lang = c.get("language", "auto")
        if new_lang != prev_lang and self._language_changed_cb:
            self._language_changed_cb(new_lang)
        self._show_saved(self._save_lbl_ast)

    def _save_voice(self) -> None:
        c = self._config
        c.set("tts_enabled", self._chk_tts.isChecked())
        c.set("voice_model", self._edit_voice.text().strip())
        c.save()
        if self._speaker:
            self._speaker.reload_config()
        self._show_saved(self._save_lbl_voice)

    def _browse_voice(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Voice Model", "", "ONNX files (*.onnx);;All files (*)"
        )
        if path:
            self._edit_voice.setText(path)

    def _load_actions(self) -> None:
        allowed = set(self._config.get("allowed_actions", []))
        _risk_colors = {"low": _SUCCESS, "medium": _WARNING, "high": _ERROR}
        self._actions_table.setRowCount(0)
        for action, desc, risk in _ALL_ACTIONS:
            row = self._actions_table.rowCount()
            self._actions_table.insertRow(row)
            chk = QCheckBox()
            chk.setChecked(action in allowed)
            chk.setStyleSheet("margin-left: 6px;")
            self._actions_table.setCellWidget(row, 0, chk)
            self._actions_table.setItem(row, 1, QTableWidgetItem(action))
            self._actions_table.setItem(row, 2, QTableWidgetItem(desc))
            rc = _risk_colors.get(risk, _MUTED)
            risk_lbl = QLabel(f" {risk} ")
            risk_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            risk_lbl.setStyleSheet(
                f"color: {rc}; font-size: 10px; font-weight: 600; "
                f"border: 1px solid {rc}; border-radius: 4px; "
                "padding: 1px 4px; background: transparent; letter-spacing: 0.2px;"
            )
            self._actions_table.setCellWidget(row, 3, risk_lbl)
            self._actions_table.setRowHeight(row, 36)

    def _set_all_actions(self, state: bool) -> None:
        for row in range(self._actions_table.rowCount()):
            chk = self._actions_table.cellWidget(row, 0)
            if chk:
                chk.setChecked(state)

    def _restore_default_actions(self) -> None:
        from utils.config import DEFAULT_CONFIG
        defaults = set(DEFAULT_CONFIG.get("allowed_actions", []))
        for row in range(self._actions_table.rowCount()):
            action = self._actions_table.item(row, 1)
            chk    = self._actions_table.cellWidget(row, 0)
            if action and chk:
                chk.setChecked(action.text() in defaults)

    def _save_permissions(self) -> None:
        enabled = []
        for row in range(self._actions_table.rowCount()):
            chk    = self._actions_table.cellWidget(row, 0)
            action = self._actions_table.item(row, 1)
            if chk and action and chk.isChecked():
                enabled.append(action.text())
        self._config.set("allowed_actions", enabled)
        self._config.save()
        if self._reload_cb:
            self._reload_cb()
        self._show_saved(self._save_lbl_perm)

    def _load_aliases(self) -> None:
        aliases = self._config.get("app_aliases", {})
        self._alias_table.setRowCount(0)
        for name, target in sorted(aliases.items()):
            self._alias_add_row(name, str(target))

    def _alias_add_row(self, name: str = "", target: str = "") -> None:
        row = self._alias_table.rowCount()
        self._alias_table.insertRow(row)
        self._alias_table.setItem(row, 0, QTableWidgetItem(name))
        self._alias_table.setItem(row, 1, QTableWidgetItem(target))
        self._alias_table.setRowHeight(row, 30)

    def _alias_del_row(self) -> None:
        rows = sorted(
            {idx.row() for idx in self._alias_table.selectedIndexes()}, reverse=True
        )
        for r in rows:
            self._alias_table.removeRow(r)

    def _save_aliases(self) -> None:
        aliases: dict[str, str] = {}
        for row in range(self._alias_table.rowCount()):
            n = self._alias_table.item(row, 0)
            t = self._alias_table.item(row, 1)
            if n and t and n.text().strip() and t.text().strip():
                aliases[n.text().strip()] = t.text().strip()
        self._config.set("app_aliases", aliases)
        self._config.save()
        self._show_saved(self._save_lbl_alias)

    def _load_dirs(self) -> None:
        self._dirs_list.clear()
        for d in self._config.get("search_dirs", []):
            self._dirs_list.addItem(str(d))

    def _dirs_add(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            self._dirs_list.addItem(path)

    def _dirs_del(self) -> None:
        for item in self._dirs_list.selectedItems():
            self._dirs_list.takeItem(self._dirs_list.row(item))

    def _dirs_restore_defaults(self) -> None:
        self._dirs_list.clear()
        from utils.config import DEFAULT_CONFIG
        for d in DEFAULT_CONFIG.get("search_dirs", []):
            self._dirs_list.addItem(d)

    def _save_directories(self) -> None:
        dirs = [self._dirs_list.item(i).text() for i in range(self._dirs_list.count())]
        self._config.set("search_dirs", dirs)
        self._config.save()
        self._show_saved(self._save_lbl_dirs)


# ── Page: History ──────────────────────────────────────────────────────────────

class HistoryTab(QWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self._state = app_state
        self._build()
        self._connect()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(10)

        root.addWidget(_section("Session History"))

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(_mono_font())
        root.addWidget(self._log, 1)

        btn_row = QHBoxLayout()
        btn_clear = QPushButton("Clear History")
        btn_clear.setObjectName("danger")
        btn_row.addStretch()
        btn_row.addWidget(btn_clear)
        root.addLayout(btn_row)

        btn_clear.clicked.connect(self._state.clear_history)

    def _connect(self) -> None:
        self._state.history_entry_added.connect(self._on_entry)
        self._state.history_cleared.connect(self._log.clear)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._reload()

    def _reload(self) -> None:
        self._log.clear()
        for entry in self._state.history:
            self._append(entry)

    @pyqtSlot(dict)
    def _on_entry(self, entry: dict) -> None:
        self._append(entry)

    def _append(self, entry: dict) -> None:
        ts = entry.get("timestamp", "")
        tx = entry.get("transcript", "")
        rx = entry.get("response",  "")
        ax = entry.get("action",    "")

        self._log.append(
            f'<span style="color:{theme.GHOST}; font-size:11px;">&#8212;&#8212;&#8212;&#8212;&#8212;</span>'
            f'&nbsp;<span style="color:{theme.MUTED}; font-size:10px; letter-spacing:0.4px;">'
            f'{_esc(ts)}</span>'
        )
        if tx:
            self._log.append(
                f'<span style="color:{theme.GHOST}; font-size:10px; font-weight:700; '
                f'letter-spacing:1.0px;">YOU</span>'
                f'&nbsp;&nbsp;'
                f'<span style="color:{theme.TEXT2}; font-size:12px;">{_esc(tx)}</span>'
            )
        if ax:
            self._log.append(
                f'<span style="color:{theme.SUCCESS}; font-size:10px; font-weight:700; '
                f'letter-spacing:1.0px;">ACTION</span>'
                f'&nbsp;&nbsp;'
                f'<span style="color:{theme.SUCCESS}; font-size:12px;">{_esc(ax)}</span>'
            )
        elif rx:
            self._log.append(
                f'<span style="color:{theme.ACCENT}; font-size:10px; font-weight:700; '
                f'letter-spacing:1.0px;">VOX</span>'
                f'&nbsp;&nbsp;&nbsp;'
                f'<span style="color:{theme.ACCENT_SOFT}; font-size:12px;">{_esc(rx)}</span>'
            )
        self._log.append("")
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())


# ── Page: Diagnostics ─────────────────────────────────────────────────────────

class DiagnosticsTab(QWidget):
    def __init__(self, app_state, validate_cb: Callable | None = None, parent=None):
        super().__init__(parent)
        self._state       = app_state
        self._validate_cb = validate_cb
        self._build()
        self._connect()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(10)

        root.addWidget(_section("Diagnostics Log"))

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(_mono_font())
        root.addWidget(self._log, 1)

        btn_row = QHBoxLayout()
        self._btn_validate = QPushButton("Re-run Validation")
        btn_clear = QPushButton("Clear Log")
        btn_clear.setObjectName("danger")
        btn_row.addWidget(self._btn_validate)
        btn_row.addStretch()
        btn_row.addWidget(btn_clear)
        root.addLayout(btn_row)

        self._btn_validate.clicked.connect(self._rerun_validation)
        btn_clear.clicked.connect(self._state.clear_diagnostics)

    def _connect(self) -> None:
        self._state.diagnostic_added.connect(self._on_entry)
        self._state.diagnostics_cleared.connect(self._log.clear)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._reload()

    def _reload(self) -> None:
        self._log.clear()
        for entry in self._state.diagnostics:
            self._append(entry)

    @pyqtSlot(dict)
    def _on_entry(self, entry: dict) -> None:
        self._append(entry)

    def _append(self, entry: dict) -> None:
        lvl = entry.get("level", "info")
        ts  = entry.get("timestamp", "")
        msg = entry.get("message", "")
        level_color = {
            "info":    theme.INFO,
            "warning": theme.WARNING,
            "error":   theme.ERROR,
        }.get(lvl, theme.TEXT2)
        icon = {"info": "●", "warning": "▲", "error": "✕"}.get(lvl, "●")
        msg_color = {
            "info":    theme.TEXT2,
            "warning": "#fcd34d",
            "error":   "#fca5a5",
        }.get(lvl, theme.TEXT2)

        self._log.append(
            f'<span style="color:{theme.MUTED}; font-size:10px;">{_esc(ts)}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:{level_color}; font-size:10px; font-weight:700; '
            f'letter-spacing:0.8px;">{icon}&nbsp;{lvl.upper()}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:{msg_color}; font-size:12px;">{_esc(msg)}</span>'
        )
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _rerun_validation(self) -> None:
        if self._validate_cb:
            self._validate_cb()
        else:
            self._state.add_diagnostic("info", "Validation callback not connected.")


# ── Navigation rail ────────────────────────────────────────────────────────────

class NavRail(QWidget):
    """Left sidebar navigation rail.

    Emits ``section_changed(int)`` when the user selects a different page.
    Call ``select(index)`` to programmatically switch sections.
    """

    section_changed = pyqtSignal(int)

    _ITEMS = [
        ("Dashboard",   "OVERVIEW"),
        ("Audio",       "DEVICES"),
        ("Settings",    "CONFIG"),
        ("History",     "SESSION"),
        ("Diagnostics", "SYSTEM"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("nav_rail")
        self.setFixedWidth(160)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 16)
        root.setSpacing(0)

        # App wordmark
        app_hdr = QLabel("VOX")
        app_hdr.setStyleSheet(
            f"color: {theme.ACCENT_SOFT}; font-size: 13px; font-weight: 700; "
            f"letter-spacing: 3.5px; padding: 20px 20px 16px 20px; "
            f"background: transparent; border-bottom: 1px solid {_BORDER};"
        )
        root.addWidget(app_hdr)
        root.addSpacing(4)

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._buttons: list[QPushButton] = []

        for i, (name, _sub) in enumerate(self._ITEMS):
            btn = QPushButton(name)
            btn.setObjectName("nav_btn")
            btn.setCheckable(True)
            self._btn_group.addButton(btn, i)
            root.addWidget(btn)
            self._buttons.append(btn)

        root.addStretch()

        self._btn_group.idClicked.connect(self.section_changed)
        self._buttons[0].setChecked(True)

    def select(self, index: int) -> None:
        """Programmatically select a nav item by zero-based index."""
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)


# ── ControlCenter ──────────────────────────────────────────────────────────────

class ControlCenter(QMainWindow):
    restart_listener_requested = pyqtSignal()
    rerun_validation_requested = pyqtSignal()
    language_changed           = pyqtSignal(str)

    _SECTION_NAMES = ["Dashboard", "Audio", "Settings", "History", "Diagnostics"]

    def __init__(self, config: Config, app_state, speaker,
                 stt_cb: Callable | None = None, executor=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VOX — Control Center")
        self.setMinimumSize(800, 580)
        self.resize(980, 660)
        self.setStyleSheet(_CC_STYLE)

        icon_path = os.path.join(_project_root(), "assets", "icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        reload_cb = executor.reload_config if executor is not None else None

        # ── Pages ─────────────────────────────────────────────────────────────
        self._dashboard   = DashboardTab(app_state, config)
        self._audio       = AudioTab(
            config, app_state, speaker,
            restart_cb=self.restart_listener_requested.emit,
            stt_cb=stt_cb,
        )
        self._settings    = SettingsPage(
            config, speaker=speaker, app_state=app_state,
            restart_cb=self.restart_listener_requested.emit,
            reload_cb=reload_cb,
            language_changed_cb=self.language_changed.emit,
        )
        self._history     = HistoryTab(app_state)
        self._diagnostics = DiagnosticsTab(
            app_state, validate_cb=self.rerun_validation_requested.emit,
        )

        # ── Stacked content area ──────────────────────────────────────────────
        self._stack = QStackedWidget()
        for page in (self._dashboard, self._audio, self._settings,
                     self._history, self._diagnostics):
            self._stack.addWidget(page)

        # ── Navigation rail ───────────────────────────────────────────────────
        self._nav = NavRail()
        self._nav.section_changed.connect(self._stack.setCurrentIndex)

        # ── Shell ─────────────────────────────────────────────────────────────
        shell = QWidget()
        layout = QHBoxLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._nav)
        layout.addWidget(self._stack, 1)
        self.setCentralWidget(shell)

    def show_section(self, name: str) -> None:
        """Show the Control Center with the named section active.

        Valid names: Dashboard, Audio, Settings, History, Diagnostics.
        """
        try:
            idx = self._SECTION_NAMES.index(name)
        except ValueError:
            idx = 0
        self._nav.select(idx)
        self._stack.setCurrentIndex(idx)
        self.show()
        self.raise_()
        self.activateWindow()

    def show_tab(self, tab_name: str) -> None:
        """Backward-compatible alias — maps legacy tab names to sections."""
        _legacy = {
            "Activation":  "Settings",
            "Assistant":   "Settings",
            "Actions":     "Settings",
            "Aliases":     "Settings",
            "Directories": "Settings",
        }
        self.show_section(_legacy.get(tab_name, tab_name))


# ── Utilities ──────────────────────────────────────────────────────────────────

def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve(project_root: str, raw: str) -> str:
    if os.path.isabs(raw):
        return raw
    return os.path.normpath(os.path.join(project_root, raw))


def _esc(text: str) -> str:
    """HTML-escape text for safe QTextEdit insertion."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _mono_font() -> QFont:
    f = QFont("Consolas")
    f.setPointSize(11)
    if not f.exactMatch():
        f.setFamily("Monospace")
    return f
