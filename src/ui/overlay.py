from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QGuiApplication

from ui.mic_meter import MicLevelWaveform
from ui import theme


# ─── Stylesheet constants ──────────────────────────────────────────────────────

_BG          = "background: transparent; border: none;"
_FONT_STACK  = f"font-family: {theme.FONT};"

# Section micro-labels (YOU / VOX) — left-accent border approach
_MUTED_LABEL = (
    f"color: rgba(210,200,240,0.78); font-size: 9px; letter-spacing: 1.8px; "
    f"font-weight: 700; {_FONT_STACK} "
    "border-top: none; border-right: none; border-bottom: none; "
    "border-left: 2px solid rgba(168,85,247,80); "
    "padding-left: 8px; background: transparent;"
)

_CONTAINER_SS = """
QFrame#container {
    background-color: rgba(10, 8, 22, 252);
    border-radius: 16px;
    border: 1px solid rgba(168, 85, 247, 115);
}
"""

_BADGE_SS = (
    "background: rgba(168,85,247,28);"
    "border: 1px solid rgba(168,85,247,120);"
    "border-radius: 5px;"
    "color: rgba(192,140,255,230);"
    "font-size: 9px;"
    "font-weight: 700;"
    "letter-spacing: 1.5px;"
    "padding: 2px 9px;"
)

_FOOTER_SS = (
    f"color: rgba(160,153,195,0.75); font-size: 10px; "
    f"letter-spacing: 0.3px; {_FONT_STACK} {_BG}"
)

W, H = 440, 284


# ─── Clickable label ──────────────────────────────────────────────────────────

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ─── Overlay ──────────────────────────────────────────────────────────────────

class OverlayWindow(QWidget):
    language_clicked = pyqtSignal()

    # Human-readable status labels and their colours (sourced from shared theme)
    _STATUS_MAP = {
        "idle":         ("idle",          "rgba(108,103,140,0.72)"),
        "monitoring":   ("monitoring",    theme.SUCCESS),
        "listening":    ("listening",     theme.LISTENING),
        "transcribing": ("transcribing",  theme.WARNING),
        "generating":   ("generating",    theme.ACTIVE),
        "responding":   ("responding",    theme.ACTIVE),
        "speaking":     ("speaking",      theme.ACTIVE),
        "error":        ("error",         theme.ERROR),
        "cancelled":    ("cancelled",     theme.ERROR),
        "done":         ("done",          theme.SUCCESS),
    }

    def __init__(self):
        super().__init__()
        self._state          = "idle"
        self._pulse_on       = True
        self._pulse_color    = "rgba(100,95,130,0.7)"
        self._pulse_dim      = "rgba(68,68,68,0.2)"
        self._footer_default = "Say VOX to activate"
        self._lang_mode_text = "AUTO"

        self._setup_window()
        self._setup_ui()

        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._auto_hide)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._tick_pulse)

    # ── Window setup ──────────────────────────────────────────────────────────

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(W, H)
        screen = QGuiApplication.primaryScreen().geometry()
        self.move(screen.width() - W - 20, screen.height() - H - 44)

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        self._container = QFrame(self)
        self._container.setObjectName("container")
        self._container.setFixedSize(W, H)
        self._container.setStyleSheet(_CONTAINER_SS)

        root = QVBoxLayout(self._container)
        root.setContentsMargins(20, 16, 20, 14)
        root.setSpacing(0)

        # ── Header row ───────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(0)

        self._title = QLabel("VOX")
        self._title.setStyleSheet(
            f"color: rgba(255,255,255,0.90); font-size: 11px; "
            f"letter-spacing: 4.5px; font-weight: 700; {_FONT_STACK} {_BG}"
        )

        # Ollama connectivity dot — right of title
        self._ollama_dot = QLabel("●")
        self._ollama_dot.setStyleSheet(
            f"color: rgba(80,80,80,0.75); font-size: 9px; {_FONT_STACK} {_BG}"
        )
        self._ollama_dot.setToolTip("Checking Ollama…")

        self._dot = QLabel("●")
        self._dot.setStyleSheet(
            f"color: rgba(108,103,140,0.72); font-size: 11px; {_FONT_STACK} {_BG}"
        )

        self._status_lbl = QLabel("idle")
        self._status_lbl.setStyleSheet(
            f"color: rgba(108,103,140,0.72); font-size: 12px; "
            f"font-weight: 600; letter-spacing: 0.4px; {_FONT_STACK} {_BG}"
        )

        status_row = QHBoxLayout()
        status_row.setSpacing(5)
        status_row.addWidget(self._ollama_dot)
        status_row.addSpacing(4)
        status_row.addWidget(self._dot)
        status_row.addWidget(self._status_lbl)

        header.addWidget(self._title)
        header.addStretch()
        header.addLayout(status_row)
        root.addLayout(header)

        root.addSpacing(10)

        # ── Thin top divider ──────────────────────────────────
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setFixedHeight(1)
        div.setStyleSheet("background: rgba(168,85,247,30); border: none;")
        root.addWidget(div)

        root.addSpacing(10)

        # ── Waveform (real mic level) ─────────────────────────
        wave_row = QHBoxLayout()
        wave_row.setSpacing(0)
        self._waveform = MicLevelWaveform()
        wave_row.addStretch()
        wave_row.addWidget(self._waveform)
        wave_row.addStretch()
        root.addLayout(wave_row)

        root.addSpacing(12)

        # ── Transcript section ────────────────────────────────
        tx_hdr = QHBoxLayout()
        tx_hdr.setSpacing(0)
        self._lbl_you = QLabel("YOU")
        self._lbl_you.setStyleSheet(_MUTED_LABEL)
        tx_hdr.addWidget(self._lbl_you)
        tx_hdr.addStretch()
        root.addLayout(tx_hdr)

        root.addSpacing(4)

        self._transcript = QLabel("–")
        self._transcript.setWordWrap(True)
        self._transcript.setMaximumHeight(48)
        self._transcript.setStyleSheet(
            f"color: rgba(235,232,248,0.30); font-size: 14px; "
            f"line-height: 1.4; {_FONT_STACK} {_BG}"
        )
        root.addWidget(self._transcript)

        root.addSpacing(10)

        # ── Subtle divider ────────────────────────────────────
        div2 = QFrame()
        div2.setFrameShape(QFrame.Shape.HLine)
        div2.setFixedHeight(1)
        div2.setStyleSheet("background: rgba(255,255,255,18); border: none;")
        root.addWidget(div2)

        root.addSpacing(10)

        # ── Response section ──────────────────────────────────
        rx_hdr = QHBoxLayout()
        rx_hdr.setSpacing(0)
        self._lbl_vox = QLabel("VOX")
        self._lbl_vox.setStyleSheet(_MUTED_LABEL)
        rx_hdr.addWidget(self._lbl_vox)
        rx_hdr.addStretch()
        root.addLayout(rx_hdr)

        root.addSpacing(4)

        self._response = QLabel("")
        self._response.setWordWrap(True)
        self._response.setMaximumHeight(64)
        self._response.setStyleSheet(
            f"color: {theme.ACCENT_SOFT}; font-size: 13px; line-height: 1.45; "
            f"{_FONT_STACK} {_BG}"
        )
        root.addWidget(self._response)

        root.addStretch()

        # ── Footer ───────────────────────────────────────────
        footer_row = QHBoxLayout()
        footer_row.setContentsMargins(0, 0, 0, 0)
        footer_row.setSpacing(0)

        self._footer = QLabel(self._footer_default)
        self._footer.setStyleSheet(_FOOTER_SS)

        self._lang_badge = ClickableLabel("AUTO")
        self._lang_badge.setStyleSheet(_BADGE_SS)
        self._lang_badge.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lang_badge.setToolTip("Click to cycle language: AUTO → PT → EN")
        self._lang_badge.clicked.connect(self.language_clicked)

        footer_row.addWidget(self._footer)
        footer_row.addStretch()
        footer_row.addWidget(self._lang_badge)
        root.addLayout(footer_row)

    # ── Dot & pulse ───────────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str, pulse: bool = False,
                    dim: str | None = None):
        self._pulse_color = color
        self._pulse_dim   = dim if dim else _darken(color)
        self._pulse_on    = True

        if pulse:
            self._pulse_timer.start(500)
        else:
            self._pulse_timer.stop()

        self._apply_dot(color)
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: 600; "
            f"letter-spacing: 0.4px; {_FONT_STACK} {_BG}"
        )

    def _tick_pulse(self):
        self._pulse_on = not self._pulse_on
        self._apply_dot(self._pulse_color if self._pulse_on else self._pulse_dim)

    def _apply_dot(self, color: str):
        self._dot.setStyleSheet(
            f"color: {color}; font-size: 10px; {_FONT_STACK} {_BG}"
        )

    # ── Mic level ─────────────────────────────────────────────────────────────

    @pyqtSlot(float)
    def set_mic_level(self, level: float):
        """Feed real RMS level (0.0–1.0) to the waveform widget."""
        self._waveform.set_level(level)

    # ── State slots ───────────────────────────────────────────────────────────

    @pyqtSlot(str)
    def set_footer_mode(self, mode: str):
        if mode == "push_to_talk":
            self._footer_default = "Press Ctrl+Shift to activate"
        else:
            self._footer_default = "Say VOX to activate"
        self._footer.setText(self._footer_default)

    @pyqtSlot(str)
    def set_footer_mode_with_key(self, mode: str, key: str):
        if mode == "push_to_talk":
            self._footer_default = f"Press {key.upper()} to activate"
        else:
            self._footer_default = "Say VOX to activate"
        self._footer.setText(self._footer_default)

    @pyqtSlot(str)
    def show_info_notice(self, message: str, duration_ms: int = 4000):
        self._footer.setText(message)
        self._footer.setVisible(True)
        self._footer.setStyleSheet(
            f"color: #f59e0b; font-size: 10px; letter-spacing: 0.3px; {_FONT_STACK} {_BG}"
        )
        QTimer.singleShot(duration_ms, self._restore_footer)

    def _restore_footer(self):
        self._footer.setText(self._footer_default)
        self._footer.setStyleSheet(_FOOTER_SS)

    @pyqtSlot()
    def set_cancelled(self):
        self._state = "idle"
        self._set_status("cancelled", theme.ERROR)
        self._waveform.set_active(False)
        self._response.setText("")
        self._footer.setVisible(False)
        self._idle_timer.start(1500)

    @pyqtSlot()
    def set_idle(self):
        self._state = "idle"
        self._set_status("idle", "rgba(100,95,130,0.7)")
        self._waveform.set_active(False)
        self._footer.setText(self._footer_default)
        self._footer.setStyleSheet(_FOOTER_SS)
        self._footer.setVisible(True)
        self._idle_timer.start(5000)

    @pyqtSlot()
    def set_monitoring(self):
        self._state = "monitoring"
        self._set_status("monitoring", theme.SUCCESS)
        self._waveform.set_active(False)
        self._footer.setText(self._footer_default)
        self._footer.setStyleSheet(_FOOTER_SS)
        self._footer.setVisible(True)
        self._idle_timer.stop()
        self.show()

    @pyqtSlot()
    def set_listening(self):
        self._state = "listening"
        self._set_status("listening", theme.LISTENING, pulse=True, dim="rgba(59,130,246,0.22)")
        self._transcript.setText("…")
        self._transcript.setStyleSheet(
            f"color: rgba(235,232,248,0.38); font-size: 14px; {_FONT_STACK} {_BG}"
        )
        self._response.setText("")
        self._response.setStyleSheet(
            f"color: {theme.ACCENT_SOFT}; font-size: 13px; {_FONT_STACK} {_BG}"
        )
        self._waveform.set_active(True)
        self._footer.setVisible(False)
        self._idle_timer.stop()
        self.show()

    @pyqtSlot()
    def set_processing(self):
        """Listening stopped — transcribing audio."""
        self._state = "transcribing"
        self._set_status("transcribing…", theme.WARNING)
        self._waveform.set_active(False)
        self._response.setText("")
        self._response.setStyleSheet(
            f"color: {theme.ACCENT_SOFT}; font-size: 13px; {_FONT_STACK} {_BG}"
        )
        self._footer.setVisible(False)

    @pyqtSlot()
    def set_generating(self):
        self._state = "generating"
        self._set_status("generating", theme.ACTIVE)

    @pyqtSlot()
    def set_speaking(self):
        self._state = "speaking"
        self._set_status("speaking", theme.ACTIVE, pulse=True, dim="rgba(168,85,247,0.22)")

    @pyqtSlot(str)
    def set_transcript(self, text: str):
        self._transcript.setText(text)
        self._transcript.setStyleSheet(
            f"color: rgba(235,232,248,0.88); font-size: 14px; {_FONT_STACK} {_BG}"
        )

    @pyqtSlot(str)
    def append_token(self, token: str):
        if self._state != "responding":
            self._state = "responding"
            self._set_status("responding", theme.ACTIVE)
            self._footer.setVisible(False)
        current = self._response.text()
        if len(current) < 300:
            self._response.setText(current + token)
        elif not current.endswith("…"):
            self._response.setText(current[:300] + "…")

    @pyqtSlot(str)
    def set_response(self, text: str):
        self._state = "responding"
        self._response.setText(text)
        self._response.setStyleSheet(
            f"color: {theme.ACCENT_SOFT}; font-size: 13px; line-height: 1.45; {_FONT_STACK} {_BG}"
        )
        self._set_status("responding", theme.ACTIVE)
        self._footer.setVisible(False)

    @pyqtSlot(str)
    def set_action(self, text: str):
        self._state = "responding"
        self._response.setText(text)
        self._response.setStyleSheet(
            f"color: {theme.SUCCESS}; font-size: 13px; line-height: 1.45; {_FONT_STACK} {_BG}"
        )
        self._set_status("done", theme.SUCCESS)
        self._footer.setVisible(False)

    @pyqtSlot(bool)
    def set_ollama_ok(self, ok: bool):
        color   = theme.SUCCESS if ok else theme.ERROR
        tooltip = "Ollama connected" if ok else "Ollama disconnected"
        self._ollama_dot.setStyleSheet(
            f"color: {color}; font-size: 8px; {_FONT_STACK} {_BG}"
        )
        self._ollama_dot.setToolTip(tooltip)

    @pyqtSlot(str)
    def set_language_mode(self, mode: str):
        labels = {"auto": "AUTO", "pt": "PT", "en": "EN"}
        text = labels.get(mode, mode.upper()[:4])
        self._lang_mode_text = text
        self._lang_badge.setText(text)

    @pyqtSlot(str)
    def show_detected_language(self, lang: str):
        labels = {"pt": "PT", "en": "EN", "portuguese": "PT", "english": "EN"}
        display = labels.get(lang.lower(), lang.upper()[:2])
        self._lang_badge.setText(display)
        QTimer.singleShot(3000, self._restore_language_badge)

    def _restore_language_badge(self):
        self._lang_badge.setText(self._lang_mode_text)

    def _auto_hide(self):
        self._transcript.setText("–")
        self._transcript.setStyleSheet(
            f"color: rgba(235,232,248,0.15); font-size: 14px; {_FONT_STACK} {_BG}"
        )
        self._response.setText("")
        self.hide()

    # ── Drag ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        self._drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if hasattr(self, "_drag_pos"):
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()

    def paintEvent(self, event):
        pass


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _darken(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},0.18)"
    return "rgba(80,80,80,0.18)"
