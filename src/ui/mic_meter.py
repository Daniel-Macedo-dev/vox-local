"""Mic level meter widgets for VOX.

MicLevelBar       – compact horizontal bar for panels / dialogs.
MicLevelWaveform  – animated bar waveform for the overlay, driven by real RMS.

Both widgets accept a `set_level(float)` slot (0.0–1.0 normalised RMS).
"""
import random

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSlot
from PyQt6.QtGui import QColor, QPainter

from ui import theme


class MicLevelBar(QWidget):
    """Horizontal RMS level bar — green / amber / red."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0
        self.setFixedHeight(8)
        self.setMinimumWidth(60)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Smooth decay when signal drops
        self._decay_timer = QTimer(self)
        self._decay_timer.timeout.connect(self._decay)
        self._decay_timer.start(50)

    @pyqtSlot(float)
    def set_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))
        self.update()

    def _decay(self) -> None:
        if self._level > 0.005:
            self._level = max(0.0, self._level - 0.04)
            self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        w, h = self.width(), self.height()
        radius = h / 2
        # Track
        p.setBrush(QColor(18, 15, 28))
        p.drawRoundedRect(0, 0, w, h, radius, radius)
        # Fill
        fill = int(w * self._level)
        if fill > 1:
            if self._level < 0.60:
                c = QColor(29, 158, 117)    # teal/green
            elif self._level < 0.85:
                c = QColor(245, 158, 11)    # amber
            else:
                c = QColor(239, 68, 68)     # red
            p.setBrush(c)
            p.drawRoundedRect(0, 0, fill, h, radius, radius)
        p.end()


class MicLevelWaveform(QWidget):
    """Animated bar waveform driven by real RMS values.

    Replaces the old random-animation WaveformWidget in overlay.py.
    When `set_active(False)` the bars decay to minimum; when active they
    grow proportionally to the last RMS level received via `set_level`.
    """

    BAR_COUNT   = 10
    BAR_W       = 3
    BAR_GAP     = 4
    BAR_MIN     = 4
    BAR_MAX     = 24
    CONTAINER_H = 28

    def __init__(self, parent=None):
        super().__init__(parent)
        total_w = self.BAR_COUNT * self.BAR_W + (self.BAR_COUNT - 1) * self.BAR_GAP
        self.setFixedSize(total_w, self.CONTAINER_H)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._level  = 0.0
        self._active = False
        self._bars   = [float(self.BAR_MIN)] * self.BAR_COUNT
        self._timer  = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(80)

    def set_active(self, active: bool) -> None:
        self._active = active
        if not active:
            self._level = 0.0

    @pyqtSlot(float)
    def set_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))

    def _tick(self) -> None:
        n = self.BAR_COUNT
        for i in range(n):
            if not self._active:
                # Decay toward minimum
                self._bars[i] = max(self.BAR_MIN, self._bars[i] * 0.72)
            else:
                # Shape: taller in the centre, shorter at the edges
                center  = (n - 1) / 2
                dist    = abs(i - center) / center if center > 0 else 0.0
                shape   = 1.0 - dist * 0.42
                target  = self.BAR_MIN + (self.BAR_MAX - self.BAR_MIN) * self._level * shape
                jitter  = random.uniform(-1.5, 1.5) * max(0.05, self._level)
                target  = max(self.BAR_MIN, min(self.BAR_MAX, target + jitter))
                self._bars[i] = self._bars[i] * 0.45 + target * 0.55
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        cy = self.height() / 2
        n  = self.BAR_COUNT
        for i, h in enumerate(self._bars):
            h = min(h, self.BAR_MAX)
            x = i * (self.BAR_W + self.BAR_GAP)
            y = cy - h / 2
            denom = (n - 1) / 2 if n > 1 else 1
            edge  = 1.0 - abs(i - (n - 1) / 2) / denom
            # Active: expressive purple peaks. Inactive: near-invisible ghost bars.
            if self._active:
                alpha = int(55 + 185 * edge)
            else:
                alpha = int(14 + 34 * edge)
            p.setBrush(QColor(168, 85, 247, alpha))
            p.drawRoundedRect(QRectF(x, y, self.BAR_W, h), 1.5, 1.5)
        p.end()
