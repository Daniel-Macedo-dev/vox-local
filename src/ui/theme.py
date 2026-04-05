"""Shared visual design tokens for the VOX UI.

Import this module from control_center.py, overlay.py, and mic_meter.py
to keep the colour palette, status semantics, and typography in one place.

Nothing here should import from other VOX modules.
"""

# ── Base palette ───────────────────────────────────────────────────────────────
BG           = "#07050f"      # window background
PANEL        = "#0c0a18"      # tab pane / content area
CARD         = "#0f0c1d"      # elevated card surface
BORDER       = "#1e1b2e"      # subtle separator / control border
BORDER_STRONG = "#2d2945"     # visible border for active/hover states
ACCENT       = "#a855f7"      # primary accent (purple)
ACCENT_SOFT  = "#c084fc"      # lighter accent for display text

# ── Text hierarchy ─────────────────────────────────────────────────────────────
TEXT         = "#eceaf6"      # primary readable text
TEXT2        = "#b0abca"      # secondary / de-emphasised text
MUTED        = "#6b6687"      # labels, placeholders, captions
GHOST        = "#45405f"      # very muted — separators, timestamps

# ── Semantic status colours (shared by overlay and control center) ─────────────
SUCCESS      = "#22c55e"      # monitoring, connected, ok
WARNING      = "#f59e0b"      # transcribing, fair signal, notice
ERROR        = "#f87171"      # error, cancelled, disconnected
INFO         = "#38bdf8"      # informational tags, hints
LISTENING    = "#3b82f6"      # listening state (distinct blue)
ACTIVE       = "#a855f7"      # generating / responding / speaking (= ACCENT)

# ── Typography ─────────────────────────────────────────────────────────────────
FONT         = "'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif"
