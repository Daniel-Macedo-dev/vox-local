import yaml
import os
from typing import Any


DEFAULT_CONFIG = {
    # ── Activation ───────────────────────────────────────────────────────────
    "activation_mode":   "wake_word",   # "wake_word" | "push_to_talk"
    "push_to_talk_key":  "ctrl+shift",  # key combo used in push_to_talk mode

    # ── Wake word ────────────────────────────────────────────────────────────
    "wake_word":         "vox",
    "silence_threshold": 0.01,   # RMS threshold below which audio is considered silent
    "silence_duration":  1.5,    # seconds of silence that ends a command

    # ── STT ──────────────────────────────────────────────────────────────────
    "language":            "en",
    "whisper_model":       "base",
    "whisper_device":      "cpu",
    "whisper_compute_type": "int8",

    # ── LLM ──────────────────────────────────────────────────────────────────
    "ollama_url":    "http://localhost:11434",
    "ollama_model":  "qwen2.5:14b",
    "max_history":   20,

    # ── TTS ──────────────────────────────────────────────────────────────────
    "tts_enabled":  True,
    "piper_path":   "piper/piper/piper.exe",
    "voice_model":  "voices/en_US-ryan-high.onnx",

    # ── Audio devices ────────────────────────────────────────────────────────
    # Legacy integer indices — kept for backward compatibility with existing configs.
    # Prefer mic_device_selector / output_device_selector for new saves (stable
    # across reboots even when PortAudio reassigns indices).
    "mic_device":             None,
    "output_device":          None,
    # Stable selectors — set automatically when saving from the Audio tab.
    # Format: "<raw device name>::<host_api_index>"
    "mic_device_selector":    None,
    "output_device_selector": None,

    # ── Executor ─────────────────────────────────────────────────────────────
    "search_dirs": [
        "~/Documents",
        "~/Downloads",
        "~/Desktop",
    ],
    "app_aliases": {
        # URI schemes — launch the installed app via Windows registry
        "discord": "discord://",
        "spotify": "spotify:",
        # Executables in PATH or App Paths registry
        "chrome":      "chrome",
        "firefox":     "firefox",
        "vscode":      "code",
        "vs code":     "code",
        "notepad":     "notepad",
        "calculator":  "calc",
        "explorer":    "explorer",
        "paint":       "mspaint",
        "steam":       "steam://open/main",
    },
    "allowed_actions": [
        "open_app",
        "close_app",
        "set_volume",
        "mute_volume",
        "play_pause_media",
        "next_track",
        "prev_track",
        "search_file",
        "open_url",
        "type_text",
        "take_screenshot",
        "show_time",
        "show_battery",
    ],
}


class Config:
    def __init__(self, path: str = None):
        if path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(base, "config", "settings.yaml")

        self._path = path
        self._data = dict(DEFAULT_CONFIG)

        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                    self._data.update(loaded)
            except yaml.YAMLError as e:
                print(f"[Config] YAML parse error in {path}: {e}")
                print("[Config] Using default settings.")
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(DEFAULT_CONFIG, f, allow_unicode=True, default_flow_style=False)
            # Use print here — logger may not be initialised yet at config load time
            print(f"[Config] Created default config at {path}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value

    def save(self):
        """Persist current in-memory values back to the YAML file."""
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    on_disk = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                on_disk = {}
        else:
            on_disk = {}
        on_disk.update(self._data)
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(on_disk, f, allow_unicode=True, default_flow_style=False)
