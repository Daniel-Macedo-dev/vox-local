"""Shared audio device resolver — normalisation, filtering, and stable resolution.

This module centralises every operation that previously duplicated or
improvised audio device logic across the UI, listener, speaker, and
startup validation:

  - Raw sounddevice inventory reading
  - Input / output direction filtering (zero-channel rejection)
  - Normalised identity key generation (stable across reboots)
  - Conservative duplicate collapsing (only identical name+host_api)
  - Display-label generation with host-API disambiguation and default markers
  - Stable persisted-selector resolution (with legacy-int fallback)
  - System-default fallback
  - Truthful ResolutionResult for UI and logging

Public surface
--------------
  enumerate_devices()   -> list[AudioDevice]
  filter_inputs(...)    -> list[AudioDevice]
  filter_outputs(...)   -> list[AudioDevice]
  resolve_input(config) -> ResolutionResult
  resolve_output(config)-> ResolutionResult

All functions are pure (no Qt, no global mutable state).
sounddevice is imported at module level; tests mock it via conftest.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sounddevice as sd

from utils.logger import get_logger

_log = get_logger("AudioDevices")


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AudioDevice:
    """Normalised representation of a single PortAudio device entry."""
    index: int
    name: str
    host_api_index: int
    host_api_name: str
    max_input_channels: int
    max_output_channels: int
    is_default_input: bool
    is_default_output: bool
    identity_key: str      # stable selector: "<name>::<host_api_index>"
    display_label: str     # human-readable — no raw index prefix


@dataclass(frozen=True)
class ResolutionResult:
    """Outcome of resolving a saved device selector to a live device.

    Possible ``status`` values:

    ``"system_default"``   — no device configured; system default in use
    ``"resolved"``         — stable selector (mic_device_selector) matched
    ``"legacy_int"``       — legacy integer index (mic_device) matched
    ``"wrong_direction"``  — stored value points to a wrong-direction device
    ``"missing"``          — stored value not in the current device list
    """
    device_index: Optional[int]   # None → pass to sounddevice as system default
    device_name: str
    status: str
    message: str
    identity_key: Optional[str] = None


# ── Identity ───────────────────────────────────────────────────────────────────

def _make_identity_key(name: str, host_api_index: int) -> str:
    """Stable identity string: raw device name + host API index.

    Does NOT strip numeric PortAudio suffixes (e.g. "2-" in "Microphone (2- HD)")
    to avoid falsely merging distinct devices with similar names.
    Two entries are the same practical device only when both the raw name
    and the host API index match exactly.
    """
    return f"{name}::{host_api_index}"


# ── Enumeration ────────────────────────────────────────────────────────────────

def enumerate_devices() -> list[AudioDevice]:
    """Return a normalised, deduplicated list of all audio devices.

    Deduplication rule
    ~~~~~~~~~~~~~~~~~~
    Only entries whose ``(name, host_api_index)`` pair is identical are
    collapsed.  Entries with the same device name but different host APIs
    (e.g. WASAPI vs MME on Windows) are preserved and distinguished by
    their display label.  This is the conservative safe path: merging by
    name similarity alone risks hiding two actually-different endpoints.

    Zero-channel entries are **not** excluded here — callers use
    ``filter_inputs`` / ``filter_outputs`` for direction-appropriate subsets.
    The system default device is never hidden.
    """
    try:
        raw_devices = sd.query_devices()
        default_in, default_out = sd.default.device
    except Exception:
        return []

    # Host API name lookup
    host_api_names: dict[int, str] = {}
    try:
        for i, api in enumerate(sd.query_hostapis()):
            host_api_names[i] = api["name"]
    except Exception:
        pass

    # First pass: collect raw info, deduplicate by identity key
    raw_list: list[dict] = []
    seen_keys: set[str] = set()
    for i, d in enumerate(raw_devices):
        hai = int(d.get("hostapi", 0))
        key = _make_identity_key(d["name"], hai)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        raw_list.append({
            "index":               i,
            "name":                d["name"],
            "host_api_index":      hai,
            "host_api_name":       host_api_names.get(hai, f"API {hai}"),
            "max_input_channels":  int(d.get("max_input_channels",  0)),
            "max_output_channels": int(d.get("max_output_channels", 0)),
            "is_default_input":    (i == default_in),
            "is_default_output":   (i == default_out),
            "identity_key":        key,
        })

    # Count how many devices share a name (for disambiguation)
    name_counts: dict[str, int] = {}
    for r in raw_list:
        name_counts[r["name"]] = name_counts.get(r["name"], 0) + 1

    # Second pass: assign display labels
    devices: list[AudioDevice] = []
    for r in raw_list:
        # Base label: raw device name — no raw-index prefix
        label = r["name"]

        # Disambiguate when the same name appears via multiple host APIs
        if name_counts[r["name"]] > 1:
            label = f"{r['name']} [{r['host_api_name']}]"

        # Mark default devices so the user always knows which one is the OS default
        markers: list[str] = []
        if r["is_default_input"] and r["is_default_output"]:
            markers.append("Default")
        elif r["is_default_input"]:
            markers.append("Default Input")
        elif r["is_default_output"]:
            markers.append("Default Output")

        if markers:
            label = f"{label}  \u2605 {', '.join(markers)}"

        devices.append(AudioDevice(
            index=r["index"],
            name=r["name"],
            host_api_index=r["host_api_index"],
            host_api_name=r["host_api_name"],
            max_input_channels=r["max_input_channels"],
            max_output_channels=r["max_output_channels"],
            is_default_input=r["is_default_input"],
            is_default_output=r["is_default_output"],
            identity_key=r["identity_key"],
            display_label=label,
        ))

    return devices


def filter_inputs(devices: list[AudioDevice]) -> list[AudioDevice]:
    """Return only devices with at least one input channel."""
    return [d for d in devices if d.max_input_channels > 0]


def filter_outputs(devices: list[AudioDevice]) -> list[AudioDevice]:
    """Return only devices with at least one output channel."""
    return [d for d in devices if d.max_output_channels > 0]


# ── Resolution helpers ─────────────────────────────────────────────────────────

def _system_default_result(direction: str) -> ResolutionResult:
    """Build a ResolutionResult for the OS-level system default device."""
    try:
        default_in, default_out = sd.default.device
        idx  = default_in if direction == "input" else default_out
        info = sd.query_devices(idx)
        name = info["name"]
    except Exception:
        name = "System Default"
    return ResolutionResult(
        device_index=None,
        device_name=name,
        status="system_default",
        message=f"Using system default {direction} device: '{name}'.",
    )


# ── Public resolver API ────────────────────────────────────────────────────────

def resolve_input(config) -> ResolutionResult:
    """Resolve the configured microphone to a live PortAudio device.

    Resolution order
    ~~~~~~~~~~~~~~~~
    1. Stable selector  — ``mic_device_selector`` matched by identity key
    2. Legacy integer   — ``mic_device`` int validated as an input device
    3. System default   — no device configured, or all lookups failed

    The returned ``status`` tells consumers exactly which path was taken so
    they can surface truthful messages in the UI, logs, and diagnostics.

    Wrong-direction protection
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    If the stored integer or selector points to a device with zero input
    channels (i.e. it is an output-only device), the status is set to
    ``"wrong_direction"`` and ``device_index`` is ``None`` (system default
    will be used by the runtime).
    """
    devices = enumerate_devices()
    inputs  = filter_inputs(devices)

    selector   = config.get("mic_device_selector", None)
    legacy_idx = config.get("mic_device", None)

    # 1. Stable selector
    if selector is not None:
        match = next((d for d in inputs if d.identity_key == selector), None)
        if match:
            return ResolutionResult(
                device_index=match.index,
                device_name=match.name,
                status="resolved",
                message=f"Saved microphone '{match.name}' resolved to index {match.index}.",
                identity_key=match.identity_key,
            )
        _log.warning(
            f"mic_device_selector '{selector}' not found — "
            "device may have been renamed or disconnected. "
            "Falling back to legacy index or system default."
        )

    # 2. Legacy integer
    if legacy_idx is not None and isinstance(legacy_idx, int):
        all_match = next((d for d in devices if d.index == legacy_idx), None)
        if all_match is not None:
            if all_match.max_input_channels > 0:
                return ResolutionResult(
                    device_index=all_match.index,
                    device_name=all_match.name,
                    status="legacy_int",
                    message=(
                        f"mic_device={legacy_idx} → '{all_match.name}'. "
                        "Open the Audio tab and save to upgrade to stable device matching."
                    ),
                    identity_key=all_match.identity_key,
                )
            else:
                return ResolutionResult(
                    device_index=None,
                    device_name=all_match.name,
                    status="wrong_direction",
                    message=(
                        f"mic_device={legacy_idx} ('{all_match.name}') has no input channels "
                        "— it is an output device. Falling back to system default."
                    ),
                )
        else:
            return ResolutionResult(
                device_index=None,
                device_name="System Default",
                status="missing",
                message=(
                    f"mic_device={legacy_idx} is not in the current device list "
                    "(device may have been disconnected or indices reassigned). "
                    "Falling back to system default."
                ),
            )

    # 3. System default
    return _system_default_result("input")


def resolve_output(config) -> ResolutionResult:
    """Resolve the configured speaker to a live PortAudio device.

    Identical resolution logic to ``resolve_input`` but for the output
    direction.  Reads ``output_device_selector`` (stable) and
    ``output_device`` (legacy integer).
    """
    devices = enumerate_devices()
    outputs = filter_outputs(devices)

    selector   = config.get("output_device_selector", None)
    legacy_idx = config.get("output_device", None)

    # 1. Stable selector
    if selector is not None:
        match = next((d for d in outputs if d.identity_key == selector), None)
        if match:
            return ResolutionResult(
                device_index=match.index,
                device_name=match.name,
                status="resolved",
                message=f"Saved output '{match.name}' resolved to index {match.index}.",
                identity_key=match.identity_key,
            )
        _log.warning(
            f"output_device_selector '{selector}' not found — "
            "device may have been renamed or disconnected. "
            "Falling back to legacy index or system default."
        )

    # 2. Legacy integer
    if legacy_idx is not None and isinstance(legacy_idx, int):
        all_match = next((d for d in devices if d.index == legacy_idx), None)
        if all_match is not None:
            if all_match.max_output_channels > 0:
                return ResolutionResult(
                    device_index=all_match.index,
                    device_name=all_match.name,
                    status="legacy_int",
                    message=(
                        f"output_device={legacy_idx} → '{all_match.name}'. "
                        "Open the Audio tab and save to upgrade to stable device matching."
                    ),
                    identity_key=all_match.identity_key,
                )
            else:
                return ResolutionResult(
                    device_index=None,
                    device_name=all_match.name,
                    status="wrong_direction",
                    message=(
                        f"output_device={legacy_idx} ('{all_match.name}') has no output channels "
                        "— it is an input-only device. Falling back to system default."
                    ),
                )
        else:
            return ResolutionResult(
                device_index=None,
                device_name="System Default",
                status="missing",
                message=(
                    f"output_device={legacy_idx} is not in the current device list "
                    "(device may have been disconnected or indices reassigned). "
                    "Falling back to system default."
                ),
            )

    # 3. System default
    return _system_default_result("output")
