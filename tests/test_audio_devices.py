"""Regression tests for src/utils/audio_devices.py.

All tests are hardware-free: sounddevice is mocked via conftest.py and
per-test monkeypatching.  No real audio hardware, no Qt, no Whisper.

Scenarios covered
-----------------
- Input filtering (max_input_channels > 0 gate)
- Output filtering (max_output_channels > 0 gate)
- Zero-channel rejection on both directions
- Deduplication: identical (name, host_api_index) → one entry
- Safe non-collapsing: same name, different host API → both preserved
- Display label: no raw index prefix, host-API disambiguation, default marker
- resolve_input: stable selector found                → status "resolved"
- resolve_input: stable selector missing, legacy int  → status "legacy_int"
- resolve_input: stable selector missing, int missing → status "missing"
- resolve_input: legacy int wrong direction           → status "wrong_direction"
- resolve_input: no config at all                     → status "system_default"
- resolve_output: same resolution paths
- Backward compatibility: old int-only config still resolves
- resolve_input: legacy int out of range              → status "missing"
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _make_raw_devices(entries: list[dict]) -> list[dict]:
    """Fill in missing fields with safe defaults so tests stay minimal."""
    defaults = {
        "hostapi": 0,
        "max_input_channels": 0,
        "max_output_channels": 0,
    }
    return [{**defaults, **e} for e in entries]


def _make_sd_mock(devices: list[dict], default_in: int = 0, default_out: int = 1,
                  hostapis: list[dict] | None = None):
    """Return a mock sounddevice namespace from raw device list."""
    hapis = hostapis or [{"name": "MME"}, {"name": "Windows WASAPI"}]

    def _query(device=None, kind=None):
        if device is None:
            return devices
        if isinstance(device, int) and 0 <= device < len(devices):
            return devices[device]
        raise ValueError(f"Unknown device: {device}")

    mock = types.SimpleNamespace(
        query_devices=_query,
        query_hostapis=lambda: hapis,
        default=types.SimpleNamespace(device=[default_in, default_out]),
    )
    return mock


@pytest.fixture
def patch_sd(monkeypatch):
    """Returns a callable that replaces the sounddevice stub for one test."""
    def _apply(devices, default_in=0, default_out=1, hostapis=None):
        import sounddevice as sd_mod
        mock = _make_sd_mock(devices, default_in, default_out, hostapis)
        monkeypatch.setattr(sd_mod, "query_devices",  mock.query_devices)
        monkeypatch.setattr(sd_mod, "query_hostapis", mock.query_hostapis)
        monkeypatch.setattr(sd_mod, "default",        mock.default)
    return _apply


# ── enumerate_devices + filtering ─────────────────────────────────────────────

class TestEnumerateAndFilter:
    def test_filter_inputs_keeps_only_input_capable(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Mic",     "max_input_channels": 2, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]))
        from utils.audio_devices import enumerate_devices, filter_inputs
        devs   = enumerate_devices()
        inputs = filter_inputs(devs)
        assert len(inputs) == 1
        assert inputs[0].name == "Mic"

    def test_filter_outputs_keeps_only_output_capable(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Mic",     "max_input_channels": 2, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]))
        from utils.audio_devices import enumerate_devices, filter_outputs
        devs    = enumerate_devices()
        outputs = filter_outputs(devs)
        assert len(outputs) == 1
        assert outputs[0].name == "Speaker"

    def test_zero_channel_entries_rejected_from_inputs(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Ghost", "max_input_channels": 0, "max_output_channels": 0},
            {"name": "Mic",   "max_input_channels": 1, "max_output_channels": 0},
        ]))
        from utils.audio_devices import enumerate_devices, filter_inputs
        inputs = filter_inputs(enumerate_devices())
        assert all(d.name != "Ghost" for d in inputs)

    def test_zero_channel_entries_rejected_from_outputs(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Ghost",   "max_input_channels": 0, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]))
        from utils.audio_devices import enumerate_devices, filter_outputs
        outputs = filter_outputs(enumerate_devices())
        assert all(d.name != "Ghost" for d in outputs)

    def test_bidirectional_device_appears_in_both_filters(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Headset", "max_input_channels": 1, "max_output_channels": 2},
        ]))
        from utils.audio_devices import enumerate_devices, filter_inputs, filter_outputs
        devs = enumerate_devices()
        assert len(filter_inputs(devs))  == 1
        assert len(filter_outputs(devs)) == 1


class TestDeduplication:
    def test_identical_name_and_host_api_collapsed(self, patch_sd):
        """Two entries with the same name and same host API → only one kept."""
        patch_sd(_make_raw_devices([
            {"name": "Mic", "hostapi": 0, "max_input_channels": 2},
            {"name": "Mic", "hostapi": 0, "max_input_channels": 2},  # exact duplicate
        ]))
        from utils.audio_devices import enumerate_devices
        devs = enumerate_devices()
        assert len(devs) == 1

    def test_same_name_different_host_api_both_kept(self, patch_sd):
        """Same name but different host API → both preserved (conservative rule)."""
        patch_sd(_make_raw_devices([
            {"name": "Mic", "hostapi": 0, "max_input_channels": 2},
            {"name": "Mic", "hostapi": 1, "max_input_channels": 2},  # different API
        ]))
        from utils.audio_devices import enumerate_devices
        devs = enumerate_devices()
        assert len(devs) == 2

    def test_similar_but_distinct_names_both_kept(self, patch_sd):
        """Names that look similar (with/without numeric prefix) are never merged."""
        patch_sd(_make_raw_devices([
            {"name": "Microphone (Realtek HD Audio)",      "hostapi": 0, "max_input_channels": 2},
            {"name": "Microphone (2- Realtek HD Audio)",   "hostapi": 0, "max_input_channels": 2},
        ]))
        from utils.audio_devices import enumerate_devices
        devs = enumerate_devices()
        assert len(devs) == 2


class TestDisplayLabels:
    def test_no_raw_index_prefix_in_label(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Microphone",   "max_input_channels": 2},
            {"name": "Speakers",     "max_output_channels": 2},
        ]))
        from utils.audio_devices import enumerate_devices
        for d in enumerate_devices():
            assert not d.display_label.startswith("[")

    def test_default_input_marker_in_label(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Mic",     "max_input_channels": 2},
            {"name": "Speaker", "max_output_channels": 2},
        ]), default_in=0, default_out=1)
        from utils.audio_devices import enumerate_devices, filter_inputs
        inputs = filter_inputs(enumerate_devices())
        default_devs = [d for d in inputs if d.is_default_input]
        assert len(default_devs) == 1
        assert "Default" in default_devs[0].display_label

    def test_ambiguous_names_disambiguated_with_host_api(self, patch_sd):
        patch_sd(
            _make_raw_devices([
                {"name": "Mic", "hostapi": 0, "max_input_channels": 2},
                {"name": "Mic", "hostapi": 1, "max_input_channels": 2},
            ]),
            hostapis=[{"name": "MME"}, {"name": "Windows WASAPI"}],
        )
        from utils.audio_devices import enumerate_devices
        devs = enumerate_devices()
        labels = [d.display_label for d in devs]
        # Both labels must contain the host API name to tell them apart
        assert any("MME" in lbl for lbl in labels)
        assert any("WASAPI" in lbl or "Windows WASAPI" in lbl for lbl in labels)

    def test_unique_name_has_no_host_api_in_label(self, patch_sd):
        patch_sd(
            _make_raw_devices([
                {"name": "Realtek Mic", "hostapi": 0, "max_input_channels": 2},
            ]),
            hostapis=[{"name": "MME"}],
        )
        from utils.audio_devices import enumerate_devices
        devs = enumerate_devices()
        assert "MME" not in devs[0].display_label

    def test_identity_key_format(self, patch_sd):
        patch_sd(_make_raw_devices([{"name": "Test Mic", "hostapi": 2, "max_input_channels": 1}]))
        from utils.audio_devices import enumerate_devices
        devs = enumerate_devices()
        assert devs[0].identity_key == "Test Mic::2"


# ── resolve_input ──────────────────────────────────────────────────────────────

class TestResolveInput:
    def _cfg(self, **kwargs):
        """Minimal config-like dict wrapper."""
        data = {"mic_device": None, "mic_device_selector": None}
        data.update(kwargs)
        class _C:
            def get(self, k, default=None):
                return data.get(k, default)
        return _C()

    def test_stable_selector_resolves(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Good Mic", "hostapi": 0, "max_input_channels": 2},
        ]))
        from utils.audio_devices import resolve_input
        cfg = self._cfg(mic_device_selector="Good Mic::0")
        res = resolve_input(cfg)
        assert res.status == "resolved"
        assert res.device_index == 0
        assert res.device_name == "Good Mic"

    def test_stable_selector_missing_falls_to_legacy_int(self, patch_sd):
        """Selector no longer matches → legacy int picks up the device."""
        patch_sd(_make_raw_devices([
            {"name": "Other Mic", "hostapi": 0, "max_input_channels": 2},
        ]))
        from utils.audio_devices import resolve_input
        cfg = self._cfg(mic_device_selector="Gone Mic::0", mic_device=0)
        res = resolve_input(cfg)
        assert res.status == "legacy_int"
        assert res.device_index == 0

    def test_legacy_int_resolves_for_input_capable_device(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "My Mic", "hostapi": 0, "max_input_channels": 2},
            {"name": "Speaker", "hostapi": 0, "max_output_channels": 2},
        ]))
        from utils.audio_devices import resolve_input
        cfg = self._cfg(mic_device=0)
        res = resolve_input(cfg)
        assert res.status == "legacy_int"
        assert res.device_index == 0
        assert res.device_name == "My Mic"

    def test_legacy_int_wrong_direction_returns_wrong_direction(self, patch_sd):
        """Integer index points to an output-only device → wrong_direction."""
        patch_sd(_make_raw_devices([
            {"name": "Speaker Only", "hostapi": 0,
             "max_input_channels": 0, "max_output_channels": 2},
        ]))
        from utils.audio_devices import resolve_input
        cfg = self._cfg(mic_device=0)
        res = resolve_input(cfg)
        assert res.status == "wrong_direction"
        assert res.device_index is None

    def test_legacy_int_out_of_range_returns_missing(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Mic", "max_input_channels": 2},
        ]))
        from utils.audio_devices import resolve_input
        cfg = self._cfg(mic_device=99)
        res = resolve_input(cfg)
        assert res.status == "missing"
        assert res.device_index is None

    def test_no_config_returns_system_default(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Default Mic", "max_input_channels": 2},
        ]), default_in=0)
        from utils.audio_devices import resolve_input
        cfg = self._cfg()   # mic_device=None, selector=None
        res = resolve_input(cfg)
        assert res.status == "system_default"
        assert res.device_index is None

    def test_backward_compat_old_int_config_works(self, patch_sd):
        """Existing configs with only mic_device: 3 (int) must still resolve."""
        raw = _make_raw_devices([
            {"name": "Mic 0", "max_input_channels": 2},
            {"name": "Mic 1", "max_input_channels": 2},
            {"name": "Mic 2", "max_input_channels": 2},
            {"name": "Mic 3", "max_input_channels": 2},
        ])
        patch_sd(raw)
        from utils.audio_devices import resolve_input
        cfg = self._cfg(mic_device=3)  # no selector
        res = resolve_input(cfg)
        assert res.status == "legacy_int"
        assert res.device_index == 3
        assert res.device_name == "Mic 3"

    def test_selector_takes_priority_over_legacy_int(self, patch_sd):
        """When both selector and int are set, selector wins."""
        patch_sd(_make_raw_devices([
            {"name": "Mic A", "hostapi": 0, "max_input_channels": 2},
            {"name": "Mic B", "hostapi": 0, "max_input_channels": 2},
        ]))
        from utils.audio_devices import resolve_input
        # selector points to index 1, legacy int points to index 0
        cfg = self._cfg(mic_device=0, mic_device_selector="Mic B::0")
        # "Mic B::0" → index 1, so resolved device is 1
        res = resolve_input(cfg)
        assert res.status == "resolved"
        assert res.device_name == "Mic B"

    def test_wrong_direction_message_is_informative(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "HDMI Out", "max_input_channels": 0, "max_output_channels": 8},
        ]))
        from utils.audio_devices import resolve_input
        cfg = self._cfg(mic_device=0)
        res = resolve_input(cfg)
        assert "output" in res.message.lower() or "no input" in res.message.lower()

    def test_missing_message_mentions_fallback(self, patch_sd):
        patch_sd(_make_raw_devices([]))
        from utils.audio_devices import resolve_input
        cfg = self._cfg(mic_device=5)
        res = resolve_input(cfg)
        assert res.status == "missing"
        assert "system default" in res.message.lower() or "fallback" in res.message.lower()


# ── resolve_output ─────────────────────────────────────────────────────────────

class TestResolveOutput:
    def _cfg(self, **kwargs):
        data = {"output_device": None, "output_device_selector": None}
        data.update(kwargs)
        class _C:
            def get(self, k, default=None):
                return data.get(k, default)
        return _C()

    def test_stable_selector_resolves(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Speakers", "hostapi": 0, "max_output_channels": 2},
        ]))
        from utils.audio_devices import resolve_output
        cfg = self._cfg(output_device_selector="Speakers::0")
        res = resolve_output(cfg)
        assert res.status == "resolved"
        assert res.device_index == 0

    def test_legacy_int_resolves_for_output_capable_device(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Mic Only",  "max_input_channels": 2},
            {"name": "Speakers",  "max_output_channels": 2},
        ]))
        from utils.audio_devices import resolve_output
        cfg = self._cfg(output_device=1)
        res = resolve_output(cfg)
        assert res.status == "legacy_int"
        assert res.device_name == "Speakers"

    def test_legacy_int_wrong_direction_input_only(self, patch_sd):
        """Output int index pointing to an input-only device → wrong_direction."""
        patch_sd(_make_raw_devices([
            {"name": "Input Only", "max_input_channels": 2, "max_output_channels": 0},
        ]))
        from utils.audio_devices import resolve_output
        cfg = self._cfg(output_device=0)
        res = resolve_output(cfg)
        assert res.status == "wrong_direction"
        assert res.device_index is None

    def test_no_config_returns_system_default(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Speakers", "max_output_channels": 2},
        ]), default_out=0)
        from utils.audio_devices import resolve_output
        cfg = self._cfg()
        res = resolve_output(cfg)
        assert res.status == "system_default"
        assert res.device_index is None

    def test_legacy_int_out_of_range_returns_missing(self, patch_sd):
        patch_sd(_make_raw_devices([
            {"name": "Speaker", "max_output_channels": 2},
        ]))
        from utils.audio_devices import resolve_output
        cfg = self._cfg(output_device=42)
        res = resolve_output(cfg)
        assert res.status == "missing"
        assert res.device_index is None

    def test_backward_compat_old_int_config_works(self, patch_sd):
        raw = _make_raw_devices([
            {"name": "Spk A", "max_output_channels": 2},
            {"name": "Spk B", "max_output_channels": 2},
        ])
        patch_sd(raw)
        from utils.audio_devices import resolve_output
        cfg = self._cfg(output_device=1)
        res = resolve_output(cfg)
        assert res.status == "legacy_int"
        assert res.device_index == 1
        assert res.device_name == "Spk B"


# ── Resolution result contract ─────────────────────────────────────────────────

class TestResolutionResultContract:
    def test_resolve_input_always_returns_resolution_result(self, patch_sd):
        patch_sd(_make_raw_devices([]))
        from utils.audio_devices import resolve_input, ResolutionResult
        class _C:
            def get(self, k, d=None): return d
        res = resolve_input(_C())
        assert isinstance(res, ResolutionResult)
        assert res.status in ("system_default", "resolved", "legacy_int", "missing", "wrong_direction")
        assert isinstance(res.device_name, str)
        assert isinstance(res.message, str)

    def test_resolve_output_always_returns_resolution_result(self, patch_sd):
        patch_sd(_make_raw_devices([]))
        from utils.audio_devices import resolve_output, ResolutionResult
        class _C:
            def get(self, k, d=None): return d
        res = resolve_output(_C())
        assert isinstance(res, ResolutionResult)
        assert res.status in ("system_default", "resolved", "legacy_int", "missing", "wrong_direction")

    def test_device_index_none_when_status_is_system_default(self, patch_sd):
        patch_sd(_make_raw_devices([{"name": "Mic", "max_input_channels": 2}]), default_in=0)
        from utils.audio_devices import resolve_input
        class _C:
            def get(self, k, d=None): return d
        res = resolve_input(_C())
        assert res.status == "system_default"
        assert res.device_index is None

    def test_device_index_is_int_when_resolved(self, patch_sd):
        patch_sd(_make_raw_devices([{"name": "Mic", "hostapi": 0, "max_input_channels": 2}]))
        from utils.audio_devices import resolve_input
        class _C:
            def get(self, k, d=None):
                return {"mic_device_selector": "Mic::0"}.get(k, d)
        res = resolve_input(_C())
        assert res.status == "resolved"
        assert isinstance(res.device_index, int)

    def test_identity_key_set_when_device_found(self, patch_sd):
        patch_sd(_make_raw_devices([{"name": "Mic", "hostapi": 0, "max_input_channels": 2}]))
        from utils.audio_devices import resolve_input
        class _C:
            def get(self, k, d=None):
                return {"mic_device_selector": "Mic::0"}.get(k, d)
        res = resolve_input(_C())
        assert res.identity_key == "Mic::0"
