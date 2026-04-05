"""
tests/test_executor.py

Tests for src/executor.py.
No hardware, GUI, or audio dependencies are imported.
"""

import glob as _glob
import os
import sys
import yaml
import pytest
from unittest.mock import patch, MagicMock

from utils.config import Config
from executor import Executor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path, extra: dict = None) -> Config:
    data = {}
    if extra:
        data.update(extra)
    p = tmp_path / "settings.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return Config(path=str(p))


# ---------------------------------------------------------------------------
# Allowlist enforcement
# ---------------------------------------------------------------------------

class TestLiveAllowlistRead:
    """run() must read allowed_actions from config on every call — no reload needed."""

    def test_run_enforces_live_config_without_reload(self, tmp_path):
        """Revoking an action in config takes effect on the next run() call,
        without calling reload_config()."""
        cfg = _make_config(tmp_path, {"allowed_actions": ["show_time"]})
        ex = Executor(cfg)

        # show_time is allowed initially
        result = ex.run("show_time", {})
        assert ":" in result

        # Revoke it in-memory without calling reload_config()
        cfg.set("allowed_actions", [])
        result = ex.run("show_time", {})
        assert "permitida" in result

    def test_run_picks_up_newly_allowed_action_without_reload(self, tmp_path):
        """Granting a new action in config takes effect on the next run() call."""
        cfg = _make_config(tmp_path, {"allowed_actions": []})
        ex = Executor(cfg)

        # show_time is not allowed — denied
        assert "permitida" in ex.run("show_time", {})

        # Grant it in-memory without reload_config()
        cfg.set("allowed_actions", ["show_time"])
        result = ex.run("show_time", {})
        assert ":" in result  # time string — action executed


class TestReloadConfig:
    def test_reload_config_picks_up_new_allowed_actions(self, tmp_path):
        cfg = _make_config(tmp_path, {"allowed_actions": ["show_time"]})
        ex = Executor(cfg)
        assert "open_app" not in ex.allowed_actions

        # Simulate user enabling open_app and saving
        cfg.set("allowed_actions", ["show_time", "open_app"])
        ex.reload_config()
        assert "open_app" in ex.allowed_actions

    def test_reload_config_respects_removed_actions(self, tmp_path):
        cfg = _make_config(tmp_path, {"allowed_actions": ["show_time", "open_app"]})
        ex = Executor(cfg)
        assert "open_app" in ex.allowed_actions

        cfg.set("allowed_actions", ["show_time"])
        ex.reload_config()
        assert "open_app" not in ex.allowed_actions

    def test_reload_config_with_no_allowed_actions(self, tmp_path):
        cfg = _make_config(tmp_path, {"allowed_actions": ["show_time"]})
        ex = Executor(cfg)
        cfg.set("allowed_actions", [])
        ex.reload_config()
        assert len(ex.allowed_actions) == 0
        result = ex.run("show_time", {})
        assert "permitida" in result or "permitted" in result.lower()


class TestAllowlist:
    def test_disallowed_action_returns_error(self, tmp_path):
        cfg = _make_config(tmp_path, {"allowed_actions": ["show_time"]})
        ex = Executor(cfg)
        result = ex.run("open_app", {"name": "notepad"})
        assert "não está permitida" in result.lower() or "not permitted" in result.lower() or "permitida" in result

    def test_allowed_action_does_not_return_permission_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        result = ex.run("show_time", {})
        assert ":" in result  # e.g. "São 14:35."

    def test_unimplemented_action_not_in_map(self, tmp_path):
        cfg = _make_config(tmp_path, {
            "allowed_actions": ["show_time", "nonexistent_action"]
        })
        ex = Executor(cfg)
        # Force it into allowed but not in _action_map
        ex.allowed_actions.add("nonexistent_action")
        result = ex.run("nonexistent_action", {})
        assert "implementada" in result or "not implemented" in result.lower()


# ---------------------------------------------------------------------------
# set_volume
# ---------------------------------------------------------------------------

class TestSetVolume:
    def test_clamps_to_100(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        # Patch out pycaw on Windows or skip OS-specific branch
        with patch.object(ex, "_set_volume", wraps=ex._set_volume):
            # We test the clamp logic directly via the internal method
            # by patching the Windows-only import block
            if os.name == "nt":
                with patch("executor.POINTER", MagicMock(), create=True), \
                     patch("executor.cast", MagicMock(), create=True):
                    try:
                        from unittest.mock import patch as p2
                        with p2.object(ex, "_set_volume") as mock_sv:
                            mock_sv.return_value = "Volume em 100%."
                            r = ex.run("set_volume", {"level": 150})
                    except Exception:
                        pass
            # Direct call — clamping happens before OS branch
            # Stub the OS-specific part
            with patch("os.name", "posix"), \
                 patch("subprocess.run", return_value=MagicMock(returncode=0)):
                result = ex._set_volume(150)
        assert "100%" in result

    def test_clamps_to_0(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        with patch("os.name", "posix"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)):
            result = ex._set_volume(-50)
        assert "0%" in result

    def test_string_level_coerced(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        with patch("os.name", "posix"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)):
            result = ex._set_volume("42")
        assert "42%" in result

    def test_non_numeric_level_returns_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        result = ex._set_volume("loud")
        assert "inválido" in result.lower() or "invalid" in result.lower()

    def test_none_params_handled(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        result = ex.run("show_time", None)
        assert ":" in result  # should still work


# ---------------------------------------------------------------------------
# show_time
# ---------------------------------------------------------------------------

class TestShowTime:
    def test_show_time_returns_time_string(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        result = ex._show_time()
        assert ":" in result

    def test_show_time_contains_hours_and_minutes(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        result = ex._show_time()
        # "São HH:MM." — extract HH:MM portion
        import re
        match = re.search(r"\d{1,2}:\d{2}", result)
        assert match is not None


# ---------------------------------------------------------------------------
# show_battery
# ---------------------------------------------------------------------------

class TestShowBattery:
    def test_show_battery_returns_string(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        mock_battery = MagicMock()
        mock_battery.percent = 75
        mock_battery.power_plugged = False
        with patch("psutil.sensors_battery", return_value=mock_battery):
            result = ex._show_battery()
        assert isinstance(result, str)
        assert "75" in result

    def test_show_battery_no_sensor(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        with patch("psutil.sensors_battery", return_value=None):
            result = ex._show_battery()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_show_battery_psutil_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        with patch("psutil.sensors_battery", side_effect=Exception("no sensor")):
            result = ex._show_battery()
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# close_app return code handling
# ---------------------------------------------------------------------------

class TestCloseApp:
    def test_close_app_not_found_returns_message(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        mock_result = MagicMock()
        mock_result.returncode = 128
        with patch("subprocess.run", return_value=mock_result), \
             patch("os.name", "nt"):
            result = ex._close_app("nonexistent")
        assert "encontrado" in result or "not found" in result.lower()

    def test_close_app_success_returns_message(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result), \
             patch("os.name", "nt"):
            result = ex._close_app("notepad")
        assert "notepad" in result.lower()


# ---------------------------------------------------------------------------
# search_file — path normalisation
# ---------------------------------------------------------------------------

class TestSearchFileNormalization:
    def test_tilde_not_in_glob_pattern(self, tmp_path):
        """search_dirs containing ~ must be expanded before globbing.

        Verify that glob.glob is never called with a literal ~ in the path —
        unexpanded ~ would cause glob to silently find nothing on most systems.
        """
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)

        captured_patterns = []

        def capture_glob(pattern, recursive=False):
            captured_patterns.append(pattern)
            return []

        with patch("glob.glob", side_effect=capture_glob):
            ex._search_file("myfile")

        for pat in captured_patterns:
            assert "~" not in pat, (
                f"Glob pattern contains unexpanded ~: {pat!r}"
            )

    def test_finds_file_in_configured_dir(self, tmp_path):
        """search_file locates a real file inside a configured search_dirs entry."""
        # Create a test file to be found.
        target = tmp_path / "my_report_2024.pdf"
        target.write_text("dummy")

        cfg = _make_config(tmp_path, {"search_dirs": [str(tmp_path)]})
        ex = Executor(cfg)

        result = ex._search_file("my_report_2024")
        assert "my_report_2024" in result

    def test_tilde_dir_is_expanded(self, tmp_path):
        """A search_dir value of '~' alone expands to the home directory."""
        cfg = _make_config(tmp_path, {"search_dirs": ["~"]})
        ex = Executor(cfg)

        home = os.path.expanduser("~")
        captured_patterns = []

        def capture_glob(pattern, recursive=False):
            captured_patterns.append(pattern)
            return []

        with patch("glob.glob", side_effect=capture_glob):
            ex._search_file("anything")

        assert any(home in p for p in captured_patterns), (
            f"Home dir '{home}' not found in glob patterns: {captured_patterns}"
        )

    def test_returns_not_found_message_when_empty(self, tmp_path):
        cfg = _make_config(tmp_path, {"search_dirs": [str(tmp_path)]})
        ex = Executor(cfg)
        result = ex._search_file("zzz_nonexistent_file_xyz")
        assert "Nenhum arquivo" in result or "nenhum" in result.lower()


# ---------------------------------------------------------------------------
# close_app — alias-aware process name derivation
# ---------------------------------------------------------------------------

class TestCloseAppAlias:
    def test_uri_scheme_alias_derives_process_name(self, tmp_path):
        """close_app("discord") with alias "discord://" must kill "discord.exe",
        not "discord://.exe"."""
        cfg = _make_config(tmp_path, {
            "app_aliases": {"discord": "discord://"},
        })
        ex = Executor(cfg)

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("os.name", "nt"):
            ex._close_app("discord")

        args = mock_run.call_args[0][0]
        # The process name argument must be "discord.exe", not "discord://.exe".
        assert "discord.exe" in args, f"Expected 'discord.exe' in args, got: {args}"
        assert "://" not in " ".join(str(a) for a in args), (
            "URI scheme leaked into taskkill arguments"
        )

    def test_colon_suffix_alias_derives_process_name(self, tmp_path):
        """close_app("spotify") with alias "spotify:" must kill "spotify.exe"."""
        cfg = _make_config(tmp_path, {
            "app_aliases": {"spotify": "spotify:"},
        })
        ex = Executor(cfg)

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("os.name", "nt"):
            ex._close_app("spotify")

        args = mock_run.call_args[0][0]
        assert "spotify.exe" in args, f"Expected 'spotify.exe' in args, got: {args}"

    def test_plain_alias_uses_alias_as_process(self, tmp_path):
        """close_app("vscode") with alias "code" must kill "code.exe"."""
        cfg = _make_config(tmp_path, {
            "app_aliases": {"vscode": "code"},
        })
        ex = Executor(cfg)

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("os.name", "nt"):
            ex._close_app("vscode")

        args = mock_run.call_args[0][0]
        assert "code.exe" in args, f"Expected 'code.exe' in args, got: {args}"

    def test_no_alias_uses_name_directly(self, tmp_path):
        """close_app with no alias entry uses the raw name as process."""
        cfg = _make_config(tmp_path, {"app_aliases": {}})
        ex = Executor(cfg)

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("os.name", "nt"):
            ex._close_app("notepad")

        args = mock_run.call_args[0][0]
        assert "notepad.exe" in args


# ---------------------------------------------------------------------------
# run() error paths — TypeError and generic Exception
# ---------------------------------------------------------------------------

class TestRunErrorPaths:
    def test_unexpected_kwarg_returns_invalid_params_message(self, tmp_path):
        """Passing an unrecognised keyword argument causes a TypeError inside the
        handler.  run() must catch it and return a user-facing error string."""
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        # show_time() takes no parameters — any kwarg causes TypeError.
        result = ex.run("show_time", {"unexpected": "value"})
        assert "inválido" in result.lower() or "invalid" in result.lower() or "parâmetro" in result.lower()
        assert "show_time" in result

    def test_generic_exception_from_handler_returns_error_message(self, tmp_path):
        """If a handler raises a non-TypeError exception, run() must catch it
        and return a structured error string — never propagate to the caller."""
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        ex._action_map["show_time"] = MagicMock(side_effect=RuntimeError("disk full"))
        result = ex.run("show_time", {})
        assert "erro" in result.lower() or "error" in result.lower()
        assert "show_time" in result


# ---------------------------------------------------------------------------
# open_url
# ---------------------------------------------------------------------------

class TestOpenUrl:
    def test_open_url_calls_webbrowser_and_returns_confirmation(self, tmp_path):
        """_open_url must call webbrowser.open with the exact URL and return
        a string that includes the URL."""
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        with patch("webbrowser.open") as mock_open:
            result = ex._open_url("https://example.com")
        mock_open.assert_called_once_with("https://example.com")
        assert "https://example.com" in result

    def test_open_url_via_run_dispatches_correctly(self, tmp_path):
        """open_url must be reachable through the normal run() dispatch path."""
        cfg = _make_config(tmp_path, {"allowed_actions": ["open_url"]})
        ex = Executor(cfg)
        with patch("webbrowser.open"):
            result = ex.run("open_url", {"url": "https://example.com"})
        assert "https://example.com" in result


# ---------------------------------------------------------------------------
# search_file — result count cap
# ---------------------------------------------------------------------------

class TestSearchFileResultCap:
    def test_search_file_caps_at_three_results(self, tmp_path):
        """search_file must never return more than 3 file names even when the
        directory contains more matching files."""
        for i in range(5):
            (tmp_path / f"report_{i}.txt").write_text("dummy")

        cfg = _make_config(tmp_path, {"search_dirs": [str(tmp_path)]})
        ex = Executor(cfg)
        result = ex._search_file("report")

        assert "Encontrei" in result
        # The result is "Encontrei: name1, name2, name3" — split on ", "
        names_part = result.replace("Encontrei: ", "")
        names = names_part.split(", ")
        assert len(names) <= 3
