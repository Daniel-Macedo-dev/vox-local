"""
tests/test_brain.py

Tests for src/brain.py — specifically _extract_action and Brain.process.
No Ollama or network calls are made; requests is mocked.
"""

import json
import sys
import yaml
import pytest
from unittest.mock import patch, MagicMock

from brain import _extract_action, _build_system_prompt, _ACTION_DOCS, Brain
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


def _make_brain(tmp_path) -> Brain:
    cfg = _make_config(tmp_path)
    ex = Executor(cfg)
    return Brain(cfg, ex)


# ---------------------------------------------------------------------------
# _extract_action
# ---------------------------------------------------------------------------

class TestExtractAction:
    def test_clean_json(self):
        text = '{"action":"show_time","params":{}}'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "show_time"
        assert result["params"] == {}

    def test_clean_json_with_params(self):
        text = '{"action":"set_volume","params":{"level":50}}'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "set_volume"
        assert result["params"]["level"] == 50

    def test_fenced_code_block_json(self):
        text = '```json\n{"action":"open_app","params":{"name":"spotify"}}\n```'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "open_app"

    def test_fenced_code_block_no_lang(self):
        text = '```\n{"action":"mute_volume","params":{}}\n```'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "mute_volume"

    def test_json_embedded_in_text(self):
        text = 'Sure, here you go: {"action":"next_track","params":{}} — done!'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "next_track"

    def test_plain_text_returns_none(self):
        text = "The weather today is sunny and warm."
        result = _extract_action(text)
        assert result is None

    def test_invalid_json_returns_none(self):
        text = '{"action": "show_time", broken json'
        result = _extract_action(text)
        assert result is None

    def test_json_without_action_key_returns_none(self):
        text = '{"name": "foo", "params": {}}'
        result = _extract_action(text)
        assert result is None

    def test_whitespace_padded_json(self):
        text = '   \n  {"action":"show_battery","params":{}}  \n  '
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "show_battery"


# ---------------------------------------------------------------------------
# Brain.process (mocked Ollama)
# ---------------------------------------------------------------------------

def _mock_stream_response(content: str):
    """Create a mock requests.Response that streams the given content as Ollama would."""
    lines = []
    for char in content:
        lines.append(json.dumps({"message": {"content": char}, "done": False}).encode())
    lines.append(json.dumps({"message": {"content": ""}, "done": True}).encode())

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines = MagicMock(return_value=iter(lines))
    return mock_resp


class TestBrainProcess:
    def test_process_plain_text_response(self, tmp_path):
        brain = _make_brain(tmp_path)
        mock_resp = _mock_stream_response("São 14:30.")
        with patch("requests.post", return_value=mock_resp):
            text, is_action = brain.process("que horas são")
        assert is_action is False
        assert "14:30" in text

    def test_process_action_response(self, tmp_path):
        brain = _make_brain(tmp_path)
        payload = '{"action":"show_time","params":{}}'
        mock_resp = _mock_stream_response(payload)
        with patch("requests.post", return_value=mock_resp):
            text, is_action = brain.process("que horas são")
        assert is_action is True
        # show_time returns a string with ":"
        assert ":" in text

    def test_process_history_trimmed_on_cancel(self, tmp_path):
        brain = _make_brain(tmp_path)
        initial_len = len(brain.history)
        mock_resp = _mock_stream_response("partial")
        # cancelled immediately
        with patch("requests.post", return_value=mock_resp):
            text, is_action = brain.process(
                "hello", cancelled=lambda: True
            )
        # User message should be rolled back
        assert len(brain.history) == initial_len

    def test_process_connection_error(self, tmp_path):
        import requests as req
        brain = _make_brain(tmp_path)
        with patch("requests.post", side_effect=req.exceptions.ConnectionError):
            text, is_action = brain.process("hello")
        assert is_action is False
        assert "ollama" in text.lower() or "running" in text.lower()

    def test_process_history_not_stores_raw_json(self, tmp_path):
        brain = _make_brain(tmp_path)
        payload = '{"action":"show_time","params":{}}'
        mock_resp = _mock_stream_response(payload)
        with patch("requests.post", return_value=mock_resp):
            brain.process("que horas são")
        # The assistant history entry should be a short summary, not raw JSON
        assistant_entries = [
            h for h in brain.history if h["role"] == "assistant"
        ]
        assert len(assistant_entries) == 1
        assert assistant_entries[0]["content"].startswith("[action:")

    def test_max_history_respected(self, tmp_path):
        """max_history=4 means 4 turns; each turn is 2 raw messages (user+assistant).
        After 5 turns the oldest is trimmed, leaving exactly 4 turns = 8 raw entries."""
        cfg = _make_config(tmp_path, {"max_history": 4})
        ex = Executor(cfg)
        brain = Brain(cfg, ex)

        for i in range(5):
            mock_resp = _mock_stream_response(f"reply {i}")
            with patch("requests.post", return_value=mock_resp):
                brain.process(f"message {i}")

        # 4 turns × 2 messages = 8 raw entries, not 4.
        assert len(brain.history) == 4 * 2
        # The oldest turn (message 0) must have been dropped.
        assert brain.history[0]["content"] == "message 1"

    def test_history_pairs_intact_after_trim(self, tmp_path):
        """After trimming, history must be properly paired (no orphaned user entries).
        This guards against off-by-one errors in the trimming slice."""
        cfg = _make_config(tmp_path, {"max_history": 3})
        ex = Executor(cfg)
        brain = Brain(cfg, ex)

        for i in range(7):
            mock_resp = _mock_stream_response(f"ok {i}")
            with patch("requests.post", return_value=mock_resp):
                brain.process(f"q {i}")

        # History must always start with a user entry and alternate user/assistant.
        assert brain.history[0]["role"] == "user"
        for j in range(0, len(brain.history), 2):
            assert brain.history[j]["role"] == "user"
            assert brain.history[j + 1]["role"] == "assistant"

    def test_timeout_rolls_back_history(self, tmp_path):
        """A Timeout exception must not leave a dangling user entry in history."""
        import requests as req
        brain = _make_brain(tmp_path)
        pre = list(brain.history)
        with patch("requests.post", side_effect=req.exceptions.Timeout):
            text, is_action = brain.process("hello")
        assert is_action is False
        assert "timed out" in text.lower() or "timeout" in text.lower() or "overloaded" in text.lower()
        assert brain.history == pre

    def test_generic_exception_rolls_back_history(self, tmp_path):
        """A generic exception during the HTTP call must not leave a dangling user entry."""
        brain = _make_brain(tmp_path)
        pre = list(brain.history)
        with patch("requests.post", side_effect=RuntimeError("unexpected")):
            text, is_action = brain.process("hello")
        assert is_action is False
        assert "error" in text.lower()
        assert brain.history == pre

    def test_empty_content_rolls_back_history(self, tmp_path):
        """When the model returns empty content, the user entry must be rolled back
        so history never accumulates a dangling user-only turn."""
        brain = _make_brain(tmp_path)
        pre = list(brain.history)
        # Stream that yields only the done sentinel with no content tokens
        done_line = json.dumps({"message": {"content": ""}, "done": True}).encode()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines = MagicMock(return_value=iter([done_line]))
        with patch("requests.post", return_value=mock_resp):
            text, is_action = brain.process("hello")
        assert text == ""
        assert is_action is False
        # History must be identical to before the call — no orphaned entry.
        assert brain.history == pre

    def test_allowed_actions_live_in_system_prompt(self, tmp_path):
        """Brain must use the current allowed_actions from config when building the
        system prompt — changes between calls are reflected without restart.
        Only the AÇÕES DISPONÍVEIS block is checked; the EXAMPLES section
        always contains a fixed set of action names regardless of the filter."""
        brain = _make_brain(tmp_path)
        brain.config.set("allowed_actions", ["show_time"])
        mock_resp = _mock_stream_response("ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            brain.process("hello")
        payload = mock_post.call_args[1]["json"]
        system_msg = payload["messages"][0]["content"]
        # Extract only the AÇÕES DISPONÍVEIS section (between header and EXEMPLOS).
        start = system_msg.find("AÇÕES DISPONÍVEIS:")
        end = system_msg.find("EXEMPLOS", start)
        action_section = system_msg[start:end] if start != -1 and end != -1 else system_msg
        assert "show_time" in action_section
        assert "open_app" not in action_section

    def test_process_uses_live_config_url(self, tmp_path):
        """Brain reads ollama_url from config on each call, not cached at init."""
        brain = _make_brain(tmp_path)
        brain.config.set("ollama_url", "http://custom-host:9999")
        mock_resp = _mock_stream_response("ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            brain.process("hello")
        call_url = mock_post.call_args[0][0]
        assert "custom-host:9999" in call_url

    def test_process_uses_live_config_model(self, tmp_path):
        """Brain reads ollama_model from config on each call."""
        brain = _make_brain(tmp_path)
        brain.config.set("ollama_model", "llama3:8b")
        mock_resp = _mock_stream_response("ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            brain.process("hello")
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "llama3:8b"


# ---------------------------------------------------------------------------
# _build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    def _action_list_section(self, prompt: str) -> str:
        """Extract the AÇÕES DISPONÍVEIS block from the prompt."""
        marker = "AÇÕES DISPONÍVEIS:"
        end_marker = "EXEMPLOS"
        start = prompt.find(marker)
        if start == -1:
            return prompt
        end = prompt.find(end_marker, start)
        return prompt[start:end] if end != -1 else prompt[start:]

    def test_includes_only_allowed_actions(self):
        allowed = ["show_time", "open_app"]
        prompt = _build_system_prompt(allowed)
        section = self._action_list_section(prompt)
        assert "show_time" in section
        assert "open_app" in section
        assert "set_volume" not in section

    def test_empty_allowed_shows_none_message(self):
        prompt = _build_system_prompt([])
        assert "none" in prompt.lower() or "disabled" in prompt.lower()

    def test_all_actions_when_all_allowed(self):
        allowed = list(_ACTION_DOCS.keys())
        prompt = _build_system_prompt(allowed)
        section = self._action_list_section(prompt)
        for action in allowed:
            assert action in section

    def test_unknown_action_ignored(self):
        prompt = _build_system_prompt(["show_time", "nonexistent_action"])
        section = self._action_list_section(prompt)
        assert "show_time" in section
        assert "nonexistent_action" not in section
