import subprocess
import tempfile
import os
import wave
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

from utils.config import Config
from utils.logger import get_logger
from utils.audio_devices import resolve_output

log = get_logger("Speaker")


class Speaker:
    def __init__(self, config: Config):
        self.config = config
        self._lock = threading.Lock()
        self._on_speak_start: Callable | None = None
        self._on_speak_end:   Callable | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reload_config(self):
        """No-op kept for API compatibility.

        All config values are now read live on every speak() call so there
        is no stale cached state to reload.
        """

    def set_speaking_callbacks(
        self,
        on_start: Callable | None = None,
        on_end:   Callable | None = None,
    ) -> None:
        """Register callbacks that fire when TTS starts and finishes.

        Both callbacks are invoked from the speaker's background thread.
        If they emit Qt signals the cross-thread queued-connection mechanism
        ensures the slot runs on the main thread automatically.
        """
        self._on_speak_start = on_start
        self._on_speak_end   = on_end

    def speak(self, text: str):
        """Synthesise and play *text* unless TTS is disabled or text is empty.

        Reads ``tts_enabled`` from config on every call so that toggling TTS
        in the UI takes effect immediately without restarting the app.
        """
        if not self.config.get("tts_enabled", True) or not text:
            return
        thread = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
        thread.start()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resolve_path(self, raw: str) -> str:
        """Resolve *raw* path relative to the project root when not absolute."""
        if os.path.isabs(raw):
            return raw
        project_root = os.path.dirname(os.path.dirname(self.config._path))
        return os.path.normpath(os.path.join(project_root, raw))

    def _speak_blocking(self, text: str):
        # Re-read all TTS config values fresh on each call.
        # This means changes made in the Assistant tab take effect immediately.
        piper_path    = self._resolve_path(self.config.get("piper_path",  "piper/piper/piper.exe"))
        voice_model   = self._resolve_path(self.config.get("voice_model", "en_US-ryan-high.onnx"))
        _out_result   = resolve_output(self.config)
        output_device = _out_result.device_index
        if _out_result.status == "wrong_direction":
            log.warning(f"output_device config: {_out_result.message}")
        elif _out_result.status == "missing":
            log.warning(f"Output device: {_out_result.message}")

        try:
            with self._lock:
                wav_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        wav_path = f.name

                    proc = subprocess.run(
                        [piper_path, "--model", voice_model, "--output_file", wav_path],
                        input=text.encode("utf-8"),
                        capture_output=True,
                        timeout=15,
                    )

                    if proc.returncode != 0:
                        stderr_msg = proc.stderr.decode("utf-8", errors="replace").strip()
                        if stderr_msg:
                            log.error(f"Piper error (exit {proc.returncode}): {stderr_msg}")
                        else:
                            log.error(f"Piper exited with code {proc.returncode} and no stderr output.")
                        return

                    if not os.path.exists(wav_path):
                        log.error("Piper exited successfully but produced no output file.")
                        return

                    # Piper succeeded — notify that real audio playback is about to begin.
                    # on_speak_start fires here, not earlier, so the app never reports
                    # "speaking" when Piper has already failed before producing any audio.
                    if self._on_speak_start:
                        try:
                            self._on_speak_start()
                        except Exception:
                            pass
                    self._play_wav(wav_path, output_device)

                except FileNotFoundError:
                    log.error("Piper not found. Check piper_path in settings.yaml")
                except subprocess.TimeoutExpired:
                    log.error("Piper timed out after 15 s.")
                except Exception as e:
                    log.error(f"TTS error: {e}")
                finally:
                    if wav_path and os.path.exists(wav_path):
                        os.unlink(wav_path)
        finally:
            # on_speak_end always fires — ensures the app returns to idle even
            # when Piper fails before on_speak_start was ever called.
            if self._on_speak_end:
                try:
                    self._on_speak_end()
                except Exception:
                    pass

    def _play_wav(self, path: str, output_device):
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            framerate  = wf.getframerate()
            raw        = wf.readframes(wf.getnframes())

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sampwidth, np.int16)
        audio = np.frombuffer(raw, dtype=dtype)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)

        audio = audio.astype(np.float32) / np.iinfo(dtype).max

        # target_rate tracks the actual sample rate of the audio array so that
        # both primary playback and the fallback path use the correct rate.
        target_rate = framerate

        if output_device is not None:
            try:
                dev_rate = int(sd.query_devices(output_device)["default_samplerate"])
                if dev_rate != framerate:
                    n_out = int(len(audio) * dev_rate / framerate)
                    audio = np.interp(
                        np.linspace(0, len(audio), n_out),
                        np.arange(len(audio)),
                        audio if audio.ndim == 1 else audio[:, 0],
                    ).astype(np.float32)
                    target_rate = dev_rate
            except Exception:
                pass

        try:
            sd.play(audio, samplerate=target_rate, device=output_device)
            sd.wait()
        except Exception as e:
            log.warning(f"Output device failed ({e}), falling back to default")
            # Use target_rate (the actual sample rate of the audio array) rather
            # than the original framerate — if the audio was already resampled,
            # playing it at framerate would produce wrong-pitch audio.
            sd.play(audio, samplerate=target_rate, device=None)
            sd.wait()
