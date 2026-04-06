import re
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal
from faster_whisper import WhisperModel

from utils.config import Config
from utils.logger import get_logger
from utils.audio_devices import resolve_input, enumerate_devices, filter_inputs
from audio_utils import (
    wake_word_in_text,
    extract_post_wake,
    compute_rms,
    normalize_level,
    has_sufficient_energy,
    update_noise_floor_gated,
)

log = get_logger("Listener")


def _extract_post_wake(text: str, wake_word: str) -> str:
    """Return text that follows the wake word in a transcription string.

    Preserves original casing of the tail.  Falls back to an accent-
    normalised search when the direct pattern fails (e.g. "vóx").
    """
    pattern = rf'\b{re.escape(wake_word)}\b'
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return text[match.end():].strip()

    # Accent-normalised fallback: find the wake word in the normalised form
    # and map the end position back to the original string approximately.
    from audio_utils import normalize_text
    norm_text = normalize_text(text)
    norm_wake = normalize_text(wake_word)
    norm_pattern = rf'\b{re.escape(norm_wake)}\b'
    norm_match = re.search(norm_pattern, norm_text)
    if norm_match:
        # The normalised text is the same length or shorter due to accent
        # stripping.  Use min() so we never exceed the original length.
        end = min(norm_match.end(), len(text))
        return text[end:].strip()
    return ""


class Listener(QThread):
    transcription_ready = pyqtSignal(str)
    language_detected   = pyqtSignal(str)
    listening_started   = pyqtSignal()
    listening_stopped   = pyqtSignal()
    monitoring_started  = pyqtSignal()          # emitted when wake-word loop is active
    mic_level           = pyqtSignal(float)     # 0.0–1.0 normalised RMS
    capture_warning     = pyqtSignal(str, str)  # (level, message)

    SAMPLE_RATE = 16000

    # ── Detection window (audio fed to Whisper for wake-word check) ───────────
    # Increased from 1.0 s → 1.5 s so that wake words spoken slowly or near
    # the edge of a window are still captured.
    _DETECT_WINDOW_S  = 1.5

    # ── Stride between consecutive detection checks ────────────────────────
    # With stride < window we get overlapping checks (sliding window).
    # stride=1.0s + window=1.5s → 33% overlap, same Whisper call rate as before.
    _DETECT_STRIDE_S  = 1.0

    # ── Rolling pre-buffer kept before the wake word ──────────────────────────
    # Must be >= _DETECT_WINDOW_S so the seed contains the full detection
    # window when the command capture phase begins.
    _PRE_BUFFER_S     = 2.0

    # Read granularity from the stream
    _READ_CHUNK_S     = 0.1

    # Noise floor EMA alpha for update_noise_floor_gated.
    # Slightly higher (slower) than before so the floor adapts less eagerly
    # when transient sounds appear.
    _NF_ALPHA         = 0.98

    # Number of consecutive empty/near-zero captures before emitting a warning.
    # Raised from 5 → 8 to reduce false "is microphone working?" warnings in
    # environments with variable silence.
    _EMPTY_WARN_AFTER = 8

    def __init__(self, config: Config, model: WhisperModel):
        super().__init__()
        self.config     = config
        self.model      = model
        self._stop_flag = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def stop_listener(self):
        self._stop_flag.set()

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        self._stop_flag.clear()
        self._print_devices()
        self._validate_mic_device()
        activation_mode = self.config.get("activation_mode", "wake_word")
        if activation_mode == "push_to_talk":
            ptt_key = self.config.get("push_to_talk_key", "ctrl+shift")
            log.info(f"Push-to-talk mode active. Press '{ptt_key}' to activate.")
            self._run_push_to_talk_loop()
        else:
            wake_word = self.config.get("wake_word", "vox").lower().strip()
            log.info(f"Wake word mode active. Say '{wake_word}' to activate.")
            self._run_wake_word_loop()

    def _validate_mic_device(self):
        try:
            result = resolve_input(self.config)
            if result.status == "wrong_direction":
                log.error(f"Microphone config error: {result.message}")
            elif result.status == "missing":
                log.warning(f"Microphone: {result.message}")
            elif result.status == "legacy_int":
                log.info(f"Microphone (legacy index): {result.message}")
            elif result.status == "resolved":
                log.info(f"Microphone resolved: '{result.device_name}' (index {result.device_index})")
            else:  # system_default
                log.info(f"Microphone: system default — '{result.device_name}'")
        except Exception as e:
            log.warning(f"Could not validate mic_device: {e}")

    # ── Device helpers ────────────────────────────────────────────────────────

    def _get_device_channels(self, mic_device) -> int:
        """Return the number of input channels supported by the device (min 1)."""
        try:
            info = sd.query_devices(mic_device, "input") if mic_device is not None else sd.query_devices(sd.default.device[0])
            return max(1, int(info["max_input_channels"]))
        except Exception:
            return 1

    # ── Wake word loop (continuous InputStream) ───────────────────────────────

    def _run_wake_word_loop(self):
        """Continuously monitor the microphone using a rolling buffer.

        Architecture:
        - One sd.InputStream per outer-loop iteration (restarted on error)
        - Rolling pre-buffer so the start of the command is preserved
        - Sliding detection window (stride < window) fed to Whisper
        - Gated noise-floor EMA: floor only updates during silence so
          speech does not raise the threshold and cause future speech to
          be silently rejected
        - Energy pre-screening: skip Whisper on silent windows
        """
        _backoff      = 0.0
        _noise_floor  = 0.0
        _empty_count  = 0

        while not self._stop_flag.is_set():
            _mic_result     = resolve_input(self.config)
            mic_device      = _mic_result.device_index
            if _mic_result.status == "wrong_direction":
                log.error(f"Microphone config error: {_mic_result.message}")
            elif _mic_result.status == "missing":
                log.warning(f"Microphone: {_mic_result.message}")
            wake_word       = self.config.get("wake_word", "vox").lower().strip()
            channels        = self._get_device_channels(mic_device)
            read_samples    = int(self.SAMPLE_RATE * self._READ_CHUNK_S)
            window_samples  = int(self.SAMPLE_RATE * self._DETECT_WINDOW_S)
            stride_samples  = int(self.SAMPLE_RATE * self._DETECT_STRIDE_S)
            pre_buf_samples = int(self.SAMPLE_RATE * self._PRE_BUFFER_S)

            # Rolling pre-buffer — keeps the last _PRE_BUFFER_S seconds of audio
            # and also serves as the detection window.
            pre_buffer  = deque(maxlen=pre_buf_samples)
            # Stride counter: how many new samples since the last Whisper check.
            acc_samples = 0

            stream = None
            try:
                stream = sd.InputStream(
                    samplerate=self.SAMPLE_RATE,
                    channels=channels,
                    dtype="float32",
                    device=mic_device,
                )
                stream.start()
                self.monitoring_started.emit()
                log.info("[wake] monitoring stream started")
                _backoff = 0.0

                while not self._stop_flag.is_set():
                    data, overflowed = stream.read(read_samples)
                    if overflowed:
                        log.debug("[wake] stream overflow — some audio dropped")

                    mono = data.mean(axis=1) if data.ndim > 1 and data.shape[1] > 1 else data.flatten()

                    rms = compute_rms(mono)
                    self.mic_level.emit(normalize_level(rms))

                    # Gated noise-floor update: only during silence so that
                    # speech does not raise the floor and tighten the energy gate.
                    _noise_floor = update_noise_floor_gated(
                        _noise_floor, rms, alpha=self._NF_ALPHA
                    )

                    # Extend pre_buffer efficiently (one deque.extend vs N appends)
                    pre_buffer.extend(mono)
                    acc_samples += len(mono)

                    # Wait until we have accumulated a full stride of new audio
                    if acc_samples < stride_samples:
                        continue

                    acc_samples = 0  # reset stride counter (pre_buffer keeps rolling)

                    # Need at least window_samples in the pre_buffer before running
                    if len(pre_buffer) < window_samples:
                        continue

                    # Build detection window from the tail of pre_buffer
                    window = np.array(list(pre_buffer), dtype=np.float32)[-window_samples:]

                    # Skip Whisper on silent windows
                    if not has_sufficient_energy(window, _noise_floor):
                        _empty_count += 1
                        if _empty_count >= self._EMPTY_WARN_AFTER:
                            self.capture_warning.emit(
                                "info",
                                f"[wake] {_empty_count} consecutive silent windows — "
                                "is the microphone working?",
                            )
                            _empty_count = 0
                        continue

                    _empty_count = 0

                    # Wake word detection — fast (beam_size=1)
                    lang_cfg   = self.config.get("language", "auto")
                    lang_param = None if lang_cfg == "auto" else lang_cfg
                    segments, _ = self.model.transcribe(
                        window, language=lang_param, beam_size=1
                    )
                    text = " ".join(s.text.strip() for s in segments).strip()
                    text_lower = text.lower()
                    if text_lower:
                        log.debug(f"[wake] heard: '{text_lower}'")

                    if not wake_word_in_text(text_lower, wake_word):
                        continue

                    # ── Wake word detected ──────────────────────────────────
                    tail_text = _extract_post_wake(text, wake_word)
                    if len(tail_text) <= 1:
                        tail_text = ""

                    log.info("Wake word detected — listening for command…")
                    self.listening_started.emit()

                    # Seed command capture with the full pre-buffer so that the
                    # beginning of the command (spoken right after the wake word)
                    # is preserved even if it overlaps the detection window.
                    seed = np.array(list(pre_buffer), dtype=np.float32)
                    command_audio = self._record_command_from_stream(
                        stream, seed, noise_floor=_noise_floor
                    )
                    self.listening_stopped.emit()

                    if command_audio is not None:
                        self._transcribe_and_emit(command_audio, fallback_text=tail_text)
                    elif tail_text:
                        log.info(f"Using wake-word chunk tail as command: '{tail_text}'")
                        self.transcription_ready.emit(tail_text)

                    # Restart the monitoring stream after handling a command
                    break

            except sd.PortAudioError as e:
                log.error(f"Audio device error: {e}")
                self.capture_warning.emit("error", f"Audio device error: {e}")
                _backoff = min(_backoff + 1.0, 5.0)
                time.sleep(_backoff)
            except Exception as e:
                log.error(f"Wake word loop error: {e}")
                _backoff = min(_backoff + 1.0, 5.0)
                if _backoff > 0:
                    log.info(f"Backing off {_backoff:.0f}s before retrying…")
                time.sleep(_backoff)
            finally:
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass

    # ── Command capture from existing stream ──────────────────────────────────

    def _record_command_from_stream(
        self,
        stream: sd.InputStream,
        seed: np.ndarray,
        noise_floor: float = 0.0,
    ) -> np.ndarray | None:
        """Capture a command on *stream*, prepending pre-buffer *seed*.

        Recording stops when silence_duration of silence is observed or
        max_record_duration is reached.

        *noise_floor* is used to compute an adaptive effective silence
        threshold: ``max(configured_threshold, noise_floor × 1.5)``.
        This prevents the recording from running indefinitely in environments
        where ambient noise exceeds the configured threshold.
        """
        all_chunks         = [seed] if seed is not None and seed.size > 0 else []
        consecutive_silent = 0
        chunk_size         = self.SAMPLE_RATE // 2
        max_samples        = int(self.SAMPLE_RATE * float(self.config.get("max_record_duration", 30)))
        min_samples        = int(self.SAMPLE_RATE * float(self.config.get("min_listen_duration", 1.0)))
        silence_duration   = float(self.config.get("silence_duration", 2.0))
        silence_samples    = int(self.SAMPLE_RATE * silence_duration)
        configured_thresh  = float(self.config.get("silence_threshold", 0.02))

        # Adaptive threshold: if the ambient noise floor is above the
        # configured threshold, use a slightly higher value so that silence
        # detection still works in noisier environments.
        silence_threshold  = max(configured_thresh, noise_floor * 1.5) if noise_floor > 0.0 else configured_thresh

        total_samples      = seed.size if seed is not None else 0

        while not self._stop_flag.is_set():
            data, _ = stream.read(chunk_size)
            mono = data.mean(axis=1) if data.ndim > 1 and data.shape[1] > 1 else data.flatten()
            all_chunks.append(mono.copy())
            total_samples += len(mono)

            rms = compute_rms(mono)
            self.mic_level.emit(normalize_level(rms))

            if total_samples >= min_samples and rms < silence_threshold:
                consecutive_silent += len(mono)
                if consecutive_silent >= silence_samples:
                    break
            else:
                consecutive_silent = 0

            if total_samples >= max_samples:
                log.warning("Max recording duration reached — stopping capture.")
                break

        if all_chunks:
            audio = np.concatenate(all_chunks, axis=0)
            return audio
        return None

    # ── Push-to-talk loop ─────────────────────────────────────────────────────

    def _run_push_to_talk_loop(self):
        try:
            import keyboard
        except ImportError:
            log.error("push_to_talk mode requires the 'keyboard' package. Install it with: pip install keyboard")
            return

        _backoff = 0.0

        while not self._stop_flag.is_set():
            try:
                _mic_result      = resolve_input(self.config)
                mic_device       = _mic_result.device_index
                if _mic_result.status == "wrong_direction":
                    log.error(f"Microphone config error: {_mic_result.message}")
                elif _mic_result.status == "missing":
                    log.warning(f"Microphone: {_mic_result.message}")
                ptt_key          = self.config.get("push_to_talk_key", "ctrl+shift")
                silence_duration = float(self.config.get("silence_duration", 2.0))
                silence_samples  = int(self.SAMPLE_RATE * silence_duration)
                keys = [k.strip() for k in ptt_key.split("+")]

                # Wait for the PTT hotkey using a short-timeout event loop so that
                # stop_listener() can interrupt this wait cleanly.  keyboard.wait()
                # blocks indefinitely and cannot be interrupted by a threading.Event,
                # which caused stop_listener() to time out and leave a ghost listener
                # thread alive during listener restarts.
                _ptt_event = threading.Event()
                _hook = None
                try:
                    _hook = keyboard.add_hotkey(ptt_key, _ptt_event.set, suppress=False)
                    while not self._stop_flag.is_set():
                        if _ptt_event.wait(timeout=0.1):
                            break
                finally:
                    if _hook is not None:
                        try:
                            keyboard.remove_hotkey(_hook)
                        except Exception:
                            pass

                if self._stop_flag.is_set():
                    break

                log.info("Push-to-talk activated — listening…")
                self.listening_started.emit()

                command_audio = self._record_push_to_talk(mic_device, silence_samples, keyboard, keys)
                self.listening_stopped.emit()

                if command_audio is not None:
                    self._transcribe_and_emit(command_audio)

                # Wait for all PTT keys to be released before next cycle.
                # Use a polling approach (same stop-flag awareness) instead of
                # keyboard.wait() to stay interruptible.
                while not self._stop_flag.is_set():
                    if not any(keyboard.is_pressed(k) for k in keys):
                        break
                    time.sleep(0.05)

                _backoff = 0.0

            except Exception as e:
                log.error(f"Push-to-talk loop error: {e}")
                _backoff = min(_backoff + 1.0, 5.0)
                if _backoff > 0:
                    log.info(f"Backing off {_backoff:.0f}s before retrying…")
                time.sleep(_backoff)

    def _record_push_to_talk(self, mic_device, silence_samples: int, keyboard, keys: list[str]):
        """Record while keys are held; stop on silence or key release."""
        all_chunks        = []
        consecutive_silent = 0
        chunk_size         = self.SAMPLE_RATE // 2
        max_samples        = int(self.SAMPLE_RATE * float(self.config.get("max_record_duration", 30)))
        total_samples      = 0
        silence_threshold  = float(self.config.get("silence_threshold", 0.02))
        channels           = self._get_device_channels(mic_device)

        stream = sd.InputStream(samplerate=self.SAMPLE_RATE, channels=channels, dtype="float32", device=mic_device)
        stream.start()
        try:
            while True:
                data, _ = stream.read(chunk_size)
                all_chunks.append(data.copy())
                total_samples += chunk_size

                if not all(keyboard.is_pressed(k) for k in keys):
                    break

                rms = compute_rms(data.mean(axis=1) if data.ndim > 1 and data.shape[1] > 1 else data.flatten())
                self.mic_level.emit(normalize_level(rms))
                if rms < silence_threshold:
                    consecutive_silent += chunk_size
                    if consecutive_silent >= silence_samples:
                        break
                else:
                    consecutive_silent = 0

                if total_samples >= max_samples:
                    log.warning("Max recording duration reached — stopping capture.")
                    break
        finally:
            stream.stop()
            stream.close()

        if all_chunks:
            audio = np.concatenate(all_chunks, axis=0)
            return audio.mean(axis=1) if audio.ndim > 1 and audio.shape[1] > 1 else audio.flatten()
        return None

    # ── Shared recording helpers ──────────────────────────────────────────────

    def _record_until_silence(self, mic_device, silence_samples: int):
        """Record audio until silence_samples consecutive silent samples.

        Kept for push-to-talk compatibility; wake word path now uses
        _record_command_from_stream instead.
        """
        all_chunks         = []
        consecutive_silent = 0
        chunk_size         = self.SAMPLE_RATE // 2
        max_samples        = int(self.SAMPLE_RATE * float(self.config.get("max_record_duration", 30)))
        min_samples        = int(self.SAMPLE_RATE * float(self.config.get("min_listen_duration", 1.0)))
        total_samples      = 0
        silence_threshold  = float(self.config.get("silence_threshold", 0.02))
        channels           = self._get_device_channels(mic_device)

        stream = sd.InputStream(samplerate=self.SAMPLE_RATE, channels=channels, dtype="float32", device=mic_device)
        stream.start()
        try:
            while True:
                data, _ = stream.read(chunk_size)
                all_chunks.append(data.copy())
                total_samples += chunk_size
                rms = compute_rms(data.mean(axis=1) if data.ndim > 1 and data.shape[1] > 1 else data.flatten())
                self.mic_level.emit(normalize_level(rms))
                if total_samples >= min_samples and rms < silence_threshold:
                    consecutive_silent += chunk_size
                    if consecutive_silent >= silence_samples:
                        break
                else:
                    consecutive_silent = 0
                if total_samples >= max_samples:
                    log.warning("Max recording duration reached — stopping capture.")
                    break
        finally:
            stream.stop()
            stream.close()

        if all_chunks:
            audio = np.concatenate(all_chunks, axis=0)
            return audio.mean(axis=1) if audio.ndim > 1 and audio.shape[1] > 1 else audio.flatten()
        return None

    def _transcribe_and_emit(self, audio: np.ndarray, fallback_text: str = ""):
        lang_cfg   = self.config.get("language", "auto")
        lang_param = None if lang_cfg == "auto" else lang_cfg
        segments, info = self.model.transcribe(audio, language=lang_param, beam_size=5)
        detected = getattr(info, "language", lang_cfg) or "auto"
        self.language_detected.emit(detected)
        text = " ".join(s.text.strip() for s in segments).strip()

        wake_word = self.config.get("wake_word", "vox").lower().strip()
        # Strip leading wake word occurrence (allow trailing punctuation like comma)
        text = re.sub(
            rf"^\s*{re.escape(wake_word)}[,.\s]*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        if not text and fallback_text:
            text = fallback_text
            log.info(f"Command audio empty — using wake-word chunk tail: '{text}'")

        if text:
            self.transcription_ready.emit(text)
        else:
            log.info("Empty command after wake word strip — ignoring.")

    # ── Device listing ────────────────────────────────────────────────────────

    def _print_devices(self):
        try:
            all_devs  = enumerate_devices()
            inputs    = filter_inputs(all_devs)
            mic_result = resolve_input(self.config)
            log.info("Available microphones:")
            for d in inputs:
                if d.index == mic_result.device_index:
                    marker = "→ ACTIVE  "
                elif d.is_default_input and mic_result.device_index is None:
                    marker = "→ DEFAULT "
                else:
                    marker = "          "
                log.info(f"  [{d.index:2d}] {marker}  {d.display_label}")
            if mic_result.status in ("wrong_direction", "missing"):
                log.warning(f"  Microphone resolution: {mic_result.message}")
            elif mic_result.status == "system_default":
                log.info(f"  Using system default microphone: '{mic_result.device_name}'")
            log.info("To change mic, open the Audio tab in the Control Center and save.")
        except Exception as e:
            log.warning(f"Could not enumerate audio devices: {e}")
