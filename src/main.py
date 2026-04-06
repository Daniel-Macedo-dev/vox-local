import sys
import os
import traceback
import requests as _requests

from utils.config import Config
from utils.logger import get_logger

log = get_logger("VOX")


# ── Pre-flight validation ──────────────────────────────────────────────────────

def _validate_startup(config: Config, app_state=None):
    """Check dependencies and emit results.

    Always logs to console; when *app_state* is provided, structured entries
    are also added to the diagnostics panel.
    """

    def _info(msg: str):
        log.info(msg)
        if app_state:
            app_state.add_diagnostic("info", msg)

    def _warn(msg: str):
        log.warning(msg)
        if app_state:
            app_state.add_diagnostic("warning", msg)

    # 1. Ollama reachability
    try:
        r = _requests.get(
            f"{config.get('ollama_url', 'http://localhost:11434')}/api/tags",
            timeout=3,
        )
        if r.status_code == 200:
            _info("Ollama is reachable.")
            if app_state:
                app_state.set_ollama_ok(True)
        else:
            _warn(f"Ollama responded with status {r.status_code}. LLM may not work.")
    except Exception as e:
        _warn(f"Ollama is not reachable ({e}). Start it with: ollama serve")

    # 2. Piper binary
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    piper_raw  = config.get("piper_path", "piper/piper/piper.exe")
    piper_path = (piper_raw if os.path.isabs(piper_raw)
                  else os.path.normpath(os.path.join(project_root, piper_raw)))
    if not os.path.exists(piper_path):
        _warn(f"Piper binary not found at '{piper_path}'. TTS will be silent.")
    else:
        _info(f"Piper found: {piper_path}")

    # 3. Voice model
    voice_raw  = config.get("voice_model", "voices/en_US-ryan-high.onnx")
    voice_path = (voice_raw if os.path.isabs(voice_raw)
                  else os.path.normpath(os.path.join(project_root, voice_raw)))
    if not os.path.exists(voice_path):
        _warn(f"Voice model not found at '{voice_path}'. TTS will be silent.")
    else:
        _info(f"Voice model found: {os.path.basename(voice_path)}")

    # 4. Audio device indices
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        n = len(devices)
        mic_idx = config.get("mic_device",    None)
        out_idx = config.get("output_device", None)
        if mic_idx is not None and (not isinstance(mic_idx, int) or mic_idx >= n):
            _warn(f"mic_device={mic_idx} is not a valid device index. Using system default.")
        if out_idx is not None and (not isinstance(out_idx, int) or out_idx >= n):
            _warn(f"output_device={out_idx} is not a valid device index. Using system default.")
    except Exception as e:
        _warn(f"Could not validate audio devices: {e}")


def load_whisper(config: Config):
    from faster_whisper import WhisperModel
    device       = config.get("whisper_device",       "cpu")
    compute_type = config.get("whisper_compute_type", "int8")
    log.info(f"Loading Whisper ({config.get('whisper_model', 'base')}, {device}/{compute_type})…")
    model = WhisperModel(config.get("whisper_model", "base"),
                         device=device, compute_type=compute_type)
    log.info("Whisper ready.")
    return model


# ── Application ────────────────────────────────────────────────────────────────

def run_app(config: Config, whisper_model):
    from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
    from PyQt6.QtGui import QIcon
    from PyQt6.QtCore import QThread, QTimer, pyqtSignal

    from listener import Listener
    from brain    import Brain
    from speaker  import Speaker
    from executor import Executor
    from app_state import AppState
    from ui.overlay import OverlayWindow
    from ui.control_center import ControlCenter

    # ── Brain worker ──────────────────────────────────────────────────────────

    class BrainWorker(QThread):
        response_ready     = pyqtSignal(str, bool)
        token_received     = pyqtSignal(str)
        generating_started = pyqtSignal()

        def __init__(self, brain, text: str):
            super().__init__()
            self._brain     = brain
            self._text      = text
            self._cancelled = False

        def cancel(self):
            self._cancelled = True

        def run(self):
            try:
                text, is_action = self._brain.process(
                    self._text,
                    on_token=self._emit_token,
                    on_generating=self.generating_started.emit,
                    cancelled=lambda: self._cancelled,
                )
            except Exception as e:
                traceback.print_exc()
                text, is_action = f"Internal error: {e}", False
            if not self._cancelled:
                self.response_ready.emit(text, is_action)

        def _emit_token(self, token: str):
            if not self._cancelled:
                self.token_received.emit(token)

    # ── Main application ──────────────────────────────────────────────────────

    class VoxApp:
        def __init__(self):
            self.app = QApplication(sys.argv)
            self.app.setQuitOnLastWindowClosed(False)

            # ── Core components
            self.state    = AppState(config)
            self.speaker  = Speaker(config)
            self.executor = Executor(config)
            self.brain    = Brain(config, self.executor)
            self.listener = Listener(config, whisper_model)

            # ── UI
            self.overlay        = OverlayWindow()
            self.control_center = ControlCenter(
                config, self.state, self.speaker,
                stt_cb=self._stt_test_transcribe,
                executor=self.executor,
            )
            self._brain_worker     = None
            # Monotonically increasing counter — bumped on every new transcription
            # and on timeout so that any response_ready still in the Qt event
            # queue from a superseded worker is recognised as stale and discarded.
            self._brain_generation = 0

            self._connect_signals()
            self._setup_tray()
            self._apply_activation_mode_ui()

            # Run validation now that AppState exists (routes issues to Diagnostics)
            _validate_startup(config, self.state)

        # ── Signal wiring ─────────────────────────────────────────────────────

        def _connect_listener_signals(self):
            # Listener status transitions drive AppState only; the overlay
            # subscribes to state.status_changed via _on_status_changed.
            self.listener.listening_started.connect(
                lambda: self.state.set_status("listening"))
            self.listener.listening_stopped.connect(
                lambda: self.state.set_status("transcribing"))
            self.listener.monitoring_started.connect(
                lambda: self.state.set_status("monitoring"))

            self.listener.capture_warning.connect(
                lambda level, msg: self.state.add_diagnostic(level, msg))

            self.listener.transcription_ready.connect(self.on_transcription)
            # Detected language flows through AppState so all subscribers update
            # from one emission (overlay badge + Dashboard).
            self.listener.language_detected.connect(
                self.state.detected_language_changed.emit)

            # Real mic level → overlay waveform (direct — high-frequency) and AppState
            self.listener.mic_level.connect(self.overlay.set_mic_level)
            self.listener.mic_level.connect(self.state.set_mic_level)

        def _connect_signals(self):
            self._connect_listener_signals()
            self.overlay.language_clicked.connect(self.on_language_cycle)

            # ── AppState → overlay ────────────────────────────────────────────
            # All status-driven and content-driven overlay updates route through
            # AppState signals.  The overlay subscribes here so main.py no longer
            # needs to call overlay methods directly for every state transition.
            self.state.status_changed.connect(self._on_status_changed)
            self.state.transcript_changed.connect(self.overlay.set_transcript)
            self.state.response_changed.connect(self.overlay.set_response)
            self.state.last_action_changed.connect(self.overlay.set_action)
            self.state.language_mode_changed.connect(self.overlay.set_language_mode)
            self.state.detected_language_changed.connect(
                self.overlay.show_detected_language)
            self.state.ollama_ok_changed.connect(self.overlay.set_ollama_ok)

            # Initialise language badge from current config
            self.state.language_mode_changed.emit(config.get("language", "auto"))

            # Control Center actions
            self.control_center.restart_listener_requested.connect(self._restart_listener)
            self.control_center.rerun_validation_requested.connect(self._rerun_validation)
            # Language changes from the Assistant tab flow through AppState so
            # overlay badge and Dashboard both update from one emission.
            self.control_center.language_changed.connect(
                self.state.language_mode_changed.emit)

            # Speaking lifecycle — callbacks fire from Speaker's background thread.
            # We route through AppState signals so Qt queues the slot calls on the
            # main thread automatically (cross-thread queued-connection safety).
            self.speaker.set_speaking_callbacks(
                on_start=self.state.speaking_started.emit,
                on_end=self.state.speaking_ended.emit,
            )
            self.state.speaking_started.connect(lambda: self.state.set_status("speaking"))
            self.state.speaking_ended.connect(lambda: self.state.set_status("idle"))

            # Ollama ping timer
            self._ollama_timer = QTimer()
            self._ollama_timer.timeout.connect(self._ping_ollama)
            self._ollama_timer.start(12000)
            self._ping_ollama()

        def _apply_activation_mode_ui(self):
            mode = config.get("activation_mode", "wake_word")
            key  = config.get("push_to_talk_key", "ctrl+shift")
            self.overlay.set_footer_mode_with_key(mode, key)

        def _on_status_changed(self, status: str):
            """Route AppState status strings to the corresponding overlay visual state.

            This is the single place that maps canonical status names to overlay
            method calls, replacing the scattered parallel overlay + state writes
            that previously appeared throughout the application code.
            """
            _map = {
                "monitoring":   self.overlay.set_monitoring,
                "listening":    self.overlay.set_listening,
                "transcribing": self.overlay.set_processing,
                "generating":   self.overlay.set_generating,
                "speaking":     self.overlay.set_speaking,
                "idle":         self.overlay.set_idle,
            }
            fn = _map.get(status)
            if fn:
                fn()

        # ── System tray ───────────────────────────────────────────────────────

        def _setup_tray(self):
            icon_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "assets", "icon.ico",
            )
            icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
            tray = QSystemTrayIcon(icon, self.app)
            tray.setToolTip("VOX")

            menu = QMenu()
            menu.addAction("Control Center", self.control_center.show)
            menu.addAction("Show Overlay",   self.overlay.show)
            menu.addSeparator()
            menu.addAction("Settings",       lambda: self.control_center.show_section("Settings"))
            menu.addSeparator()
            menu.addAction("Quit",           self.app.quit)

            tray.setContextMenu(menu)
            tray.activated.connect(
                lambda reason: self.control_center.show()
                if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None
            )
            tray.show()

        # ── Listener lifecycle ────────────────────────────────────────────────

        def _restart_listener(self):
            self.listener.stop_listener()
            if not self.listener.wait(3000):
                log.warning("Listener did not stop cleanly within 3 s — proceeding anyway.")
            self.listener.start()
            # Refresh overlay footer and language badge to reflect whatever config
            # is now active (activation mode, PTT key, language).
            self._apply_activation_mode_ui()
            self.state.language_mode_changed.emit(config.get("language", "auto"))
            log.info("Listener restarted.")
            self.state.add_diagnostic("info", "Listener restarted.")

        # ── Validation ───────────────────────────────────────────────────────

        def _rerun_validation(self):
            log.info("Re-running startup validation…")
            self.state.add_diagnostic("info", "Running validation…")
            _validate_startup(config, self.state)

        # ── Ollama ping ───────────────────────────────────────────────────────

        def _ping_ollama(self):
            try:
                r = _requests.get(
                    f"{config.get('ollama_url', 'http://localhost:11434')}/api/tags",
                    timeout=2,
                )
                ok = r.status_code == 200
            except Exception:
                ok = False
            if not ok:
                log.warning("Ollama ping failed — LLM unavailable.")
            # overlay updates via state.ollama_ok_changed → overlay.set_ollama_ok
            self.state.set_ollama_ok(ok)

        # ── Language cycling ──────────────────────────────────────────────────

        def on_language_cycle(self):
            order   = ["auto", "pt", "en"]
            current = config.get("language", "auto")
            nxt     = order[(order.index(current) + 1) % len(order)] if current in order else "auto"
            config.set("language", nxt)
            config.save()
            self.state.language_mode_changed.emit(nxt)  # overlay updates via signal
            log.info(f"Language switched to: {nxt}")

        # ── STT test callback (called from AudioTab worker thread) ───────────

        def _stt_test_transcribe(self, audio) -> str:
            """Transcribe *audio* with Whisper and return the transcript string.

            Called from a background thread in AudioTab — must not touch Qt.
            """
            lang_cfg   = config.get("language", "auto")
            lang_param = None if lang_cfg == "auto" else lang_cfg
            segments, _ = whisper_model.transcribe(audio, language=lang_param, beam_size=5)
            return " ".join(s.text.strip() for s in segments).strip()

        # ── Transcription ─────────────────────────────────────────────────────

        def on_transcription(self, text: str):
            # Stop the previous timeout before doing anything else.  Without this,
            # an old 35 s timer firing after the new worker starts would call
            # _on_brain_timeout with self._brain_worker pointing to the *new* worker
            # and cancel it prematurely.
            if hasattr(self, "_brain_timeout"):
                self._brain_timeout.stop()

            if self._brain_worker and self._brain_worker.isRunning():
                log.info("New transcription — cancelling running worker.")
                self._brain_worker.cancel()
                # Wait briefly for cooperative cancellation.  The cancel flag is
                # checked on every stream chunk so this almost always resolves in
                # well under 500 ms.  A full 2 s wait was blocking the Qt event
                # loop unnecessarily and making barge-in feel unresponsive.
                self._brain_worker.wait(500)
                # "cancelled" is a transient visual — not routed through status_changed
                # because it is immediately followed by "generating" and should not
                # trigger the status dispatcher's overlay.set_generating call twice.
                self.overlay.set_cancelled()
                self.state.add_diagnostic("info", "Generation cancelled — new command received.")

            # Transcript and status update: overlay subscribes via state signals.
            self.state.set_transcript(text)    # → overlay.set_transcript via transcript_changed
            self.state.set_status("generating")

            # Advance the generation counter.  The lambda below captures the
            # current value so that any response_ready already in the Qt event
            # queue from the previous worker (which may have completed before
            # cancel() was called) is silently discarded rather than processed
            # as if it were the answer to the new transcription.
            self._brain_generation += 1
            _gen = self._brain_generation

            self._brain_worker = BrainWorker(self.brain, text)
            self._brain_worker.token_received.connect(self.overlay.append_token)
            self._brain_worker.generating_started.connect(
                lambda: self.state.set_status("generating"))
            self._brain_worker.response_ready.connect(
                lambda resp, is_act, g=_gen:
                    self.on_response(resp, is_act) if g == self._brain_generation
                    else log.info(
                        f"Discarding stale brain response "
                        f"(gen {g}, current {self._brain_generation})."
                    )
            )
            self._brain_worker.start()

            self._brain_timeout = QTimer()
            self._brain_timeout.setSingleShot(True)
            self._brain_timeout.timeout.connect(self._on_brain_timeout)
            self._brain_timeout.start(35000)

        def _on_brain_timeout(self):
            if self._brain_worker and self._brain_worker.isRunning():
                log.error("Brain worker timed out — cancelling cooperatively.")
                self._brain_worker.cancel()
                if not self._brain_worker.wait(2000):
                    # Ollama may be hanging mid-chunk and unresponsive to the cancel
                    # flag (the flag is only checked between received lines).
                    # Terminate as a last resort to unblock the thread.
                    log.warning("Brain worker did not stop after cancel — terminating.")
                    self._brain_worker.terminate()
                    self._brain_worker.wait(1000)
                # Advance the generation counter and clear the worker reference so
                # that any response_ready that was already queued before cancel()
                # took effect is rejected by the lambda guard in on_transcription.
                self._brain_generation += 1
                self._brain_worker = None
                self.state.set_status("idle")   # → overlay.set_idle via _on_status_changed
                self.state.add_diagnostic("error", "Brain worker timed out after 35 s.")

        # ── Response ──────────────────────────────────────────────────────────

        def on_response(self, response: str, is_action: bool):
            if hasattr(self, "_brain_timeout"):
                self._brain_timeout.stop()

            transcript = self.state.transcript

            if is_action:
                # overlay.set_action fires via last_action_changed signal
                self.state.set_last_action(response)
                self.state.add_history_entry(transcript, "", action=response)
            else:
                # overlay.set_response fires via response_changed signal
                self.state.set_response(response)
                # Skip history entry when the model returned nothing — an empty
                # entry in session history is misleading and carries no information.
                if response:
                    self.state.add_history_entry(transcript, response)

            self.speaker.speak(response)
            # speaking_ended callback drives the idle transition when TTS plays.
            # When TTS is disabled OR the response is empty, speak() is a no-op
            # and speaking_ended never fires — go idle immediately in both cases.
            if not config.get("tts_enabled", True) or not response:
                self.state.set_status("idle")   # → overlay.set_idle via _on_status_changed

        # ── Run ───────────────────────────────────────────────────────────────

        def run(self):
            self.overlay.show()
            self.listener.start()
            sys.exit(self.app.exec())

    vox = VoxApp()
    vox.run()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vox_error.log")
    try:
        config        = Config()
        _validate_startup(config)          # early console-only pre-flight
        whisper_model = load_whisper(config)
        run_app(config, whisper_model)
    except Exception:
        err = traceback.format_exc()
        log.error(err)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(err)
        log.error(f"Error saved to: {os.path.abspath(log_path)}")
