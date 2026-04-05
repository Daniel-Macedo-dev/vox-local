# VOX

**Topics:** voice-assistant, local-ai, ollama, whisper, python, offline-ai, piper-tts

A fully **offline**, **privacy-first** voice assistant for your PC — no cloud APIs, no data leaving your machine.

Built with [faster-whisper](https://github.com/guillaumekynast/faster-whisper), [Ollama](https://ollama.com), and [Piper TTS](https://github.com/rhasspy/piper).

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why VOX?

- **100% offline** — speech recognition, reasoning, and TTS all run locally
- **No cloud APIs** — your voice never leaves your machine
- **Works without internet** after initial model setup
- **Declarative permission system** — the AI can only run actions you explicitly allow; no arbitrary shell access

---

## Features

- **Wake word activation** — say "VOX" to activate; no hotkey required
- **Push-to-talk mode** — optional `Ctrl+Shift` press-to-talk (configurable)
- **Same-utterance commands** — "vox open spotify" said in one breath is correctly captured
- **Accent-tolerant wake word matching** — NFD normalisation + word-boundary check (no substring false positives)
- **Continuous audio capture** — rolling InputStream with 1.5 s pre-buffer preserves the start of every command
- **Energy pre-screening** — silent windows skip Whisper entirely, saving CPU
- **Adaptive noise floor** — exponential moving average tracks ambient noise for reliable VAD
- **Monitoring state** — overlay and Dashboard show "monitoring" (green) when the wake-word loop is active and listening for the wake word
- **Control Center** — full desktop GUI covering all settings, diagnostics, history, and audio testing
- **Real mic level meter** — live RMS-based input meter in the overlay and Control Center (not fake animation)
- **Mic calibration** — 2-phase flow (3 s silence + 3 s speech) measures noise floor, SNR, and suggests a silence threshold
- **Signal health card** — noise floor, SNR, and clipping fraction visible in the Audio tab after calibration
- **STT test** — record 5 s, run Whisper, see transcript and quality diagnosis directly in the UI
- **Local LLM** — runs on Ollama (qwen2.5, llama3, mistral, etc.)
- **Local TTS** — Piper voices, zero latency
- **Permission system** — editable allowlist of allowed actions, configurable from the GUI
- **App aliases** — editable spoken-name → executable mapping, configurable from the GUI
- **Floating HUD overlay** — compact satellite display, always on top, draggable
- **Session history** — recent interactions visible in the Control Center
- **Diagnostics panel** — structured in-app log of warnings, errors, and validation results
- **Mic test and TTS test** — available directly from the Audio tab
- **Fully configurable from the UI** — no YAML editing needed for normal use

---

## Platform Support

| Feature | Windows | Linux |
|---------|---------|-------|
| Wake word / push-to-talk | ✅ Full | ✅ Full |
| Volume control | ✅ WASAPI/pycaw | ✅ pactl / amixer |
| Mute toggle | ✅ keyboard | ✅ pactl |
| Media keys (play/pause/next/prev) | ✅ keyboard | ⚠️ requires `playerctl` |
| Open/close apps | ✅ Full (PATH + registry) | ⚠️ URI schemes via xdg-open; plain apps via PATH |
| TTS (Piper) | ✅ `.exe` binary | ✅ native binary |
| Screenshot | ✅ | ✅ |

**Linux note for media keys:** install `playerctl` (`sudo apt install playerctl`) for reliable media control. The `keyboard` module requires root privileges on Linux and may not work without it.

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- [Piper TTS](https://github.com/rhasspy/piper/releases) binary in `piper/piper/`
- NVIDIA GPU recommended (CUDA) — CPU works but is slower
- Windows 10/11 or Linux

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/DeD-TechStack/vox-local.git
cd vox-local
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull an Ollama model

```bash
ollama pull qwen2.5:7b    # recommended for most systems
# or
ollama pull qwen2.5:14b   # higher quality (requires ~9 GB VRAM)
```

### 4. Download Piper + voice model

Download the Piper binary and place it at `piper/piper/piper.exe` (Windows) or `piper/piper/piper` (Linux).
Download a voice model and place it in `voices/`:

```
vox-local/
├── piper/
│   └── piper/
│       └── piper.exe          ← binary here
└── voices/
    ├── en_US-ryan-high.onnx
    └── en_US-ryan-high.onnx.json
```

### 5. Run

```bash
python src/main.py
```

---

## Running Tests

The test suite covers pure logic and critical runtime rules without requiring hardware, Ollama, Whisper, or Piper.

```bash
# From the vox-local/ directory:
python -m pytest tests/ -v
```

A quick sanity-compile pass before testing:

```bash
python -m compileall src
python -m pytest tests/ -v
```

Tests live in `tests/` and are discovered automatically by pytest. No extra configuration is needed — `tests/conftest.py` stubs all hardware and GUI dependencies so the suite runs in any environment.

---

## Usage

### Interface overview

VOX has two UI surfaces:

| Surface | Role |
|---------|------|
| **Control Center** | Full configuration and monitoring window. Open via tray icon or double-click. |
| **Overlay HUD** | Compact floating status display. Shows real-time state, transcript, and response. |

### Tray icon actions

| Action | How |
|--------|-----|
| Open Control Center | Double-click tray icon, or right-click → Control Center |
| Show Overlay | Right-click → Show Overlay |
| Open Control Center (Activation tab) | Right-click → Settings |

### Voice interaction

| Action | How |
|--------|-----|
| Activate (wake word mode) | Say "VOX" — the overlay turns blue and listens |
| Activate (push-to-talk mode) | Press and hold `Ctrl+Shift`, speak, release |
| Switch language | Click the `AUTO`/`PT`/`EN` badge in the overlay |
| Move the overlay | Click and drag |

### Control Center tabs

| Tab | What you can do |
|-----|-----------------|
| **Dashboard** | See runtime state, dependency health, active config, last interaction |
| **Audio** | Select mic/output devices, view live mic level, test mic (3 s recording), test TTS |
| **Activation** | Set activation mode, wake word, PTT key, silence/chunk parameters |
| **Assistant** | Set Ollama URL, model, history size, language, TTS on/off, voice model |
| **Actions** | Enable/disable individual actions in the allowlist |
| **Aliases** | Add, edit, or remove spoken name → app/URI mappings |
| **Directories** | Manage directories searched by `search_file` |
| **History** | View recent transcript/response/action entries; clear session history |
| **Diagnostics** | View structured warning/error log; re-run validation; clear log |

### Example commands

```
"Open Spotify"
"Close Discord"
"Volume 40"
"Mute"
"Next track"
"Take a screenshot"
"What time is it?"
"Search for file report"
"Open YouTube"
```

### Language badge

The badge in the overlay lower-right corner shows the **configured** language mode (`AUTO`, `PT`, or `EN`). After each command, it briefly flashes the **detected** transcription language, then restores to the configured mode. Clicking it cycles through the modes and saves the selection immediately.

---

## Security Model

VOX uses a **declarative permission system**. The LLM cannot execute arbitrary shell commands — it can only call functions explicitly listed in `allowed_actions`. This prevents prompt injection attacks and accidental destructive operations.

```yaml
allowed_actions:
  - open_app       # allowed
  - set_volume     # allowed
  # delete_files   # not in the list = never runs
  # shell_command  # not in the list = never runs
```

The allowlist is fully editable from the **Actions** tab in the Control Center.

---

## Architecture

```
Microphone
    ↓
sd.InputStream — continuous rolling stream (100 ms reads)
    ↓
audio_utils — compute_rms, has_sufficient_energy, update_noise_floor
  · 1.5 s pre-buffer (deque) preserves audio before wake word
  · energy pre-screening skips Whisper on silent 1 s windows
  · exponential moving average tracks ambient noise floor
    ↓
faster-whisper — Speech-to-Text (wake word detection, beam_size=1)
  · accent-normalised word-boundary match via audio_utils.wake_word_in_text
    ↓
Listener — emits monitoring_started / listening_started / capture_warning
  · emits real RMS mic level → overlay waveform + Control Center meter
    ↓
_record_command_from_stream — captures command audio on existing stream
  · seeds with pre-buffer to include audio before the wake word was recognised
    ↓
faster-whisper — Speech-to-Text (command transcription, beam_size=5)
    ↓
AppState — central state hub (status, transcript, response, diagnostics, history)
    ↓
Brain (Ollama API) — LLM reasoning (streamed)
    ↓
Executor — validates action against allowlist
    ↓
Action runs (open app / set volume / etc.)
    ↓
Piper TTS — speaks the response
    ↓
Overlay HUD + Control Center — show transcript, response, history, diagnostics
```

---

## Configuration (`config/settings.yaml`)

**Most settings can be changed from the Control Center** and are saved automatically. Direct YAML editing is only needed for advanced settings not exposed in the UI (`whisper_model`, `whisper_device`, `whisper_compute_type`).

| Key | Default | Description |
|-----|---------|-------------|
| `activation_mode` | `wake_word` | `wake_word` or `push_to_talk` |
| `wake_word` | `vox` | Phrase that activates wake-word mode |
| `push_to_talk_key` | `ctrl+shift` | Key combo for push-to-talk |
| `language` | `en` | Transcription language (`auto`, `pt`, `en`) |
| `whisper_model` | `base` | Whisper model size (**YAML only — requires restart**) |
| `whisper_device` | `cpu` | Device for Whisper (`cpu` or `cuda`) (**YAML only — requires restart**) |
| `whisper_compute_type` | `int8` | Compute type for Whisper (**YAML only — requires restart**) |
| `ollama_model` | `qwen2.5:14b` | Ollama model to use |
| `ollama_url` | `http://localhost:11434` | Ollama API base URL |
| `tts_enabled` | `true` | Enable/disable voice responses |
| `piper_path` | `piper/piper/piper.exe` | Path to Piper binary (relative to project root) |
| `voice_model` | `voices/en_US-ryan-high.onnx` | Piper voice (.onnx, relative to project root) |
| `mic_device` | `null` | Microphone device index (`null` = system default) |
| `output_device` | `null` | Speaker device index (`null` = system default) |
| `max_history` | `20` | Max conversation turns kept in memory |
| `silence_threshold` | `0.01` | RMS below which audio is silent |
| `silence_duration` | `1.5` | Seconds of silence that ends a command |
| `app_aliases` | see file | Map spoken names → executables or URI schemes |
| `allowed_actions` | see file | Allowlist of executable actions (applies immediately on next LLM call) |
| `search_dirs` | `~/Documents`, `~/Downloads`, `~/Desktop` | Directories for file search |

### Settings that apply immediately (no restart)

- `language`, `wake_word` — next recognition cycle
- `tts_enabled`, `voice_model`, `piper_path` — next TTS call
- `ollama_model`, `ollama_url` — next LLM request
- `allowed_actions` — next LLM call (both the prompt and executor allowlist update together)
- `silence_threshold`, `silence_duration` — next command capture cycle
- Audio devices — output: immediate; microphone: listener restarts automatically

### Settings that restart the listener automatically (on Save in the UI)

- `activation_mode`, `push_to_talk_key` — the listener is stopped and restarted when these are saved from the Activation tab; a confirmation message is shown in the tab

### Settings that require restarting VOX

- `whisper_model`, `whisper_device`, `whisper_compute_type` — Whisper is loaded once at startup

---

## Troubleshooting

**Ollama not running**
```
Ollama is not reachable. Start it with: ollama serve
```
Start Ollama:
```bash
ollama serve
```
Check the **Diagnostics** tab in the Control Center for details.

**CUDA not available**
VOX falls back to CPU automatically. To force CPU explicitly in `settings.yaml`:
```yaml
whisper_device: cpu
whisper_compute_type: int8
```

**Piper not found**
Check that `piper_path` in the **Assistant** tab (or `config/settings.yaml`) points to the correct binary. Default: `piper/piper/piper.exe` (relative to project root).

**No audio input / microphone not detected**
- Check microphone permissions in your OS settings
- Open the **Audio** tab in the Control Center and use **Test Microphone** to verify the device
- Select a specific device from the Input dropdown and save
- Run a validation from **Diagnostics** → **Re-run Validation**

**Mic level meter shows nothing**
The real-time meter is driven by actual RMS from the microphone. If it never moves:
- Check that the correct input device is selected in Audio settings
- Use **Test Microphone** to record 3 seconds and check the peak reading
- Run **Calibrate** in the Audio tab — a noisy or low-SNR environment shows up in the Signal Health card

**Wake word not detected reliably**
- Run **Calibrate** to measure your noise floor and apply the suggested silence threshold
- Run the **STT Test** to verify Whisper can hear and transcribe your voice
- Check Diagnostics for repeated "silent windows" warnings — this means the mic level is too low
- VOX now uses accent normalisation and word-boundary matching, so "vóx" and "vox" both work

**VOX says "vox open spotify" instead of "open spotify"**
This is handled automatically — the listener strips any leading wake-word echo. If it persists, lower `silence_threshold` so command capture starts sooner.

**Media keys not working on Linux**
```bash
sudo apt install playerctl   # Debian/Ubuntu
sudo pacman -S playerctl      # Arch
```

**search_file doesn't find files**
Check the **Directories** tab in the Control Center — verify the correct directories are listed.

---

## Known Limitations

- **Push-to-talk on Linux** requires the `keyboard` package to have root privileges (or `uinput` access). Wake word mode is recommended on Linux.
- **open_app on Linux** for plain executable names uses PATH directly. Apps not in PATH must be aliased with their full path in the **Aliases** tab.
- **TTS interruption** is not implemented within a single response — Piper generates full audio before playback begins.
- **Barge-in** (speaking while VOX is responding) cancels LLM generation but the in-flight TTS audio plays to completion.
- **Speaking state** reflects real playback: the overlay enters "speaking" only after Piper synthesis succeeds and audio playback is about to begin. If Piper fails (binary missing, model error, timeout), the app returns to idle without ever entering "speaking". When TTS is disabled, the transition to idle is immediate.
- **Whisper settings** (`whisper_model`, `whisper_device`, `whisper_compute_type`) are not exposed in the Control Center UI — they require a YAML edit and application restart.

---

## Roadmap

- [x] Wake word support (always-on)
- [x] Push-to-talk activation mode
- [x] Settings GUI
- [x] Cancel/interrupt running LLM request
- [x] Same-utterance wake-word + command capture
- [x] Listener restart on microphone change
- [x] Truthful Linux platform behavior
- [x] Control Center with all settings accessible from UI
- [x] Real RMS-based mic level meter (overlay + Control Center)
- [x] Session history visible in UI
- [x] Structured diagnostics panel
- [x] Mic test and TTS test from UI
- [x] Allowlist and app aliases editable from UI
- [x] Central AppState model
- [x] Continuous InputStream with rolling pre-buffer (no lost command starts)
- [x] Energy pre-screening (skip Whisper on silent audio)
- [x] Adaptive noise floor tracking
- [x] Accent-tolerant, word-boundary wake word matching
- [x] Monitoring state (overlay + Dashboard show active wake-word loop)
- [x] Microphone calibration flow (noise floor + SNR + suggested threshold)
- [x] Signal health card in Audio tab
- [x] STT test (record + Whisper + quality diagnosis in UI)
- [x] Capture warnings routed to Diagnostics panel
- [ ] TTS barge-in / cancellation
- [ ] Custom action plugins
- [ ] Conversation memory persistence across restarts
- [ ] Linux audio (PipeWire) full integration

---

## License

MIT — use it, fork it, make it yours.
