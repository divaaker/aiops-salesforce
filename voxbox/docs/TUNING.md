# Tuning the live voice agent

All knobs are environment variables on the `scripts/talk.py` (and `serve.py`)
command. The live scripts apply forgiving defaults (`live_env_defaults`); anything
you set explicitly wins.

## Backend selection
| Var | Default | Values |
|-----|---------|--------|
| `VOX_STT` | `mock` | `mock`, `faster_whisper` |
| `VOX_LLM` | `mock` | `mock`, `ollama` |
| `VOX_TTS` | `mock` | `mock`, `piper` |
| `VOX_VAD` | `energy` | `energy`, `silero` |

## Turn-taking / endpointing (the EnergyVAD)
| Var | Live default | What it does |
|-----|--------------|--------------|
| `VOX_VAD_THRESHOLD` | `-40` (lib) | dBFS above which audio counts as speech. **Raise toward 0** if it self-triggers on noise; **lower toward -50** if it doesn't hear you. Run `scripts/calibrate.py` to find yours. |
| `VOX_VAD_HANGOVER_MS` | `600` | Silence needed to end your turn. **Higher** = won't cut you off when you pause mid-thought (but slower to respond). **Lower** = snappier. |
| `VOX_VAD_MIN_SPEECH_MS` | `200` | Utterances shorter than this are ignored (rejects coughs/clicks). |

## Barge-in (talk over the agent)
| Var | Default | What it does |
|-----|---------|--------------|
| `VOX_BARGE_IN` | `1` | `0` disables talk-over. |
| `VOX_BARGE_THRESHOLD` | `-30` | dBFS to trigger an interrupt. Raise it if the agent interrupts itself. |
| `VOX_ONSET_FRAMES` | `6` | Consecutive loud 20 ms frames (~120 ms) needed to interrupt. Higher = less twitchy. |
| `VOX_PLAYBACK_MARGIN` | `8` | Extra dB added to the barge-in threshold **while the agent is speaking** (so its own voice doesn't self-trigger). |

## Neural VAD (Silero) — no threshold to tune
`VOX_VAD=silero` swaps the energy detector for the Silero neural model, which tells
speech from noise so you can skip `VOX_VAD_THRESHOLD`/calibration entirely. CPU-ok.
```bash
pip install "voxbox[silero]"
VOX_VAD=silero VOX_STT=faster_whisper VOX_LLM=ollama VOX_TTS=piper \
VOX_OLLAMA_MODEL=gemma3 VOX_PIPER_VOICE=en_US-amy-medium.onnx python scripts/talk.py
```
| Var | Default | What it does |
|-----|---------|--------------|
| `VOX_SILERO_THRESHOLD` | `0.5` | Speech probability (0–1) to count as speech. Raise to be stricter. |
| `VOX_VAD_HANGOVER_MS` / `VOX_VAD_MIN_SPEECH_MS` | shared with energy | endpointing, same as above |

## STT model (faster-whisper)
| Var | Live default | Notes |
|-----|--------------|-------|
| `VOX_WHISPER_MODEL` | `base.en` | Speed↑accuracy↓: `tiny.en` < `base.en` < `small.en` < `medium.en`. Drop the `.en` for non-English. First run downloads the model. |
| `VOX_WHISPER_DEVICE` | `auto` | `cpu`, `cuda`, `auto`. |
| `VOX_WHISPER_COMPUTE` | `default` | e.g. `int8` (fast on CPU), `float16` (GPU). |

## LLM (Ollama)
| Var | Default | Notes |
|-----|---------|-------|
| `VOX_OLLAMA_MODEL` | `gemma3` | Any model you've `ollama pull`ed. `gemma3:1b` is lighter. |
| `VOX_OLLAMA_HOST` | `http://localhost:11434` | Point at a remote Ollama if needed. |

## TTS (Piper)
| Var | Notes |
|-----|-------|
| `VOX_PIPER_VOICE` | Path to a Piper `.onnx` voice (the `.onnx.json` must sit next to it). |

## Other
| Var | Default | Notes |
|-----|---------|-------|
| `VOX_STREAM` | `1` | `0` = wait for the whole reply before speaking. |
| `VOX_BUDGET_MS` | `1200` | Latency budget; turns over it are flagged in metrics. |

## Recipes
**Quietest, snappiest (good mic, headphones):**
```bash
VOX_VAD_THRESHOLD=-35 VOX_VAD_HANGOVER_MS=400 VOX_WHISPER_MODEL=base.en \
VOX_STT=faster_whisper VOX_LLM=ollama VOX_TTS=piper \
VOX_OLLAMA_MODEL=gemma3 VOX_PIPER_VOICE=en_US-amy-medium.onnx python scripts/talk.py
```
**Noisy room / laptop mic (more forgiving):**
```bash
VOX_VAD_THRESHOLD=-28 VOX_VAD_HANGOVER_MS=700 VOX_BARGE_THRESHOLD=-22 \
VOX_STT=faster_whisper VOX_LLM=ollama VOX_TTS=piper \
VOX_OLLAMA_MODEL=gemma3 VOX_PIPER_VOICE=en_US-amy-medium.onnx python scripts/talk.py
```

## Troubleshooting
- **It keeps interrupting itself / triggers on silence** → raise `VOX_VAD_THRESHOLD` (toward 0), use headphones, or raise `VOX_PLAYBACK_MARGIN`.
- **It never responds to me** → lower `VOX_VAD_THRESHOLD`; check `scripts/calibrate.py` and the mic meter.
- **It cuts me off mid-sentence** → raise `VOX_VAD_HANGOVER_MS`.
- **Slow first reply** → smaller `VOX_WHISPER_MODEL` (`tiny.en`) and/or `gemma3:1b`.
- **"Connection refused" to Ollama** → start it (`brew services start ollama` / `ollama serve`).
