"""Tests for env-driven config and the live capture->orchestrate->play loop,
using a fake speaker (no audio hardware)."""
from voxbox.pipeline import build_pipeline
from voxbox.runtime import config_from_env, live_env_defaults, run_local
from tests.helpers import utterance_chunks


class _FakeSpeaker:
    def __init__(self):
        self.played = []

    def play(self, reply):
        self.played.append(reply)


def test_config_from_env_defaults_to_mock():
    cfg = config_from_env(env={})
    assert (cfg.vad, cfg.stt, cfg.llm, cfg.tts) == ("energy", "mock", "mock", "mock")
    assert cfg.options == {}


def test_live_env_defaults_fill_endpointing():
    e = live_env_defaults(env={})
    assert e["VOX_VAD_HANGOVER_MS"] == "600"
    assert e["VOX_VAD_MIN_SPEECH_MS"] == "200"
    assert "VOX_WHISPER_MODEL" not in e  # only when whisper is selected


def test_live_env_defaults_add_whisper_model_when_selected():
    e = live_env_defaults(env={"VOX_STT": "faster_whisper"})
    assert e["VOX_WHISPER_MODEL"] == "base.en"


def test_live_env_defaults_do_not_override_user_values():
    e = live_env_defaults(env={"VOX_VAD_HANGOVER_MS": "300", "VOX_STT": "faster_whisper",
                               "VOX_WHISPER_MODEL": "small"})
    assert e["VOX_VAD_HANGOVER_MS"] == "300"
    assert e["VOX_WHISPER_MODEL"] == "small"


def test_live_defaults_reach_the_vad():
    cfg = config_from_env(live_env_defaults(env={}))
    orch = build_pipeline(cfg)
    assert orch.vad.hangover_ms == 600.0
    assert orch.vad.min_speech_ms == 200.0


def test_config_from_env_silero_options():
    cfg = config_from_env(env={"VOX_VAD": "silero", "VOX_SILERO_THRESHOLD": "0.6",
                               "VOX_VAD_HANGOVER_MS": "500"})
    assert cfg.vad == "silero"
    assert cfg.options["vad"]["threshold"] == 0.6
    assert cfg.options["vad"]["hangover_ms"] == 500.0
    assert "threshold_dbfs" not in cfg.options["vad"]


def test_config_from_env_vad_threshold_tuning():
    cfg = config_from_env(env={"VOX_VAD_THRESHOLD": "-28", "VOX_VAD_HANGOVER_MS": "500"})
    assert cfg.options["vad"]["threshold_dbfs"] == -28.0
    assert cfg.options["vad"]["hangover_ms"] == 500.0
    # and it actually reaches the EnergyVAD
    from voxbox.pipeline import build_pipeline
    orch = build_pipeline(cfg)
    assert orch.vad.threshold_dbfs == -28.0


def test_config_from_env_wires_real_backends_and_options():
    cfg = config_from_env(env={
        "VOX_STT": "faster_whisper", "VOX_WHISPER_MODEL": "small",
        "VOX_LLM": "ollama", "VOX_OLLAMA_MODEL": "gemma3", "VOX_OLLAMA_HOST": "http://h:1",
        "VOX_TTS": "piper", "VOX_PIPER_VOICE": "amy.onnx",
        "VOX_BUDGET_MS": "2500",
    })
    assert cfg.stt == "faster_whisper" and cfg.options["stt"]["model_size"] == "small"
    assert cfg.llm == "ollama" and cfg.options["llm"] == {"model": "gemma3", "host": "http://h:1"}
    assert cfg.tts == "piper" and cfg.options["tts"]["voice_path"] == "amy.onnx"
    assert cfg.budget_ms == 2500.0


def test_run_local_plays_each_completed_turn():
    orch = build_pipeline(config_from_env(env={}))  # all mock
    speaker = _FakeSpeaker()
    seen = []

    chunks = []
    ts = 0.0
    for line in ["hello", "what is your name"]:
        u = utterance_chunks(line, start_ts=ts)
        chunks.extend(u)
        ts = u[-1].ts + 0.5

    turns = run_local(orch, chunks, speaker, on_turn=lambda t: seen.append(t.transcript.text))

    assert turns == 2
    assert len(speaker.played) == 2
    assert seen == ["hello", "what is your name"]
    assert all(r.num_samples > 0 for r in speaker.played)


def test_run_local_respects_max_turns():
    orch = build_pipeline(config_from_env(env={}))
    speaker = _FakeSpeaker()
    chunks = []
    ts = 0.0
    for line in ["a", "b", "c"]:
        u = utterance_chunks(line, start_ts=ts)
        chunks.extend(u)
        ts = u[-1].ts + 0.5
    turns = run_local(orch, chunks, speaker, max_turns=1)
    assert turns == 1
    assert len(speaker.played) == 1
