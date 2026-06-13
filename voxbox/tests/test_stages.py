from voxbox.contracts import SpeechSegment
from voxbox.stt import MockSTT
from voxbox.llm import RuleBasedLLM
from voxbox.tts import MockTTS


def test_stt_uses_label_when_present():
    t = MockSTT().transcribe(SpeechSegment(pcm=b"\x00\x00" * 1600, sample_rate=16000, label="hi"))
    assert t.text == "hi"
    assert 0.0 <= t.confidence <= 1.0


def test_stt_falls_back_to_length_description():
    t = MockSTT().transcribe(SpeechSegment(pcm=b"\x00\x00" * 16000, sample_rate=16000))
    assert "utterance" in t.text


def test_llm_greeting_and_name():
    llm = RuleBasedLLM(name="VoxBox")
    assert "VoxBox" in llm.respond("hello", []).text
    assert "VoxBox" in llm.respond("what is your name?", []).text


def test_llm_is_deterministic():
    llm = RuleBasedLLM()
    a = llm.respond("say something", [])
    b = llm.respond("say something", [])
    assert a.text == b.text


def test_llm_uses_history_for_turn_count():
    llm = RuleBasedLLM()
    hist = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    out = llm.respond("again", hist)
    assert "turn 2" in out.text


def test_tts_length_grows_with_word_count():
    tts = MockTTS(sample_rate=24000, seconds_per_word=0.3, min_seconds=0.3)
    short = tts.synthesize("hi")
    long = tts.synthesize("one two three four five six")
    assert long.num_samples > short.num_samples
    assert short.sample_rate == 24000
    assert short.duration_ms >= 300.0  # min floor honored
