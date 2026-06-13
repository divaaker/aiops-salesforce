"""
VoxBox — a fully-local, private, real-time voice agent.

Public API:
    from voxbox import build_pipeline, Config
    orch = build_pipeline(Config())          # all-mock, no GPU, runs anywhere
    result = orch.handle_segment(segment)    # or orch.feed(chunk) in a loop
"""
from .config import Config
from .contracts import (
    AudioChunk,
    AudioReply,
    LLMResponse,
    SpeechSegment,
    Transcript,
)
from .orchestrator import Orchestrator, TurnResult
from .pipeline import build_pipeline

__version__ = "0.1.0"

__all__ = [
    "Config",
    "AudioChunk",
    "AudioReply",
    "SpeechSegment",
    "Transcript",
    "LLMResponse",
    "Orchestrator",
    "TurnResult",
    "build_pipeline",
]
