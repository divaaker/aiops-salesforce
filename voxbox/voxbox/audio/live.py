"""
voxbox.audio.live
=================
Microphone capture and speaker playback via sounddevice — the audio I/O of the
Mac client in the diagram. Imports are lazy and the streams use raw int16 buffers
(no numpy needed). These wrap real hardware, so they're exercised on a real
machine, not in CI; the loop/protocol logic around them is tested separately.

Install with:  pip install "voxbox[client]"
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

from ..contracts import AudioChunk, AudioReply


class MicSource:
    """Iterable of AudioChunk from the default input device (mono int16)."""

    def __init__(self, sample_rate: int = 16000, chunk_ms: float = 20.0,
                 device: Optional[int] = None) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.device = device
        self.blocksize = max(1, int(sample_rate * chunk_ms / 1000.0))

    def __iter__(self) -> Iterator[AudioChunk]:  # pragma: no cover - needs hardware
        import queue

        import sounddevice as sd  # lazy

        q: "queue.Queue[bytes]" = queue.Queue()

        def _cb(indata, frames, t, status):
            q.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=self.sample_rate, blocksize=self.blocksize,
            dtype="int16", channels=1, device=self.device, callback=_cb,
        ):
            while True:
                pcm = q.get()
                yield AudioChunk(pcm=pcm, sample_rate=self.sample_rate, ts=time.monotonic())


class Speaker:
    """Plays an AudioReply on the default output device at the reply's own rate.
    Blocking — fine for the file demo; use StreamingSpeaker for the live agent."""

    def __init__(self, device: Optional[int] = None) -> None:
        self.device = device

    def play(self, reply: AudioReply) -> None:  # pragma: no cover - needs hardware
        import sounddevice as sd  # lazy

        with sd.RawOutputStream(
            samplerate=reply.sample_rate, dtype="int16", channels=1, device=self.device
        ) as out:
            out.write(reply.pcm)


class StreamingSpeaker:
    """Non-blocking, interruptible playback for barge-in.

    `play` starts playback on a background thread and returns immediately; `stop`
    halts it mid-stream; `is_active` reports whether audio is still playing. This is
    the speaker the live loop watches so the user can talk over the agent.
    """

    def __init__(self, device: Optional[int] = None, chunk_frames: int = 1024) -> None:
        import threading

        self.device = device
        self.chunk_frames = chunk_frames
        self._stop = threading.Event()
        self._active = threading.Event()
        self._thread = None
        self._threading = threading

    def is_active(self) -> bool:
        return self._active.is_set()

    def play(self, reply: AudioReply) -> None:  # pragma: no cover - needs hardware
        self.stop()  # cancel anything currently playing
        self._stop.clear()
        self._active.set()
        self._thread = self._threading.Thread(
            target=self._run, args=(reply,), daemon=True
        )
        self._thread.start()

    def _run(self, reply: AudioReply) -> None:  # pragma: no cover - needs hardware
        import sounddevice as sd  # lazy

        step = self.chunk_frames * 2  # bytes (int16)
        try:
            with sd.RawOutputStream(
                samplerate=reply.sample_rate, dtype="int16",
                channels=1, device=self.device,
            ) as out:
                for i in range(0, len(reply.pcm), step):
                    if self._stop.is_set():
                        break
                    out.write(reply.pcm[i : i + step])
        finally:
            self._active.clear()

    def stop(self) -> None:  # pragma: no cover - needs hardware
        if self._thread is not None and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=1.0)
        self._active.clear()

    def wait(self) -> None:  # pragma: no cover - needs hardware
        if self._thread is not None:
            self._thread.join()
