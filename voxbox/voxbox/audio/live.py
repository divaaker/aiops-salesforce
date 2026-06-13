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
from typing import Callable, Iterator, Optional

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


class QueueingSpeaker:
    """Plays a queue of AudioReply clips back-to-back on a worker thread — what the
    streaming agent needs so consecutive sentences flow without gaps — while still
    being instantly interruptible for barge-in (`stop` flushes the queue).

    The actual audio writing is a pluggable `writer(reply, should_stop)` callable, so
    the queue/interrupt logic is testable without audio hardware.
    """

    def __init__(self, writer: Optional[Callable] = None, device: Optional[int] = None,
                 chunk_frames: int = 1024) -> None:
        import queue
        import threading

        self.device = device
        self.chunk_frames = chunk_frames
        self._writer = writer or self._sd_writer
        self._q: "queue.Queue" = queue.Queue()
        self._interrupt = threading.Event()
        self._playing = threading.Event()
        self._queue_mod = queue
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            reply = self._q.get()
            if reply is None:
                self._q.task_done()
                break
            self._playing.set()
            try:
                if not self._interrupt.is_set():
                    self._writer(reply, self._interrupt.is_set)
            finally:
                self._playing.clear()
                self._q.task_done()

    def play(self, reply: AudioReply) -> None:
        """Enqueue a clip (plays after whatever is already queued)."""
        self._q.put(reply)

    enqueue = play

    def is_active(self) -> bool:
        return self._playing.is_set() or not self._q.empty()

    def stop(self) -> None:
        """Barge-in: drop everything queued and cut off the current clip."""
        self._interrupt.set()
        while True:
            try:
                self._q.get_nowait()
                self._q.task_done()
            except self._queue_mod.Empty:
                break
        while self._playing.is_set():
            time.sleep(0.002)
        self._interrupt.clear()

    def wait(self) -> None:
        self._q.join()
        while self._playing.is_set():
            time.sleep(0.002)

    def close(self) -> None:
        self._q.put(None)
        self._worker.join(timeout=1.0)

    def _sd_writer(self, reply: AudioReply, should_stop) -> None:  # pragma: no cover - hardware
        import sounddevice as sd  # lazy

        step = self.chunk_frames * 2
        with sd.RawOutputStream(
            samplerate=reply.sample_rate, dtype="int16", channels=1, device=self.device
        ) as out:
            for i in range(0, len(reply.pcm), step):
                if should_stop():
                    return
                out.write(reply.pcm[i : i + step])
