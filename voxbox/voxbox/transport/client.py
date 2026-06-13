"""
LAN client: streams mic audio to the GPU-box inference server and plays replies.

The message-handling (`ReplyHandler` + `dispatch_message`) is pure and tested; the
live websocket pump (`run_client`) wraps it and is exercised on real hardware.
"""
from __future__ import annotations

import json
from typing import Union

from ..contracts import AudioReply


class ReplyHandler:
    """Turns incoming server frames into playback. `speaker` needs `.play(reply)`."""

    def __init__(self, speaker, default_sample_rate: int = 24000) -> None:
        self.speaker = speaker
        self.sample_rate = default_sample_rate
        self.last_meta: dict | None = None

    def on_text(self, data: dict) -> None:
        self.last_meta = data
        sr = data.get("sample_rate")
        if sr:
            self.sample_rate = int(sr)

    def on_bytes(self, pcm: bytes) -> None:
        self.speaker.play(AudioReply(pcm=pcm, sample_rate=self.sample_rate))


def dispatch_message(raw: Union[str, bytes, bytearray, dict], handler: ReplyHandler) -> None:
    """Route one received frame to the handler (JSON meta vs binary audio)."""
    if isinstance(raw, (bytes, bytearray)):
        handler.on_bytes(bytes(raw))
    elif isinstance(raw, dict):
        handler.on_text(raw)
    else:  # str
        handler.on_text(json.loads(raw))


async def run_client(uri: str, mic, speaker) -> None:  # pragma: no cover - needs net+hw
    import asyncio

    import websockets  # lazy

    handler = ReplyHandler(speaker)

    async with websockets.connect(uri, max_size=None) as ws:

        async def sender():
            loop = asyncio.get_event_loop()
            it = iter(mic)
            while True:
                chunk = await loop.run_in_executor(None, next, it)
                await ws.send(chunk.pcm)

        async def receiver():
            async for msg in ws:
                dispatch_message(msg, handler)

        await asyncio.gather(sender(), receiver())
