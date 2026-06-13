"""
voxbox.transport.server
========================
Skeleton of the GPU-box inference service (the "WINDOWS / 3x Blackwell" side of
the diagram). It exposes the pipeline over a WebSocket so the Mac orchestrator can
stream mic audio in and receive synthesized replies — all on the LAN, nothing
leaving the network.

FastAPI/uvicorn are imported lazily; install with:  pip install voxbox[server]
The protocol is intentionally tiny:
    client -> server : binary frames of 16-bit PCM @16k (mic chunks)
    server -> client : binary frames of 16-bit PCM @24k (TTS reply) + JSON meta
"""
from __future__ import annotations

from ..config import Config
from ..pipeline import build_pipeline


def create_app(cfg: Config | None = None):  # pragma: no cover - needs server extras
    try:
        from fastapi import FastAPI, WebSocket
    except Exception as exc:
        raise RuntimeError("Server requires fastapi+uvicorn. Install voxbox[server].") from exc

    app = FastAPI(title="VoxBox Inference")
    orch = build_pipeline(cfg)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "turns": orch.metrics.count}

    @app.get("/metrics")
    def metrics():
        return orch.metrics.summary()

    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        from ..contracts import AudioChunk
        from .protocol import turn_to_meta

        await ws.accept()
        try:
            while True:
                pcm = await ws.receive_bytes()
                result = orch.feed(AudioChunk(pcm=pcm, sample_rate=16000))
                if result is not None:
                    await ws.send_json(turn_to_meta(result))
                    await ws.send_bytes(result.reply.pcm)
        except Exception:
            await ws.close()

    return app
