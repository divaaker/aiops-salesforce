"""
Run the GPU-box inference server (the WINDOWS / 3x Blackwell side). Backends are
chosen via the same VOX_* environment variables as scripts/talk.py.

Install:  pip install "voxbox[server]"   (plus whichever model extras you use)
Run:      VOX_STT=faster_whisper VOX_LLM=ollama VOX_TTS=piper \
          VOX_PIPER_VOICE=en_US-amy-medium.onnx python3 scripts/serve.py
          # then connect from the Mac:  python3 scripts/talk_remote.py ws://<gpu-ip>:8765/stream
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxbox.runtime import config_from_env  # noqa: E402


def main() -> None:
    host = os.environ.get("VOX_HOST", "0.0.0.0")
    port = int(os.environ.get("VOX_PORT", "8765"))
    try:
        import uvicorn

        from voxbox.transport.server import create_app
    except Exception:
        print("❌ Server needs fastapi+uvicorn. Install with: pip install 'voxbox[server]'")
        sys.exit(1)

    app = create_app(config_from_env())
    print(f"🚀 VoxBox inference server on ws://{host}:{port}/stream")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
