"""
Mac client: stream the local mic to a remote VoxBox inference server and play the
replies — the split topology from the diagram (thin Mac + fat GPU box).

Install:  pip install "voxbox[client]"
Run:      python3 scripts/talk_remote.py ws://<gpu-ip>:8765/stream
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    uri = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "VOX_SERVER", "ws://localhost:8765/stream"
    )
    try:
        from voxbox.audio.live import MicSource, Speaker
        from voxbox.transport.client import run_client
    except ImportError:
        print("❌ Needs sounddevice + websockets. Install with: pip install 'voxbox[client]'")
        sys.exit(1)

    print(f"🔌 connecting to {uri} — speak, then pause. Ctrl-C to quit.\n")
    try:
        asyncio.run(run_client(uri, MicSource(), Speaker()))
    except KeyboardInterrupt:
        print("\n👋 bye")


if __name__ == "__main__":
    main()
