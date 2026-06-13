.PHONY: help install dev test demo stream-demo serve talk docker up down

help:
	@echo "VoxBox — common tasks:"
	@echo "  make test         run the test suite (stdlib only)"
	@echo "  make demo         run the mock conversation demo"
	@echo "  make stream-demo  run the streaming-reply demo"
	@echo "  make serve        run the inference server (VOX_* env selects backends)"
	@echo "  make talk         run the live mic agent (needs voxbox[client])"
	@echo "  make up / down    docker compose up/down (server + local Ollama)"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	python3 -m pytest -q

demo:
	python3 scripts/demo.py

stream-demo:
	python3 scripts/stream_demo.py

serve:
	python3 scripts/serve.py

talk:
	python3 scripts/talk.py

docker:
	docker build -t voxbox --build-arg EXTRAS="server,whisper,piper" .

up:
	docker compose up --build

down:
	docker compose down
