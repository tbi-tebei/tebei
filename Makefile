PYTHON = python3
PIP    = pip

.PHONY: install run test lint build-index help

help:
	@echo "Available commands:"
	@echo "  make install      Install all dependencies"
	@echo "  make run          Start the dev server (hot-reload)"
	@echo "  make test         Run test suite"
	@echo "  make lint         Check code style with ruff"
	@echo "  make build-index  Trigger index build via API"

install:
	$(PYTHON) -m venv env
	. env/bin/activate && $(PIP) install -r requirements.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check app/ tests/

build-index:
	curl -s -X POST http://localhost:8000/api/index/build | python3 -m json.tool
