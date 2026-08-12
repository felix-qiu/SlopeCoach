UV ?= uv
PYTHON_PROJECT := python/pyproject.toml
FIXTURE := fixtures/golden_pose_001.json

.PHONY: doctor python test lint golden benchmark

doctor:
	@command -v git >/dev/null && git --version
	@command -v $(UV) >/dev/null && $(UV) --version
	@command -v ffmpeg >/dev/null && ffmpeg -version 2>/dev/null | head -n 1
	@command -v ffprobe >/dev/null && ffprobe -version 2>/dev/null | head -n 1
	@$(UV) run --project python python --version
	@$(UV) lock --project python --check

python:
	$(UV) sync --project python --dev

test:
	$(UV) run --project python pytest

lint:
	$(UV) run --project python ruff format --check python tests
	$(UV) run --project python ruff check python tests

golden:
	$(UV) run --project python python -m slopecoach_ml.cli golden

benchmark:
	$(UV) run --project python python -m slopecoach_ml.cli benchmark $(FIXTURE)

