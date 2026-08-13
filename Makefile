UV ?= uv
PYTHON_PROJECT := python/pyproject.toml
FIXTURE := fixtures/golden_pose_001.json

.PHONY: doctor python test lint golden benchmark pose-doctor pose-smoke benchmark-real-pose benchmark-target-identity openmmlab-macos

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

pose-doctor:
	$(UV) run --project python python -m slopecoach_ml.cli pose-doctor

pose-smoke:
	@test -n "$(IMAGE)" || (echo 'usage: make pose-smoke IMAGE=/path/to/image' >&2; exit 2)
	$(UV) run --project python python -m slopecoach_ml.cli pose-image "$(IMAGE)" --input-non-mirrored

benchmark-real-pose:
	@test -n "$(VIDEO)" || (echo 'usage: make benchmark-real-pose VIDEO=/path/to/video' >&2; exit 2)
	$(UV) run --project python python -m slopecoach_ml.cli benchmark-real-pose "$(VIDEO)" --sample-fps "$(or $(SAMPLE_FPS),2)" --input-non-mirrored $(if $(OUTPUT),--output "$(OUTPUT)",) $(if $(DEBUG_DIR),--debug-dir "$(DEBUG_DIR)",)

benchmark-target-identity:
	@test -n "$(VIDEO)" || (echo 'usage: make benchmark-target-identity VIDEO=/path/to/video' >&2; exit 2)
	$(UV) run --project python python -m slopecoach_ml.cli benchmark-target-identity "$(VIDEO)" --sample-fps "$(or $(SAMPLE_FPS),2)" --input-non-mirrored $(if $(OUTPUT),--output "$(OUTPUT)",) $(if $(DEBUG_DIR),--debug-dir "$(DEBUG_DIR)",)

openmmlab-macos:
	bash scripts/bootstrap_openmmlab_macos.sh
