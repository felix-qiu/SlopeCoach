UV ?= uv
PYTHON_PROJECT := python/pyproject.toml
FIXTURE := fixtures/golden_pose_001.json

.PHONY: doctor python test lint golden temporal-golden turn-golden biomechanics-golden sport-type-golden benchmark pose-doctor sport-equipment-doctor pose-smoke benchmark-real-pose benchmark-target-identity benchmark-temporal-turns benchmark-biomechanics benchmark-sport-type prepare-biomechanics-dataset benchmark-biomechanics-dataset prepare-target-gt openmmlab-macos

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

temporal-golden:
	$(UV) run --project python python -m slopecoach_ml.cli temporal-golden

turn-golden:
	$(UV) run --project python python -m slopecoach_ml.cli turn-golden

biomechanics-golden:
	$(UV) run --project python python -m slopecoach_ml.cli biomechanics-golden

sport-type-golden:
	$(UV) run --project python python -m slopecoach_ml.cli sport-type-golden

benchmark:
	$(UV) run --project python python -m slopecoach_ml.cli benchmark $(FIXTURE)

pose-doctor:
	$(UV) run --project python python -m slopecoach_ml.cli pose-doctor

sport-equipment-doctor:
	$(if $(OPENML_PY),PYTHONPATH=python $(OPENML_PY),$(UV) run --project python python) -m slopecoach_ml.cli sport-equipment-doctor $(if $(EQUIPMENT_CONFIG),--equipment-config "$(EQUIPMENT_CONFIG)",) $(if $(EQUIPMENT_CHECKPOINT),--equipment-checkpoint "$(EQUIPMENT_CHECKPOINT)",)

pose-smoke:
	@test -n "$(IMAGE)" || (echo 'usage: make pose-smoke IMAGE=/path/to/image' >&2; exit 2)
	$(UV) run --project python python -m slopecoach_ml.cli pose-image "$(IMAGE)" --input-non-mirrored

benchmark-real-pose:
	@test -n "$(VIDEO)" || (echo 'usage: make benchmark-real-pose VIDEO=/path/to/video' >&2; exit 2)
	$(UV) run --project python python -m slopecoach_ml.cli benchmark-real-pose "$(VIDEO)" --sample-fps "$(or $(SAMPLE_FPS),2)" --input-non-mirrored $(if $(OUTPUT),--output "$(OUTPUT)",) $(if $(DEBUG_DIR),--debug-dir "$(DEBUG_DIR)",)

benchmark-target-identity:
	@test -n "$(VIDEO)" || (echo 'usage: make benchmark-target-identity VIDEO=/path/to/video' >&2; exit 2)
	$(UV) run --project python python -m slopecoach_ml.cli benchmark-target-identity "$(VIDEO)" --sample-fps "$(or $(SAMPLE_FPS),2)" --input-non-mirrored $(if $(or $(TARGET_GT),$(GT)),--target-gt "$(or $(TARGET_GT),$(GT))",) $(if $(OUTPUT),--output "$(OUTPUT)",) $(if $(DEBUG_DIR),--debug-dir "$(DEBUG_DIR)",)

benchmark-temporal-turns:
	@test -n "$(VIDEO)" || (echo 'usage: make benchmark-temporal-turns VIDEO=/path/to/video' >&2; exit 2)
	$(UV) run --project python python -m slopecoach_ml.cli benchmark-temporal-turns "$(VIDEO)" --sample-fps "$(or $(SAMPLE_FPS),5)" --input-non-mirrored $(if $(OUTPUT),--output "$(OUTPUT)",) $(if $(DEBUG_DIR),--debug-dir "$(DEBUG_DIR)",)

benchmark-biomechanics:
	@test -n "$(VIDEO)" || (echo 'usage: make benchmark-biomechanics VIDEO=/path/to/video' >&2; exit 2)
	$(UV) run --project python python -m slopecoach_ml.cli benchmark-biomechanics "$(VIDEO)" --sample-fps "$(or $(SAMPLE_FPS),5)" --input-non-mirrored $(if $(OUTPUT),--output "$(OUTPUT)",) $(if $(DEBUG_DIR),--debug-dir "$(DEBUG_DIR)",)

benchmark-sport-type:
	@test -n "$(VIDEO)" || (echo 'usage: make benchmark-sport-type VIDEO=/path/to/video' >&2; exit 2)
	$(if $(OPENML_PY),PYTHONPATH=python $(OPENML_PY),$(UV) run --project python python) -m slopecoach_ml.cli benchmark-sport-type "$(VIDEO)" --sample-fps "$(or $(SAMPLE_FPS),5)" --sport-type "$(or $(SPORT_TYPE),auto)" --equipment-provider "$(or $(EQUIPMENT_PROVIDER),none)" --input-non-mirrored $(if $(EQUIPMENT_CONFIG),--equipment-config "$(EQUIPMENT_CONFIG)",) $(if $(EQUIPMENT_CHECKPOINT),--equipment-checkpoint "$(EQUIPMENT_CHECKPOINT)",) $(if $(OUTPUT),--output "$(OUTPUT)",) $(if $(DEBUG_DIR),--debug-dir "$(DEBUG_DIR)",)

prepare-biomechanics-dataset:
	@test -n "$(VIDEO_DIR)" || (echo 'usage: make prepare-biomechanics-dataset VIDEO_DIR=benchmarks/ski_bench/videos OUTPUT=artifacts/manifests/biomechanics_real.local.json' >&2; exit 2)
	@test -n "$(OUTPUT)" || (echo 'OUTPUT is required' >&2; exit 2)
	$(UV) run --project python python -m slopecoach_ml.cli prepare-biomechanics-dataset --video-dir "$(VIDEO_DIR)" --output "$(OUTPUT)"

benchmark-biomechanics-dataset:
	@test -n "$(MANIFEST)" || (echo 'MANIFEST is required' >&2; exit 2)
	@test -n "$(OUTPUT)" || (echo 'OUTPUT is required' >&2; exit 2)
	@test -n "$(CLIP_OUTPUT_DIR)" || (echo 'CLIP_OUTPUT_DIR is required' >&2; exit 2)
	$(UV) run --project python python -m slopecoach_ml.cli benchmark-biomechanics-dataset "$(MANIFEST)" --output "$(OUTPUT)" --per-clip-output-dir "$(CLIP_OUTPUT_DIR)" $(if $(DEBUG_DIR),--debug-dir "$(DEBUG_DIR)",)

prepare-target-gt:
	@test -n "$(VIDEO)" || (echo 'usage: make prepare-target-gt VIDEO=/path/to/video TARGET_GT=/path/to/template.json' >&2; exit 2)
	@test -n "$(or $(TARGET_GT),$(GT))" || (echo 'TARGET_GT output path is required' >&2; exit 2)
	$(UV) run --project python python -m slopecoach_ml.cli prepare-target-gt "$(VIDEO)" --sample-fps "$(or $(SAMPLE_FPS),5)" --output "$(or $(TARGET_GT),$(GT))" $(if $(or $(GT_REVIEW_DIR),$(REVIEW_DIR)),--review-dir "$(or $(GT_REVIEW_DIR),$(REVIEW_DIR))",)

openmmlab-macos:
	bash scripts/bootstrap_openmmlab_macos.sh
