# SlopeCoach

SlopeCoach is currently in **Phase A1: AI Research / Reference Foundation**. The code in
`python/` supports algorithm research, deterministic golden fixtures, validation, and
benchmarks. It is not a production mobile application and does not replace the product
architecture.

> **Python reference implementation is not the production Domain Kernel.** The future Rust
> `contracts` crate will be the single source of truth for cross-language contracts, and the
> future Rust Domain Kernel will own production biomechanics and other domain behavior.

## Architecture

The implementation baseline is
[`docs/SlopeCoach - Ski AI Coach.md`](docs/SlopeCoach%20-%20Ski%20AI%20Coach.md), App Local
Architecture V1.1 — Freeze Candidate. Its runtime boundary remains:

```text
Swift / Kotlin native video and model runtime
  -> canonical pose
  -> UniFFI
  -> Rust production Domain Kernel
```

Python is limited to research, training, experiments, model export, benchmark, dataset
tooling, validation, and calibration. All Python data models in this phase are provisional
reference models for research and future Rust parity validation.

## Repository structure

```text
python/slopecoach_ml/       Research/reference package
  video/                    ffprobe-backed video metadata
  quality/                  READY/PARTIAL_ANALYSIS/NOT_ANALYZABLE gate
  detection/, pose/         Provider interfaces, mocks, canonical pose contract
  biomechanics/             2D knee-angle reference slice
  reference/                Provisional ReferenceAnalysisResult pipeline
  benchmark/                SkiBench reference harness
  cli/                      Command-line interface
fixtures/                   Human-verifiable deterministic pose input
benchmarks/ski_bench/       Benchmark documentation and future inputs
tests/                      Contract, quality, biomechanics, benchmark and CLI tests
docs/                       Product architecture
```

`configs/`, `models/`, and `scripts/` may be added only when real phase work requires them;
the project intentionally avoids speculative empty scaffolding and large ML frameworks.

## Python and setup

The project supports Python `>=3.11,<3.14`. Python 3.11–3.13 provide a stable compatibility
range for the broader scientific/ML ecosystem; the host's newer Python 3.14 is deliberately
not frozen as the project baseline. `python/.python-version` pins the validated development
interpreter to Python 3.12. Runtime code uses only the standard library. Development
dependencies are limited to pytest and Ruff and are reproducibly pinned by `python/uv.lock`.

Prerequisites: Git, `uv`, and FFmpeg/ffprobe. Then run:

```bash
make doctor
make python
make lint
make test
```

`make python` creates the uv-managed environment under `python/.venv`. No system Python
packages are modified.

## Coordinate and joint contracts

Every pose entering reference biomechanics is validated as `SourcePixel2D`: top-left origin,
x right, y down, source-frame pixels, canonical upright orientation, and corrected mirroring.
Crop, resize, and letterbox transforms must already have been inverted by a concrete provider.
BBoxes and keypoints must share this coordinate space. Invalid or unknown geometry fails
explicitly.

Canonical joints use `COCO17_V1` identities and lookup APIs; biomechanics does not interpret
array indices. The model-schema adapter is an explicit, currently unconfigured boundary.

## CLI

Run commands from the repository root:

```bash
uv run --project python python -m slopecoach_ml.cli golden
uv run --project python python -m slopecoach_ml.cli inspect-video /path/to/video.mp4
uv run --project python python -m slopecoach_ml.cli benchmark fixtures/golden_pose_001.json
uv run --project python python -m slopecoach_ml.cli benchmark /path/to/video.mp4
```

All commands emit JSON to stdout. Add `--output artifacts/name.json` to write an artifact.
Invalid inputs return a non-zero exit code. Exceptions are reported rather than converted into
fake success.

## Golden fixture

`fixtures/golden_pose_001.json` uses a visually checkable left hip `(200,150)`, knee
`(200,250)`, and ankle `(300,250)`, producing a 90-degree image-plane knee angle. The pipeline
parses and validates the canonical pose, uses joint identity lookup, calculates
`knee_angle_2d`, and serializes a provisional reference result. Tests use an explicit floating
point tolerance and verify deterministic serialization.

## Benchmark and artifacts

`make benchmark` runs the golden benchmark. Real-video benchmark inputs are inspected using
ffprobe and receive a real metadata/quality result. Because no real pose provider is configured,
real-video output explicitly contains `pose_provider: "NOT_CONFIGURED"`; it never claims pose
or biomechanics completion.

Artifacts may be written under:

```text
artifacts/reference_analysis_result.json
artifacts/benchmark.json
artifacts/debug/
```

The debug skeleton renderer is deferred. No expert-labeled ground truth is present, so
`REAL_GT_STATUS = NOT_AVAILABLE` and diagnosis precision, recall, and F1 remain JSON `null`.

## Provider status and deferred work

`DetectorProvider` and `PoseProvider` interfaces plus deterministic mock implementations exist.
No RTMDet, RTMW, YOLO, or other real model/provider is claimed or downloaded.

Not implemented in Phase A1: iOS/Android apps, Swift/Kotlin, UniFFI, Rust mobile integration,
the production Rust Domain Kernel, real tracking/identity/ReID/temporal/turn/diagnosis
pipelines, LLMs, QNN, TensorRT, physical 3D edge angle, live camera coaching, complex UI, or
first-party C++. Mobile integration and the Rust production implementation are explicitly
deferred.
