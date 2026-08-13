# SlopeCoach

SlopeCoach is currently in **Phase A4: Temporal Pose + Turn Foundation**. The code in
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
  temporal/                 Identity-safe interpolation and One Euro stabilization
  turns/                    Image-space proxy, extrema, crossings and provisional segments
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
The reference coordinate adapter records crop, resize, and letterbox and inverts them in reverse
order before the canonical boundary. Orientation and mirror correction remain the responsibility
of a future concrete provider/native preprocessing path. BBoxes and keypoints share the canonical
space, and no coordinate is automatically clamped.

Finite canonical coordinates may lie outside the visible source rectangle. Contract validity and
visibility are deliberately separate: keypoints expose `is_inside_frame`, while bboxes expose
intersection and visible-fraction helpers. Current `knee_angle_2d` conservatively requires its
hip, knee, and ankle evidence to be visible; unrelated out-of-frame joints do not invalidate it.

`FrameGeometry` truthfully represents non-square source pixels. The current image-plane knee
angle only supports pixel aspect ratios approximately equal to 1.0; unsupported non-square
geometry produces `null` plus `NON_SQUARE_PIXEL_ASPECT_RATIO_UNSUPPORTED`, never a misleading
angle.

Canonical joints use `COCO17_V1` identities and lookup APIs; biomechanics does not interpret
array indices. The model-schema adapter is an explicit, currently unconfigured boundary.

## CLI

Run commands from the repository root:

```bash
uv run --project python python -m slopecoach_ml.cli golden
uv run --project python python -m slopecoach_ml.cli inspect-video /path/to/video.mp4
uv run --project python python -m slopecoach_ml.cli benchmark fixtures/golden_pose_001.json
uv run --project python python -m slopecoach_ml.cli benchmark /path/to/video.mp4
uv run --project python python -m slopecoach_ml.cli temporal-golden
uv run --project python python -m slopecoach_ml.cli turn-golden
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

A4 adds two independent algorithm Goldens. `golden_temporal_pose_001.json` has irregular
timestamps, deterministic jitter, a short fillable gap, a long unfilled gap, and an identity
boundary. `golden_turn_signal_001.json` has timestamped positive/negative extrema, exact and
interpolated zero crossings, a weak wiggle, and a hard missing boundary. They validate numerical
engineering behavior—not skiing or product accuracy.

## Temporal pose and provisional turns

The A4 reference flow is `TargetIdentity -> TargetPoseSample -> temporal continuity segment ->
timestamp-weighted short-gap interpolation -> per-joint One Euro -> signed image-space proxy ->
peak/trough/zero-crossing analysis -> provisional segment`. Only sufficiently confident `LOCKED`
identity input is trusted. `UNINITIALIZED`, `SUSPECT`, `LOST`, `RECOVERING`, `AMBIGUOUS`, active
track changes, incompatible geometry, explicit discontinuity, and hard timestamp gaps reset the
temporal boundary. Interpolation never crosses those boundaries.

Raw `PoseFrame` evidence is immutable. Temporal joints separately preserve raw coordinates,
derived interpolation support, stabilized coordinates, support confidence, and `OBSERVED`,
`INTERPOLATED`, or `MISSING` provenance. The default interpolation endpoint gap is 300000 us;
support confidence is the minimum of its endpoints and is never called RTMW confidence. One Euro
defaults are min cutoff 1.0 Hz, beta 0.05, derivative cutoff 1.0 Hz, with frequency derived only
from timestamps.

`signed_lateral_body_proxy` is a dimensionless normalized 2D cross product between the torso axis
and hip-to-lower-body displacement. It uses stabilized bilateral shoulders, hips, knees, and
ankles. Its signs are `POSITIVE_PHASE`/`NEGATIVE_PHASE`, not skiing left/right, and it is never a
physical edge angle. The dependency-free `ReferencePeakDetector` is the default. A lazy SciPy
adapter exists but SciPy remains optional and absent from base dependencies.

```bash
make temporal-golden
make turn-golden
make benchmark-temporal-turns \
  VIDEO=benchmarks/ski_bench/videos/example.mp4 SAMPLE_FPS=5 \
  OUTPUT=artifacts/benchmarks/a4/example_5fps.json \
  DEBUG_DIR=artifacts/debug/a4_temporal_turns/example_5fps
```

The A4 benchmark continues through the A3 identity manager and target-focused RTMW scheduler. It
does not pick `persons[0]`, the largest box, or a manual target. Target Identity annotation remains
deferred and the existing template remains `UNLABELED`; identity accuracy is unknown. Turn GT is
also unavailable, so precision/recall/F1 remain JSON `null`. This blocks product validation, not
deterministic temporal engineering work.

## Real ski-video benchmark and artifacts

`make benchmark` runs the deterministic Golden benchmark. The explicit local A2.2 path samples
decoded timestamps, normalizes supported display rotation, runs the configured RTMDet-m and
RTMW-L providers on CPU, adapts 133 joints to `COCO17_V1`, and records technical pose quality,
raw temporal observations, 2D knee-angle coverage, failure reasons, and measured latency.

```bash
make benchmark-real-pose \
  VIDEO=benchmarks/ski_bench/videos/example.mp4 \
  SAMPLE_FPS=2 \
  OUTPUT=artifacts/benchmarks/a2_2/example.json \
  DEBUG_DIR=artifacts/debug/a2_2_real_ski/example
```

The command requires the isolated OpenMMLab environment and config/checkpoint variables
documented below. Direct CLI use additionally supports `--max-debug-frames 0..10`. One warm-up
sample is excluded from per-frame timing. P95 is `null` below 20 observations. Throughput is
labeled sampled-pipeline throughput and is never presented as realtime or mobile FPS.

Artifacts may be written under:

```text
artifacts/reference_analysis_result.json
artifacts/benchmark.json
artifacts/debug/
```

Representative canonical overlays and a contact sheet can be written under the debug directory;
the harness deliberately does not dump every frame. No expert-labeled ground truth is present,
so `REAL_GT_STATUS = NOT_AVAILABLE` and diagnosis precision, recall, and F1 remain JSON `null`.

Benchmark inputs are explicitly classified as `GOLDEN_FIXTURE`, `SYNTHETIC_METADATA_SMOKE`, or
`REAL_VIDEO`. Synthetic ffprobe integration media must never be reported as a real ski-video
benchmark. Real user videos are local-only inputs. `benchmarks/ski_bench/videos/` and
`fixtures/real/` are
ignored, are never required by pytest or CI, and must not be committed. Generated benchmark JSON,
overlays, OpenMMLab source, environments, and weights stay under ignored `artifacts/`.

## Target safety and CI

Target Identity is not implemented. Zero persons yields no target feature; exactly one person may
use the single-person reference path; more than one person yields `null` target biomechanics and
`MULTIPLE_PERSONS_TARGET_IDENTITY_UNRESOLVED`. Detection ordering, bbox size/position, and
confidence are never treated as Target Identity. Track ID is not Target Identity.

Phase A3 adds a separate, provisional identity-aware research path without changing the A2.2
raw benchmark. Candidate quality conservatively rejects invalid, nearly invisible, implausibly
small, or malformed boxes; it is not target selection and is not an accuracy claim. A
deterministic `REFERENCE_MOTION_IOU` tracker builds ephemeral tracks, while `TargetIdentity`
remains a separate object that can recover onto a different Track ID.

The auto selector evaluates a timestamp-based initialization window using relative foreground
area, center proximity, persistence, motion, detector confidence, and candidate quality. Missing
pose or appearance evidence remains `null` and weights are renormalized. Close winners produce
`AMBIGUOUS`, never a track-ID or detection-order tie break. Identity states are `UNINITIALIZED`,
`LOCKED`, `SUSPECT`, `LOST`, `RECOVERING`, and `AMBIGUOUS`; only a sufficiently confident
`LOCKED` target may produce reference biomechanics.

At the final initialization decision, historical observations remain diagnostic evidence, but
only a viable, non-missing track present on that exact frame may win. Selector history expires by
timestamp after the configured staleness limit. Identity trajectory evidence predicts
`last_center + velocity * dt`; `dt` is capped at the configured horizon and distance is normalized
by the last trusted bbox diagonal. Missing velocity/history yields `null`, not zero similarity.

The default appearance descriptor is a bounded HSV histogram with safely clipped crops. It is
lightweight research evidence, not deep ReID (`DEEP_REID_STATUS = NOT_CONFIGURED`). Pose
scheduling is bounded: initialization/ambiguity may probe at most two candidates, a locked frame
poses only the active target, and uncertain states suppress target biomechanics. Manual user
target correction remains architecturally possible but deferred.

All A3.1 thresholds are centralized, validated, serialized into benchmark provenance, and labeled
research defaults; they are conservative provisional defaults, not scientifically validated
values. Benchmark contract `ski-bench-target-identity-v2` reports tracker terminations as
`terminated_track_count`. Historical v1 `track_fragments` was a misnamed termination count and
must not be reinterpreted as person fragmentation. Fragmentation is `null` until identity GT
supports it.

Target GT contract `target-identity-gt-v1` is timestamp-based, manually authored
`SourcePixel2D` truth with video SHA256 integrity. Generate an empty review template (all labels
remain `UNLABELED`, no model boxes are copied) and then have a human annotate it:

```bash
make prepare-target-gt VIDEO=benchmarks/ski_bench/videos/example.mp4 \
  TARGET_GT=benchmarks/ski_bench/annotations/example.target.json \
  GT_REVIEW_DIR=artifacts/debug/a3_1_gt/example SAMPLE_FPS=5

make benchmark-target-identity VIDEO=benchmarks/ski_bench/videos/example.mp4 \
  TARGET_GT=benchmarks/ski_bench/annotations/example.target.json SAMPLE_FPS=5 \
  OUTPUT=artifacts/benchmarks/a3_1/example.json \
  DEBUG_DIR=artifacts/debug/a3_1_identity/example
```

`PRESENT` requires a human bbox; `ABSENT` requires null; `UNCERTAIN` and `UNLABELED` are excluded.
The research match threshold defaults to IoU 0.5 and is not scientifically validated. Metrics are
`CORRECT_LOCK / PRESENT` for present lock coverage, wrong locks divided by system locks on
evaluable timestamps for wrong-target rate, and correctly handled `PRESENT+ABSENT` divided by all
evaluable frames for frame accuracy. Until reviewed labels exist,
`TARGET_IDENTITY_GT_STATUS = TEMPLATE_CREATED_REQUIRES_HUMAN_LABELING` (or `NOT_AVAILABLE`) and all
accuracy, presence-conditioned, false-lock, fragmentation, and real-recovery metrics stay JSON
`null`.

```bash
make benchmark-target-identity \
  VIDEO=benchmarks/ski_bench/videos/example.mp4 \
  SAMPLE_FPS=2 \
  OUTPUT=artifacts/benchmarks/a3/example.json \
  DEBUG_DIR=artifacts/debug/a3_target_identity/example
```

Pass `TARGET_GT=/path/to/reviewed.target.json` only after manual review. Human annotation images
and model-comparison overlays are separate artifact layers; model output is never GT.

The provisional `ReferenceAnalysisConfig` centrally defines the joint confidence and square-pixel
tolerance. Callers explicitly provide analysis/provider/model provenance, preventing future real
providers from inheriting Golden labels.

Pull requests and pushes to `main` run `.github/workflows/python-ci.yml`: lock validation, Python
3.12 environment sync, formatting, lint, tests, Golden CLI, and Golden benchmark. The workflow has
been validated with locally equivalent commands; that does not claim a GitHub-hosted run passed.

## Provider status and deferred work

Phase A2 defines an optional OpenMMLab research stack isolated from the minimal Python 3.12
environment. The validated candidate is Python 3.11 with PyTorch 2.1.2, TorchVision 0.16.2,
MMCV 2.1.0, MMEngine 0.10.7, MMDetection 3.2.0, and MMPose 1.3.2. Install it separately:

```bash
uv venv artifacts/openmmlab-venv --python 3.11
uv pip install --python artifacts/openmmlab-venv/bin/python "setuptools<81" wheel pip packaging
uv pip install --python artifacts/openmmlab-venv/bin/python \
  -r python/openmmlab-requirements.txt --no-build-isolation
```

On Apple Silicon, MMCV 2.1.0 needs a source build with compiled ops after PyTorch is present.
The validated macOS bootstrap checks out the official v2.1.0 tag under ignored artifacts, builds
with `MMCV_WITH_OPS=1`, and verifies both `mmcv._ext` and `mmcv.ops.nms`. Xcode 26 / Clang 21
requires `-Wno-invalid-specialization` for the pinned PyTorch 2.1.2 headers; this is a compiler
diagnostic compatibility flag and does not patch third-party source. Reproduce the runtime with:

```bash
make openmmlab-macos
```

The bootstrap never downloads checkpoints. It modifies only the ignored isolated environment and
official source checkout under `artifacts/`; it does not use sudo or modify system Python.

The local reference runtime has passed real CPU inference with the exact registered RTMDet-m and
RTMW-L checkpoints: `REAL_POSE_MODEL_STATUS = READY_CPU`. This is research/reference validation,
not a production runtime claim. MPS hardware is available but full model inference remains
`MPS_STATUS = NOT_TESTED` because CPU is the reproducible P0 baseline.

The configured detector is official RTMDet-m 640 COCO/Objects365 person detection; pose is
official RTMW-L Cocktail14 256x192 (`20231122`). Exact sources and nullable, post-download SHA256
fields live in `models/registry.json`. Weights belong under ignored `artifacts/models/` and are
never committed. MMPose code is Apache-2.0; checkpoint license fields remain null where the
official listing does not explicitly provide separate terms.

```bash
make pose-doctor
make pose-smoke IMAGE=/path/to/image.jpg
make benchmark-real-pose VIDEO=/path/to/ski.mov
```

These Make targets explicitly attest that input media is non-mirrored. Direct CLI calls must pass
`--input-non-mirrored`; otherwise they fail with `MIRROR_STATE_UNRESOLVED` rather than guessing.

The real commands require explicit `SLOPECOACH_DETECTOR_CONFIG`,
`SLOPECOACH_DETECTOR_CHECKPOINT`, `SLOPECOACH_POSE_CONFIG`, and
`SLOPECOACH_POSE_CHECKPOINT` paths. Help and package imports never download weights.
Real providers default explicitly to `SLOPECOACH_DEVICE=cpu`. `mps` may be requested, but there is
no silent fallback; hardware availability alone is not treated as validated OpenMMLab support.

RTMW uses the official COCO-WholeBody 133-keypoint metainfo. A single named mapping converts body
identities 0–16 to `COCO17_V1`, with tested left/right semantics. Official MMPose top-down
inference restores predictions to the original input image; the provider therefore treats that
boundary as SourcePixel2D and does not apply a second inverse transform.

Video sampling uses decoded timestamps. Supported display rotations are explicitly normalized
before inference; ambiguous rotations fail. Mirror state must already be known non-mirrored.
Canonical debug overlays are written under ignored `artifacts/debug/`.

Target Identity remains unavailable: detections are observations, not tracks or target IDs.
Multi-person frames retain every pose but return null target biomechanics. Real-pose benchmarks
measure coverage, confidence, coordinate visibility, and latency—not diagnosis accuracy.

Phase A2.2 validates a user-provided local ski clip without making it a repository fixture. The
local OpenMMLab runtime and weights must pass `pose-doctor` before any real inference may be
claimed READY. Per-frame observations are not Track IDs, multi-person frames never select a
target, raw temporal metrics apply no smoothing, and `left_knee_angle_2d_degrees` is image-plane
evidence only—not physical 3D flexion or diagnosis.

Not implemented in Phase A2.2: iOS/Android apps, Swift/Kotlin, UniFFI, Rust mobile integration,
the production Rust Domain Kernel, tracking/Target Identity/ReID, temporal smoothing, turn/diagnosis
pipelines, LLMs, QNN, TensorRT, physical 3D edge angle, live camera coaching, complex UI, or
first-party C++. Mobile integration and the Rust production implementation are explicitly
deferred.
