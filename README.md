# SlopeCoach

SlopeCoach is currently in **Phase A9: Unified AnalysisResult v1 and ProductReport v1**. The code in
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
  sport_type/               Auto-first sport contracts, fusion, providers, cues and routing gate
    calibration/            Manual GT, source aggregation, Platt calibration and diagnostic LLR fusion
  diagnosis/                Turn-window, multi-frame provisional research rule engine
  scoring/                  Structure-only nullable scorecard and diagnosis-dimension mapping
  coach/                    Top-two issues, controlled drills and deterministic zh-CN templates
  analysis_result/          Unified machine result and pure app-facing report projection
  product/                  Explicit-user MVP request boundary; no automatic SportType inference
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
uv run --project python python -m slopecoach_ml.cli sport-type-golden
uv run --project python python -m slopecoach_ml.cli scorecard-golden
uv run --project python python -m slopecoach_ml.cli coach-golden
uv run --project python python -m slopecoach_ml.cli benchmark-scoring-coach /path/to/a7.json
uv run --project python python -m slopecoach_ml.cli analysis-result-golden
uv run --project python python -m slopecoach_ml.cli benchmark-analysis-result /path/to/a7.json
uv run --project python python -m slopecoach_ml.cli analyze-video /path/to/video.mp4 \
  --sport-type SKI --input-non-mirrored --output artifacts/local/mvp_analysis.json
uv run --project python python -m slopecoach_ml.cli sport-visual-doctor \
  --visual-checkpoint artifacts/models/a6_2/openai_clip/ViT-B-32.pt
```

All commands emit JSON to stdout. Add `--output artifacts/name.json` to write an artifact.
Invalid inputs return a non-zero exit code. Exceptions are reported rather than converted into
fake success.

## B3 MVP analyze-video product pipeline

The product `analyze-video` request requires an explicit user selection. The only accepted
values are `SKI` (双板) and `SNOWBOARD` (单板); omission, `AUTO`, and `UNKNOWN` are command errors.
This user selection is product truth and enters the analysis directly with:

```text
effective_sport_type = SKI | SNOWBOARD
effective_source = USER
resolution_status = RESOLVED_USER
```

`analyze-video` is the single MVP product entry point. It performs one video/model pass and then
orchestrates existing reference modules without recomputation:

```text
video + explicit SportType + optional target seed
  -> A3 Target Identity
  -> A4 Temporal Pose / Turn Segmentation
  -> A5 Biomechanics
  -> A7 Diagnosis
  -> A8 nullable ScoreCard / deterministic Coach
  -> A9 AnalysisResult v1 / ProductReport v1
```

The command does not perform a separate warmup inference pass or reopen the video for downstream
stages. A7–A9 consume the facts produced by the single A5 pass. The resulting JSON artifact contains
the compact `analysis_result`, final `product_report`, model/runtime provenance, and the explicit
SportType provenance. Its ProductReport and AnalysisResult have independent deterministic semantic
SHA256 fingerprints; runtime timing is outside those semantic fingerprints.

If trusted target evidence and qualified turns are available, the availability Quality Gate may
be `READY`. A safe target with no qualified turns is `PARTIAL_ANALYSIS` with
`NO_QUALIFIED_TURNS`; Diagnosis, ScoreCard, and Coach remain unavailable rather than being
fabricated. No trusted target pose is `NOT_ANALYZABLE` with `TARGET_IDENTITY_UNCERTAIN`, and
product recommendations are suppressed. Numeric ScoreCard values remain `null`.

The product path does **not** construct the A6 Equipment or CLIP providers, run
`ReferenceSportTypeFusion` or calibrated fusion, load the CLIP checkpoint, or perform extra
equipment inference. Its artifact records `automatic_sport_type_research.executed=false` and
`status=DEFERRED_RESEARCH_ONLY`.

`benchmark-sport-type` remains a separate research command and retains the pre-B2 A6 AUTO,
RTMDet equipment, CLIP visual evidence, original reference fusion, calibration diagnostics,
Goldens, and benchmarks. Automatic SportType is research-only and deferred from the MVP product
path. The proposed B2 equipment-first/visual-fallback hierarchy is not implemented and remains
deferred. The Python `analyze-video` artifact is a provisional MVP/reference artifact, not the
future Rust production Domain Kernel contract.

## A9 end-to-end product contract

A9 introduces a stable research/reference boundary without introducing a new AI capability:

```text
validated upstream truth -> AnalysisResult v1 -> ProductReport v1
```

`AnalysisResult` is the machine semantic truth envelope. It carries a fixed ordered registry of
eight compact sections (`SOURCE`, `TARGET_IDENTITY`, `SPORT_TYPE`, `TURNS`, `BIOMECHANICS`,
`DIAGNOSIS`, `SCORECARD`, and `COACH`), explicit availability, blockers, limitations, Ground Truth
status, semantic provenance, and a canonical content SHA. Missing legacy metadata remains missing;
source identity is never inferred from a filename and unavailable sections never receive fake
payloads.

`ProductReport` is a pure projection of `AnalysisResult`. It cannot create or modify Diagnosis,
scores, Top Issues, drills, evidence, or coaching copy. It reuses the validated ScoreCard and
CoachReport exactly and is fingerprinted independently while binding to its source
`analysis_result_sha256`. Runtime timing, local paths, and export timestamps are outside both
semantic fingerprints, avoiding circular identity.

Top-level statuses are `READY`, `PARTIAL_ANALYSIS`, and `NOT_ANALYZABLE`. `READY` means only that
the currently implemented research output has enough trusted evidence; **READY does not mean
scientific accuracy, Ground Truth validation, or production readiness**. The current
`ski_test_001` A7 artifact remains `PARTIAL_ANALYSIS` because it contains zero qualified turns and
does not embed Source, Target Identity, or compact Biomechanics summaries. The artifact-only A9
benchmark performs deterministic downstream assembly and never reruns RTMDet, RTMW, equipment,
CLIP, tracking, turn, or biomechanics models.

All five ScoreCard dimensions remain structurally present. Numeric scoring is disabled, every
dimension score and the overall score remain JSON `null`, and no-trigger output is never promoted
to `GOOD_FORM`. Target Identity, SportType, turn segmentation, Diagnosis, and score Ground Truth
are not created by A9. Normal CI remains video-free, checkpoint-free, Torch/OpenMMLab/CLIP-free,
network-free, and LLM-free.

```bash
make analysis-result-golden
make benchmark-analysis-result \
  ARTIFACT=artifacts/benchmarks/a7/ski_test_001_artifact_only.json \
  OUTPUT=artifacts/benchmarks/a9/ski_test_001_analysis_result.json
```

Python validates these provisional contracts and enables deterministic parity fixtures. A future
Rust production Domain Kernel should emit equivalent `AnalysisResult v1`; native apps should
primarily consume `ProductReport v1`. Rust and mobile implementation remain deferred.

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

The A4 benchmark continues through the A3 identity manager and target-focused RTMW scheduler. Its
default path does not pick `persons[0]`, the largest box, or a manual target. Target Identity
annotation remains deferred and the existing template remains `UNLABELED`; identity accuracy is
unknown. Turn GT is also unavailable, so precision/recall/F1 remain JSON `null`. This blocks
product validation, not deterministic temporal engineering work.

A4.1 makes both extrema generation and acceptance local to a `ValidSignalRun`: a maximal,
strictly timestamp-increasing sequence with one temporal segment, finite signal values, and
sufficient confidence. Missing/low-confidence values and segment changes are hard boundaries;
later runs are never merged even if they reuse the same temporal segment ID. Peak separation,
same-sign replacement, zero crossings, and segment boundaries therefore cannot suppress or bridge
across a signal gap. Exact/near-zero plateaus emit one crossing only when they connect opposite
known signs; its timestamp is the integer midpoint between the first and last zero timestamps.

New artifacts use `ski-bench-temporal-turns-v2`. They report signal sufficiency separately from
qualified extrema, so a continuous flat/low-amplitude signal is
`EXECUTED_NO_QUALIFIED_TURN_CANDIDATES`, not automatically insufficient pose evidence. The default
backend remains dependency-free `ReferencePeakDetector`; `SCIPY_USAGE = NOT_USED`.

V2 stability metrics estimate each frame's raw symmetric shoulder-center to ankle-center distance,
take its median per temporal segment, and use that single segment scale for both raw and stabilized
differences. Segments without a truthful scale are excluded and counted. These normalization
semantics differ from v1's historical asymmetric per-frame denominator, so v1 and v2 stability
numbers are not directly comparable and old v1 artifacts must not be rewritten.

## A5 temporal biomechanics facts

A5 adds the provisional `temporal-biomechanics-v1` research contract after trusted stabilized
target pose. Its deterministic registry contains 14 frame facts grouped as stance, balance,
symmetry, timing, and edge-control proxies. These families are organizational labels—not scores,
diagnoses, or coaching conclusions. `FIXED_ML_FEATURE_VECTOR_STATUS = NOT_FROZEN`.

The flow is `StabilizedPoseSample -> frame facts -> temporal-segment aggregates -> A4.1-gated turn
facts`. Every fact records unit, required joints, support confidence, observed/interpolated counts,
status, and limitations. Missing, low-confidence, out-of-frame, non-square-pixel, or degenerate
evidence produces JSON `null`, never zero. Normalized distances reuse the A4.1 constant segment
median raw shoulder-center-to-ankle-center body scale. Derivatives use actual timestamps and never
bridge missing facts or temporal segments.

A5.1 introduces `temporal-biomechanics-v2` and `ski-bench-biomechanics-v2`. Historical v1
artifacts retain the original A5 semantics and must not be rewritten. The ordered 30-definition
registry is identified by `biomechanics-feature-schema-v1` and SHA256
`2777c3fbf7513e7537122f897f1901e61baf7eeddcee927937decb7476953048`. This schema provenance
does not freeze an ML input vector: `FIXED_ML_FEATURE_VECTOR_STATUS = NOT_FROZEN`.

V2 enforces `AVAILABLE` if and only if a fact has a finite, non-null, non-bool value; unavailable
facts must be null. Numeric settings reject booleans and non-finite values. A complete turn window
requires ordered non-null start/apex/end boundaries, positive duration, and a signal-run ID. Only
complete windows expose minimum-knee timing facts. A `PARTIAL` turn may still expose matched
apex-local evidence and whichever start/end boundary exists; absent-boundary facts are
`TURN_BOUNDARY_UNAVAILABLE`. Apex and boundary matching use separate tolerances, stay within every
known boundary and signal run, and break equal-distance ties toward the earlier timestamp.
Single-source turn facts preserve source joint evidence; two-source deltas use minimum confidence
and unique-joint support without double-counting observations. Turn-segment-derived duration and
peak proxy do not claim joint support confidence.

## A5.2 real-motion dataset robustness

A5.2 keeps the exact `biomechanics-feature-schema-v1` registry and formulas unchanged. It adds the
local-only `biomechanics-real-dataset-v1` manifest and
`ski-bench-biomechanics-dataset-v1` report to measure evidence availability across independent
`source_video_id` values. Temporal slices from one source never inflate the independent-video
count. Newly discovered videos default disabled; only the already established `ski_test_001` clip
may be enabled automatically by the local template-preparation policy.

Dataset output reports frame macro/clip-weighted and micro/frame-weighted coverage separately,
temporal facts per eligible segment, and turn facts per eligible turn. Zero-trusted-target clips
are classified as upstream non-evaluable and excluded from feature robustness denominators. The
failure matrix, mathematical domain checks, identity lock, target bbox ratios, candidate density,
joint visibility, and motion signal summaries are engineering covariates—not labels or accuracy.

```bash
make prepare-biomechanics-dataset \
  VIDEO_DIR=benchmarks/ski_bench/videos \
  OUTPUT=artifacts/manifests/biomechanics_real.local.json

make benchmark-biomechanics-dataset \
  MANIFEST=artifacts/manifests/biomechanics_real.local.json \
  OUTPUT=artifacts/benchmarks/a5_2/dataset_report.json \
  CLIP_OUTPUT_DIR=artifacts/benchmarks/a5_2/clips \
  DEBUG_DIR=artifacts/debug/a5_2_biomechanics
```

Enabled clips execute sequentially. Per-clip failures remain explicit instead of being silently
discarded. The default threshold is five independent source videos; it is a configurable project
engineering-evidence threshold, not statistical proof. Ten or more varied clips are preferred
before feature-retention research decisions.
`HIGH_FEATURE_COVERAGE_DOES_NOT_IMPLY_ACCURACY`: coverage cannot validate physical correctness,
biomechanics accuracy, diagnosis, or coaching. The ML feature vector remains `NOT_FROZEN`, and no
feature is automatically deleted.

All measurements remain image-space 2D and camera/viewpoint dependent. No output is a physical
center of mass, physical edge angle, diagnosis, technique classification, or score. Run:

```bash
make biomechanics-golden
make benchmark-biomechanics \
  VIDEO=benchmarks/ski_bench/videos/example.mp4 SAMPLE_FPS=5 \
  OUTPUT=artifacts/benchmarks/a5/example_5fps.json \
  DEBUG_DIR=artifacts/debug/a5_biomechanics/example_5fps \
  OVERLAY_VIDEO=artifacts/local/pose_overlay.mp4
```

The optional `OVERLAY_VIDEO` / `--overlay-video` output is a
`SAMPLED_DEBUG_VIDEO`: its frame rate is the requested benchmark `--sample-fps`, not the source
video frame rate. It reuses the sampled JPEG frames and existing target, temporal, turn-proxy, and
2D biomechanics evidence collected during the same benchmark run; it never rereads the video or
reruns RTMDet/RTMW. Missing target poses, joints, turn segments, and feature facts stay visibly
missing rather than being inferred. The overlay is a research/debug visualization only—not a
physical edge-angle, carving, diagnosis, or production analysis output.
It does not establish scientific accuracy or Ground Truth validation.

For a manual target, the overlay also distinguishes direct current-frame RTMW `RAW` pose from the
identity-safe `STABILIZED` pose. A thin RAW skeleton may remain visible while the HUD says
`analysis=GATED`; this is observability only and does not create a temporal segment, biomechanics
fact, turn, diagnosis, or score. Missing or low-confidence RAW joints are not drawn. The current
associated target bbox remains visible while its detection is observable, including `SUSPECT`
frames. `LOST` without a current association fabricates neither bbox nor pose. The emitted
`raw_target_pose_debug` counts are non-semantic debug metadata; thresholds, feature registries, and
analysis fingerprints are unchanged.

Target Identity keeps two deliberately separate memories. Trusted `last_bbox/last_seen_us` and
the HSV appearance gallery advance only after identity evidence meets the existing lock threshold.
Observed continuity records the latest physically associated active-track bbox/timestamp/velocity,
including `SUSPECT` frames, solely for motion proposals and for timing an actual association loss.
Thus `OBSERVED != TRUSTED`: a fresh observation can keep the short missing timer current without
creating a temporal segment or biomechanics. Debug observations/HUD expose a non-carried-forward
`latest_identity_match_score` and `last_observed_age_us`; `null` means no match was calculated on
that frame.

Direct CLI equivalent:

```bash
python -m slopecoach_ml.cli benchmark-biomechanics "$VIDEO" \
  --sample-fps 5 --input-non-mirrored \
  --output artifacts/local/a5.json \
  --debug-dir artifacts/local/a5_debug \
  --overlay-video artifacts/local/pose_overlay.mp4
```

For a crowded video, `benchmark-biomechanics` optionally accepts a manual **initial target seed**:

```bash
VIDEO="/Users/felix/Desktop/1794bbe778d0e9f6f34e2d2bd010f2e7.mp4"

SLOPECOACH_DEVICE=cpu \
SLOPECOACH_DETECTOR_CONFIG="$PWD/artifacts/openmmlab-src/mmpose-1.3.2/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py" \
SLOPECOACH_DETECTOR_CHECKPOINT="$PWD/artifacts/models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth" \
SLOPECOACH_POSE_CONFIG="$PWD/artifacts/openmmlab-src/mmpose-1.3.2/projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmw-l_8xb1024-270e_cocktail14-256x192.py" \
SLOPECOACH_POSE_CHECKPOINT="$PWD/artifacts/models/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth" \
PYTHONPATH="$PWD/python" \
"$PWD/artifacts/openmmlab-venv/bin/python" -m slopecoach_ml.cli benchmark-biomechanics \
  "$VIDEO" \
  --sample-fps 5 \
  --input-non-mirrored \
  --target-seed-time 1.6 \
  --target-seed-point 820,460 \
  --output "$PWD/artifacts/local/a5_manual_target.json" \
  --debug-dir "$PWD/artifacts/local/a5_manual_target_debug" \
  --overlay-video "$PWD/artifacts/local/pose_overlay_manual_target.mp4"
```

The point is an upright, non-mirrored source-frame pixel coordinate (`SourcePixel2D`: top-left
origin, x right, y down). Both seed flags are required together. The click selects one viable
candidate only at the nearest eligible sampled initialization frame; it is not a permanent Track
ID pin. Subsequent `LOCKED`, `SUSPECT`, `LOST`, `RECOVERING`, and `AMBIGUOUS` behavior remains
automatic through the existing tracker and `TargetIdentityManager`, including recovery onto a new
Track ID. The seed is not Ground Truth and does not validate identity accuracy. An explicit seed
whose frame, point, or containing viable person cannot be resolved fails closed and never falls
back to another automatically selected person. Omitting both flags preserves the existing AUTO
path and output shape.

## A6.3 SportType calibration research

A6.3 does not change `ReferenceSportTypeFusion`, its weights, or its thresholds. RAW_V1 remains
the effective AUTO routing path and `CALIBRATED_FUSION_CONTROLS_ROUTING = false`. The separate
CALIBRATED_V1 path is a research diagnostic that converts each provider's source-level raw
direction (`snowboard_support - ski_support`) through a provider-specific scalar Platt model,
removes the training prior into estimated calibrated log-likelihood-ratio space, averages
same-kind provider LLRs, and sums independent EQUIPMENT and VISUAL_CLASSIFIER kind LLRs.

Frame observations and temporal subclips are correlated; each `source_video_id` contributes at
most one sample per provider channel. Calibration labels are local manual annotations under
`sport-type-gt-v1`. Only `USER_MANUAL` SKI/SNOWBOARD labels with `CONFIRMED` intended-target
confirmation and matching video SHA are eligible. Generated templates are always `UNLABELED` and
`UNCONFIRMED`; filenames, detector output, CLIP output, and AUTO decisions never prefill GT.

```bash
make prepare-sport-type-gt \
  MANIFEST=artifacts/manifests/biomechanics_real.local.json \
  OUTPUT_DIR=artifacts/annotations/sport_type

make build-sport-calibration-dataset \
  ARTIFACTS=artifacts/benchmarks/a6_2/ski_test_001_5fps.json \
  ANNOTATIONS_DIR=artifacts/annotations/sport_type \
  OUTPUT=artifacts/benchmarks/a6_3/calibration_dataset.json

make fit-sport-evidence-calibration \
  DATASET=artifacts/benchmarks/a6_3/calibration_dataset.json \
  OUTPUT=artifacts/calibration/sport_evidence_a6_3.json

make sport-calibration-golden
```

The research fit minimum is 10 confirmed independent SKI sources and 10 confirmed independent
SNOWBOARD sources; 20 per class is preferred before fusion decisions. The current local corpus
contains one source and no human SportType label, so its honest status is
`INSUFFICIENT_LABELED_SPORT_TYPE_GT`: no real calibration coefficients or calibrated probability
are produced. A future corpus should vary view, subject size, crowding, visibility, occlusion, and
environment. This is an engineering evidence goal, not proof of generalization or product
accuracy. Product threshold selection requires independent held-out GT.

Calibration artifacts are model/checkpoint/provider specific; visual channels additionally bind
the visual prompt SHA. Model, checkpoint, prompt, preprocessing, or raw-support semantic drift
requires recalibration. Artifacts and calibrated fusion remain research-only. User SportType is
authoritative; production SportType fusion remains deferred to Rust/mobile implementation.

## A7 provisional diagnosis foundation

A7 adds a deterministic, evidence-backed research diagnosis layer after effective SportType,
qualified complete turns, and A5 biomechanics facts. It consumes upstream facts without changing
target identity, SportType, turn segmentation, or biomechanics formulas. UNKNOWN SportType blocks
sport-specific diagnosis. USER-selected SportType permits research routing but is not ground
truth; unvalidated RAW_V1 AUTO resolution is explicitly limited. A6.3 calibrated diagnostics do
not control A7 routing.

The registry contains exactly three provisional image-space rules:

- `LIMITED_KNEE_FLEXION_MODULATION_2D`
- `BILATERAL_KNEE_ASYMMETRY_2D`
- `KNEE_FLEXION_TIMING_OFFSET_2D`

Every rule requires a complete qualified turn and at least five AVAILABLE samples with 0.60
coverage inside the exact turn, temporal segment, and signal run. Defaults—12 degrees knee-angle
range, 10 degrees median bilateral difference, and 0.20 absolute phase offset—are engineering
research thresholds, not coach-certified or product-validated semantics. Missing evidence is
`NOT_EVALUABLE`; sufficient evidence below threshold is `NOT_TRIGGERED`. No trigger does not mean
GOOD_FORM.

```bash
make diagnosis-golden
make benchmark-diagnosis \
  ARTIFACT=artifacts/benchmarks/a5_1/ski_test_001_5fps.json \
  SPORT_TYPE=ski OUTPUT=artifacts/benchmarks/a7/ski_test_001.json
```

The artifact benchmark is model-free and refuses inputs lacking persisted turn and biomechanics
facts. Diagnosis severity and confidence are JSON `null`; neither coverage nor threshold margin is
renamed as confidence. Diagnosis GT and Turn GT are unavailable, so precision, recall, F1, and
agreement remain null. `DIAGNOSIS_SEVERITY_STATUS` is the canonical status field;
`SEVERITY_STATUS` remains a deprecated matching alias for A7 artifact compatibility. There is no
physical COM, pressure, edge angle, calibrated scoring, validated drills, free-form coaching
text, XGBoost, or real diagnosis-accuracy claim. Python remains research/reference only; the
production Rust/mobile implementation is deferred.

## A8 structure-only scorecard and controlled coach

A8 stabilizes the research/reference path `DiagnosisResult -> ScoreCard -> Top 1–2 Issues ->
CoachContext -> Controlled Drill Library -> deterministic zh-CN Template Coach -> CoachReport`.
It does not read pose, pixels, frame facts, or raw biomechanics after DiagnosisResult exists and
never recalculates the A7 thresholds. This is a downstream reference bridge for future Rust/mobile
parity—not a production Domain Kernel.

The scorecard has exactly `BALANCE`, `EDGE_CONTROL`, `STANCE`, `SYMMETRY`, and `TIMING` dimensions.
The three current rules map only to Stance, Symmetry, and Timing. Balance and Edge Control remain
`NOT_IMPLEMENTED`; evidence is not copied across dimensions. Every dimension `score_value`, scale,
and the overall score are deliberately JSON `null`, with numeric scoring disabled. Diagnosis GT,
Turn GT, and Score GT do not exist, so numeric calibration is deferred. A recurrence ratio is only
an observed diagnostic statistic: it is not severity, confidence, probability, or a score.

`NO_PROVISIONAL_ISSUE_DETECTED` means only that currently implemented, evaluable research rules did
not trigger. It is not `GOOD_FORM`, a skill level, or proof that technique is correct. Missing
evidence remains `NOT_EVALUABLE`. Top issues use deterministic evidence recurrence (ratio, count,
then A7 registry order), never threshold distance. Severity and confidence remain null.

The controlled library contains exactly three low-risk research practice focuses for the three A7
signals. The deterministic `zh-CN` template coach makes provisional 2D evidence and safety limits
explicit; it does not claim that a drill fixes an error. Drill effectiveness and coach preference
have not been validated. There is no LLM, network call, XGBoost, physical COM, physical pressure,
physical edge angle, 3D inference, personalization, or skill classification.

```bash
make scorecard-golden
make coach-golden
make benchmark-scoring-coach \
  ARTIFACT=artifacts/benchmarks/a7/example.json \
  OUTPUT=artifacts/benchmarks/a8/example.json
```

The A8 benchmark is artifact-only and accepts a compatible `ski-bench-diagnosis-v1` artifact. It
does not rerun video or any model and refuses to reconstruct diagnosis from upstream biomechanics.
Ground-truth score metrics remain null. A future LLM may only rephrase, translate, or summarize a
controlled CoachContext; it may not create diagnosis, change scores or priority facts, or invent
evidence. Mobile integration and the production Rust implementation remain deferred.

## A8.1 downstream provenance and contract invariants

A8.1 adds no coaching capability. It binds every downstream ScoreCard and Coach artifact to the
exact Diagnosis semantics that produced it. A Diagnosis registry SHA alone is insufficient because
research runs may use an explicit custom `DiagnosisRuleConfig`. The additive
`diagnosis-semantics-provenance-v1` block therefore records the Diagnosis contract and registry SHA,
the complete config, a config SHA, and a combined semantic SHA. Persisted artifact ingestion fails
closed on missing provenance, an unsupported registry, inconsistent configs, invalid fingerprints,
or contradictory Diagnosis entries and rule evaluations.

New A7 artifacts carry explicit semantic provenance. Existing `ski-bench-diagnosis-v1` artifacts
may be accepted as `LEGACY_EXPLICIT_FIELDS_DERIVED` only when their top-level registry/config and
`diagnosis_result.config` are all present and consistent; current Python defaults are never used to
silently relabel old artifacts. In-memory typed `DiagnosisResult` values bind their actual config
to the current process registry.

The complete `IssuePriorityPolicy`—including `max_top_issues`—and its deterministic SHA now survive
ScoreCard-to-Coach reporting. All deterministic zh-CN headlines, evidence formatting, issue copy,
and controlled warning semantics live in one language policy. Expanding this semantic coverage
intentionally changes `coach_template_registry_sha256` from
`4b7af86d4364b516cca265e6ff23b3f0f2704b80393f531157ba518a1fd7d549` to
`b7ffb27da4f2179f45cf83bf08175593e4233362ebdef6f3202808d2138aa202`; runtime counts and timing are
excluded from that fingerprint.

Contract constructors now reject numeric score or scale leakage, invalid dimension arithmetic,
invalid recurrence ratios, fabricated issue severity/confidence, incompatible practice plans, and
unsafe drill metadata. Numeric dimension and overall scores remain intentionally null because
Diagnosis, Turn, and Score ground truth are unavailable.

```bash
make a8-provenance-golden
make benchmark-scoring-coach \
  ARTIFACT=artifacts/benchmarks/a7/ski_test_001_artifact_only.json \
  OUTPUT=artifacts/benchmarks/a8_1/ski_test_001_artifact_only.json
```

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
correction remains deferred. The optional A5 manual target seed is initialization provenance only,
not correction, unconditional Track ID pinning, or Ground Truth. After initialization, a viable
active target track receives first claim on its best geometric association only when the configured
preferred-association threshold is met. Preferred ownership is not unconditional: it may override
a stronger track only when the two pre-assignment track histories are duplicate-like under the
configured bbox IoU, normalized center-distance, scale-ratio, and available velocity-direction
checks. A stronger non-duplicate competitor keeps normal global priority, so the target may become
`MISSING`, `SUSPECT`, or `LOST` instead of silently attaching to another person. Lost is safer than
Wrong Target. These are conservative research defaults, not validated tracking-accuracy claims.
The benchmark reports preferred conflicts, accepted duplicate overrides, and rejected
non-duplicate overrides without presenting any of them as Ground Truth wrong-target metrics.
The provisional duplicate-like gate uses only the two tracks' pre-assignment state: prior bbox
IoU must be at least `0.55`, center distance divided by the larger prior bbox diagonal must be at
most `0.25`, and prior bbox area ratio must be at most `2.0`. When both tracks have measured
non-zero velocity, their velocity cosine must be at least `0.50`. All four defaults live in
`TrackingConfig`, are validated, and are serialized into benchmark provenance.
`preferred_association_conflict_count` counts qualifying preferred pairs that face a stronger
competitor for the same detection. `preferred_association_override_count` counts the subset that
actually win after passing the duplicate-like gate, while
`preferred_association_rejected_non_duplicate_count` counts pairs denied that privilege by the
gate. These are deterministic research diagnostics, not wrong-target accuracy metrics.

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

## A6 SportType foundation

A6 adds provisional `sport-type-v1` contracts and deterministic, dependency-free
`ReferenceSportTypeFusion` for `SKI`, `SNOWBOARD`, and unresolved `UNKNOWN`. The policy is Auto
First, Ask on Ambiguity, User Override Wins. `EQUIPMENT` and `VISUAL_CLASSIFIER` are primary
evidence; future calibrated `POSE_GEOMETRY` and `TEMPORAL_MOTION` evidence is secondary. Auto
resolution requires active primary evidence, sufficient engineering support, and a sufficient
margin. Ties, conflicts, weak evidence, and secondary-only input remain `UNKNOWN`. Support values
are engineering values, not calibrated probabilities.

The current RTMDet provider remains person-only and is never interpreted as equipment evidence.
No dedicated equipment or visual sport classifier is configured, so current real AUTO benchmarks
honestly report both providers `NOT_CONFIGURED` and normally recommend user confirmation. A user
selection controls effective routing while retaining the auto decision and disagreement. User
input is not SportType ground truth.

Four existing A5 aggregate facts are exposed as viewpoint-dependent, uncalibrated measurements
for future calibration. They preserve nulls and have `contributes_to_auto_fusion=false`; there are
no pose-width/body-side classification rules and the 30-entry biomechanics schema is unchanged.

```bash
make sport-type-golden
make benchmark-sport-type \
  VIDEO=benchmarks/ski_bench/videos/ski_test_001.mp4 SAMPLE_FPS=5 SPORT_TYPE=auto \
  OUTPUT=artifacts/benchmarks/a6/ski_test_001_5fps.json \
  DEBUG_DIR=artifacts/debug/a6_sport_type/ski_test_001_5fps
```

The `ski-bench-sport-type-v1` benchmark composes the existing single-pass A5.1 pipeline. It does
not rerun RTMDet/RTMW or infer from filenames. Generic pose and image-space biomechanics remain
available when SportType is `UNKNOWN`; only future sport-specific analysis is gated.
`SPORT_TYPE_GT_STATUS = NOT_AVAILABLE`, so accuracy, precision, and recall remain JSON `null`.

### A6.1 real primary equipment evidence

A6.1 adds the first real primary SportType evidence provider. The existing
`rtmdet-m-640-coco-obj365-person` remains an unchanged person-only detector. A separate official
full-COCO `rtmdet_tiny_8xb32-300e_coco` provider detects only the `skis` and `snowboard` classes
for SportType evidence. Their indices are resolved dynamically from `model.dataset_meta["classes"]`;
no numeric COCO class IDs are hard-coded.

Only `LOCKED` target frames are eligible. The single-pass identity observer retains those target
contexts, deterministically selects at most 12 distinct timestamps, constructs a wider/lower
source-frame crop, maps detections back to `SourcePixel2D`, and accepts equipment whose center is
inside a target-relative association zone. That association is geometric only and has no
equipment-to-person GT. Detector scores are engineering support—not calibrated SportType
probabilities. Zero detections emit no fake observation, and `UNKNOWN` remains valid.

The provider is optional and never auto-enabled merely because weights exist. Normal Python 3.12
CI remains OpenMMLab-, video-, checkpoint-, and network-free. Validate and run it explicitly in
the isolated runtime:

```bash
PYTHONPATH=python artifacts/openmmlab-venv/bin/python -m slopecoach_ml.cli \
  sport-equipment-doctor \
  --equipment-config /path/to/rtmdet_tiny_8xb32-300e_coco.py \
  --equipment-checkpoint /path/to/rtmdet_tiny_checkpoint.pth

PYTHONPATH=python artifacts/openmmlab-venv/bin/python -m slopecoach_ml.cli \
  benchmark-sport-type benchmarks/ski_bench/videos/ski_test_001.mp4 \
  --sample-fps 5 --input-non-mirrored --sport-type auto \
  --equipment-provider rtmdet-coco \
  --equipment-config /path/to/rtmdet_tiny_8xb32-300e_coco.py \
  --equipment-checkpoint /path/to/rtmdet_tiny_checkpoint.pth \
  --output artifacts/benchmarks/a6_1/ski_test_001_5fps.json \
  --debug-dir artifacts/debug/a6_1_sport_type/ski_test_001_5fps
```

The v2 benchmark aggregates every provider of the same evidence kind, keeps failures separately,
and conditionally reports provider limitations. Equipment evidence always flows through the
existing `ReferenceSportTypeFusion`; it never directly assigns SKI/SNOWBOARD. User override still
wins while retaining the auto decision. SportType GT, diagnosis, scoring, and product accuracy
validation remain unavailable.

### A6.2 visual SportType evidence

A6.2 adds a second independent primary evidence kind without changing `sport-type-v1` fusion
weights or thresholds:

```text
LOCKED target -> RTMDet equipment ---------+
              -> CLIP target-crop visual --+-> ReferenceSportTypeFusion
                                               -> SKI / SNOWBOARD / UNKNOWN
```

The visual provider is the official OpenAI CLIP `ViT-B/32` zero-shot baseline pinned to commit
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. It uses a versioned, fixed English
`SKI`/`SNOWBOARD`/`NEUTRAL` prompt taxonomy. Neutral support remains diagnostic evidence; it is
not a new `SportType`. CLIP supports are normalized zero-shot engineering supports, not calibrated
probabilities, and require in-domain SportType validation before any production claim. Results
depend on the fixed prompt taxonomy.

CLIP, Torch, TorchVision, Pillow, and ftfy remain absent from the normal Python 3.12 dependency
path. Install the official pinned source only in the ignored Python 3.11 OpenMMLab runtime, then
explicitly prepare the local checkpoint. Neither import nor benchmark silently downloads it:

```bash
git clone https://github.com/openai/CLIP.git artifacts/openai-clip-src
git -C artifacts/openai-clip-src checkout d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
artifacts/openmmlab-venv/bin/python -m pip install --no-deps artifacts/openai-clip-src
artifacts/openmmlab-venv/bin/python -m pip install 'ftfy>=6,<7' 'regex>=2024,<2027'

make prepare-visual-sport-model OPENML_PY=artifacts/openmmlab-venv/bin/python
make sport-visual-doctor OPENML_PY=artifacts/openmmlab-venv/bin/python \
  VISUAL_CHECKPOINT=artifacts/models/a6_2/openai_clip/ViT-B-32.pt
```

Run the v4 benchmark with both providers explicitly enabled:

```bash
make benchmark-sport-type \
  OPENML_PY=artifacts/openmmlab-venv/bin/python \
  VIDEO=benchmarks/ski_bench/videos/ski_test_001.mp4 SAMPLE_FPS=5 SPORT_TYPE=auto \
  EQUIPMENT_PROVIDER=rtmdet-coco EQUIPMENT_CONFIG=/path/to/rtmdet_tiny_config.py \
  EQUIPMENT_CHECKPOINT=artifacts/models/a6_1/rtmdet_tiny_coco/checkpoint.pth \
  VISUAL_PROVIDER=openai-clip \
  VISUAL_CHECKPOINT=artifacts/models/a6_2/openai_clip/ViT-B-32.pt \
  OUTPUT=artifacts/benchmarks/a6_2/ski_test_001_5fps.json \
  DEBUG_DIR=artifacts/debug/a6_2_sport_type/ski_test_001_5fps
```

`ski-bench-sport-type-v4` preserves the v3 equipment-only, visual-only, and combined diagnostic auto
decisions from one model pass. Only the combined result plus an optional authoritative user
override controls routing. Same-frame evidence IDs are deterministic and provider-qualified.
Dataset-level LOCKED context counts are unique; per-provider selection and inference counts are
separate. SportType GT remains unavailable, so accuracy, precision, and recall remain JSON `null`
even if AUTO resolves on a real clip.

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

Not implemented in Phase A8.1: iOS/Android apps, Swift/Kotlin, UniFFI, Rust mobile integration,
the production Rust Domain Kernel, validated numeric scores or overall score, severity/confidence
calibration, skill classification, diagnosis/turn/score GT, validated production SportType models,
3D or physical edge-angle/COM/pressure measurement, LLM/VLM coaching, XGBoost, QNN, TensorRT,
live camera coaching, complex UI, or first-party C++. Mobile integration and the Rust production
implementation remain explicitly deferred.
