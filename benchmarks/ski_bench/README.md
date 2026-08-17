# SkiBench reference harness

This directory is the stable home for benchmark inputs and policy notes. The executable
harness is `slopecoach_ml.benchmark`. Golden fixtures measure deterministic reference
behavior. Phase A2.2 real-video runs measure RTMDet-m + RTMW-L technical pose quality and CPU
performance; no expert diagnosis ground truth is configured.

Inputs are labeled `GOLDEN_FIXTURE`, `SYNTHETIC_METADATA_SMOKE`, or `REAL_VIDEO`. Generated
media must never be described as real ski-video benchmarks. Real user media belongs only in
ignored `videos/` or `fixtures/real/`, is never a pytest/CI dependency, and must not be added to
Git.

Generated JSON belongs under `artifacts/` and is intentionally git-ignored.

Phase A2 real-pose mode samples decoded frames, runs the configured RTMDet-m and RTMW-L providers,
and reports technical pose/coordinate/latency metrics. It is frame-independent: observations are
not tracks, and multi-person frames remain unresolved instead of choosing a target. Use
`REAL_VIDEO` only for user-provided real videos. Decoder/provider smoke media must use
`SYNTHETIC_PIPELINE_SMOKE`. Debug overlays and contact sheets use canonical SourcePixel2D poses
and belong in `artifacts/debug/<analysis-id>/`. Raw temporal observations apply no smoothing;
knee angles remain 2D image measurements, not physical 3D or diagnosis.

Phase A3 adds the separate `benchmark-target-identity` comparison path. It preserves raw
detections, applies conservative candidate quality, builds ephemeral reference tracks, and
maintains a distinct TargetIdentity state machine. Track IDs are not user identity. Ambiguous,
suspect, recovering, and lost frames suppress target biomechanics. The lightweight HSV
appearance descriptor is not deep ReID, and target-identity accuracy metrics remain `null`
without formal frame annotations. A3 artifacts belong under ignored `artifacts/`; local videos
remain ignored and are never CI inputs.

Phase A3.1 uses benchmark contract `ski-bench-target-identity-v2`. It requires the final initial
winner to be present on the decision frame, bounds historical evidence by timestamp, and uses a
capped constant-velocity prediction normalized by the last target bbox diagonal. Tracker
termination is `terminated_track_count`; v1's `track_fragments` was a misnamed termination count,
not GT-backed fragmentation.

Optional identity annotations live under `annotations/` and use `target-identity-gt-v1`. A template
contains the source video SHA256 and actual decoded sample timestamps, but every generated entry is
`UNLABELED` with `bbox: null`. A human must label the intended subject independently of detector,
tracker, pose, and identity output. Benchmark use verifies the video hash and reports only
timestamp-matched, unique samples within the configured tolerance. Without reviewed `PRESENT` or
`ABSENT` labels, identity accuracy and recovery results remain null.

```bash
make prepare-target-gt VIDEO=benchmarks/ski_bench/videos/ski_test_001.mp4 \
  TARGET_GT=benchmarks/ski_bench/annotations/ski_test_001.target.json \
  GT_REVIEW_DIR=artifacts/debug/a3_1_gt/ski_test_001 SAMPLE_FPS=5
```

Human-labeling review frames contain only the source image, timestamp, and frame index. Separate
`gt_comparison/` model overlays are produced only when a reviewed GT file is evaluated.

Phase A4 adds `benchmark-temporal-turns`. It reuses the A3 identity manager and target-focused pose
scheduling before applying timestamp-driven short-gap interpolation, per-joint One Euro filtering,
an image-space signed lower-body proxy, and provisional peak/zero-crossing segmentation. Identity
uncertainty is always a hard temporal boundary; no gap is filled across it. Raw model pose and
interpolated/stabilized evidence stay distinguishable.

Phase A4.1 changes the report contract to `ski-bench-temporal-turns-v2`. Valid signal runs split at
missing/low-confidence signal or temporal-segment changes; non-increasing timestamps inside a run
fail validation explicitly. Extrema acceptance, zero crossings, and provisional boundaries are
all run-local. Zero plateaus crossing
opposite signs use the integer midpoint of the first/last zero timestamps; same-side plateaus emit
nothing. Reports distinguish no signal, insufficient continuous signal, no qualified extrema,
rejected candidates, and provisional candidates. No thresholds are reduced to manufacture turns.

V2 stability uses one median raw symmetric shoulder-center to ankle-center scale per temporal
segment for both raw and stabilized metrics. A4 v1 used a different per-frame asymmetric scale;
the values are historical and not numerically comparable. Keep v1 JSON unchanged and write A4.1
locally under `artifacts/benchmarks/a4_1/`.

```bash
make benchmark-temporal-turns \
  VIDEO=benchmarks/ski_bench/videos/ski_test_001.mp4 SAMPLE_FPS=5 \
  OUTPUT=artifacts/benchmarks/a4_1/ski_test_001_5fps.json \
  DEBUG_DIR=artifacts/debug/a4_1_temporal_turns/ski_test_001_5fps
```

The signed proxy is `IMAGE_SPACE_2D_PROXY_ONLY`; phase sign is not skiing left/right and no output
is physical edge angle or diagnosis. The current target template must remain `UNLABELED` during
A4. `TURN_SEGMENTATION_GT_STATUS = NOT_AVAILABLE`, and turn precision, recall, and F1 stay null.
The target template remains wholly `UNLABELED`; target identity accuracy remains unknown. A4.1 is
engineering/reference validation only and implements neither diagnosis nor true skiing-direction
semantics.

Phase A5 adds `benchmark-biomechanics` and contract `ski-bench-biomechanics-v1`. It composes the
existing A3/A4/A4.1 pipeline rather than selecting a target again, then emits 14 frame-level 2D
facts, per-temporal-segment statistics and timestamp derivatives, plus turn facts only for A4.1
`VALID`/`PARTIAL` segments within the same signal run. A run with trusted frame/segment evidence
and no qualified turn is honestly `EXECUTED_FRAME_AND_SEGMENT_FEATURES_NO_TURNS`.

```bash
make benchmark-biomechanics \
  VIDEO=benchmarks/ski_bench/videos/ski_test_001.mp4 SAMPLE_FPS=5 \
  OUTPUT=artifacts/benchmarks/a5/ski_test_001_5fps.json \
  DEBUG_DIR=artifacts/debug/a5_biomechanics/ski_test_001_5fps
```

Biomechanics GT is unavailable, so feature accuracy and MAE remain `null`. Coverage is evidence
availability, not accuracy. Facts are camera-dependent image-space proxies without diagnosis,
scores, physical COM, physical edge angle, or a frozen ML feature vector.

Phase A5.1 hardens new output to `ski-bench-biomechanics-v2`, containing
`temporal-biomechanics-v2`, feature schema `biomechanics-feature-schema-v1`, and registry SHA256
`2777c3fbf7513e7537122f897f1901e61baf7eeddcee927937decb7476953048`. The 14 frame, 4 temporal,
and 12 turn feature IDs and their order are unchanged. Historical v1 artifacts remain historical.
V2 makes available/null invariants symmetric, rejects bool-as-number configuration, limits the
three minimum-knee timing facts to complete bounded turns, preserves valid apex/local boundary
facts for partial turns, and propagates conservative source evidence. Matching is signal-run-local,
inside all known boundaries, and deterministically prefers an earlier timestamp on equal distance.
The ML vector remains `NOT_FROZEN`; no diagnosis or score is produced.

Phase A5.2 adds the tracked disabled template
`manifests/biomechanics_real.example.json`; actual manifests remain local under ignored
`artifacts/manifests/`. The `prepare-biomechanics-dataset` command scans only the designated
directory, accepts `.mp4`, `.mov`, and `.m4v`, computes provenance SHA256, and never auto-enables
unknown clips. `source_video_id`, not clip rows or subclips, defines independent-source evidence.

The sequential `benchmark-biomechanics-dataset` command preserves each complete A5.1 report,
isolates clip failures, and writes macro/micro frame coverage, segment-based temporal coverage,
turn-based coverage, failure matrices, mathematical domain checks, CSV matrices, performance, and
strict-null GT status. A single real source honestly reports
`INSUFFICIENT_DATASET_SINGLE_CLIP`; with the default threshold of five, 2–4 sources report
`LIMITED_MULTICLIP_EVIDENCE` and five or more report project-level
`MULTICLIP_ENGINEERING_EVIDENCE`. The threshold is config-driven. These levels are engineering governance,
not statistical generalization or accuracy evidence. No diagnosis, score, feature tuning, or
automatic schema deletion occurs.

Phase A6 adds `benchmark-sport-type` and contract `ski-bench-sport-type-v1`. It composes the A5.1
real-video benchmark once, then runs dedicated SportType provider boundaries, collects existing A5
aggregates as uncalibrated cues, performs deterministic `sport-type-v1` fusion, and applies an
optional user override. Current RTMDet detections remain person-only; there is no second detector
pass and no hard-coded ski/snowboard class ID.

`EQUIPMENT` and `VISUAL_CLASSIFIER` are primary evidence. Pose geometry and temporal motion are
secondary and cannot resolve AUTO alone. Both current primary providers are `NOT_CONFIGURED`, so
the honest result is normally `UNKNOWN / INSUFFICIENT_PRIMARY_EVIDENCE` with user confirmation
recommended. Filenames and directories are never evidence; user selection is routing input, not
GT, and the auto result remains observable after override.

```bash
make benchmark-sport-type \
  VIDEO=benchmarks/ski_bench/videos/ski_test_001.mp4 SAMPLE_FPS=5 SPORT_TYPE=auto \
  OUTPUT=artifacts/benchmarks/a6/ski_test_001_5fps.json \
  DEBUG_DIR=artifacts/debug/a6_sport_type/ski_test_001_5fps
```

SportType GT is unavailable, so accuracy, precision, and recall remain null. A real UNKNOWN result
is an honest provider-availability outcome, not a failed execution. A6 produces no diagnosis or
score and does not reinterpret turn phase signs as skiing left/right.

A6.1 advances new reports to `ski-bench-sport-type-v2` while retaining the `sport-type-v1` domain
contract. A separate full-COCO RTMDet-tiny equipment provider operates only on deterministic
`LOCKED` target contexts collected during the existing single upstream pass. It does not modify or
reuse the person detector as an equipment classifier. Runtime class names `skis` and `snowboard`
are resolved from model metadata, never numeric constants.

The target crop extends wider and lower than the person bbox. Equipment detections are mapped back
to SourcePixel2D and accepted only when their centers enter a target-relative geometric association
zone. Debug output distinguishes the target, crop, zone, and accepted detections and labels them
`EQUIPMENT EVIDENCE ONLY`. Association correctness and SportType accuracy remain unknown without
GT. The default research configuration evaluates at most 12 evenly-spaced eligible locked frames
with score threshold 0.25; these values are not tuned to the local clip.

Use `--equipment-provider rtmdet-coco` plus explicit config/checkpoint paths (or the matching
`SLOPECOACH_EQUIPMENT_DETECTOR_*` environment variables). The model is never auto-enabled and
normal CI does not need OpenMMLab. `sport-equipment-doctor` verifies the checkpoint hash, model
load, 80-class metadata, and required names before real use. Historical v1 artifacts are not
rewritten.

A6.2 advances new artifacts to `ski-bench-sport-type-v3` while retaining the compatible
`sport-type-v1` domain contract. A pinned official OpenAI CLIP `ViT-B/32` zero-shot provider
classifies only deterministic crops around `LOCKED` targets; the full frame is never classified.
Fixed English `SKI`, `SNOWBOARD`, and `NEUTRAL` prompt groups are fingerprinted under
`visual-sport-prompts-v1`. Neutral support stays diagnostic and never becomes a SportType.

Equipment and visual providers execute independently and failures are isolated. Their frame
observations enter the unchanged `ReferenceSportTypeFusion`. Reports expose equipment-only,
visual-only, and combined diagnostic decisions without rerunning models; only combined evidence
plus optional user override controls the effective result. Dataset LOCKED-context counts are
unique and distinct from per-provider selected/inferred counts.

CLIP remains optional in the ignored Python 3.11 runtime. The checkpoint must be prepared and
passed explicitly; normal CI is CLIP-free and network-free. Zero-shot supports are not calibrated
probabilities, results depend on the fixed prompt taxonomy, and no accuracy claim is possible
while `SPORT_TYPE_GT_STATUS = NOT_AVAILABLE`. One local source clip is engineering evidence only.

A6.3 advances new artifacts to `ski-bench-sport-type-v4` without changing the RAW_V1 effective
routing result. Provider frame observations are quality-weighted into one source sample per
`KIND::provider_name` channel. A separate research-only calibrated diagnostic uses manual
`sport-type-gt-v1` labels, provider-specific Platt transforms, grouped source-level OOF
evaluation, and same-kind-mean/cross-kind-sum estimated LLR fusion. Generated GT starts
UNLABELED/UNCONFIRMED, and no calibration is fitted until both classes meet the independent-source
minimum. The existing A6.2 v3 artifact can be extracted without rerunning any vision model.

A7 adds the artifact-only `ski-bench-diagnosis-v1` report. It consumes persisted effective
SportType, complete turn segments, and A5 frame/turn facts; it never reruns detection, pose,
identity, SportType evidence, or segmentation. Reports separate evaluable, triggered,
not-triggered, and not-evaluable turns for each of the three provisional rules. The current real
clip has zero qualified turns, so it cannot validate turn-window diagnosis and must not be used to
tune A4 or A7 thresholds. Turn and Diagnosis GT remain unavailable and all accuracy metrics null.

A9 adds `ski-bench-analysis-result-v1`, a deterministic downstream-only assembly benchmark. It
first applies the A8.1 diagnosis compatibility gate, then builds the nullable ScoreCard and
controlled CoachReport before assembling the fixed eight-section `AnalysisResult v1` and its pure
`ProductReport v1` projection. It does not rerun any detector, pose, tracking, SportType, turn, or
biomechanics model. Runtime timings are diagnostic only and are excluded from both semantic SHAs.

Legacy artifacts contribute only explicitly embedded facts. The current A7 clip provides
SportType, turn count, and Diagnosis facts but no stable Source identity, Target Identity summary,
or compact Biomechanics summary. Those sections therefore remain `UNAVAILABLE`; with zero
qualified turns, the result is `PARTIAL_ANALYSIS / NO_QUALIFIED_TURNS`, with zero issues, zero
practice items, and all scores null.

```bash
make benchmark-analysis-result \
  ARTIFACT=artifacts/benchmarks/a7/ski_test_001_artifact_only.json \
  OUTPUT=artifacts/benchmarks/a9/ski_test_001_analysis_result.json
```

```bash
make benchmark-biomechanics \
  VIDEO=benchmarks/ski_bench/videos/ski_test_001.mp4 SAMPLE_FPS=5 \
  OUTPUT=artifacts/benchmarks/a5_1/ski_test_001_5fps.json \
  DEBUG_DIR=artifacts/debug/a5_1_biomechanics/ski_test_001_5fps
```
