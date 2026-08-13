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
