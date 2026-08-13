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
