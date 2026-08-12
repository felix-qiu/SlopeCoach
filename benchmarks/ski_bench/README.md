# SkiBench reference harness

This directory is the stable home for benchmark inputs and policy notes. The executable
harness is `slopecoach_ml.benchmark`. Golden fixtures measure deterministic reference
behavior. Real videos currently measure metadata and quality only; no real pose provider
or expert diagnosis ground truth is configured.

Inputs are labeled `GOLDEN_FIXTURE`, `SYNTHETIC_METADATA_SMOKE`, or `REAL_VIDEO`. Generated
metadata smoke videos must never be described as real ski-video benchmarks. Phase A1.1 had no
user-provided real ski video, so the real-video benchmark status is
`NOT_EXECUTED_NO_REAL_VIDEO_INPUT`.

Generated JSON belongs under `artifacts/` and is intentionally git-ignored.

Phase A2 real-pose mode samples decoded frames, runs the configured RTMDet-m and RTMW-L providers,
and reports technical pose/coordinate/latency metrics. Use `REAL_VIDEO` only for user-provided
real videos. Decoder/provider smoke media must use `SYNTHETIC_PIPELINE_SMOKE`. Debug overlays use
canonical SourcePixel2D poses and belong in `artifacts/debug/<analysis-id>/`.
