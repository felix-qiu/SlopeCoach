# SlopeCoach / Ski AI Coach
# App Local Architecture V1.1 — Freeze Candidate

**文档类型：** Mobile Local Architecture  
**版本：** V1.1  
**目标平台：** iOS / Android  
**运行模式：** Offline First / On-device Analysis  
**架构状态：** CONDITIONALLY APPROVED — FREEZE CANDIDATE

---

# 1. Architecture Decision Summary

SlopeCoach App Local 第一方技术栈正式定义为：

```text
Swift
+
Kotlin
+
Rust
+
Python
```

正式 ADR 原则：

```text
No First-party C++ by Default
```

平台职责：

```text
Swift / Kotlin
    ↓
Native UI
Native Video
Native Model Runtime
Pixel / GPU
    ↓
Canonical Pose Contract
    ↓
UniFFI
    ↓
Rust Domain Kernel
    ↓
AnalysisResult
    ↓
Template Coach / LLM Coach
```

Python：

```text
Training
Experiment
Benchmark
Model Export
Calibration
```

不进入 App Runtime。

---

# 2. 核心架构原则

## Native

Native 层负责：

```text
Video
Pixel
GPU
Platform Runtime
UI
Storage
Export
```

原则：

> **Native 管像素和设备。**

---

## Rust

Rust 负责：

```text
Tracking
Target Identity
Sport Type Fusion
Temporal Processing
Biomechanics
Turn Segmentation
Diagnosis
Scoring
Evidence
Analysis Orchestration
```

原则：

> **Rust 管动作和事实。**

---

## Python

Python 负责：

```text
Training
Dataset
Experiment
Model Export
Benchmark
XGBoost
Calibration
```

原则：

> **Python 管模型研发，不进入 Local Runtime。**

---

## LLM

LLM 负责：

```text
Explanation
Report
Drill Explanation
Progress Summary
Language Adaptation
```

原则：

> **LLM 是 Language Layer，不是 Truth Layer。**

---

# 3. 总体架构

```text
                         VIDEO
                           │
                           ▼
                  Native Video Layer
                 Swift / Kotlin
                           │
                           ▼
                    PoseProvider
                           │
                  Model Coordinate
                           │
                           ▼
                  Coordinate Adapter
                           │
                           ▼
              Canonical Pose Coordinate
                           │
                           ▼
                       PoseBatch
                           │
                         UniFFI
                           │
                           ▼
                      mobile-api
                     FFI Façade
                           │
                           ▼
                        domain
                    Orchestrator
                           │
        ┌──────────────────┼───────────────────┐
        ▼                  ▼                   ▼
     Tracking          Target Identity      Temporal
        │                                      │
        └──────────────────┬───────────────────┘
                           ▼
                      Sport Type
                           │
                           ▼
                      Biomechanics
                           │
                           ▼
                         Turns
                           │
                           ▼
                       Diagnosis
                           │
                           ▼
                        Scoring
                           │
                           ▼
                        Evidence
                           │
                           ▼
                    AnalysisResult
                    ──────────────
                       FACT LAYER
                    ──────────────
                           │
                           ▼
                     CoachContext
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
         Template Coach             LLM Coach
               │                       │
               └───────────┬───────────┘
                           ▼
                      CoachReport
                    ──────────────
                    LANGUAGE LAYER
                    ──────────────
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             App          Voice       Share/PDF
```

---

# 4. First-party Language Stack

| Layer | Language | Responsibility |
|---|---|---|
| iOS | Swift | SwiftUI、AVFoundation、Metal、Inference Adapter、Video、Charts、Export |
| Android | Kotlin | Compose、Media3、Inference Adapter、Video、Charts、Export |
| Shared Core | Rust | Domain / Tracking / Identity / Feature / Diagnosis |
| AI Development | Python | Training / Export / Benchmark / Dataset |
| Contract Generation | Rust | FFI DTO + JSON Schema |
| Configuration | YAML / JSON | Threshold / Model Manifest / Runtime Config |

第一方 P0 / P1：

```text
C++ = NOT REQUIRED
```

第三方 Runtime 内部使用 C/C++：

```text
Allowed
```

但属于 implementation detail。

---

# 5. Monorepo

正式采用：

```text
SlopeCoach/
│
├── apps/
│   ├── ios/
│   └── android/
│
├── crates/
│   ├── contracts/
│   ├── mobile-api/
│   ├── domain/
│   ├── tracking/
│   ├── target-identity/
│   ├── sport-type/
│   ├── temporal/
│   ├── biomechanics/
│   ├── turns/
│   ├── diagnosis/
│   ├── scoring/
│   └── evidence/
│
├── bindings/
│   ├── ios/
│   └── android/
│
├── python/
│   ├── pyproject.toml
│   ├── uv.lock
│   └── slopecoach_ml/
│       ├── detection/
│       ├── pose/
│       ├── sport_type/
│       ├── diagnosis/
│       ├── training/
│       ├── export/
│       └── benchmark/
│
├── coach/
│   ├── prompts/
│   ├── drill-library/
│   ├── templates/
│   └── validators/
│
├── models/
│   ├── mobile/
│   ├── cloud/
│   └── registry/
│
├── schemas/
│   └── generated/
│
├── configs/
│
├── datasets/
│
├── benchmarks/
│   ├── mobile/
│   ├── tracking/
│   ├── identity/
│   ├── pose/
│   ├── sport-type/
│   └── diagnosis/
│
├── scripts/
│
├── docs/
│   ├── architecture/
│   └── adr/
│
├── toolchains/
│   └── versions.toml
│
├── Cargo.toml
├── rust-toolchain.toml
├── Makefile
└── README.md
```

正式不建立：

```text
cpp/
```

---

# 6. Coordinate Contract — P0 Blocking

PoseFrame 在进入 Rust 前必须完成完整坐标标准化。

禁止：

```text
Model Coordinate
直接进入
Rust Feature Engine
```

正确：

```text
Original Video
     ↓
Crop / Resize / Letterbox
     ↓
Model Coordinate
     ↓
Model Inference
     ↓
Inverse Transform
     ↓
Canonical Coordinate
     ↓
Rust
```

---

# 7. Canonical Coordinate Space

Rust Domain 的 Canonical Pose Space：

```text
CoordinateSpace = SourcePixel2D
```

正式定义：

```text
Origin      = Top Left

X Direction = Right

Y Direction = Down

Unit        = Source-frame Pixels

Orientation = Upright / Applied

Mirror      = Corrected

Crop        = Inverted

Resize      = Inverted

Letterbox   = Inverted
```

即：

> **进入 Rust 的 Pose 已经恢复到显示方向正确的原始视频帧坐标系。**

---

# 8. 禁止非等比例 Normalized Coordinate 直接做 Geometry

例如：

```text
x_norm = x / width
y_norm = y / height
```

对于：

```text
1920 × 1080
```

会造成：

```text
X/Y scale 不一致
```

因此这种坐标不能直接计算：

```text
body angle
segment angle
```

Canonical Geometry 默认基于：

```text
SourcePixel2D
```

距离类指标随后通过：

```text
Torso Length
Shoulder Width
Hip Width
BBox Height
```

等人体尺度进行归一化。

---

# 9. PoseFrame Contract

正式定义：

```rust
pub struct PoseFrame {
    pub contract_version: String,

    pub timestamp_us: u64,

    pub frame_index: u32,

    pub geometry: FrameGeometry,

    pub joint_schema: JointSchemaId,

    pub persons: Vec<PersonPose2D>,
}
```

---

# 10. FrameGeometry

```rust
pub struct FrameGeometry {
    pub width_px: u32,

    pub height_px: u32,

    pub pixel_aspect_ratio: f32,

    pub coordinate_space: CoordinateSpace,

    pub orientation: FrameOrientation,

    pub mirrored: bool,
}
```

进入 Rust 时必须满足：

```text
coordinate_space = SourcePixel2D

orientation = CanonicalUpright

mirrored = false
```

---

# 11. Keypoint Contract

```rust
pub struct Keypoint2D {
    pub x_px: f32,

    pub y_px: f32,

    pub confidence: f32,
}
```

Person：

```rust
pub struct PersonPose2D {
    pub detection_id: Option<u64>,

    pub bbox: BoundingBox2D,

    pub person_confidence: f32,

    pub keypoints: Vec<Keypoint2D>,
}
```

关键原则：

> **BBox 和 Keypoints 必须处于完全相同的 Coordinate Space。**

---

# 12. Joint Schema Contract

禁止业务代码：

```text
keypoints[15]
```

隐式代表某个人体关节。

必须：

```text
joint_schema = COCO17_V1
```

或者未来：

```text
SLOPECOACH_SKIER_V1
```

PoseProvider Adapter 负责：

```text
YOLO Schema
      ↓
Canonical Joint Schema
```

```text
RTMPose Schema
      ↓
Canonical Joint Schema
```

Rust Domain 只消费：

```text
Canonical Joint Schema
```

---

# 13. PoseProvider Coordinate Responsibility

PoseProvider 必须保存推理预处理 Transform。

例如：

```text
1920 × 1080
      ↓
Letterbox
      ↓
640 × 640
```

模型输出后执行：

```text
Model Coordinate
      ↓
inverse letterbox
      ↓
inverse crop
      ↓
inverse resize
      ↓
orientation correction
      ↓
mirror correction
      ↓
SourcePixel2D
```

Rust Core：

```text
不处理模型预处理坐标恢复
```

---

# 14. 2D / 3D Contract

严格分离：

```rust
pub struct Keypoint2D {
    pub x_px: f32,
    pub y_px: f32,
    pub confidence: f32,
}
```

真实 3D：

```rust
pub struct Keypoint3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub confidence: f32,
}
```

PoseSpace：

```rust
pub enum PoseSpace {
    Image2D,
    Camera3D,
    World3D,
}
```

禁止：

```text
2D
+
z = 0
=
3D
```

---

# 15. Contract Source of Truth — P0 Blocking

正式废弃：

```text
Rust Types
+
人工维护 JSON Schema
```

双 Source-of-Truth。

V1.1 正式定义：

> **Rust Contract Types = Single Source of Truth**

---

# 16. contracts crate

建立：

```text
crates/contracts/
│
├── pose.rs
├── analysis.rs
├── diagnosis.rs
├── evidence.rs
├── coach.rs
├── sport.rs
├── model.rs
└── lib.rs
```

其中保存所有：

```text
Cross-language Public Contracts
```

---

# 17. Contract Generation

例如：

```rust
#[derive(
    Serialize,
    Deserialize,
    JsonSchema
)]
pub struct AnalysisResult {
    // ...
}
```

生成：

```text
                Rust Contracts
                      │
         ┌────────────┼─────────────┐
         │            │             │
         ▼            ▼             ▼
      UniFFI       JSON Schema    Domain
         │            │
    ┌────┴────┐       ├── Python
    ▼         ▼       ├── Cloud
  Swift     Kotlin     └── Validation
```

---

# 18. Generated Schema

目录：

```text
schemas/generated/
```

例如：

```text
analysis-result.schema.json

pose-frame.schema.json

diagnosis.schema.json

coach-report.schema.json

model-manifest.schema.json
```

规则：

```text
Generated Files
=
DO NOT EDIT MANUALLY
```

---

# 19. Schema CI Gate

CI：

```text
cargo run -p schema-generator
```

然后：

```text
git diff --exit-code schemas/generated/
```

如果 Contract 修改但生成 Schema 未同步：

```text
CI FAIL
```

---

# 20. mobile-api / domain Boundary — P0 Blocking

正式定义：

## mobile-api

职责：

> **FFI Façade Only**

只允许：

```text
UniFFI annotations

FFI-safe DTO boundary

Input validation

Error translation

API version

Session handles

Domain delegation
```

禁止：

```text
Tracking algorithm

Feature formula

Threshold

Diagnosis logic

Scoring

Sport logic
```

---

# 21. domain crate

正式定义：

> **Application / Domain Orchestrator**

完整链路：

```text
PoseBatch
    ↓
Tracking
    ↓
Target Identity
    ↓
Temporal
    ↓
SportType
    ↓
Biomechanics
    ↓
Turns
    ↓
Diagnosis
    ↓
Scoring
    ↓
Evidence
    ↓
AnalysisResult
```

核心入口：

```text
domain::AnalysisPipeline
```

---

# 22. Rust Dependency Direction

```text
                   mobile-api
                       │
                       ▼
                     domain
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
  tracking          temporal        sport-type
      │                │
      ▼                ▼
target-identity   biomechanics
                       │
                       ▼
                     turns
                       │
                       ▼
                   diagnosis
                       │
                       ▼
                    scoring
                       │
                       ▼
                    evidence
```

公共数据：

```text
contracts
   ↑
   │
all public-data consumers
```

禁止：

```text
domain
↓
mobile-api
```

---

# 23. UniFFI Contract — P0 Blocking

P0 UniFFI 采用：

```text
Synchronous FFI
+
Native Background Worker
```

暂不采用：

```text
UniFFI async public API
```

---

# 24. Threading Contract

`AnalysisEngine`：

```text
Long-lived

Thread-safe
```

`AnalysisSession`：

```text
Single Writer

Non-reentrant
```

禁止：

```text
Thread A
push

Thread B
finish

Thread C
push
```

并发操作同一个 Session。

---

# 25. iOS Threading

```text
SwiftUI MainActor
       │
       ▼
Analysis Worker
Actor / Serial Executor
       │
       ▼
Synchronous UniFFI
       │
       ▼
Rust
```

禁止 Heavy Analysis：

```text
MainActor
```

运行。

---

# 26. Android Threading

```text
Compose
   │
   ▼
ViewModel
   │
   ▼
Coroutine
Dispatchers.Default
   │
   ▼
Synchronous UniFFI
   │
   ▼
Rust
```

禁止：

```text
Main Thread
```

执行长时间 Rust Analysis。

---

# 27. PoseBatch

原：

```text
push_pose_frame(frame)
```

调整为：

```text
push_pose_batch(batch)
```

减少跨 FFI 调用。

例如：

```text
Frame 0
Frame 1
...
Frame 15
   ↓
PoseBatch
   ↓
one FFI call
```

---

# 28. Batch Contract

配置：

```yaml
ffi:
  pose_batch_max_frames: ...
```

实际 Batch Size：

```text
不得作为 Architecture Constant
```

由 Benchmark 决定。

考虑：

```text
FFI overhead

Memory

Latency

Cancellation responsiveness
```

---

# 29. Backpressure

P0 使用：

> **Synchronous Bounded Ingestion**

```text
Native Producer
     ↓
Pose Batch
     ↓
Rust
     ↓
Process
     ↓
Return
     ↓
Next Batch
```

如果 Rust 较慢：

```text
Producer Wait
```

即为天然 Backpressure。

---

# 30. P0 不允许无限 Queue

禁止：

```text
Native
↓
不断 push
↓
Rust unbounded queue
↓
Memory grows
↓
OOM
```

P0 Offline Analysis 不需要这种架构。

---

# 31. P0 不静默 Drop Frame

Offline Analysis：

```text
Buffer Full
↓
WAIT
```

不是：

```text
DROP
```

未来：

```text
Live Camera Coaching
```

可以增加：

```text
FrameDropPolicy
```

但不属于 V1.1。

---

# 32. AnalysisSession Lifecycle

```rust
pub enum SessionState {
    Created,
    Running,
    Finishing,
    Finished,
    Cancelled,
    Failed,
}
```

状态：

```text
CREATED
   ↓ first batch
RUNNING
   ↓ finish()
FINISHING
   ↓
FINISHED
```

取消：

```text
RUNNING
   ↓ cancel()
CANCELLED
```

异常：

```text
RUNNING
   ↓ fatal error
FAILED
```

---

# 33. Lifecycle Rules

终态：

```text
FINISHED
CANCELLED
FAILED
```

之后禁止继续：

```text
push_pose_batch()
```

`cancel()`：

```text
Idempotent
```

`finish()`：

```text
Final Result cached
```

重复读取可以返回已有结果，而不重复分析。

---

# 34. Cancellation Contract

Native 发起：

```text
cancel()
```

Rust 必须在合理边界检查：

```text
Cancellation Token
```

例如：

```text
between batches

between pipeline stages

before expensive finalization
```

Cancellation 不要求中断到单个浮点运算。

---

# 35. Swift 6 Compatibility Gate — P0 Blocking

新增：

```text
Swift6FFIGate
```

测试：

```text
Swift 6 Language Mode

Strict Concurrency

Actor Isolation

Sendable Boundaries

Session Lifetime

Cancellation

Background / Foreground

Repeated Analysis

Deinit

Error Propagation
```

---

# 36. P0 Swift / UniFFI Rule

Architecture Freeze 前：

```text
NO public UniFFI async functions
```

```text
NO callback-heavy FFI design
```

```text
NO Rust future directly exposed to Swift
```

优先：

```text
Native concurrency
+
Synchronous FFI
```

---

# 37. PoseProvider

```text
PoseProvider
│
├── MobilePoseProviderA
├── MobilePoseProviderB
├── NativePoseProvider
└── MockPoseProvider
```

业务层：

```text
不绑定具体模型
```

PoseProvider 负责：

```text
Preprocess

Inference

Coordinate Recovery

Canonical Joint Mapping

Pose Confidence
```

---

# 38. iOS Pipeline

```text
Video
↓
AVFoundation
↓
CVPixelBuffer
↓
Swift PoseProvider
↓
Model Runtime
↓
Coordinate Adapter
↓
Canonical PoseBatch
↓
UniFFI
↓
Rust
```

不增加：

```text
Swift
↓
Custom C++
```

---

# 39. Android P0 Pipeline

正式收敛为：

```text
Video
↓
Media3
↓
Frame
↓
Kotlin PoseProvider
↓
ONNX Runtime
↓
XNNPACK / CPU
↓
Coordinate Adapter
↓
Canonical PoseBatch
↓
UniFFI
↓
Rust
```

---

# 40. Android P0 Backend

正式定义：

```text
ONNX Runtime
      │
 ┌────┴────┐
 ▼         ▼
XNNPACK    CPU
```

P0 不以：

```text
QNN
```

作为必要依赖。

---

# 41. Android Backend Selection

候选策略：

```text
Non-quantized model
↓
Benchmark XNNPACK

Quantized model
↓
Benchmark CPU / supported path
```

最终选择：

```text
Model × Device Benchmark
```

而不是静态规则永久写死。

---

# 42. QNN Status

正式状态：

```text
QNN
=
Optimization Candidate
```

不是：

```text
P0 Requirement
```

进入条件：

```text
QNN Feasibility Gate
```

---

# 43. QNN Feasibility Gate

至少验证：

```text
Supported Device Matrix

Operator Coverage

Model Partition

Quantization Requirements

SDK Dependency

Native Packaging

AAR Packaging

App Size

Initialization Time

Inference P50/P95

Thermal

Battery

Fallback

CI Reproducibility
```

通过后：

```text
QNN
↓
Candidate Backend
```

---

# 44. Tracking

正式采用：

```text
First-party Rust Tracking
```

但不能因为算法实现完成就宣称等价于 ByteTrack。

必须建立：

```text
Tracking Reference Benchmark
```

---

# 45. Tracking Reference Architecture

```text
Recorded Detector Observations
             │
      ┌──────┴─────────┐
      ▼                ▼
Reference Tracker    Rust Tracker
      │                │
      └──────┬─────────┘
             ▼
          Compare
```

两个 Tracker：

> **必须消费完全相同的 Detection / Pose Observations。**

避免 Detector 差异污染 Tracking Benchmark。

---

# 46. Tracking Reference Dataset

```text
benchmarks/tracking/
└── tracking-reference-v1/
    ├── detections/
    ├── pose_observations/
    ├── ground_truth/
    └── scenarios/
```

场景：

```text
Single Skier

Multiple People

Crossing

Short Occlusion

Long Occlusion

Snow Spray

Camera Pan

Zoom

Exit Frame

Re-entry

Similar Clothing

Partial Body
```

---

# 47. Tracking Metrics

通用：

```text
ID Switches

Fragmentation

Identity Consistency
```

SlopeCoach 业务指标：

```text
Target Frame Accuracy

Wrong Target Rate

Target Coverage

Re-acquisition Rate

Time to Re-lock

False Lock Duration

Ambiguous Rate
```

最高优先级：

```text
Wrong Target Rate
```

原则：

> **宁可 Lost，不要 Wrong Target。**

---

# 48. Tracking Release Gate

Rust Tracking 进入 Production 前必须：

```text
Rust Tracker
vs
Reference Tracker
```

在：

```text
tracking-reference-v1
```

上证明：

```text
No unacceptable regression
```

阈值由 Benchmark Policy 定义。

---

# 49. Tracking / Identity 分层

Tracking：

> 某一小段连续运动轨迹属于哪个 Track？

Identity：

> Track ID 改变以后，是否还是同一个目标人物？

因此：

```text
Track ID != Target Identity
```

---

# 50. Target Identity Manager

```text
Target Identity Manager
│
├── Initial Target Selector
└── Identity Continuity & Recovery
```

整个 Normal Workflow 自动完成。

用户默认不需要：

```text
选择本人
```

---

# 51. Target Identity States

```text
LOCKED

SUSPECT

LOST

RECOVERING

AMBIGUOUS
```

例如：

```text
Track 7
↓
LOST
↓
Track 12
↓
Identity Recovery
↓
TARGET_001
```

`target_id` 不变：

```text
TARGET_001
```

`active_track_id`：

```text
7 → null → 12
```

---

# 52. Target Ambiguity UX

原则：

> **Auto First, Skip on Ambiguity, Correct Only If Needed**

```text
LOCKED
↓
AMBIGUOUS
↓
Skip Target-specific Analysis
↓
Continue Search
↓
RECOVER
↓
LOCKED
```

不默认询问用户。

---

# 53. Target Correction

只有结果明显错人时提供：

```text
分析对象不对？
↓
选择正确主体
↓
重新分析
```

属于：

```text
Correction Workflow
```

---

# 54. Appearance ReID Status

正式从：

```text
P1 mandatory
```

修改为：

```text
CONDITIONAL
```

即：

> **Benchmark-triggered Capability**

---

# 55. ReID Trigger

```text
Tracking / Identity Benchmark
            ↓
Occlusion / Crossing subset
            ↓
Release target missed?
      ┌─────┴─────┐
      │           │
     NO          YES
      │           │
      │       ReID Prototype
      │           │
      │       Benchmark
      │           │
      └─────┬─────┘
            ▼
     measurable gain?
```

只有：

```text
Wrong Target ↓
```

或者：

```text
Re-acquisition ↑
```

达到明确收益，并且：

```text
Latency
Thermal
Model Size
Battery
License
```

可接受时才纳入产品。

---

# 56. SportType

Sport Type 的策略与 Target Identity 不同。

SportType：

> **Auto First, Ask on Ambiguity**

Target：

> **Auto First, Skip on Ambiguity, Correct Only If Needed**

---

# 57. SportType 不阻塞 Pose

```text
Video
↓
Generic Pose
↓
Generic Tracking
↓
Generic Features
↓
Sport Type
↓
Sport-specific Features
↓
Diagnosis
```

Pose 是：

```text
Generic Perception Layer
```

---

# 58. SportType Fusion

候选：

```text
Visual Evidence

Pose Evidence

Temporal Evidence

Equipment Evidence
```

融合：

```text
S_sport =
quality-aware fusion
```

输出：

```text
SKI

SNOWBOARD

UNKNOWN
```

---

# 59. Temporal Pose Pipeline

```text
Raw Pose
↓
Confidence Gate
↓
Outlier Detection
↓
Left/Right Consistency
↓
Short-gap Interpolation
↓
One Euro Filter
↓
Canonical Pose
```

Long Gap：

```text
null
```

不得伪造。

---

# 60. UI Smooth 与 Analysis Smooth

严格分离：

```text
Feature Series
      │
      ├────→ Display Smoothing → UI
      │
      ▼
Canonical Analysis
      ↓
Diagnosis
```

UI Slider 不改变 Diagnosis。

---

# 61. Biomechanics Feature Layer

```text
Canonical Pose
      ↓
Generic Features
      │
 ┌────┴───────────┐
 ▼                ▼
Ski Features   Snowboard Features
```

---

# 62. Measurement Contract

每个 Feature：

```text
value

confidence

unit

measurement_space

estimated

source_frames
```

Measurement Space：

```text
Image2D

Normalized2D

Estimated3D

Physical3D
```

---

# 63. Edge Angle Safety

普通单目 2D Pose：

```text
禁止
Physical Edge Angle = xx°
```

允许：

```text
Angulation Proxy

Torso Tilt

Edge Control Proxy
```

真实物理刃角必须有：

```text
Board Plane

Slope Plane

Calibration

Reliable 3D Geometry
```

---

# 64. Turn Segmentation

Turn：

```text
start_us

apex_us

end_us

direction

confidence

valid
```

Diagnosis：

```text
Turn Window
+
Multi-frame Evidence
```

不得依赖单个 Apex Frame。

---

# 65. Diagnosis Truth Layer

正式事实链：

```text
Video Quality
↓
Target Identity
↓
Pose Confidence
↓
SportType
↓
Turn Confidence
↓
Features
↓
Rules / ML
↓
Diagnosis
```

证据不足：

```text
null
```

---

# 66. Evidence Engine

Diagnosis 输出至少包含：

```text
error_code

severity

confidence

phase

affected_turns

evidence_frames

feature_evidence
```

用于回答：

> **AI 为什么这样判断？**

---

# 67. AnalysisResult

`AnalysisResult`：

> **Fact Layer**

包括：

```text
analysis_id

contract_version

analysis_mode

video_metadata

video_quality

target_identity

sport_type

pose_summary

turns

features

diagnoses

scores

evidence

warnings

limitations

model_versions
```

---

# 68. Coach Layer

```text
AnalysisResult
↓
CoachContext
↓
┌──────────────┐
▼              ▼
Template      LLM
Coach         Coach
│              │
└──────┬───────┘
       ▼
CoachReport
```

---

# 69. LLM Boundary

正式 ADR：

> **LLM is the language layer, not the truth layer.**

禁止：

```text
Video
↓
LLM
↓
正式 Diagnosis
```

正式：

```text
Video
↓
Perception
↓
Biomechanics
↓
Diagnosis
↓
AnalysisResult
↓
LLM
↓
Explanation
```

---

# 70. Drill Library

训练方案优先来自：

```text
Controlled Drill Library
```

Pipeline：

```text
Diagnosis
↓
Drill Retrieval
↓
Skill / Context Filter
↓
Candidate
↓
LLM Explanation
```

LLM 不自由创造核心训练事实。

---

# 71. Offline / Cloud Coach

Offline：

```text
AnalysisResult
↓
Template Coach
↓
CoachReport
```

Connected：

```text
AnalysisResult
+
Coach Context
+
History
+
Drill Library
↓
LLM
↓
CoachReport
```

没有网络：

```text
仍然能完成完整基础分析
```

---

# 72. Video Pipeline

独立：

```text
             Video Asset
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
    Playback   Analysis    Export
```

---

# 73. Playback / Analysis FPS

例如：

```text
Source = 120fps

Playback = 120fps

Analysis = 15 / 20 / 30fps
```

原则：

> **Playback FPS != Analysis FPS**

---

# 74. Pixel / FFI Rule

永久原则：

```text
PixelBuffer
HardwareBuffer
Raw Video Frame
```

不得通过普通 UniFFI DTO 进入 Rust Domain。

正确：

```text
Native Pixel
↓
Native Model Runtime
↓
Canonical PoseBatch
↓
UniFFI
↓
Rust
```

---

# 75. Mobile Public API

P0 public API 尽量冻结为：

```text
AnalysisEngine
    │
    └── create_session(config)
             │
             ▼
       AnalysisSession
             │
             ├── push_pose_batch(...)
             ├── progress()
             ├── finish()
             └── cancel()
```

这是 Native 唯一需要依赖的 Domain 接口。

---

# 76. Development Environment

主开发机推荐：

```text
Apple Silicon Mac

32 GB RAM recommended

1 TB SSD recommended
```

工具：

```text
Xcode

Android Studio

JDK

Android SDK

Android NDK

Rust / rustup / Cargo

Python / uv

Git

Docker
```

安装 NDK：

```text
不代表项目编写 C++
```

它仍用于 Rust Android Target 和第三方 Native Dependencies。

---

# 77. Toolchain Pinning

根目录：

```text
rust-toolchain.toml

.python-version

uv.lock

Gradle Wrapper

libs.versions.toml

toolchains/versions.toml
```

原则：

> **Validated Version > Latest Version**

---

# 78. Build Entry Points

```text
make bootstrap

make rust

make bindings

make ios

make android

make python

make schemas

make test

make benchmark

make all
```

开发者不需要手工记忆各语言所有底层构建命令。

---

# 79. First Vertical Slice

第一阶段不要立即接真实复杂 AI。

先打通：

```text
Test Video
↓
FakePoseProvider
↓
Coordinate Adapter
↓
Canonical PoseBatch
↓
UniFFI
↓
mobile-api
↓
domain
↓
Biomechanics
↓
AnalysisResult
↓
iOS / Android
```

要求：

```text
iOS Result
=
Android Result
```

---

# 80. Implementation Stages

```text
L0
Monorepo / Toolchains

↓

L1
contracts / JSON Schema Generator

↓

L2
Rust domain + mobile-api

↓

L3
UniFFI + Swift 6 Compatibility Gate

↓

L4
iOS / Android App Shell

↓

L5
FakePose Vertical Slice

↓

L6
Coordinate Adapter + Real Pose

↓

L7
Rust Tracking + Reference Benchmark

↓

L8
Target Identity

↓

L9
Temporal / Features

↓

L10
SportType

↓

L11
Turn / Diagnosis / Evidence

↓

L12
Coach

↓

L13
Export / SkiBench Mobile

↓

L14
Optimization Backends / Conditional ReID
```

---

# 81. ReID Roadmap Rule

不再写：

```text
Stage X = ReID
```

改成：

```text
Identity Benchmark
↓
Need ReID?
```

如果：

```text
No
```

则不增加模型。

如果：

```text
Yes
```

才进入：

```text
ReID Feasibility
↓
Benchmark
↓
Production Decision
```

---

# 82. QNN Roadmap Rule

同样不写：

```text
P1 = QNN
```

而是：

```text
Android Performance Benchmark
↓
Need Hardware EP?
↓
QNN Feasibility Gate
↓
Benchmark
↓
Production Decision
```

---

# 83. Mobile SkiBench

必须覆盖：

## Pose

```text
Coverage
Stability
Confidence
Coordinate Accuracy
Feature Drift
```

## Tracking

```text
Reference Comparison
ID Switch
Fragmentation
```

## Identity

```text
Wrong Target Rate
Coverage
Re-acquisition
False Lock
Ambiguous
```

## Sport

```text
Precision
Recall
Unknown Rate
False Auto-confirm Rate
```

## Diagnosis

```text
Precision
Recall
F1
Agreement
Confidence
```

## Performance

```text
P50
P95
RTF
Memory
Battery
Thermal
Load Time
Package Size
```

## FFI

新增：

```text
Batch Throughput
FFI Calls / Video
Memory Growth
Cancellation Latency
Repeated Session Stability
```

---

# 84. Architecture Freeze Gate

V1.1 只有满足以下条件才能改为：

```text
ARCHITECTURE FROZEN
```

| Gate | Requirement |
|---|---|
| Coordinate Contract | SourcePixel2D、Orientation、Mirror、Crop、Resize、Letterbox 定义完成 |
| Joint Contract | Canonical Joint Schema 定义完成 |
| Contract SOT | Rust Contract 为唯一 SOT |
| Schema Generation | JSON Schema 自动生成 + CI Drift Gate |
| mobile-api Boundary | Façade Only |
| domain Boundary | Orchestrator |
| UniFFI Threading | Single Writer / Background Worker |
| UniFFI Batching | `push_pose_batch()` |
| Backpressure | Bounded synchronous ingestion |
| Lifecycle | Created → Running → Finished / Cancelled / Failed |
| Swift 6 Gate | Strict Concurrency / Lifetime / Cancellation 验证 |
| Pixel Boundary | Pixel 不跨 UniFFI |
| Tracking Gate | Rust vs Reference Benchmark |
| Identity Gate | Wrong Target / Re-acquisition Metrics |
| ReID | Benchmark-triggered only |
| Android P0 | ORT + XNNPACK / CPU |
| QNN | Conditional optimization |
| 2D/3D | 严格 Measurement Contract |
| Diagnosis | Fact Layer |
| LLM | Language Layer |

---

# 85. Freeze 前禁止事项

在 Architecture Frozen 前禁止：

```text
新增大量 UI Feature

新增复杂 ReID 模型

接 QNN Production

做真实 3D

做 Physical Edge Angle

做 Live Camera Coaching

暴露 UniFFI async API

增加 First-party C++

把业务逻辑放 mobile-api

手工维护 JSON Schema

把 Model Coordinate 直接传 Rust
```

优先把架构边界打稳。

---

# 86. 最终语言职责

| Capability | Swift | Kotlin | Rust | Python |
|---|---:|---:|---:|---:|
| iOS UI | ✅ | ❌ | ❌ | ❌ |
| Android UI | ❌ | ✅ | ❌ | ❌ |
| Video | ✅ | ✅ | ❌ | Test |
| Pixel Preprocess | ✅ | ✅ | ❌ | Research |
| Mobile Model Runtime | ✅ | ✅ | ❌ P0 | ❌ |
| Coordinate Adapter | ✅ | ✅ | Validate | Test |
| Tracking | ❌ | ❌ | ✅ | Reference |
| Target Identity | ❌ | ❌ | ✅ | Benchmark |
| Temporal | ❌ | ❌ | ✅ | Validation |
| Biomechanics | ❌ | ❌ | ✅ | Validation |
| Turns | ❌ | ❌ | ✅ | Research |
| Diagnosis | ❌ | ❌ | ✅ | Train/Validate |
| Scoring | ❌ | ❌ | ✅ | Validate |
| Evidence | ❌ | ❌ | ✅ | Validate |
| Contracts | Generated | Generated | ✅ SOT | Generated |
| Model Training | ❌ | ❌ | ❌ | ✅ |
| Model Export | ❌ | ❌ | ❌ | ✅ |
| Charts | ✅ | ✅ | ❌ | ❌ |
| Export | ✅ | ✅ | ❌ | ❌ |
| First-party C++ | ❌ | ❌ | ❌ | ❌ |

---

# 87. 最终 Architecture Contract

```text
                Native Platform

             Swift       Kotlin
               │           │
               ▼           ▼
              Video / Pixel
                    │
                    ▼
                PoseProvider
                    │
                    ▼
             Coordinate Adapter
                    │
                    ▼
             Canonical PoseBatch
                    │
                  UniFFI
                    │
                    ▼
                mobile-api
                 FAÇADE
                    │
                    ▼
                  domain
              ORCHESTRATOR
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Tracking   Identity   Temporal
                    │
                    ▼
                 Features
                    │
                    ▼
                   Turns
                    │
                    ▼
                Diagnosis
                    │
                    ▼
                 Evidence
                    │
                    ▼
             AnalysisResult
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    Template Coach         LLM Coach
          │                   │
          └─────────┬─────────┘
                    ▼
                CoachReport
```

---

# 88. Architecture Status

当前：

```text
APP LOCAL ARCHITECTURE V1.1

STATUS:

CONDITIONALLY APPROVED
FREEZE CANDIDATE
```

完成六项 Blocking Review：

```text
1. Coordinate Contract

2. Single Contract Source of Truth

3. UniFFI Thread / Batch /
   Backpressure / Lifecycle +
   Swift 6 Gate

4. domain / mobile-api Boundary

5. Tracking Reference Benchmark +
   Conditional ReID

6. Android ORT/XNNPACK/CPU P0 +
   Conditional QNN
```

并完成对应 Vertical Slice / CI Gate 后：

```text
STATUS:

ARCHITECTURE FROZEN
```

---

# 89. 最终原则

### Language

> **Swift + Kotlin + Rust + Python。**

### C++

> **No First-party C++ by Default。**

### Contracts

> **Rust Contract Types 是唯一 Source of Truth。**

### Coordinate

> **进入 Rust 的 Pose 必须已经转换为 Canonical Coordinate Space。**

### FFI

> **窄接口、同步、批处理、有界 Backpressure、明确 Lifecycle。**

### Swift

> **Swift 6 Strict Concurrency 必须作为独立 Compatibility Gate。**

### Crates

> **domain 是 Orchestrator；mobile-api 只做 Façade。**

### Tracking

> **Rust 实现必须经过 Reference Benchmark。**

### ReID

> **Benchmark-triggered，不是 Roadmap-triggered。**

### Android

> **P0 = ORT + XNNPACK / CPU。**

### QNN

> **Optimization Candidate，不是 Architecture Dependency。**

### Identity

> **Track ID != Target Identity。**

### Target UX

> **Auto First, Skip on Ambiguity, Correct Only If Needed。**

### SportType UX

> **Auto First, Ask on Ambiguity。**

### Measurement

> **2D Proxy 不冒充 Physical 3D。**

### Diagnosis

> **Diagnosis / Evidence 是 Fact Layer。**

### LLM

> **LLM 是 Language Layer，不是真相层。**

### Performance

> **Playback FPS 与 Analysis FPS 解耦。**

### Product

> **用户的正常流程仍然只有：Upload → Analyze → Result。**

---

**本版本正式替代此前 App Local Architecture V1.1，并作为进入 Architecture Freeze 前的唯一实施基线。**