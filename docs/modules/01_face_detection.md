# Module 1 — Face Detection

## 1. Requirement

From the specification:

> Detect faces · Reject no-face images · Reject multiple-face images ·
> Detect face visibility · Detect face angle · Detect occlusion
>
> Return: `face_detected`, `face_count`, `face_visibility_score`,
> `bounding_box`, `pose`, `confidence`

## 2. Design

### 2.1 Separation of concerns

```
DETECTION            ANALYSIS                     POLICY
─────────            ────────                     ──────
Where are            How is this face             Is this image
the faces?           oriented / obstructed        acceptable?
                     / visible?
scrfd.py             pose.py                      policy.py
yolo_face.py         occlusion.py
opencv_dnn.py        visibility.py
haar.py
   │                     │                           │
   └──────── factory.py ─┴───── face_detection_service.py
                (chain)              (orchestration)
```

A detector answers *"where are the faces"*. The policy answers *"is this image
acceptable for identity verification"* — pure business logic, driven entirely by
`configs/thresholds.yaml`. Keeping them apart means the rules are unit-testable
against synthetic detections with no model weights present, which is why
`tests/unit/test_policy.py` has 24 tests and runs in milliseconds.

### 2.2 Pipeline for one image

```
normalise size ──► detect ──► per-face pose
                           ──► per-face occlusion
                           ──► per-face visibility
                           ──► acceptance policy ──► FaceDetectionResult
```

## 3. Model selection

### 3.1 Primary: SCRFD-10GF

`det_10g.onnx` from the InsightFace `buffalo_l` pack.

| Reason | Detail |
|---|---|
| Accuracy per FLOP | ~95.4% AP on WIDER FACE *hard*. RetinaFace-R50 needs roughly 10× the compute for a comparable number. On the CPU-only nodes this service must fall back to, that is the difference between a 300 ms and a 3 s detection stage. |
| Small-face recall | Sample-redistribution training deliberately reweights supervision towards small faces — exactly the failure mode that matters for a CNIC portrait occupying 4% of the frame. |
| Landmarks for free | The `bnkps` variant emits five keypoints in the same forward pass. Those drive ArcFace alignment (Module 3), pose estimation, and the occlusion region layout. A separate landmark model would add a second inference per face. |
| Shared provenance | Detector and embedder come from one versioned artefact, so they cannot drift apart across deployments. |

**Implementation note.** Post-processing is implemented directly against ONNX
Runtime rather than through `insightface.model_zoo`. That package pulls a large
dependency tree, downloads weights implicitly at import time into a user-home
cache, and gives no control over execution providers or session options — all
disqualifying in a container that must be reproducible and offline-capable. The
decoder is ~120 lines, fully under our control, and verified to sub-pixel
accuracy against synthetic tensors in `tests/unit/test_scrfd_decode.py`.

### 3.2 Fallback chain

| Order | Adapter | Landmarks | Why it is in the chain |
|---|---|---|---|
| 1 | `scrfd` | 5, predicted | Primary. |
| 2 | `yolo` | 5, predicted | Different failure surface — better on large, close-up, off-angle faces where the anchor-free FPN sometimes gives a loose box. |
| 3 | `opencv_dnn` | none | Different lineage entirely: Caffe, different training set, different failure modes from the two anchor-free detectors above. |
| 4 | `haar` | 5, **derived** | Ships *inside* the OpenCV wheel. Needs no download, no network, no GPU, no model store. Whatever else fails — wiped volume, firewall, corrupt artefact — this still runs. |

The chain handles two distinct failures: **construction** (weights absent →
adapter skipped at start-up) and **inference** (a forward pass raises → next
adapter tried for that image). Every result reports `detector`,
`detector_version` and `used_fallback`, so degradation is visible, never silent.

**On the Haar landmarks.** Viola-Jones produces no keypoints. The adapter
recovers an eye pair from a second cascade and *constructs* the remaining three
points from the canonical face template. Those points are flagged
`landmarks_derived=True`, and the pose estimator refuses to run a PnP solve on
them — a constructed nose tip would confidently report a perfectly frontal pose
for a face at 30° of yaw. It reports roll only, and the result carries a
`LANDMARKS_DERIVED` warning.

## 4. Analysis components

### 4.1 Head pose (`pose.py`)

Two estimators, in order:

1. **Perspective-n-Point.** Solves for the rigid transform projecting a
   canonical 3D face model onto the detected landmarks, then decomposes the
   rotation into Euler angles. `SOLVEPNP_SQPNP` is used rather than the classic
   iterative solver: it is global and non-iterative, and does not fall into the
   mirror-ambiguity local minimum that `ITERATIVE` hits on near-frontal faces —
   the case that dominates this workload.
2. **Landmark geometry.** Closed-form approximation used when PnP fails to
   converge, when its reprojection error exceeds 0.35 × interocular distance,
   or when the landmarks were derived.

**Camera model.** Phone EXIF rarely survives the Flutter upload path, so
intrinsics are unknown. The standard substitute is used — focal length = image
width, principal point = image centre, which is a ~53° horizontal FOV, close to
essentially every phone main camera. Absolute angles carry a few degrees of
systematic error for unusual optics, but the thresholds are calibrated against
the same assumption, so the accept/reject boundary is unaffected.

**Sign convention** — enforced by test, not by comment
(`test_pnp_recovers_known_orientation` projects the 3D model at known angles and
asserts recovery):

- `yaw > 0` — subject turned towards the viewer's right
- `pitch > 0` — chin raised
- `roll > 0` — head tilted so the viewer-right eye moves down

**Deviation score** is the *maximum* per-axis ratio to that axis's hard limit,
not an average. 45° of yaw makes a face unusable no matter how perfect its pitch
and roll are, and an average would dilute exactly the signal that matters.

### 4.2 Occlusion (`occlusion.py`)

There is no public, redistributable occlusion classifier of production quality,
and training one demands a labelled dataset this project does not have.
Inventing a model would mean shipping something untested. So Module 1 uses a
**multi-signal geometric estimator** — fully implemented, deterministic,
weight-free and honest about being an estimator.

Every face is first warped into a **canonical 128×128 frame** using the eye pair
and the mouth centre (three points → full affine, which also absorbs the
vertical foreshortening pitch introduces). Region boxes are then fixed
rectangles in that frame, so head tilt, scale and position stop mattering.

Three signals per region:

| Signal | Weight | What it catches |
|---|---|---|
| **Flat fraction** — share of pixels whose gradient is below half the face-wide median | 0.50 | Any occluder, *regardless of colour*. Measured as a fraction rather than mean gradient energy because a hard-edged occluder (a sunglasses bar) contributes strong gradients along its own boundary, so mean energy of a covered region can be *higher* than bare skin. Measured separation on a real portrait: 0.06 bare eye vs 1.00 covered. |
| **Skin coverage** — fraction inside the YCrCb skin ellipse | 0.32 | Non-skin coverings. Chrominance-only, so illumination-invariant and robust across skin tones — essential for a Pakistani user base. |
| **Colour uniformity** — inverse CIELAB a*/b* spread | 0.18 | Manufactured surfaces, which are far more uniform in colour than skin. |

No single signal suffices: a dark-skinned subject in low light drags skin
coverage down, a smooth forehead is genuinely low-texture, and a skin-coloured
occluder defeats chrominance entirely. The weighted evidence is sharpened by a
logistic; without it the estimator produces a mush of mid-range probabilities
that no threshold separates.

Two special cases:

- **Sunglasses** get a dedicated term. Dark lenses are neither colourful nor
  textured enough to trip the three general signals decisively, so a markedly
  dark *and* flat eye region raises the evidence directly.
- **Forehead** evidence is damped to 0.55 and its configured weight is only
  0.10, because hair, a fringe or a headscarf legitimately covers it for a large
  fraction of genuine users. Treating that as fraud would be a serious product
  failure.

A **symmetry** term is computed separately: a hand covering one cheek produces
large left/right asymmetry with only a moderate overall score, which the
per-region scores alone would miss.

An optional ONNX classifier can be enabled in `configs/models.yaml`; when
present its outputs take precedence and the geometric estimator becomes a
cross-check supplying the measured signals that make a flag explainable. No
weights are shipped, because shipping unvalidated weights is worse than
shipping none.

### 4.3 Visibility (`visibility.py`)

`face_visibility_score` is the number a human reviewer looks at first, so the
composite is published *with its full breakdown*.

| Component | Default weight |
|---|---|
| `detector_confidence` | 0.20 |
| `occlusion` (1 − occlusion score) | 0.30 |
| `pose` (1 − deviation) | 0.22 |
| `face_size` | 0.16 |
| `framing` (1 − truncation) | 0.12 |

`face_size` uses a **plateau** over the ideal 6–45% area band rather than a ramp
to a single ideal point — otherwise a perfectly good portrait is penalised for
being 8% of the frame instead of 20%, which is not a real defect.

**Missing components are redistributed, not defaulted.** The landmark-free
detectors produce no pose or occlusion analysis. Substituting a neutral 1.0
would flatter a face nobody examined, so the affected weights are spread
proportionally over the components that *were* measured and the result carries a
`LANDMARKS_UNAVAILABLE` warning. A face scored on three components is scored
honestly on three components.

Weights are normalised to sum to 1.0 at config load, so an operator can raise
one without rebalancing the rest.

### 4.4 Policy (`policy.py`)

Checks run cheapest-and-most-actionable first, and the **first** failure
determines the error code. That ordering is a product decision: telling a user
"no face detected" when the real problem is sunglasses sends them round a loop
they cannot exit.

1. Any detections at all? → `FACE_NOT_DETECTED`
2. Per-face admissibility: confidence, pixel floor, area ratio, truncation
3. Bystander filter, then the single-person rule → `MULTIPLE_FACES_DETECTED`
4. Primary-face selection (area, weighted by centrality and confidence)
5. Quality gates: pose → `FACE_POSE_OUT_OF_RANGE`, occlusion → `FACE_OCCLUDED`,
   visibility → `FACE_NOT_VISIBLE`

**Bystanders.** A face smaller than 35% of the largest is treated as background,
not a second person. Without this, every selfie taken in a public place is
rejected — which is the single most common false-reject in consumer KYC.

**Near-miss messaging.** When every face is inadmissible, the face that came
*closest* to passing determines the message, because that is the one the user
has a realistic chance of fixing.

## 5. Optimisation

| Technique | Effect |
|---|---|
| Anchor grids precomputed per input size | ~1.5 ms/image saved — 15–20% of CPU decode cost |
| Vectorised NMS (no Python loop over survivors) | Detectors emit thousands of candidates; the naive version dominates CPU latency |
| Single vectorised coordinate un-map for all survivors | vs per-face transform |
| Source downscaled to `max_source_long_side` (1920) | A 12 MP phone photo costs the same as a 2 MP one — asserted by `test_large_source_is_downscaled_not_processed_at_full_size` |
| Occlusion analysis capped at the largest 8 faces | A 32-face crowd image costs <4× a single face — asserted by test |
| Expensive analysis skipped for faces that cannot pass admissibility | Confidence and pixel floor checked before the warp |
| Eager start-up warm-up (`runtime.warmup_iterations`) | Moves 300–800 ms of allocator + kernel-autotune cost out of the first real request |
| ONNX graph optimisation `ORT_ENABLE_ALL`, sequential execution mode | Intra-session parallelism would oversubscribe the CPU against the request-level thread pool |
| cgroup CPU quota honoured for thread counts | ORT otherwise sees host cores and oversubscribes badly in a limited container |

Measured on the reference machine (Windows 11, CPU-only, `CPUExecutionProvider`,
640×640 input):

```
test_benchmark_no_face_path           min 269.9 ms   median 277.2 ms
test_benchmark_detector_only          min 276.9 ms   median 285.1 ms
test_benchmark_end_to_end_detection   min 283.7 ms   median 292.7 ms
```

The whole analysis stack — pose, occlusion, visibility, policy — costs ~7 ms.
The forward pass dominates entirely, which is the expected shape on CPU and
means GPU deployment will move the needle by roughly an order of magnitude.

## 6. Tests

| File | Tests | Covers |
|---|--:|---|
| `unit/test_redaction.py` | 49 | The PII guarantee: key dropping, masking, regex scrubbing, nesting limits, pseudonymisation |
| `unit/test_image_ops.py` | 39 | Letterbox coordinate round-trip, cropping, blobs, alignment, skin mask |
| `unit/test_geometry.py` | 38 | Boxes, landmarks, NMS, anchor decoding |
| `unit/test_config.py` | 34 | Layering, normalisation, validation, production hardening |
| `unit/test_image_io.py` | 30 | Codec hardening, decompression bombs, EXIF, base64 transport |
| `unit/test_pose.py` | 29 | PnP round-trip at known angles, sign convention, fallback tolerance |
| `unit/test_tempfiles.py` | 27 | The no-persistence guarantee: shredding, traversal, janitor |
| `unit/test_policy.py` | 24 | Every rule and every ordering guarantee |
| `unit/test_haar_detector.py` | 20 | Terminal fallback; derived-landmark honesty |
| `unit/test_scrfd_decode.py` | 19 | Decoder vs synthetic tensors, all 3 FPN levels, un-mapping |
| `unit/test_visibility.py` | 19 | Weighting, plateau, redistribution, limiting factor |
| `unit/test_occlusion.py` | 15 | Clean/sunglasses/mask/one-sided; roll and scale invariance |
| `integration/…` | 25 | Real SCRFD weights, real photograph, tampered-artefact refusal |
| `performance/…` | 11 | Benchmarks plus scaling and concurrency assertions |

Current state: **343 unit + 25 integration + 11 performance = 379 passing, 80%
statement coverage, `ruff` clean.**

Two of these files exist because the guarantee they protect is a *security
control*, not a feature. `test_redaction.py` is what makes "access logging"
(section 21) and "CNIC data is confidential" (agreement section 10) coexist:
the log pipeline must be physically incapable of emitting the confidential
parts. `test_tempfiles.py` covers all four cleanup mechanisms separately,
because each handles a failure the others cannot — in particular the janitor,
which is the only one that survives a `SIGKILL` or an OOM kill.

Unit tests use **synthetic faces** — procedurally drawn images whose ground-truth
landmarks are known exactly and which can be perturbed deterministically. That
makes them fast, hermetic and reproducible, and it means no real identity
document ever enters the repository. Integration tests use the US Navy portrait
of Rear Admiral Grace Hopper bundled inside matplotlib: a genuine photograph of
a real face, public domain, already on disk.

The decoder is tested **by construction** rather than by eyeballing a demo image:
an off-by-one in the anchor grid or a forgotten stride multiplication produces
boxes that are *plausible* rather than obviously broken, and no amount of visual
inspection reliably catches a two-pixel systematic bias.

## 7. Demonstration

```bash
python scripts/download_models.py --write-lock
python scripts/demo_face_detection.py --output-dir reports/demo
```

Output on the reference machine:

```
scenario           passed   faces   vis     error
clean_portrait     True     1       89.2    -
no_face            False    0       0.0     FACE_NOT_DETECTED
two_people         False    2       0.0     MULTIPLE_FACES_DETECTED
sunglasses         False    1       80.9    FACE_OCCLUDED
face_mask          False    0       0.0     FACE_NOT_DETECTED
tiny_face          False    0       0.0     FACE_TOO_SMALL
truncated          False    0       0.0     FACE_NOT_DETECTED
```

Annotated images are written showing boxes, landmarks, pose axes and the
occlusion verdict per region.

Two observations from this run worth recording:

- **`face_mask`** reports `FACE_NOT_DETECTED` rather than `FACE_OCCLUDED`. SCRFD
  emitted one candidate but below the confidence floor, so it never reached
  occlusion analysis. The message — *"No face could be identified with
  sufficient confidence"* — is accurate, but a mask-specific message would serve
  the user better. Addressed in Module 7, which analyses masks explicitly.
- **`truncated`** reports `FACE_NOT_DETECTED` rather than `FACE_TRUNCATED`,
  because SCRFD found nothing at all in that crop. The `FACE_TRUNCATED` path is
  exercised in `tests/unit/test_policy.py` instead.

## 8. Deployment

### 8.1 Model artefacts

```bash
python scripts/download_models.py            # fetch + compute digests
python scripts/download_models.py --write-lock   # pin them
git add configs/model_digests.lock.yaml      # commit and review
python scripts/download_models.py --verify-only  # CI gate
```

`buffalo_l.zip` is 288 MB and yields both `det_10g.onnx` (17 MB, Module 1) and
`w600k_r50.onnx` (174 MB, Module 3) — one download for both.

In a container image, run the download at **build time** and bake the artefacts
in. Fetching at start-up makes pod scheduling depend on GitHub availability.

### 8.2 Configuration

| Variable | Production value | Why |
|---|---|---|
| `HQ_APP__ENVIRONMENT` | `production` | Activates the hardening gate |
| `HQ_RUNTIME__DEVICE` | `auto` | Degrades to CPU with a visible warning |
| `HQ_RUNTIME__INFERENCE_WORKERS` | ≈ CPU limit | Bounded; each session holds an arena |
| `HQ_RUNTIME__WARMUP_ON_STARTUP` | `true` | Keeps first-request latency off the user |
| `HQ_DETECTION__PRIMARY` | `scrfd` | |
| `HQ_MODEL_STORE__VERIFY_CHECKSUM` | `true` | Enforced by the hardening gate |

The production hardening gate refuses to boot with `debug=true`, empty API keys,
wildcard CORS, console logging, disabled redaction or disabled checksum
verification. Failing at start-up is correct: a pod that crash-loops with a
clear error is far better than one quietly serving unverified results.

### 8.3 Sizing

CPU-only, 640×640: ~290 ms per image, ~3.4 images/s per worker. A verification
request carries up to seven images, so budget ~2 s of detection per request at
concurrency 1, less with the thread pool overlapping. `inference_workers: 4` on
a 4-core pod sustains roughly 12 images/s.

GPU (CUDA) moves the forward pass into the 10–20 ms range; the ~7 ms analysis
stack then becomes a meaningful share and the thread pool, not the GPU, becomes
the limiting factor.

### 8.4 Health

`/health` (Module 10) reports per-model load state, the resolved device plan and
whether it is `degraded`. Readiness fails until every model marked `required`
has loaded. A `degraded: true` on a node expected to have a GPU should page —
that node is silently running 20× slower than its capacity plan assumes.

## 9. Known limitations

| Limitation | Mitigation |
|---|---|
| Occlusion is a geometric estimator, not a learned classifier | Fully documented; ONNX classifier plug-in point implemented and ready for weights |
| Pose assumes a generic pinhole camera | Thresholds calibrated against the same assumption; error is systematic, not random |
| Haar-derived landmarks encode no yaw or pitch | Flagged `landmarks_derived`; PnP refused; `roll_only` reported; warning raised |
| YOLO-face has no canonical signed release | Source set to `manual`; adapter fully implemented and will activate if an operator supplies a vetted export |
| Thresholds are principled but not yet dataset-calibrated | Requires a labelled Hamqadam validation set; the calibration protocol is the first task of the accuracy work in Module 10 |
