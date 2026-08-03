# Module 3 — Face Embeddings

Turns a detected face into the 512-dimensional vector everything downstream
depends on. Matching (Module 4), CNIC comparison (6), duplicate search (8) and
therefore the final recommendation are all functions of these numbers, so this
module has more leverage over the verification outcome than any other.

---

## 1. Model selection

`w600k_r50.onnx` from the InsightFace **buffalo_l** pack: ResNet-50 trained
with ArcFace loss on Glint360K (360k identities, 17M images).

| Alternative | Why not |
|---|---|
| ArcFace **r100** | Marginally better; roughly 2× the compute. On the CPU-only fallback path this service must support, that is not a trade worth making. |
| FaceNet / older MS1M models | Materially weaker, and MS1M is substantially less balanced across ethnicities — which matters directly for a Pakistani user base. |
| A separate detector's recogniser | Detector and recogniser would drift apart across deployments. buffalo_l ships both in one versioned artefact. |

**A quirk of this artefact.** The exported graph declares its output shape as
`[1, 512]` while the computation is genuinely batch-correct — a batch of four
returns `(4, 512)`. ONNX Runtime notices and logs a warning per call. The
session runs at `log_severity_level=3` so it stays quiet, and the decoder reads
the real array shape rather than the declared one. Worth knowing before someone
"fixes" the batching on the strength of that warning.

---

## 2. Why the vectors are L2-normalised

ArcFace is trained with an **angular margin on the hypersphere**: the loss only
ever sees the *direction* of the embedding, never its magnitude. Comparing
un-normalised vectors would mix a quantity the model optimised (angle) with one
it did not (length). Normalising also makes cosine similarity a plain dot
product, which Qdrant indexes directly in Module 8.

The discarded magnitude is reported as `raw_norm` — see §5 for why nothing is
decided on it.

---

## 3. Alignment is the whole ball game

ArcFace was trained exclusively on faces warped onto a fixed five-point
template at 112×112. The warp is estimated as a **similarity** transform
(rotation, uniform scale, translation — four degrees of freedom) using the
closed-form **Umeyama** solution.

**Not a full affine.** Six degrees of freedom would additionally shear and
independently scale the axes to fit the template exactly. That sounds like a
better fit and is actively harmful: it normalises away the inter-landmark
geometry that distinguishes one person's face from another's.

**Not an iterative estimator.** `cv2.estimateAffinePartial2D` with RANSAC or
LMEDS is non-deterministic. The embedding cache is keyed on the aligned crop,
so a warp that varied between calls would silently produce a different vector
for the same face. Umeyama is exact, closed-form and bit-reproducible — verified
by test.

### How much it matters — measured

| Comparison | Cosine |
|---|---|
| Same face: template-aligned vs box-only fallback | **0.786** |
| 25° rotated capture vs upright, **template-aligned** | **0.985** |
| 25° rotated capture vs upright, **box-only** | **0.573** |

That last row is the argument. At 0.573 the system would call the person a
stranger — purely because alignment was skipped. The box fallback exists so a
landmark-free detector still produces *something*, but it is flagged
`aligned=false` and forced to low confidence everywhere.

---

## 4. Measured behaviour

Reproduce with `python scripts/_calibrate_embeddings.py`.

### Identity survives degradation

Same person, one controlled defect at a time, against the pristine original:

| Variant | cos(original) | residual | raw_norm |
|---|---|---|---|
| original | 1.0000 | 0.0402 | 22.77 |
| rotated 12° | 0.9790 | 0.0560 | 22.63 |
| rotated 25° | 0.9767 | 0.0613 | 23.20 |
| blur 5×5 | 0.9817 | 0.0428 | 23.33 |
| blur 11×11 | 0.9597 | 0.0449 | 24.54 |
| blur 21×21 | 0.9006 | 0.0521 | 25.16 |
| dark ×0.4 | 0.9624 | 0.0406 | 22.69 |
| noise σ=20 | 0.9505 | 0.0340 | 22.87 |
| JPEG q30 | 0.9701 | 0.0462 | 22.92 |
| JPEG q8 | 0.9004 | 0.0490 | 24.10 |
| downscaled 4× | 0.9396 | 0.0411 | 25.15 |

### Separation from a different person

| | |
|---|---|
| Worst genuine pair | **0.8978** |
| Best impostor pair | **0.0112** |
| **Margin** | **+0.887** |

Impostor similarity sits at essentially zero. This is the property the entire
verification decision rests on, and it is exercised in
`tests/integration/test_embedding_service.py`.

> **Caveat.** This is a two-identity sample. It demonstrates the model is wired
> up correctly; it is **not** an accuracy evaluation. Module 4's thresholds
> need a proper labelled dataset with ROC analysis before production use.

---

## 5. What was tried and rejected

### `raw_norm` does not track quality on this model

The first design derived per-embedding confidence from ArcFace's
pre-normalisation L2 norm, on the MagFace/AdaFace premise that the magnitude
correlates with face quality.

**Measurement refuted it.** Across the twelve-step degradation ladder above the
norm moved only between **22.8 and 25.5** — and a 21-pixel Gaussian blur
measured **25.2 against the pristine face's 22.8**, i.e. the wrong direction.

The reason is straightforward in hindsight: MagFace and AdaFace *train* for
that correlation with a magnitude-aware loss. Vanilla ArcFace leaves the
magnitude unconstrained, so the property simply is not there to exploit.

`raw_norm` is still reported as a diagnostic. Nothing is decided on it.

### Confidence comes from the alignment residual instead

The residual — mean landmark-to-template distance after warping, in interocular
units — does carry signal, and sits in a tight, stable band:

| | Residual |
|---|---|
| Correctly-detected face, any degradation | **0.034 – 0.061** |
| A different individual's face geometry | 0.132 |
| Deliberately corrupted landmarks | rejected above 0.28 |

> **Known conflation.** The residual measures fit against a *population-average*
> template, so an individual with unusual facial proportions scores higher
> without anything being wrong. The second person in the demo reads 0.132 →
> confidence 0.68, comfortably above the 0.35 floor. The threshold is set with
> that headroom deliberately; tightening it would start penalising people for
> the shape of their face.

### Flip augmentation is off by default

Embedding the mirrored crop and averaging is standard InsightFace practice and
it does help — but the measured gain is small and the cost is exact:

| Variant | no-flip | flip-averaged | Δ |
|---|---|---|---|
| blur 5×5 | 0.9817 | 0.9872 | +0.0055 |
| JPEG q30 | 0.9701 | 0.9784 | +0.0083 |
| rotated 12° | 0.9790 | 0.9865 | +0.0074 |
| dark ×0.4 | 0.9624 | 0.9672 | +0.0048 |
| **mean** | | | **+0.0065** |

+0.0065 mean cosine for exactly **2× inference**. At ~148 ms/face on CPU that
turns a seven-image request from ~0.85 s into ~1.7 s. Off by default; worth
switching on for GPU deployments where the second pass is nearly free.

### Batching helps less than expected

| Batch size | ms/face |
|---|---|
| 1 | 148 |
| 8 | 121 |

**1.22×**, not an order of magnitude. ONNX Runtime already parallelises a single
112×112 ResNet-50 across all 12 cores, so batching mostly saves per-call
overhead rather than unlocking idle compute.

> A first measurement suggested 610 ms/face and no batching benefit at all.
> That was an artefact of the calibration script itself — three ArcFace sessions
> plus the detector alive simultaneously, contending for the same cores. Worth
> recording, because it is an easy trap: **ONNX latency measurements are only
> meaningful on an otherwise-idle machine.**

---

## 6. Caching

Keyed on the **SHA-256 of the aligned crop**, not the source image. Two
slightly different detections of the same face genuinely produce different
crops and different embeddings, so they must not share an entry; a genuine
repeat produces a byte-identical crop and is a guaranteed hit. The model
version is mixed in, so a recogniser upgrade invalidates everything rather than
silently serving vectors from the old space.

Measured: **591 ms cold → 2 ms warm** for a three-face batch (276×).

### A data-protection decision, not a performance one

An embedding is biometric data and is reversible enough through model inversion
to fall under GDPR Article 9.

| Backend | Behaviour |
|---|---|
| `memory` (default) | In-process, LRU-bounded, 600 s TTL. Dies with the pod. |
| `redis` | Shared store with its own retention and access controls. **Requires a deliberate decision.** |
| `none` | No caching. |

The vector is also excluded from the API response by default and never appears
in a log line — `FaceEmbedding.describe()` and `EmbeddingResult.summary()` both
omit it structurally, and there is a test asserting so.

---

## 7. Failure behaviour

There is **no fallback chain** here, unlike detection. An embedding from a
different model is not a degraded answer — it is an *incomparable* one,
occupying an unrelated vector space. `build_embedding_service` therefore raises
if the recogniser cannot be loaded, and `FaceEmbedding.similarity_to` refuses
to compare across model versions rather than returning a plausible-looking
number.

Within a batch, failures are per-face: one image with unusable landmarks
produces a failed result in its own slot while the other six embed normally.

---

## 8. Performance

| Operation | Measured |
|---|---|
| Single face, cold | ~148 ms |
| Batch of 8 | ~121 ms/face |
| Seven-image request | ~0.85 s |
| Fully cached repeat | ~2 ms |
| With flip augmentation | ~2× the above |

This is the pipeline's most expensive stage by a wide margin — Module 2's full
nine-dimension quality assessment costs ~75 ms for comparison. GPU deployment
is the obvious lever; the `DevicePlan` already selects CUDA/TensorRT
automatically when available.

---

## 9. Testing

| Suite | Count | Covers |
|---|---|---|
| `tests/unit/test_embedding_alignment.py` | 27 | Umeyama solve, residual, reflection guard, fallback, determinism |
| `tests/unit/test_embedding_cache_and_vectors.py` | 40 | Normalisation, cosine, cache LRU/TTL/keys, Redis degradation |
| `tests/integration/test_embedding_service.py` | 22 | Real ArcFace: separation, degradation floors, alignment, batching, cache |
| `tests/performance/test_embedding_performance.py` | 7 | Latency budgets, flip cost, memory, cache bounds |

The integration suite pins the measured floors with headroom, so it catches a
genuine regression rather than tracking noise.

---

## 10. Running it

```bash
python scripts/demo_embeddings.py
```

```bash
python scripts/demo_embeddings.py a.jpg b.jpg --save-crops out/
```

```bash
python scripts/_calibrate_embeddings.py
```

---

## 11. Configuration surface

Everything lives in `configs/thresholds.yaml` under `embedding:`.

```bash
HQ_EMBEDDING__FLIP_AUGMENTATION=true        # worth it on GPU
HQ_EMBEDDING__CACHE__BACKEND=redis          # needs a DPIA
HQ_EMBEDDING__BATCH__MAX_SIZE=32
```

`alignment.output_size` is validated to be exactly `[112, 112]`: the template's
five reference coordinates are absolute pixels in that frame, so any other size
would silently mis-place all of them and quietly degrade recognition rather
than failing.
