# Module 2 — Image and Face Quality Assessment

Measures nine quality dimensions on every submitted image and combines them
into the `image_quality_score` the Backend's rules engine consumes, together
with the `blur_score`, `brightness_score` and `noise_score` the requirements
name explicitly.

Needs **no model weights**. It is pure OpenCV and NumPy, so it cannot fail to
start and it runs identically on every host.

---

## 1. Why this module is not a set of thresholds

The obvious implementation — Laplacian variance below 80 means blurred — fails
immediately in production, for three reasons that shaped every decision here.

**Raw focus measures are not comparable across resolutions.** The same face
photographed at 4000 px and at 300 px produces Laplacian variances an order of
magnitude apart. A single threshold cannot be right for both.

**A threshold discards the information downstream needs most.** "Blur: fail"
tells the user nothing they can act on and tells the fraud engine nothing about
*how far* from acceptable the attempt was.

**An arithmetic mean launders a fatal defect into a passing grade.** With the
configured weights, an image that is perfectly exposed, perfectly contrasted,
noise-free, high-resolution and completely out of focus scores **78** under a
plain mean. That image is worthless for face recognition.

The three answers, in order:

| Problem | Answer |
|---|---|
| Cross-resolution comparability | Focus metrics measured on the face crop resampled to a fixed `canonical_face_size` |
| Information loss | Every raw measurement reported alongside a smoothstep-mapped `[0,1]` score |
| Fatal-defect laundering | Weighted **power mean** with exponent < 1, plus per-dimension **critical floors** |

---

## 2. The nine dimensions

| Dimension | Measured on | Sub-metrics | Weight |
|---|---|---|---|
| `blur` | Resolution-capped whole image | Laplacian, Tenengrad, spectral HF ratio | 0.22 |
| `sharpness` | Eye band of the canonical face crop | Eye-region Laplacian, face-vs-scene ratio | 0.16 |
| `resolution` | Face box + spectrum | Face short side, interocular, detail energy | 0.16 |
| `brightness` | Face region | Mean luminance (gate), shadow & highlight clipping | 0.12 |
| `noise` | Face region | Immerkær sigma, flat-block sigma, noise-to-signal | 0.11 |
| `contrast` | Face region | RMS, dynamic range, histogram entropy | 0.10 |
| `pixelation` | Whole image + native face crop | Phase-invariant JPEG blockiness | 0.09 |
| `distortion` | Whole image + landmarks | Banding, chromatic aberration, geometric anisotropy | 0.04 |

`blur` and `sharpness` are deliberately separate. Global blur answers *is the
photograph in focus*; face sharpness answers *is the face in focus*. A pin-sharp
background with a motion-blurred subject passes the first and fails the second,
and that case is both a common capture error and a weak signal of a replayed
video.

Exposure, contrast and noise are measured on the **face region** rather than
the frame. A selfie against a bright window has a perfectly healthy global
histogram and a silhouetted, unusable face.

---

## 3. Calibration — measured, not assumed

Every anchor in `configs/thresholds.yaml` was derived from measurement against
a reference portrait and a ladder of single controlled degradations. Reproduce
with:

```bash
python scripts/_calibrate_quality.py
```

### Measured raw values

| Measurement | pristine | gauss 3×3 | gauss 9×9 | gauss 21×21 |
|---|---|---|---|---|
| Laplacian variance | 10 014 | 2 150 | 275 | 37 |
| Tenengrad | 17 119 | 10 804 | 5 023 | 1 947 |
| HF energy ratio (>0.25 Nyquist) | 0.0189 | 0.0089 | 0.0012 | 0.00002 |
| Eye-band Laplacian | 19 058 | 9 445 | 2 536 | 396 |

| Measurement | pristine | 2× upscaled | 4× upscaled |
|---|---|---|---|
| Detail energy (>0.5 Nyquist) | 4.2 × 10⁻³ | 5.2 × 10⁻⁴ | 2.5 × 10⁻⁵ |

| Blockiness | q95 | q70 | q50 | q30 | q20 | q8 | q5 |
|---|---|---|---|---|---|---|---|
| value | 0.18 | 0.13 | 0.49 | 0.65 | 0.75 | 1.22 | 1.81 |

### Why log scale

Focus measures span **three orders of magnitude**. A linear ramp across
40 → 4000 assigns everything below ~1000 a score under 0.25 and cannot rank the
degradations at all. `log_scale: true` interpolates in log₁₀ space, so the
midpoint is the geometric mean and each decade gets equal resolution.

### Resulting end-to-end scores

| Variant | Overall | Verdict | Limiting |
|---|---|---|---|
| pristine | 99.7 | usable | noise |
| gauss 3×3 | 94.5 | usable | resolution |
| gauss 9×9 | 70.4 | usable | blur |
| **gauss 21×21** | **49.9** | **REJECTED** | blur — critical |
| camera shake (15 px) | 74.8 | usable | blur |
| underexposed (×0.35) | 60.9 | usable | brightness |
| washed out | 64.8 | usable | blur |
| noise σ=20 | 94.1 | usable | noise |
| JPEG q30 | 97.0 | usable | pixelation |
| JPEG q8 | 83.8 | usable | pixelation |
| 2× upscaled | 95.9 | usable | resolution |
| 4× upscaled | 75.5 | usable | blur |
| posterised | 95.5 | usable | brightness |
| stretched 1.6× | 96.9 | usable | distortion |

---

## 4. Four defects found by measurement

Each of these was discovered by running the metrics against real pixels and
finding they did not discriminate. They are recorded because the fixes are
non-obvious and a future change could reintroduce any of them.

### 4.1 Blockiness returned 0.0 on a perfect block pattern

A divide-by-zero guard returned `0.0` when off-grid variation was zero — which
is the *strongest* possible blocking signal, a flat image made entirely of hard
block edges. The guard was protecting the division and inverting the metric at
its own extreme.

### 4.2 Blockiness assumed grid phase 0

A face crop taken at an arbitrary offset shifts the JPEG 8×8 grid by
`offset % 8`. Only testing phase 0 reported a severely blocked quality-8 JPEG
as clean **seven times out of eight**. The metric now searches all eight phases
and takes the maximum. Verified: a synthetic block pattern reads 2.0 at phase 0
and 2.0 shifted by 3.

### 4.3 Isotropic focus operators hide half of directional blur

A 41-px horizontal smear left the Laplacian variance at **2665** — comfortably
"sharp" — because horizontal blurring removes only horizontal-frequency content
and every vertical edge survives. Both derivative measures are now evaluated
per axis and scored on the **weaker** one:

| | isotropic | worst-axis |
|---|---|---|
| sharp | 88 582 | 78 853 |
| **41-px horizontal smear** | **2 665** | **38** |
| 9×9 Gaussian defocus | 99 | 77 |

The correction is a near-no-op on isotropic content, so the same anchors remain
valid. End-to-end, `camera shake` moved from 86.6 to **74.8**.

### 4.4 Mean luminance must gate clipping, not average with it

A silhouetted face scored **45**: its mean-luminance term was zero, but both
clipping terms were perfect and two thirds of the weight outvoted the one
measurement that mattered. Whether a silhouette's shadows are technically
clipped is meaningless — there is no signal in that region either way.

Now:

```
score = mean_score × (1 − clipping_influence × (1 − clipping_retention))
```

`dark_x0.35` brightness moved from 36.3 to **5.5**.

---

## 5. What this module does *not* claim

**It cannot distinguish upscaling from blur.** An earlier version reported an
inferred "upscale factor" from a spectral cliff. Measurement withdrew it: on
already-compressed photographs a 2× cubic upscale produces no clean cliff,
because interpolation ringing and codec noise repopulate the band above the
theoretical cutoff. Its radial profile was indistinguishable in shape from a
mildly blurred capture. The module now reports **detail energy** — how much
genuine high-frequency content is present relative to the nominal pixel count —
and says exactly that. For the question this service exists to answer, the two
defects are equivalent.

**Geometric anisotropy has a noise floor around 0.12.** The comparison template
is a population average, so an individual face legitimately differs from it: the
reference portrait reads 0.12 with no distortion whatsoever. Only stretches
beyond roughly **1.35×** are reliably detectable.

**Anisotropy is withheld beyond 20° of yaw.** Perspective foreshortening is
geometrically indistinguishable from an editor's horizontal squeeze, so the term
is reported as unmeasured rather than guessed at.

**Absolute scores are not calibrated against verification accuracy.** The scores
are monotone in degradation and the ordering is sound, but mapping "quality 84"
onto an expected false-match rate needs a labelled dataset this project does not
have. The role thresholds in `configs/thresholds.yaml` are engineering defaults
and should be re-derived against production traffic.

**Two known confounds**, both documented rather than hidden:
darkening depresses gradient magnitude and so drags `sharpness` down with
`brightness`; and additive noise inflates the chromatic-aberration measure.

---

## 6. Aggregation

```
composite = ( Σ wᵢ · scoreᵢ^p )^(1/p)     with p = 0.5
```

The power mean with `p < 1` is pulled towards its smallest component. Five
perfect dimensions and one at zero give 0.80 under an arithmetic mean and
**0.64** under this one.

A power mean still cannot express "this dimension is disqualifying whatever the
others say", so any component in `critical_components` at or below
`critical_floor` (0.12) sets `usable=False` outright. Only `blur`, `sharpness`
and `resolution` are critical: exposure and contrast degrade recognition but do
not destroy the signal, and a dim-but-sharp photograph is still workable.

Unmeasured dimensions are **dropped and the remaining weights renormalised**.
Scoring an unmeasured dimension zero would punish an image for a measurement
nobody took; scoring it one would flatter it.

Both the composite and the plain arithmetic mean are returned, so the power-mean
adjustment is auditable rather than mysterious.

### Per-role thresholds

| Role | Minimum |
|---|---|
| `live_selfie` | 55 |
| `profile_image` | 50 |
| `secondary_image` | 45 |
| `cnic_image` | 35 |
| `cnic_portrait` | 25 |

A CNIC portrait is a sub-300-dpi print photographed through a laminate. Holding
it to the live-selfie bar would reject every genuine document.

---

## 7. Performance

Measured on the CPU-only development host, 512×600 reference portrait:

| Stage | Median |
|---|---|
| Full assessment | **75 ms** |
| 1080p upload | < 700 ms (budget) |
| 12 MP upload | < 1500 ms (budget) |
| Seven-image request, concurrent | < 4000 ms (budget) |

Three optimisations carry that:

1. **Shared context.** Grayscale, the canonical crop and the radial spectrum are
   each needed by three or four analysers and are computed once
   (`functools.cached_property`). A performance test asserts the FFT is cached
   rather than recomputed.
2. **Analysis-size cap.** Global metrics run on the image reduced to
   `global_analysis_long_side` (1024). A regression test asserts that 16× the
   pixels costs less than 8× the time.
3. **Off-loop execution.** The whole assessment runs on a worker thread; OpenCV
   and NumPy release the GIL, so concurrent requests get real parallelism.

---

## 8. Testing

| Suite | Count | What it covers |
|---|---|---|
| `tests/unit/test_quality_scoring.py` | 49 | Smoothstep, ramp/band mappings, log scale, power mean |
| `tests/unit/test_quality_metrics.py` | 68 | Each analyser against one controlled degradation |
| `tests/unit/test_quality_aggregation.py` | 34 | Power mean, critical floors, weight renormalisation |
| `tests/unit/test_quality_service.py` | 35 | Orchestration, roles, contract, async, containment |
| `tests/integration/test_quality_pipeline.py` | 12 | Detection → quality on a real photograph |
| `tests/performance/test_quality_performance.py` | 7 | Latency budgets, caching, memory |

Every metric test applies **one** defect and asserts the responsible dimension
moves while the others hold. That cross-check matters as much as the direction —
a blur metric that also fires on darkness is not measuring blur.

---

## 9. Running it

```bash
python scripts/demo_quality.py --role live_selfie
```

```bash
python scripts/demo_quality.py selfie.jpg --measurements --annotate out/
```

```bash
python scripts/_calibrate_quality.py
```

---

## 10. Configuration surface

Every number lives in `configs/thresholds.yaml` under `quality:`. Nothing is
hard-coded. Anchors carry the measured values they were derived from as
comments, so a future engineer retuning them can see what the current numbers
were fitted to.

Override any of it from the environment:

```bash
HQ_QUALITY__AGGREGATION__POWER=0.4
HQ_QUALITY__ROLES__LIVE_SELFIE__MIN_OVERALL=60
```

Weights are normalised to sum to 1.0 on load, so raising one does not require
rebalancing the rest by hand.
