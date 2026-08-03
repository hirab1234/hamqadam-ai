# Module 6 — CNIC Face Matching

Locates the portrait printed on a Pakistani CNIC and compares it with the live
selfie.

Module 4 already owns the *comparison*: `selfie_vs_cnic` thresholds, calibration
and fusion weights. What nobody had built is the part before it — deciding which
pixels on a photographed card are the portrait. That decision is the whole of
this module, and getting it wrong is not a degraded result but a bypassed check.

---

## 1. The attack

A user holds their CNIC up in front of their own face and photographs it. That
is not an exotic attack. It is how people naturally photograph a card they are
holding.

A detector run over that frame finds **two** faces: the small printed portrait,
and the large live face behind it. Take the largest — as almost every naive
implementation does — and you take the live face. The comparison becomes "is
this selfie the same person as this selfie", which passes at near-perfect
similarity **whoever the card belongs to**.

Measured on the fixture: the live face scores **cosine 0.71** against the selfie.
The CNIC strong-match threshold is 0.42. Picking it passes, every time, with
room to spare.

### Three layers stop it

**1. Rectification.** Warping the card to its own detected quadrilateral crops
away everything that is not on it. A face behind the card ceases to exist. This
is the real defence — it rests on the pixels no longer being there, not on a
threshold. The other two cover the case where no card border could be found,
which is common: a card held against a *face* has no clean contour, whereas a
card on a desk does.

**2. An upper area bound.** An ID-1 card is 85.6 × 53.98 mm and its photo box is
a fixed physical size, so the face inside occupies a bounded fraction of the card
at *any* capture resolution. That makes the bound a geometric property of the
document rather than a tuned number.

| | face area, as a share of the frame |
|---|---|
| genuine printed portrait | **0.0125** |
| live face, held-up card | **0.183** |
| configured ceiling | 0.10 |

The ceiling is derived from proportion — a generous 30 × 40 mm photo box is 26%
of the card, and a passport-style face fills roughly a third of its box, so ~10%
is the top of the plausible range. It lands an order of magnitude above the
genuine case and well below the attack.

**3. Template bands.** The photo box sits in a known horizontal band. A face over
the printed fields in the middle of the card is not the portrait. Containment is
graded, not binary — a hard test discards a genuine portrait whose card was
cropped a few percent tight, and grading lets it through with the confidence
penalty it deserves.

### The lower bound is pixels, not a fraction

My first version expressed both bounds as area ratios. That was a formulation
error, and the ghost-portrait fixture caught it: a genuine second face was
rejected at 0.35% against a 0.4% floor.

Whether a face can be recognised depends on **how many pixels it has**, not what
fraction of the frame it covers. A ratio floor conflates "this is not a face"
with "the card was photographed small" — and rejects the genuine portrait on any
card that does not fill the frame, which is most of them. Guarding against
detector hallucinations is what the confidence floor is for.

So the minimum is `min_face_pixels: 40` on the shorter side: below roughly that,
the upscale into ArcFace's 112 × 112 input is pure interpolation.

### Priors rank and validate; they never crop

Nothing blind-crops the expected region. Template revisions move the photo box,
and a prior that overrode the detector would fail closed on a layout nobody
anticipated — silently, and on genuine cards. Both sides of the card are listed
as candidate bands because the layout has moved between CNIC generations, and
this module declines to assert one.

---

## 2. Two faces on a card is normal

Modern NADRA cards print a faded secondary reproduction of the portrait as a
security feature. Treating "more than one face" as tampering would reject every
one of them.

The distinction that matters is *where*:

| | meaning |
|---|---|
| second face **on** the card | ghost reproduction — expected, reported, not a signal |
| face **not plausibly on** the card | `foreign_face_count` — the fraud-relevant count |

`foreign_face_count` is the field to read when a match score looks too good. A
live face held behind the card produces a near-perfect comparison *and* a
non-zero count; the score alone cannot distinguish that from a genuine
verification, and the count can.

It is a signal, not proof. A person in shot behind a card on a desk produces the
same reading, which is why it is surfaced for Module 9 to weigh rather than acted
on here.

A face rejected merely for low detector confidence is **not** counted as foreign
— it is probably still on the card and just faint, and counting it would raise a
fraud signal on every dim photograph.

---

## 3. A defect this module found in Module 2

Module 2 already varied `min_overall` by role: 55 for a live selfie, 25 for a
CNIC portrait. But `critical_components` — the floor below which any single
component makes an image unusable outright — was **global**. So a CNIC portrait
tripped the sharpness floor and was marked unusable before its own carefully-set
threshold of 25 was ever consulted. Half the role policy was doing nothing.

The critical floor on focus is right for a live capture, where softness means
camera shake and the image really is worthless. A print is a different object:
low high-frequency content is the medium, not a fault.

Measured, downsampling a reference face harder and harder before printing it:

| downsample | quality | sharpness | cosine (same) | cosine (impostor) |
|---|---|---|---|---|
| 1× | 89.3 | 67.93 | 0.976 | 0.004 |
| 2× | 78.9 | 22.17 | 0.945 | −0.018 |
| 3× | 70.0 | 2.08 | 0.881 | −0.010 |
| 4× | 67.2 | 0.14 | 0.710 | −0.009 |
| 6× | 66.9 | **0.00** | **0.551** | −0.028 |

Sharpness reaches zero while ArcFace still separates the same person at 0.551
from an impostor at −0.028, against a threshold of 0.42. Enforcing the focus
floor there refuses a decisive match on the grounds that a print looks like a
print.

`RoleQualityConfig` now carries an optional `critical_components` override, and
the two CNIC roles set it to `[resolution]` — a portrait genuinely too small to
resolve is a real failure and keeps its floor, while the composite floor of 25
still catches a card photographed out of focus.

### One threshold, one place

Module 6 has deliberately **no** portrait-quality threshold of its own. It reads
`quality.usable`, which already encodes the role's composite floor and its
critical components. An earlier draft had a `min_portrait_quality` in this
module's config; that would have been two places to change one policy, and they
would have drifted.

---

## 4. Pipeline order

```
rectify  ->  upscale  ->  detect every face  ->  locate_portrait
                                                       |
                                          quality (CNIC role)
                                                       |
                                             embed (CNIC role)
                                                       |
                                   compare_pair against the selfie
```

**Rectify before detecting**, not after. Filtering afterwards would work, but it
leaves the containment argument resting on a threshold rather than on the pixels
no longer existing.

**Upscale before detecting.** On a 1012 px rectified card the portrait's face is
around 120 px across, close to where detectors start missing. Enlarging to
`detection_min_width` costs milliseconds and recovers real portraits.

**Quality before embedding, and quality is allowed to say no.** An illegible
print embeds to a vector that is not wrong so much as meaningless, and a
meaningless vector produces a match score somebody will act on.

**Landmarks are matched by detector index**, not by geometry. Matching by
proximity could hand the primary portrait the ghost's keypoints, which would
misalign the crop and silently degrade the comparison.

---

## 5. Measured

Synthetic cards, public-domain reference portraits, ONNX Runtime on CPU:

| scenario | verdict | score | cosine | faces | ghost | foreign | rectified | ms |
|---|---|---|---|---|---|---|---|---|
| my card, flat | MATCH | 87.3 | 0.706 | 1 | – | 0 | – | 2516 |
| my card, with ghost print | MATCH | 87.2 | 0.704 | 2 | **y** | 0 | – | 2521 |
| a stranger's card | differ | 16.1 | 0.096 | 1 | – | 0 | – | 2568 |
| my card, bystander in shot | MATCH | 86.6 | 0.690 | 1 | – | 0 | **y** | 2555 |
| **ATTACK** stolen card @0.52 | **none** | – | – | 2 | – | **1** | – | 893 |
| **ATTACK** stolen card @0.70 | **differ** | 11.2 | 0.067 | 2 | – | **1** | – | 2066 |
| **ATTACK** stolen card @0.88 | **differ** | 14.7 | 0.088 | 1 | – | 0 | – | 2396 |
| my own card held up @0.70 | MATCH | 87.8 | 0.717 | 2 | – | **1** | – | 2538 |
| card with no portrait | none | – | – | 0 | – | 0 | – | 764 |

**At every holding distance, a stolen card never matches.** Either the printed
portrait is too small to read and the request is refused, or it is compared
against the stranger actually on the card and fails at cosine ≈0.07 — while the
live face, sitting at 0.71, is never once selected.

The bystander row is worth reading twice: `rectified: y`, `foreign: 0`. The card
was on a desk, the border was found, and the bystander was cropped away before
the detector ever saw it. That is layer 1 doing the work, and the reason layers
2 and 3 are a backstop rather than the plan.

### What these numbers are not

Synthetic cards, and a print degradation model I wrote — a downsample, a tonal
compression, a synthetic sheen. Real NADRA cards have holographic overlays, a
guilloche background and physical wear, and real captures have real glare.

More importantly, **the operating point is not validated.** No Pakistani
print-versus-live dataset has been used, because none is available to this
project. `thresholds_validated` is `false` in every response, and it should stay
false until the Client measures it on their own data. The separation shown above
(0.71 against 0.07) is wide enough that the *mechanism* is clearly sound; where
exactly the threshold belongs is a different question and this module does not
answer it.

---

## 6. Architecture

A new capability package, `documents/`, for document-image understanding that is
not text recognition. It shares no machinery with `ocr/` — no engine, no
language, no text — only the subject. A passport MRZ extractor would go here too.

`locate_portrait` takes `(box, confidence)` pairs rather than the detector's own
model. Explicit rather than duck-typed: this package has no business importing
Module 1's schema, and the plain tuple makes the unit tests free of fakes — all
28 of them run with no weights installed.

---

## 7. Failure messages

| condition | message says |
|---|---|
| no face anywhere | photograph the front of the CNIC, the side with the portrait |
| a face far too large | **photograph the card on its own, lying flat, rather than holding it up in front of your face** |
| a face off-template | photograph the whole front, flat and square to the camera |
| portrait too degraded | retake in even light, avoiding glare on the laminate |
| no selfie | the card was read; there is no selfie to compare it against |

The second row is the one that earns its place. When the diagnostic signature of
the attack is present, the guidance names it — a generic "no portrait found"
would send the user round a loop that cannot succeed.

---

## 8. Testing

| suite | count | what it covers |
|---|---|---|
| `tests/unit/test_cnic_portrait.py` | 28 | candidate selection with no models: the attack, ghosts, both size bounds, graded band containment, template configurability, reporting |
| `tests/unit/test_quality_role_policy.py` | 12 | per-role critical floors, including the shipped YAML policy |
| `tests/integration/test_cnic_face_service.py` | 26 | real weights: genuine, impostor, ghost, bystander, the stolen-card attack at three distances, strict mode, missing selfie, noise, determinism |
| `tests/performance/test_cnic_face_performance.py` | 5 | latency budgets, and that a card with no portrait fails faster than one with |

The unit tests hand the locator hand-built boxes rather than round-tripping a
rendered card. That is deliberate: it pins the document-specific reasoning
against exactly the geometries that matter — including ones no fixture produces —
rather than against whatever one detector emits today.

Three defects were found this way. Two came from measuring: the ratio-based
lower bound rejecting a genuine ghost, and the role-blind critical floor in
Module 2. The third came from reading the code back — `allow_unrectified:
false` was **inert**. It skipped the upscale and then searched the frame
anyway, so a deployment that had asked for the strong containment guarantee
silently did not get it. It now refuses, and a test says so.

A fourth was caught by looking at the demo's own output: the overlay renderer
drew boxes on the image the caller passed, but after rectification those boxes
are in the *warped card's* coordinates. The picture contradicted a correct
result. Fixing it turned `prepare_card` into public API, which Module 10 wants
anyway so it can rectify a card once for both OCR and the portrait rather than
twice.

---

## 9. Running it

```bash
python scripts/demo_cnic_face.py
```

```bash
python scripts/demo_cnic_face.py --overlays out/
```

```bash
python scripts/demo_cnic_face.py --cnic card.jpg --selfie me.jpg --json
```

```bash
python -m pytest tests/unit/test_cnic_portrait.py tests/unit/test_quality_role_policy.py -q
```

The `--overlays` run writes an annotated image per scenario: the chosen portrait
in green, refused faces in red, and the template bands in grey. On the attack
image it shows the large live face boxed red and the small printed portrait boxed
green, which is the module's whole argument in one picture.

---

## 10. Configuration surface

Under `cnic_face:` in `configs/thresholds.yaml`, overridable by `HQ_CNIC_FACE__*`.

| key | default | effect |
|---|---|---|
| `rectify_first` | `true` | Isolate the card before searching. The primary containment defence |
| `allow_unrectified` | `true` | Search the whole frame when no border was found. A flat scan has none |
| `detection_min_width` | 1200 | Upscale target before detection |
| `min_detector_confidence` | 0.35 | Objectness floor; below the live-image floor by design |
| `geometry.candidate_bands` | `[[0, 0.38], [0.62, 1.0]]` | Where the photo box may sit, as fractions of width |
| `geometry.vertical_band` | `[0.10, 0.95]` | Its vertical extent |
| `geometry.min_face_pixels` | 40 | Shorter side, in pixels of the searched image |
| `geometry.max_face_area_ratio` | 0.10 | Ceiling, derived from ID-1 proportions |
| `geometry.band_tolerance` | 0.08 | How far outside a band before containment reaches zero |

Three things are deliberately **not** configured here, each because another
section already owns it and two places to change one policy is how they drift:

| | owned by |
|---|---|
| portrait quality | `quality.roles.cnic_portrait` |
| comparison threshold | `matching.selfie_vs_cnic` |
| alignment crop margin | `embedding.alignment.box_fallback_margin` |
