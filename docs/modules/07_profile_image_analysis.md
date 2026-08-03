# Module 7 — Profile Image Analysis

Answers a question none of the identity modules do: **is this a genuine camera
photograph, taken by the person uploading it?**

A profile photo that is a screenshot of somebody else's social media, a picture
of a laptop screen, or a cartoon avatar is a different failure from a photo of
the wrong person. It needs different evidence, and the user needs to be told
something different about it.

---

## 1. What this module does not claim

Stated first, because the omissions matter more than the inclusions.

| not implemented | why |
|---|---|
| content moderation | needs a trained classifier this project does not have |
| deepfake / GAN detection | same, and the field moves faster than any fixed detector |
| reverse image search | needs a reference corpus of stock and celebrity imagery |

A function returning `False` for each of these would look exactly like a
working check while providing none of the protection somebody would rely on.
`describe()` reports all three as `false` so a caller cannot assume otherwise
by accident, and a test asserts it.

There is also no trained classifier behind what *is* here. A CNN would very
likely beat these measurements. What is shipped is signal processing whose
separation has been measured against controlled fixtures, with every margin
written down — so the next engineer knows exactly how much confidence the
numbers support.

---

## 2. Reading the score correctly

`authenticity_score` is an **upper bound on suspicion, not a measure of trust**.

100 means four specific tests found nothing. It does not mean the image is
genuine. A cropped screenshot, a screen capture taken far enough away to lose
its moiré, and a competent composite all score 100.

Treat a low score as evidence and a high score as the *absence* of evidence.
The internal type says so too — the property is called `clean`, not `genuine`,
and a test pins that naming, because `genuine` would invite exactly the
inference the module spends its docstrings warning against.

---

## 3. The measurement that failed, and the one that worked

The obvious idea for screenshot detection is that rendered UI has flat regions
no camera could produce. **Measured, that is false.**

| | exactly-zero-variance 8×8 blocks |
|---|---|
| genuine photograph | 0.000 |
| **quality-12 JPEG** | **0.355** |
| screenshot | 0.359 |

JPEG quantisation zeroes every AC coefficient in a smooth block, so a heavily
compressed photograph is indistinguishable from a screenshot on flatness alone.

What works is a **full-width constant row**. A photograph's content varies
horizontally *somewhere* along any given row, so quantisation cannot flatten the
whole span; rendered chrome is written from a constant and does.

| | constant rows |
|---|---|
| genuine photograph | 0.000 |
| quality-12 JPEG | **0.000** |
| twice-recompressed | 0.000 |
| photograph of a screen | 0.000 |
| photograph of a print | 0.068 |
| **screenshot** | **0.335 – 0.454** |

### Then measurement found the false positive

Social apps pad uploads to a square. A letterboxed photograph scores **0.333** —
identical to a screenshot. The first version of this detector would have flagged
a large share of perfectly honest uploads.

The fix is to ask *where* the constant rows are. Padding is contiguous at the
frame edges; application chrome is distributed through the interior. Stripping
the contiguous constant borders before measuring separates them completely:

| | interior constant rows |
|---|---|
| genuine photograph | 0.000 |
| letterboxed photograph | **0.000** ← fixed |
| square-padded then JPEG | **0.000** ← fixed |
| padded at the top only | 0.000 |
| photograph of a print | 0.000 |
| **screenshot** | **0.175 – 0.187** |

Exact device resolutions are deliberately **not** used. A genuine photograph
resized to 1080×1920 matches one, so the test convicts the innocent, and any
attacker defeats it by cropping a single row.

---

## 4. The other three detectors

### Photograph of a screen — moiré

A display is a regular pixel grid; a sensor is another. Photograph one with the
other and they beat together, putting an isolated symmetric pair of peaks
off-axis in the Fourier transform. Natural spectra are smooth and decay
monotonically.

Peak prominence above the locally-smoothed log spectrum:

| | prominence |
|---|---|
| genuine photograph | 1.410 |
| quality-12 JPEG | 1.456 |
| twice-recompressed | 1.437 |
| screenshot | 1.428 |
| photograph of a print | 1.411 |
| synthetic render | 1.443 |
| **photograph of a screen** | **3.461 – 4.220** |

Everything that is not a screen capture sits in a band 0.05 wide; a screen
capture sits at two and a half times its top. The widest separation any
measurement in this module achieves, which is why this detector is allowed to
trigger on its own evidence.

The axes are excluded from the search: JPEG's 8×8 blocking puts a comb of energy
exactly there, and so does any axis-aligned texture — the one place a false
positive is likely.

### Photograph of a print

Share of spectral power above half Nyquist:

| | high-frequency share |
|---|---|
| twice-recompressed | 0.0206 |
| genuine photograph | 0.0178 |
| synthetic render | 0.0153 |
| quality-12 JPEG | 0.0121 |
| **photograph of a print** | **0.0021** |

An order of magnitude below the genuine case, and below the quality-12 JPEG that
is the hard negative everywhere else. Corroborated by compressed tonal range —
ink reaches neither true black nor paper white — and a uniform border.

### Rendered artwork

| | palette | flat blocks |
|---|---|---|
| genuine photograph | 0.286 | 0.000 |
| quality-12 JPEG | 0.173 | 0.355 |
| screenshot | 0.235 | 0.359 |
| **synthetic render** | **0.000** | **0.911** |

Both must agree, combined as a geometric mean. A small palette alone fires on a
posterised photograph; a high flat share alone fires on that quality-12 JPEG.

Note the asymmetry with §3: flat blocks are worthless at 0.355 and decisive at
0.911. **The same measurement is diagnostic at one magnitude and useless at
another** — which is why every anchor in this module is written beside the
population it separates rather than left as a bare number.

---

## 5. Operating envelope, measured

Where each detector stops working. Published because a null result is not
evidence of a genuine capture, and a caller needs to know how much a silence is
worth.

**Screen recapture:**

| condition | result |
|---|---|
| moiré strength | triggers to 0.05, graded 0.03–0.05, blind below 0.02 |
| downscaling | survives 0.50× (256×300), lost at 0.35× (179×210) |
| re-encoding | survives JPEG quality 15 (confidence 0.933) |

Robust to compression, fragile to heavy downscaling.

**Screenshot — a known evasion:**

| crop | confidence |
|---|---|
| full frame | 1.000 |
| centre 90% | **0.172** |
| centre 70% | 0.152 |
| centre 50% | 0.000 |

Cropping 10% off a screenshot defeats the detector. This is a direct consequence
of the edge-stripping in §3: the flat chrome then touches the new frame edge and
is stripped as padding. The two are geometrically identical and cannot be told
apart without semantic understanding.

The trade favours **not accusing innocent users**, which is the right default for
a check that calls somebody dishonest. Both the limitation and the reasoning are
pinned by a test so neither can be quietly forgotten.

---

## 6. Three questions, kept separate

```
authenticity  ->  is it a genuine camera capture?      this module
subject       ->  is there exactly one person in it?   Module 1
quality       ->  is it good enough to recognise?      Module 2
```

Separate because a caller needs to tell them apart. *"Upload a photo instead of
a screenshot"*, *"crop this so only you are in it"* and *"retake this somewhere
brighter"* are three different things to say to a user, and a single merged
score can say none of them.

**Authenticity runs first and unconditionally**, because it is the only one that
does not need a face. A cartoon avatar has no face for Module 1 to find and no
meaningful quality score — but the useful thing to tell the user is not "no face
detected", it is "that is a drawing".

The corollary is equally deliberate: a landscape photograph scores **100** on
authenticity and fails on `FACE_NOT_DETECTED`. It is a genuine capture. Someone
who uploaded a photo of their cat has not been dishonest, and the response must
not imply they have.

---

## 7. Failure messages

Ordered by what the user should do first. Authenticity findings come before
everything else because they are the only ones where retaking the same
photograph cannot possibly help.

| condition | code | message says |
|---|---|---|
| authenticity finding | `INVALID_IMAGE` | upload a *different* image — the finding names which kind |
| no face | `FACE_NOT_DETECTED` | upload a clear photo of yourself facing the camera |
| several faces | `MULTIPLE_FACES_DETECTED` | upload one where you are alone, or crop it |
| low quality | `LOW_IMAGE_QUALITY` | retake in better light, holding the camera steady |

Every finding message contains the word "Upload" — asserted by a test. Each of
these accuses somebody of uploading something dishonest; the least it can do is
say what would fix it.

---

## 8. Measured end to end

| group | case | verdict | authenticity | faces | ms |
|---|---|---|---|---|---|
| genuine | a real photograph | OK | 100.0 | 1 | 447 |
| **hard negative** | JPEG quality 12 | **OK** | 100.0 | 1 | 328 |
| **hard negative** | recompressed 92→74 | **OK** | 100.0 | 1 | 944 |
| **hard negative** | square-padded upload | **OK** | 100.0 | 1 | 984 |
| impostor | screenshot of an app | FAIL | 0.0 | 1 | 1138 |
| impostor | photograph of a screen | FAIL | 0.0 | 1 | 863 |
| impostor | photograph of a print | FAIL | 0.0 | 1 | 885 |
| impostor | cartoon avatar | FAIL | 0.0 | 0 | 547 |
| **evasion** | screenshot, cropped 10% | OK | **100.0** | 1 | 1126 |
| **evasion** | screen capture, downscaled 4× | FAIL* | **100.0** | 0 | 536 |
| not a portrait | a landscape | FAIL | 100.0 | 0 | 631 |
| not a portrait | a flat colour field | FAIL | 100.0 | 0 | 290 |

\* fails on "no face" after downscaling, not on authenticity — the moiré is gone.

The evasion rows are in the demo deliberately. A demonstration that only shows
its successes is marketing.

### What these numbers are not

Synthetic impostors, built from models I wrote of what a screenshot, a screen
capture and a print look like. Real screenshots come from thousands of app
layouts; real screen captures vary with panel type, distance and angle; real
prints vary with paper and printer.

The **separations** are large enough — an order of magnitude on three of the four
detectors — that the mechanisms are clearly sound. The **thresholds** should be
re-derived by the Client against their own upload traffic before anyone relies on
the exact operating point. Nothing here reports a validated false-positive rate,
because none has been measured.

---

## 9. Timing

Measured on an idle machine, median of five runs:

| | cost |
|---|---|
| four detectors, 512x600 portrait | 150 ms |
| four detectors, 1080x1920 screenshot | 210 ms |
| four detectors, 3000x4000 phone photo | 725 ms |
| full analysis, portrait | 467 ms |
| **detector share of the full analysis** | **0.34** |

An earlier draft of this section claimed the detectors cost "~55 ms" and were
"cheaper by an order of magnitude". Both were wrong — I derived them by
subtraction rather than measuring. They are about **three times** cheaper, not
ten, and the numbers above are measured.

That changes the justification for exposing `assess_authenticity()`
separately, and it is worth being clear about what the real one is: **it needs
no model weights.** It can run where no models are installed, and it can reject
a screenshot before the face models are loaded at all. A 3× saving would not on
its own have earned a second entry point; not needing the models does.

Cost scales with pixel count, because the row and block measurements run at
full resolution. The spectral work does not — it runs on a fixed 512x512
resample. Nothing here downscales the input the way Module 1 does, deliberately:
downsampling changes flat-block fraction and palette size, so it would shift two
of the four detectors' anchors and every measurement above would need
re-deriving.

---

## 10. Testing

| suite | count | what it covers |
|---|---|---|
| `tests/unit/test_authenticity_detectors.py` | 54 | every detector against every fixture, both hard negatives, the documented evasions, degenerate input, determinism |
| `tests/unit/test_authenticity_aggregate.py` | 24 | strongest-wins aggregation, finding order, message coverage, unmeasured detectors |
| `tests/integration/test_profile_service.py` | 32 | the real pipeline: impostors, hard negatives, roles, subject rules, disclaimers |
| `tests/performance/test_profile_performance.py` | 7 | latency, scaling with pixel count, and the detector share of the total |

The unit tests need **no model weights** — these are signal-processing
measurements and they run on a bare install, given a public-domain reference
photograph.

Three defects were found by building the fixture and measuring:

1. **The flat-block premise was wrong.** I asserted in the fixture docstring that
   rendered UI has flat regions "which no camera sensor can produce, and neither
   can JPEG". The measurement refuted it immediately — quality-12 JPEG gives 0.355
   against a screenshot's 0.359 — and the constant-row measure replaced it.
2. **`AuthenticitySignalModel` was missing `measured`.** The internal dataclass
   had it; the response model did not, so every API consumer would have
   re-derived `note is None` themselves. Found by the demo crashing on it.
3. **The timing figures in this document were fabricated.** §9 claimed the
   detectors cost "~55 ms" and were "cheaper by an order of magnitude"; I had
   derived both by subtraction rather than measuring. The real figures are
   150 ms and a 0.34 share. Found by going back to check a number I had
   written down without evidence — which is exactly the standard the rest of
   this module is held to.

---

## 11. Running it

```bash
python scripts/demo_profile.py
```

```bash
python scripts/demo_profile.py --image me.jpg --detectors-only
```

```bash
python -m pytest tests/unit/test_authenticity_detectors.py tests/unit/test_authenticity_aggregate.py -q
```

---

## 12. Configuration surface

Under `profile:` in `configs/thresholds.yaml`, overridable by `HQ_PROFILE__*`.
Every anchor is printed there beside the measured populations it separates.

| key | default | separates |
|---|---|---|
| `screenshot.interior_rows_floor` / `_ceiling` | 0.04 / 0.15 | 0.000 from 0.175–0.187 |
| `moire.prominence_floor` / `_ceiling` | 1.90 / 3.20 | 1.41–1.46 from 3.46–4.22 |
| `moire.axis_exclusion` | 6 | keeps JPEG blocking out of the search |
| `print_recapture.high_frequency_floor` / `_ceiling` | 0.0090 / 0.0030 | 0.0121–0.0206 from 0.0021 |
| `synthetic.palette_floor` / `_ceiling` | 0.090 / 0.010 | 0.173–0.286 from 0.000 |
| `synthetic.flat_floor` / `_ceiling` | 0.50 / 0.85 | 0.355 from 0.911 |
| `min_authenticity_score` | 55.0 | verdict threshold |
| `require_face` | `true` | a cat photo cannot be verified against a selfie |
| `flag_multiple_faces` | `true` | a group photo is ambiguous, not dishonest |

Each detector's `min_confidence` defaults to 0.55. Every anchor pair runs
*downwards* where less of the quantity means more suspicion — `ramp()` handles
either direction, and a test covers both.
