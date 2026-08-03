# Module 5 — CNIC OCR

Reads a Pakistani Computerised National Identity Card and returns its fields as
structured, typed data with per-field provenance, plus a set of consistency
checks the card itself makes possible.

The recognition engine is a commodity — PP-OCR reads Latin glyphs about as well
as anything else will. What is *not* commodity, and what most of this module
is, is everything around it: getting a photographed card into a state the
recogniser can work with, deciding which recognised string is which field, and
knowing what a CNIC's own structure permits.

---

## 1. What the document tells you about itself

Two facts about the Pakistani CNIC number do real work here.

**The final digit encodes gender.** Odd is male, even is female. This is the
single most valuable check in the module, because the identity number and the
printed gender field are physically separate parts of the card, recognised
independently. When they disagree, either a digit was misread or the document
was altered — and nothing else on the card reveals it.

**The first digit encodes region.** 1 Khyber Pakhtunkhwa, 2 FATA, 3 Punjab,
4 Sindh, 5 Balochistan, 6 Islamabad Capital Territory, 7 Gilgit-Baltistan,
8 Azad Jammu & Kashmir. A leading 0 or 9 is unallocated and is reported as no
region rather than guessed at.

### There is no checksum

Stated explicitly because it is exactly the kind of thing that gets assumed.
The thirteen digits carry no check digit, so **any** correctly-shaped number
parses. This module does not claim to verify a number against anything beyond
its structure, its region code and its gender parity, and no amount of wanting
a checksum will produce one.

### The circularity trap

When the gender glyph cannot be read, it can be *derived* from the number's
final digit. That is genuinely useful — but a derived value must never then be
"cross-checked" against the number it came from. It would agree every time and
prove nothing.

So `FieldValue.source` records where every value came from, and
`gender_cross_check_passed` returns:

| value | meaning |
|---|---|
| `true` | a gender was **read** and agrees with the number's parity |
| `false` | a gender was **read** and disagrees — misread digit, or tampering |
| `null` | no gender was read; the check could not be performed |

`null` is not `false`. Collapsing them would report every unreadable glyph as a
fraud signal.

---

## 2. Reading a lone glyph: detection, not recognition

The gender field is a single `F` or `M` sitting in a wide margin. On the
synthetic card, PP-OCR's page detector returned **nothing at all** for that
region — at every padding tried (0, 12, 40 px) and every scale (2×, 4×).

Handing the *same crop* straight to the recognition head returned `'F'` at
confidence 0.24. **Recognition was never the problem; detection was.** Text
detectors are trained on lines, and one character does not look like a line.

So `OcrEngine.recognise_crop()` exists as a first-class port method: when a
label has been located but its value is missing, the region beside the label is
cropped and fed directly to the recogniser, skipping detection entirely. All
three adapters implement it natively (`text_recognizer` on rapidocr, `det=False`
on PaddleOCR, `recognize` on EasyOCR), and the default falls back to full
detection.

Its confidence is **not comparable** with `recognise()`. An isolated glyph
scored **0.24** where the same character inside a line scores above **0.8** —
the recogniser has almost no context to work with. Judging it against
`min_line_confidence: 0.35` would discard every recovery this path exists to
make, so it gets its own floor, `min_crop_confidence: 0.12`.

This is what makes the parity cross-check real rather than theoretical: without
it the gender is always derived, and the check always returns `null`.

---

## 3. Orientation: three passes, not four

Users photograph cards whichever way is convenient, and the recogniser reads
only upright text. Recognition costs roughly **3.5 seconds a pass** on CPU, so
trying all four rotations exhaustively costs about fourteen seconds — an
unacceptable share of the request budget for one image.

The geometry of the detected text boxes narrows it. A line of Latin script is
wider than it is tall; when most detected regions are *taller* than wide, the
page is a quarter turn out. That distinguishes {0°, 180°} from {90°, 270°} but
cannot separate the members of either pair — both produce identical box shapes.

### A hypothesis that failed

I expected the *point ordering* of each returned quadrilateral to encode reading
direction, which would have separated 90 from 270 in a single pass. Measured
across all four rotations, the mean reading vector came out at ≈0° every time.
The hypothesis was wrong, the measurement said so, and the three-pass search
stands.

### The early exit, and the canary it needs

The loop stops as soon as a pass yields a usable document. Getting the accept
condition right took a correction: the first version accepted the upright pass
on a rotated card, because the identity number and the dates are extracted by
**pattern** and patterns match at any rotation.

The fix is to require `full_name` — the one field that can only be found
*spatially*, by locating its label and looking beside it. It is the canary for
"the page is actually the right way up".

Measured cost:

| capture | passes | wall clock |
|---|---|---|
| upright | 1 | ~3.5 s |
| half turn | 2 | ~5.6 s |
| quarter turn | 3 | ~11.0 s |
| unreadable | 4 | ~1.5 s (nothing to recognise) |

Preprocessing is 35–40 ms in a fresh process, and 150–175 ms once the process
has done a few thousand OpenCV operations — which is what a long-lived service
is at all times. That degradation is uniform across every stage, survives
`gc.collect()`, and does not recover after 25 s of idle, so it is neither a leak
here nor thermal throttling; it is a property of sustained OpenCV work in one
process, and the warm figure is the one that describes production.

Recognition still dominates by more than twenty to one either way, which is the
assumption the whole pipeline ordering rests on — and
`test_preprocessing_is_negligible_beside_recognition` pins it, budgeted for the
warm case.

---

## 4. Getting the card ready

| stage | what it does | why it earns its place |
|---|---|---|
| **rectify** | Canny → `approxPolyDP` → perspective warp to ID-1 proportions | A card on a desk is a trapezium. Warping it back makes every downstream spatial rule (beside/below) mean something |
| **deskew** | Hough over text lines, ±12° | Two degrees of tilt scatters a label and its value onto different rows and breaks row grouping |
| **enhance** | CLAHE on the L channel only | Local contrast for dim capture and laminate glare. Colour channels are left alone — the recogniser reads grey, and tinting the card helps nobody |
| **upscale** | to 700 px minimum width | PP-OCR's recognition head has a fixed 48 px input; source text below ~10 px degrades sharply once resampled into it |

Rectification refuses shapes that are not card-like — a phone or a book lying on
the same desk is a strong contour with the wrong aspect ratio, and warping it
into ID-1 proportions would distort whatever text it carries.

Upscaling adds no information and does not pretend to. It recovers a readable
field from a small capture; the quality module independently penalises the low
native resolution.

---

## 5. Field extraction

Three strategies, in order of how much they can be trusted:

**By shape.** The identity number and the dates have unambiguous forms and are
found anywhere on the card, with no label needed. This is why the number is the
field most likely to survive a bad capture.

**By label.** Labels are matched by `difflib` similarity at a **0.78** floor,
not by equality — the probe genuinely returned `ldentity` for `Identity`
(capital I read as lowercase l), and exact matching would have lost the whole
row. A matched label's value is then sought inline on the same line, beside it
on the same row, or below it.

The inline case needed the same fuzzy treatment. The first version required an
*exact* prefix, so `Fathor Namo MUHAMMAD KHAN` — one box, two glyph errors —
lost the name entirely. Prefix matching is now word-count-wise and fuzzy: a
two-word label consumes exactly two words, however badly either was spelled.

**By order.** With no labels at all, three unlabelled dates are assigned
chronologically. Birth precedes issue precedes expiry on every card ever issued,
so this is sound. **Two** unlabelled dates are *not* assigned — they could be
any of three pairings, and a coin toss dressed up as a reading is worse than a
reported gap.

### Glyph confusion is corrected per-field, never in names

`O`↔`0`, `I`/`l`↔`1`, `S`↔`5`, `B`↔`8`, `Z`↔`2`. Applied to numbers and dates,
where the context is known to be numeric and the correction is safe.

**Never applied to a name.** A name is free text; "correcting" it would silently
alter the holder's identity. `AYESHA 0KHAN` is reported as read, with its
confidence multiplied by 0.7 to say that something in it is certainly wrong. An
OCR error a reviewer can see beats a plausible-looking invention.

A line that is *mostly* digits is still rejected — one stray digit in a twelve
character name is a slip; a third of the characters being digits means it is a
serial number that happened to start with a letter.

---

## 6. Validation

| finding | severity | penalty |
|---|---|---|
| `CNIC_GENDER_MISMATCH` | error | 0.25 |
| `CNIC_ISSUE_BEFORE_BIRTH` | error | 0.25 |
| `CNIC_EXPIRY_BEFORE_ISSUE` | error | 0.25 |
| `CNIC_BIRTH_IN_FUTURE` | error | 0.30 |
| `CNIC_IMPLAUSIBLE_AGE` | error | 0.25 |
| `CNIC_TOO_FEW_FIELDS` | error | 0.20 |
| `CNIC_UNUSUAL_VALIDITY_TERM` | warning | 0.05 |
| `CNIC_UNKNOWN_PROVINCE` | warning | 0.05 |
| `CNIC_EXPIRED` | info | **0.00** |
| `CNIC_GENDER_CONSISTENT` | info | 0.00 |
| `CNIC_GENDER_DERIVED` | info | 0.00 |

**An expired card carries no penalty.** It is a *correct reading* of an
out-of-date document. Whether an expired CNIC is acceptable is a policy question
for the Backend's rules engine; degrading the OCR confidence would smuggle that
policy into the wrong module.

An unusual validity term is a warning rather than an error because NADRA does
issue non-standard terms. It is most likely a misread year digit, which is worth
flagging without rejecting.

### Document confidence

Multiplicative rather than additive:

```
confidence = weighted_field_confidence
           × completeness_factor
           × (1 − validation_penalty)
```

Multiplicative because the three are not interchangeable. A card where every
field was read crisply but two are missing should not average its way to a
respectable score.

---

## 7. Failure messages must be actionable

Every failure branch has to survive one question: *what should the user do
next?*

The demo caught a violation. A card whose printed gender contradicted its own
identity number was read perfectly — 6/6 fields, all correct — but confidence
fell below the floor from the validation penalty, and the message said "Retake
the photograph in even light". That advice **cannot work**: the contradiction is
in the document, so every retake reproduces it exactly.

There are now four branches:

| condition | message says |
|---|---|
| not a CNIC | photograph the front of your CNIC |
| fields missing | retake — even light, card flat and in focus |
| **validation errors** | **every field was read; this needs manual review, not another photograph** |
| low confidence | retake — avoid glare on the laminate |

The same reasoning produced a distinct error code. `CNIC_NOT_RECOGNISED` was
added alongside the contract's `CNIC_OCR_FAILED`, because "unreadable" and
"wrong document" ask the user for different things, and collapsing them sends
half of them round a loop that cannot succeed.

---

## 8. Engines

| engine | role | why here |
|---|---|---|
| `onnx_ppocr` | default | Same ONNX Runtime the detector and recogniser already use — inherits device selection and adds no second inference runtime to the image. Models bundled in the wheel |
| `paddleocr` | named by the specification | Fully implemented, selected when installed. Second only because `paddlepaddle` is a large second runtime with its own device handling |
| `easyocr` | named fallback | Torch-based, stronger than PP-OCR on some degraded captures |

All three sit behind one port emitting the same `TextLine` in source
coordinates, so the parser above them is engine-agnostic and fully testable with
none of them installed.

The `ocr` extra installs the **default only** — `rapidocr-onnxruntime`, which
runs on the ONNX Runtime already in the base dependencies and bundles its own
weights. That is the whole cost of giving a production image the ability to
read a CNIC. `ocr-fallbacks` adds the other two, and is a much larger install:
`paddlepaddle` is a second inference runtime and `easyocr` pulls in Torch.

---

## 9. Measured

Synthetic card, ONNX PP-OCR on CPU, 1012 px wide:

| capture | read | confidence | fields | rotation | passes | ms |
|---|---|---|---|---|---|---|
| clean scan | ✓ | 79.9 | 6/6 | 0 | 1 | 3490 |
| on a desk, tilted | ✓ | 81.7 | 6/6 | 0 | 1 | 2885 |
| dim lighting | ✓ | 80.3 | 6/6 | 0 | 1 | 3457 |
| glare on laminate | ✓ | 78.3 | 6/6 | 0 | 1 | 3864 |
| low resolution | ✓ | 78.7 | 6/6 | 0 | 1 | 3171 |
| jpeg quality 25 | ✓ | 81.3 | 6/6 | 0 | 1 | 3435 |
| sensor noise | ✓ | 78.5 | 6/6 | 0 | 1 | 4194 |
| rotated 90 | ✓ | 76.5 | 6/6 | 270 | 3 | 10964 |
| rotated 180 | ✓ | 80.2 | 6/6 | 180 | 2 | 5646 |
| rotated 270 | ✓ | 79.6 | 6/6 | 90 | 2 | 6196 |
| expired card | ✓ | 78.5 | 6/6 | 0 | 1 | 2839 |
| gender contradicts number | rejected | 60.4 | 6/6 | 0 | 1 | 2716 |
| utility bill | rejected | 0.0 | 0/6 | — | 4 | 13267 |
| blank page | rejected | 0.0 | 0/6 | — | 4 | 1536 |

Every field on the clean read is exact: number, both names, gender **read from
the card** (not derived), and all three dates.

### What these numbers are not

They are measured on **synthetic cards rendered with a clean sans-serif font**.
Real NADRA cards have holographic overlays, a guilloche background, a different
typeface and physical wear. Accuracy on real documents will be lower, and how
much lower is not knowable from this fixture set. No real identity document
appears in this repository, and none may — a CNIC is precisely the category of
personal data this service exists to avoid retaining.

The pipeline is what has been validated here, not a claimed field-level accuracy
on production documents. That number has to be measured by the Client on their
own data.

---

## 10. Privacy

The recognised text of a CNIC *is* the holder's name, father's name and identity
number. Accordingly:

- Raw recognised text is **never** returned in the response and never logged.
  `TextLine.as_dict()` is diagnostics-only; `OcrOutput.describe()` deliberately
  omits the text.
- `FieldValue.as_dict()` redacts `raw` by default; un-redacting is an explicit
  opt-in.
- `CnicFields.summary()` — the object that reaches the log sink — contains only
  which fields are present, completeness, mean confidence and province. A test
  asserts that no field *value* can appear in it.
- Nothing is persisted. The image and every intermediate live for the duration
  of the call.

---

## 11. Testing

| suite | count | what it covers |
|---|---|---|
| `tests/unit/test_cnic_domain.py` | 62 | number structure, region codes, gender parity, dates, glyph confusion, all validation rules, field container, PII in summaries |
| `tests/unit/test_cnic_parser.py` | 43 | field extraction against hand-built line sets: mangled labels, merged rows, stacked layouts, unlabelled dates, name handling, recovery, document identification |
| `tests/unit/test_ocr_preprocessing.py` | 40 | rectification on real pixels, orientation geometry, rotation losslessness and direction, deskew, CLAHE, upscaling |
| `tests/integration/test_ocr_service.py` | 46 | the real engine on rendered cards: clean, six degradations, three rotations, lifetime, expired, tampered, non-CNIC, determinism |
| `tests/performance/test_ocr_performance.py` | 7 | latency budgets, and that rotation costs *passes* rather than multiples |

The parser tests build `TextLine` objects by hand rather than round-tripping a
rendered card. That is deliberate: it tests the CNIC-specific logic against
exactly the failure modes real engines produce — a dropped label, a merged row,
a substituted glyph — rather than only against whatever one engine happens to
emit today.

Three defects in this module were found by writing the test first and measuring:
the rotation early-exit accepting the wrong pass, the exact-prefix requirement
losing merged rows, and names being discarded outright over a single misread
digit.

---

## 12. Running it

```bash
python scripts/demo_ocr.py
```

```bash
python scripts/demo_ocr.py --image card.jpg --json
```

```bash
python -m pytest tests/unit/test_cnic_domain.py tests/unit/test_cnic_parser.py tests/unit/test_ocr_preprocessing.py -q
```

```bash
python -m pytest tests/integration/test_ocr_service.py -q
```

---

## 13. Configuration surface

Everything below lives in `configs/thresholds.yaml` under `ocr:` and is
overridable by `HQ_OCR__*` environment variables.

| key | default | effect |
|---|---|---|
| `engine_chain` | `[onnx_ppocr, paddleocr, easyocr]` | Fallback order; first importable wins |
| `language` | `en` | Recognition language for the card's English side |
| `min_confidence` | 0.55 | Document confidence below which the read fails |
| `min_required_fields` | 4 | Fields needed before a read counts as successful |
| `min_line_confidence` | 0.35 | Recognition floor for full-page lines |
| `min_crop_confidence` | 0.12 | Separate, lower floor for detection-free crops |
| `field_weights` | see file | Per-field weighting in the confidence aggregate |
| `preprocessing.rectify` / `.deskew` / `.enhance` | `true` | Individually disableable |
| `preprocessing.min_card_coverage` | 0.18 | Smallest card-to-frame ratio worth warping |
| `preprocessing.aspect_tolerance` | 0.35 | How far from ID-1 proportions a quad may be |
| `orientation.enabled` | `true` | Turn the rotation search off entirely |
| `orientation.early_exit` | `true` | Stop at the first acceptable pass |
| `orientation.max_attempts` | 4 | Cap on recognition passes per document |

No accept/reject threshold in this module is hard-coded.
