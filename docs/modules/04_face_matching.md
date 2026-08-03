# Module 4 — Face Matching

Turns pairs of embeddings into the scores and decisions the Backend's rules
engine consumes: live selfie against the profile image, against each secondary
image, and against the portrait on the CNIC.

Needs **no model weights** of its own — it consumes what Module 3 produced. That
makes the fusion rules fully unit-testable against synthetic vectors, so they
are pinned exactly rather than sampled from whatever a real model happens to
output.

---

## 1. The calibration problem

The API reports match scores on 0–100, but the underlying quantity is a cosine
in `[-1, 1]`. The obvious conversion is a linear rescaling:

```
score = (cosine + 1) / 2 * 100
```

**This is actively misleading.** Two embeddings of different people are close to
*orthogonal*, not opposed — the measured impostor pair in Module 3 scored
**0.011** — so that formula reports **50.6** for a confident non-match. A
reviewer reading "50" reasonably infers "borderline, half a match". The truth is
"certainly a different person".

### What is done instead

The score is interpolated piecewise-linearly through the **decision boundaries
themselves**:

| cosine | reported |
|---|---|
| ≤ `floor_similarity` (0.0) | 0 |
| `review` threshold | 50 |
| `strong_match` threshold | 75 |
| 1.0 | 100 |

Two properties fall out, both of them the point:

**The score is self-describing.** 75+ is a strong match, 50–75 needs review,
below 50 failed — regardless of which comparison produced it. A reader does not
need to know that the CNIC threshold is 0.42 and the profile threshold 0.62.

**Comparison types become commensurable.** A CNIC score of 80 and a profile
score of 80 mean the same thing about the strength of the evidence, despite
coming from very different cosines. That is what makes the weighted identity
fusion defensible — averaging raw cosines across comparison types would be
meaningless.

The cost is that the reported score is **not** a linear function of the cosine.
Both are returned on every comparison so nothing has to reverse-engineer one
from the other; `uncalibrate_score` exists for the evaluation harness, which
computes ROC curves in cosine space.

---

## 2. Identity fusion

| Comparison | Weight | Why |
|---|---|---|
| CNIC | **0.45** | The only state-issued anchor in the request |
| Profile | 0.35 | User-chosen; proves internal consistency, not identity |
| Secondary | 0.20 | Corroborating |

Weights renormalise over whatever comparisons actually succeeded, so a request
without secondary images is not penalised for their absence.

### Three rules worth explaining

**A failed CNIC comparison caps the result** at 45. Without it, a user could
upload three selfies of themselves alongside somebody else's identity document
and score highly on internal consistency while the one comparison that matters
had failed. The cap makes the document a gate rather than merely a heavy vote.

**Secondary images are averaged, not maximised.** A secondary image that does
not match is a fraud signal — somebody else's photograph on the profile — not
noise to discard, and `max` would silently ignore it. It is not `min` either,
because people legitimately upload old or unflattering photographs of
themselves. The worst individual score is reported separately as
`secondary_worst_score` so Module 9 can weigh it without it dominating.

**A comparison inherits the *minimum* of the two embedding confidences**, not
their mean. An excellent selfie against a box-aligned profile photo is not a
moderately confident comparison; it is limited by the bad one. Module 3
quantified how badly: a box-aligned face scores 0.786 against a properly
aligned embedding of the same person, and 0.573 once the head is tilted.

---

## 3. Two questions, two fields

The demo's second scenario is instructive. A user uploads **somebody else's
profile photograph** but their **own** CNIC:

| | |
|---|---|
| profile | 0.0 — FAILED |
| cnic | 99.2 — STRONG_MATCH |
| **identity_confidence** | **55.8** |
| **any_comparison_failed** | **true** |

Is 55.8 wrong? No — and this is the design point. The CNIC proves the person
*is who they claim to be*. What failed is that the profile photograph is not
them, which for a matrimonial platform is the fraud (catfishing), but it is not
an *identity* failure.

So the two fields answer two different questions and Module 10 needs both:

- `identity_confidence_score` — **is this the claimed person?**
- `any_comparison_failed` — **is anything inconsistent?**

Collapsing them would either approve a catfished profile on a valid document,
or reject a genuine user for one odd photograph.

Likewise `NOT_COMPARED` is kept strictly distinct from `FAILED` throughout.
"There was no CNIC portrait" and "the CNIC portrait is a different person" are
opposite findings; collapsing them would let a missing document read as a
passing one.

---

## 4. Thresholds are **not** validated

The operating points are engineering defaults from the ArcFace/IJB-C
literature. They have **not** been derived from a labelled corpus of this
deployment's traffic, because none exists yet.

| Comparison | strong_match | review |
|---|---|---|
| profile | 0.62 | 0.45 |
| secondary | 0.60 | 0.43 |
| cnic | 0.42 | 0.30 |

The CNIC point is deliberately laxest: a CNIC portrait is a sub-300-dpi print
photographed through a laminate, often years old, and holding it to the selfie
bar would refuse a large share of genuine documents.

**The service says so in every response** — `thresholds_validated: false` plus a
`MATCHING_THRESHOLDS_UNVALIDATED` warning. That is more useful than a note in a
document nobody reads at 3am.

### The harness that fixes this

`hamqadam_ai.matching.evaluation` implements ROC, AUC, EER, TAR@FAR and
confusion matrices in pure NumPy — no scikit-learn, so it runs inside the
production image.

```bash
python scripts/evaluate_matching.py --corpus path/to/corpus --plot reports/
```

Corpus layout is one directory per identity; two images in the same directory
are a genuine pair, two in different directories an impostor pair.

It reports **what the corpus can and cannot resolve**:

```
      target FAR       TAR   threshold   resolvable
            0.01    0.9995      0.2707   yes
           0.001    0.9938      0.3363   yes
          0.0001    0.9830      0.4037   yes
           1e-05    0.8925      0.5175   NO (need 100,000)
```

Resolving a FAR of 1e-4 needs at least 10,000 impostor pairs. An all-pairs
protocol over *n* identities with *k* images gives roughly `n(n-1)k²/2`, so 50
identities × 3 images ≈ 11,000 — enough for 1e-4, not for 1e-5. Printing a
confident number from too little data would be spurious precision, so the
harness refuses to.

**Thresholds are recommended at a fixed FAR, not at the EER.** The EER weights
admitting an impostor and refusing a genuine user equally; they are not equal.
One is a security incident, the other an inconvenience the user can retry. The
strong-match boundary is set at FAR ≤ 1e-4 because crossing it admits somebody
to an account; the review boundary at FAR ≤ 1e-2 because crossing it only routes
the case to a human.

`--self-test` exercises the whole report path on synthetic distributions, and
says loudly that they are invented.

---

## 5. Measured end to end

Real weights, two identities:

| Scenario | identity | capped | failed |
|---|---|---|---|
| Consistent identity | **99.5** | – | no |
| Impostor profile photo | 55.8 | – | **yes** |
| Stranger's CNIC | **45.0** | `cnic_failure` | **yes** |
| One bad secondary of two | 89.5 | – | **yes** |

Genuine comparisons land at 99.2–99.8 and impostor comparisons at 0.0, which is
what the calibration predicts from Module 3's measured cosines (0.898–0.977 and
0.011).

> As in Module 3: this is a **two-identity sample**. It demonstrates the rules
> fire correctly. It is not an accuracy evaluation.

---

## 6. Testing

| Suite | Count | Covers |
|---|---|---|
| `tests/unit/test_matching_similarity.py` | 21 | Calibration anchors, monotonicity, round-trip, commensurability |
| `tests/unit/test_matching_service.py` | 48 | Comparison, fusion rules, caps, service contract |
| `tests/unit/test_matching_evaluation.py` | 28 | ROC/AUC/EER/TAR@FAR against analytically-known cases |
| `tests/integration/test_matching_pipeline.py` | 9 | Detection → embedding → matching on real weights |

The evaluation metrics are tested against cases whose answers are known
analytically rather than against a reference implementation. A subtly wrong ROC
would produce a confident, wrong operating point — the worst failure mode here,
because nothing would look broken.

---

## 7. Running it

```bash
python scripts/demo_matching.py
```

```bash
python scripts/demo_matching.py --selfie s.jpg --profile p.jpg --cnic c.jpg --secondary a.jpg
```

```bash
python scripts/evaluate_matching.py --self-test --plot reports/matching
```

---

## 8. Configuration surface

Everything is in `configs/thresholds.yaml` under `matching:`.

```bash
HQ_MATCHING__SELFIE_VS_CNIC__STRONG_MATCH=0.48
HQ_MATCHING__IDENTITY__CNIC_FAILURE_CAP=30
HQ_MATCHING__IDENTITY__SECONDARY_AGGREGATION=min
```

`MatchThresholds` enforces `review < strong_match`, and
`recommend_thresholds()` guards the same invariant so a recommendation from a
tiny corpus cannot produce an unloadable configuration.
