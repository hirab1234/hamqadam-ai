# Module 9 — Fraud Risk Engine

Takes the results of Modules 1 through 8 and produces one score, one band, and
the reasons for both.

No model, no images, no I/O. All the difficulty is in the aggregation, which is
where the interesting mistakes live: every upstream module reports what it saw,
and none of them is in a position to know that four of those reports are the
same fact, or that a missing check is neither good news nor bad.

---

## 1. The mistake this module is built to avoid

A single fact usually produces several findings. A CNIC photographed off a
laptop screen makes Module 7 report a screen recapture, Module 2 report low
quality, Module 5 report low OCR confidence, and Module 6 report a degraded
portrait — **four findings, one fact**.

So every signal carries a **family**. The strongest member of a family is what
that family contributes; families combine by **noisy-OR** because they are
independent evidence.

| case | additive | naive OR | **family + OR** |
|---|---|---|---|
| one fact: CNIC shot off a screen | 100.0 | 77.0 | **69.4** |
| one fact: a blurry photograph | **65.0** | 51.0 | **20.0** |
| three genuinely independent facts | 100.0 | 94.6 | **94.6** |
| two independent facts | 100.0 | 90.0 | **90.0** |

The second row decides the design. Additive scoring puts an ordinary
out-of-focus snapshot at **65 — the boundary of HIGH risk** — purely because
four quality sub-scores each contributed. It is not fraud, it is a bad photo,
and a system that cannot tell the difference will reject honest users all day.
Grouping by family puts it at 20 and leaves the genuinely independent cases
exactly where they were.

Reproduce it:

```bash
python scripts/demo_fraud.py --compare
```

### Why noisy-OR and not a weighted mean

Three properties, all of which matter:

- two moderate findings exceed either alone;
- nothing ever exceeds 100;
- **no finding can be diluted by the presence of others.** A weighted mean
  would let an attacker lower their score by adding innocent noise. Noisy-OR is
  monotonic — more evidence never helps — and a test pins that.

---

## 2. The seven families

| family | what it means | example |
|---|---|---|
| `document_integrity` | the card contradicts itself | gender against the number's parity |
| `document_authenticity` | the *document image* is not a photograph of a document | reserved; see below |
| `image_authenticity` | a user photograph is not a genuine capture | screenshot, screen recapture |
| `identity_consistency` | the faces are not all the same person | selfie ≠ CNIC portrait |
| `presentation` | how the submission was staged | a face held behind the card |
| `duplication` | already enrolled elsewhere | gallery hit |
| `capture_quality` | the captures are poor | blur, pose, occlusion |

`document_authenticity` is deliberately empty for now. Module 7's detectors run
on user photographs; applying them to the CNIC image is Module 10's composition
to make. The family exists so that when it does, the finding lands somewhere
other than `image_authenticity` — where it would wrongly merge with a screenshot
of a selfie and one of the two would vanish.

---

## 3. Caps, floors, and the difference

A **cap** bounds how far one family can push the score alone. `capture_quality`
is capped at 0.25 — exactly the LOW boundary — so no amount of blur can on its
own read as dishonesty. It still combines with real findings; it just cannot
manufacture one.

A **floor** is the opposite: a finding that is not "some risk" but a conclusion,
raising the band to HIGH regardless of the arithmetic.

**Nothing is marked decisive in the shipped catalogue.** The two obvious
candidates — a confirmed duplicate and a face mismatch — are precisely the
findings whose limits are documented most heavily elsewhere: identical twins are
indistinguishable at any threshold, and the 1:N duplicate threshold is
explicitly uncalibrated. Quietly hard-coding either to HIGH here would undo that
care. The mechanism exists because a deployment with its own data may reasonably
decide otherwise.

When a floor does fire, it raises the **band** without overwriting the **score**
— the arithmetic keeps saying what the evidence weighed, because that is the only
number a reviewer can audit.

---

## 4. A missing check is not a pass

If the duplicate gallery was unreachable, that is not evidence of innocence and
not evidence of guilt. It is an absence.

- the score is **unchanged**;
- `assessment_confidence` **drops** — the share of intended checks that ran;
- the check is **named** in `unavailable_checks`.

Eleven infrastructure codes (`VECTOR_DB_ERROR`, `MODEL_NOT_LOADED`,
`INFERENCE_FAILED`, …) route here rather than being scored. Scoring the
resulting silence as "nothing suspicious" is how a fraud engine gets quietly
switched off by an outage while continuing to emit confident low-risk verdicts.

---

## 5. The gap this engine cannot see in itself

If a module gains a warning and nobody scores it, the engine ignores it, every
request still produces a plausible number, and **nothing anywhere goes red**.
The gap gets discovered when somebody wonders why a fraudulent account scored 12.

So codes are enumerated from the source and each must be *deliberately* placed:

| classification | count | meaning |
|---|---|---|
| scored | 40 | in the catalogue with a weight and a family |
| benign | 33 | decided harmless — passing checks, service self-description, bad input |
| infrastructure | 11 | a check could not run |
| **unknown** | **0** | a test fails if this is ever non-zero |

An unrecognised code at runtime is still surfaced on the response as
`unrecognised_findings` and logged at WARNING, because the catalogue can lag the
code that emits into it.

**The sweep found 47 unclassified codes when first written** — including every
infrastructure error, which meant an unreachable vector database was about to be
scored as silence rather than as absence. It also caught `CNIC_UNKNOWN_PROVINCE`,
a code I had classified that the service never emits; the real one is
`CNIC_UNKNOWN_REGION_CODE`, and it had been going unscored.

---

## 6. The weights are policy, not measurement

There is no labelled fraud data in this project, so **nothing here is
calibrated**. `weights_validated` is `false` in every response.

What *has* been reasoned about and measured is the **structure** — which findings
are independent evidence and which are the same fact seen twice. Retune the
numbers against real outcomes; keep the grouping.

Three ordering properties are pinned by tests, because they encode judgements
that should not drift silently:

- no single `capture_quality` finding may reach 0.30;
- identity and document findings outweigh quality ones;
- no message states a conclusion it cannot support — no "proves", no
  "definitely". These are shown to somebody deciding whether a person is
  dishonest.

The two gravest findings carry their caveats *in the message itself*, asserted by
test, so the limitation travels with the accusation rather than staying in a
design document:

> **DUPLICATE_FACE_DETECTED** — …Not proof of one person: face recognition cannot
> separate identical twins at any threshold, and siblings score well above chance.

---

## 7. Measured end to end

| scenario | score | band | confidence |
|---|---|---|---|
| everything passes | 0.0 | LOW | 1.00 |
| **a blurry photograph (4 findings, 1 cause)** | **15.0** | **LOW** | 1.00 |
| the wrong document entirely | 61.0 | MEDIUM | 1.00 |
| a screenshot of somebody's profile | 60.0 | MEDIUM | 1.00 |
| card held up in front of a stranger | 85.4 | HIGH | 1.00 |
| an altered card | 70.0 | HIGH | 1.00 |
| three independent facts | 96.4 | HIGH | 1.00 |
| **the gallery was unreachable** | **0.0** | LOW | **0.83** |
| a finding nobody has scored yet | 0.0 | LOW | 1.00 → reported |

The two rows in bold are the ones a naive engine gets wrong.

---

## 8. What this module does not do

It does not decide. `fraud_risk_level` is evidence for the Backend's rules
engine, which owns the accept/reject outcome and knows things this service never
will — a manual allow-list, a regulatory hold, an account's history. Module 10
turns this into a *recommendation*; the Backend turns that into an outcome.

---

## 9. Testing

| suite | count | what it covers |
|---|---|---|
| `tests/unit/test_fraud_aggregation.py` | 30 | the arithmetic: family suppression, compounding, caps, floors, banding, monotonicity |
| `tests/unit/test_fraud_catalogue.py` | 89 | **every reachable code is classified**, message quality, weight ordering, no stale entries |
| `tests/unit/test_fraud_service.py` | 28 | reading each module's results, missing stages, overrides, determinism |

No model weights and no images — the whole module runs on a bare install.

Defects found while building:

1. **47 codes unclassified**, including all 11 infrastructure errors.
2. **`CNIC_UNKNOWN_PROVINCE` classified but never emitted** — the real code,
   `CNIC_UNKNOWN_REGION_CODE`, was unscored. Caught by the stale-entry test.
3. **`INVALID_IMAGE` scored as its own finding**, double-counting Module 7's
   specific authenticity finding in a *different* family — exactly what the
   family grouping exists to prevent.
4. **`cnic_face` counted twice as unavailable**, because two collectors read it,
   understating `assessment_confidence`.
5. **Two catalogue messages that only restated their code**, giving a reviewer
   nothing to act on.

---

## 10. Running it

```bash
python scripts/demo_fraud.py
```

```bash
python scripts/demo_fraud.py --compare
```

```bash
python scripts/demo_fraud.py --catalogue
```

```bash
python -m pytest tests/unit/test_fraud_aggregation.py tests/unit/test_fraud_catalogue.py tests/unit/test_fraud_service.py -q
```

---

## 11. Configuration surface

Under `fraud:` in `configs/thresholds.yaml`, overridable by `HQ_FRAUD__*`.

| key | default | effect |
|---|---|---|
| `levels.low_max` | 30.0 | highest score still LOW (inclusive) |
| `levels.medium_max` | 65.0 | highest score still MEDIUM |
| `family_caps.capture_quality` | 0.25 | blur alone cannot leave the LOW band |
| `family_caps.presentation` | 0.60 | staging is suggestive, not conclusive |
| `signal_weights` | `{}` | per-code overrides, so policy retunes without a code change |
| `expected_checks` | 6 | denominator for `assessment_confidence` |

Per-signal weights live in `src/hamqadam_ai/fraud_detection/signals.py`, each
beside an explanation of what the finding means — because a weight without its
reasoning is a number nobody can safely change.
