# Module 8 — Duplicate Face Detection

Asks whether the person verifying is already enrolled under another account.

This is the **only stateful module** in the service. Everything else takes
images and returns scores; this one keeps a gallery of biometric templates,
which changes what "correct" means for it.

---

## 1. A 1:N search is not a 1:1 verification

Module 4 asks *"are these two the same person?"* — one comparison against one
claimed identity. This module asks *"is this person already in the gallery?"* —
one comparison against **everyone enrolled**. Reusing the verification threshold
is the standard way to get this wrong, because a query compared against N
templates has N chances to match one by accident:

```
system false-match rate  ~  1 - (1 - per_comparison_far) ** N
```

At a per-comparison FAR of 1e-4 — unremarkable for a face recogniser:

| gallery | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|
| **system FMR** | 1.0% | 9.5% | **63.2%** | **100.0%** |

By ten thousand enrolled faces, an ordinary recogniser FAR produces a false
duplicate on two queries in three.

---

## 2. Why the threshold is not validated, and cannot be here

### The naive model says there is no problem

Modelled as isotropic random unit vectors in 512 dimensions, impostors never
come close: max cosine over 3,000 templates is **0.214**, against a configured
threshold of 0.68. On that model, gallery growth is harmless.

### That model is wrong in the unsafe direction

Real face embeddings do not fill the sphere. Every enrolled vector is a *face*,
so they share a manifold of far lower intrinsic dimension. Confining the
population to a subspace and re-measuring the impostor distribution over 2,000
templates:

| effective dim | 99.9th pct | max | verdict |
|---|---|---|---|
| 512 | 0.136 | 0.214 | safe |
| 128 | 0.300 | 0.468 | safe |
| 64 | 0.395 | 0.576 | safe |
| **32** | 0.531 | **0.750** | **exceeds 0.68** |
| **16** | 0.700 | **0.882** | **exceeds 0.68** |
| **8** | 0.877 | **0.986** | **exceeds 0.68** |

The answer swings from *entirely safe* to *fails always* across one parameter —
a property of the recogniser and the enrolled population that **cannot be
estimated from two public-domain reference faces**.

So `thresholds_validated` is `false` in every response, and this module ships a
calibration tool rather than a number:

```bash
python scripts/calibrate_duplicate_threshold.py --explain
```

```bash
python scripts/calibrate_duplicate_threshold.py --vectors templates.npy --gallery-size 250000
```

### What no threshold fixes

Everything above concerns *random* impostors. The pairs that actually defeat a
duplicate check are not random: **identical twins are indistinguishable to face
recognition at any operating point**, and siblings and cousins score well above
chance. A matrimonial platform serving extended families will enrol exactly
that population.

`describe()` reports this under `known_limits`, and every duplicate carries a
`DUPLICATE_MAY_BE_A_RELATIVE` warning. Only a second factor addresses it, and
choosing one is the Client's decision.

---

## 3. Three traps this module is built around

### Self-exclusion

A user re-verifying matches **their own enrolled template at cosine 1.0**.
Without excluding the querying reference, every returning user is reported as a
duplicate of themselves — and the bug looks exactly like a working detector.

`check(embedding, reference=...)` excludes it. Omitting the reference is legal
(the caller may not know it) but raises `DUPLICATE_SELF_NOT_EXCLUDED`, because
the consequence is silent and severe.

### Model versioning

A search never crosses recogniser versions. Two embeddings from different
ArcFace builds are not comparable, and comparing them anyway produces scores
that look ordinary and mean nothing. `gallery_size` is likewise per-version: a
caller cannot interpret a similarity without knowing how many *comparable*
entries it beat.

### Check never enrols

`check()` does not write, and `enrol()` does not search. That looks like an
inconvenience and is a safety property: enrolling as a side effect of checking
would put a **rejected** applicant's face permanently in the gallery, where it
would then match their next legitimate attempt.

---

## 4. Storing biometric data

The one place in this service where privacy is a design problem rather than a
matter of deleting temp files.

| decision | why |
|---|---|
| keyed by an **opaque reference the Backend supplies** | this service never invents an identifier and cannot resolve one to an account |
| **erasure is on the port**, and idempotent | a service that stores face templates and cannot delete one on request cannot lawfully be deployed; a retried erasure must not error the second time |
| the vector **never** appears in a response or a log | `VectorRecord.describe()` omits it; a test asserts the response JSON contains no `vector` |
| the log summary names **no matched account** | a log line saying which account a face matched is a linkage nobody asked for |
| `close()` clears the gallery | releasing the store releases the templates, rather than leaving them for the garbage collector |

Metadata attached at enrolment is stored verbatim. The docstring says plainly
that it **must be PII-free** and that this service cannot enforce that — rather
than implying a guarantee it does not provide.

---

## 5. Two adapters

| | in-process | Qdrant |
|---|---|---|
| search | exact, over a materialised matrix | approximate index |
| durable | **no** — lost on restart | yes |
| shared between replicas | **no** | yes |
| use for | tests, single node | anything with more than one replica |

The in-process store is not a stub — it is exact and fast. Its limitation is
durability: several replicas would each hold a different partial gallery, so
whether a duplicate was found would depend on which pod answered. The fallback
to it is logged as a **warning naming that consequence**, because an in-process
gallery behind a load balancer looks exactly like a working system.

Both adapters are tested against **the same contract suite**, parametrised over
both. The Qdrant cases use the client's embedded engine — the real query path,
not a hand-written fake.

That distinction earned its keep immediately. The real engine returns cosine
**1.0000000158616884** for a vector matched against itself; the response schema
bounds similarity to `[-1, 1]` and rejected it, taking the whole verification
down. A fake would have returned exactly 1.0 and hidden it until production.
`SearchHit` now clamps, so no adapter can reintroduce it.

---

## 6. Three performance defects, each found by measuring the last fix

I wrote 0.6 / 4 / 19 ms into the performance-test docstring as **guesses**.
Checking them turned up a defect, and fixing that one exposed the next.

**1. The gallery was stacked into a fresh matrix on every query** — 200 MB of
copying at 100,000 templates.

| gallery | per-query stack | materialised |
|---|---|---|
| 1,000 | 1.46 ms | **0.18 ms** |
| 10,000 | 15.23 ms | **0.96 ms** |
| 50,000 | 80.55 ms | **6.86 ms** |
| 100,000 | **285.28 ms** | **17.13 ms** |

Seventeen times faster at 100,000, and the growth became linear. It had been
superlinear — the wrong *shape* as well as the wrong number, for a structure
whose whole justification is that exact search is linear.

**2. The fix moved the cost rather than removing it.** Appending with
`np.vstack` copies the whole buffer, so building a gallery was O(n²) and a
realistic check-then-enrol cycle over 20,000 templates took **60 ms** — of
which 58 ms was the copy, against a 2 ms search. A capacity-doubling buffer
made appends amortised constant at **0.010 ms** each, and the cycle 6.8 ms.

**3. A returning user still rebuilt the whole gallery.** Re-verifying re-enrols
under the same reference, which is a *replacement*, and replacement invalidated
the index. Caught by a test that used the same reference twice and failed.
Adding a reference→row map made replacement O(1) — and the same map removed an
O(n) list scan that was running on **every query** to exclude the caller.

Final, over a 20,000 gallery:

| cycle | before | after |
|---|---|---|
| new user (check + append) | 60 ms | **5.5 ms** |
| returning user (check + replace) | ~50 ms | **5.3 ms** |

Both are now dominated by the search itself, which is what should dominate.

---

## 7. A numerical bug worth naming

`per_comparison_far` inverts `system = 1 - (1 - far) ** n`. Written directly,
that expression suffers catastrophic cancellation for a large gallery: at
`n = 1e12` the intermediate rounds to exactly 1.0, so the result is **zero** —
which reads as "no threshold can achieve this target" and is off by eighteen
orders of magnitude.

Computed through `log1p` and `expm1` it returns 1e-18, which is correct. A
parametrised test covers 1e6 through 1e15.

---

## 8. Testing

| suite | count | what it covers |
|---|---|---|
| `tests/unit/test_vector_store_contract.py` | 50 | the port contract, **parametrised over both adapters**: enrolment, replacement, self-exclusion, version isolation, erasure, capacity, thread safety |
| `tests/unit/test_duplicate_calibration.py` | 37 | the compounding arithmetic, the manifold simulation, threshold recommendation, the underflow fix |
| `tests/unit/test_duplicate_service.py` | 34 | verdicts, warnings, gallery outage, and that checking never enrols |
| `tests/integration/test_duplicate_service.py` | 12 | real ArcFace templates, both adapters agreeing on the number as well as the verdict |
| `tests/performance/test_duplicate_performance.py` | 6 | latency and the **shape** of its growth |

Defects found while building:

1. **`enrol` inferred replacement from a top-1 search**, which returns the
   *closest* record rather than the one asked for — so it reported "new
   enrolment" whenever somebody else's template happened to be nearer. Fixed by
   adding `exists()` to the port.
2. **Qdrant's cosine exceeded 1.0** and broke the response schema.
3. **Three successive performance defects** (§6), each exposed by measuring the
   previous fix rather than trusting it.
4. **The `per_comparison_far` underflow** (§7).

Four of the six were found by going back to check a number I had written down
without measuring it.

---

## 9. Configuration surface

Under `duplicate:` in `configs/thresholds.yaml`, overridable by `HQ_DUPLICATE__*`.
The block carries the full argument above, so an operator changing a threshold
meets the reasoning before the number.

| key | default | effect |
|---|---|---|
| `backend` | `memory` | `memory` or `qdrant` |
| `qdrant.url` | `http://localhost:6333` | `:memory:` or a path uses the embedded engine |
| `qdrant.collection` | `hamqadam_faces` | collection name |
| `max_memory_records` | 250,000 | in-process ceiling; refuses rather than evicting |
| `similarity_threshold` | 0.68 | **unvalidated** — cosine at or above which a hit is a duplicate |
| `review_threshold` | 0.58 | **unvalidated** — worth a human look; a validator forbids it exceeding the above |
| `top_k` | 10 | neighbours returned, so a reviewer sees what else was close |
| `on_duplicate` | `manual_review` | recommendation only; `manual_review` is the defensible default given the twin limit |
