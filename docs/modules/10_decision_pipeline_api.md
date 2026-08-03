# Module 10 — Decision Engine, Pipeline, API and Deployment

The module that turns nine analyses into one answer, and exposes it over HTTP.

Nothing here computes a new measurement. Everything here is about composition:
what runs when, what happens when a piece breaks, how the pieces combine into a
recommendation, and how that recommendation reaches the Backend.

---

## 1. Design

### The pipeline is the only place that knows the whole picture

Every module up to now analyses one thing about one image and reports what it
found. None of them decides anything, and none of them knows what the others
saw. That was deliberate, and this module is why: a service where every
component can veto a verification has its policy scattered across ten files and
no single place to read it.

### Three phases, ordered by dependency and nothing else

```
phase A — everything that depends on one image, all concurrent
  live selfie   detect → quality → embed
  profile       detect → embed ; analyse (authenticity, subject, quality)
  secondary[i]  detect → embed ; analyse
  CNIC          OCR ; portrait extract ; moiré check

phase B — needs templates from phase A, concurrent
  duplicate     needs the selfie template
  matching      needs every template

phase C — needs every finding
  fraud         aggregate the signals
  decision      identity confidence + fraud risk → recommendation
```

Phase A is where the wall-clock goes and every item in it is independent, so it
is dispatched together. ONNX Runtime releases the GIL inside its kernels, so
`asyncio.to_thread` over a bounded pool gives real parallelism rather than the
appearance of it.

Measured on a coherent submission (selfie + profile + CNIC), the stages sum to
15.5 s of work and complete in 7.9 s wall-clock:

| stage | ms |
|---|---|
| cnic_ocr | 7946 |
| profile | 2689 |
| cnic_portrait | 2347 |
| selfie | 2256 |
| cnic_authenticity | 234 |
| duplicate | 1.9 |
| matching | 1.9 |

The CNIC read dominates and sets the floor. Duplicate search and matching are
noise by comparison — they are arithmetic over vectors that already exist.

### One deadline, shared

The whole request gets a single budget from `server.request_timeout_seconds`,
and every stage asks how much is left before starting. The alternative — a
generous timeout per stage — lets their sum quietly exceed the caller's own
timeout, which is how a service ends up holding connections nobody is waiting
on any more. A stage that starts past the budget is skipped and marked so, and
the response carries `VERIFICATION_BUDGET_EXHAUSTED`.

### Degrade, never abort

No stage failure ends the request. A verification whose CNIC was unreadable
still has a face comparison worth reporting, and a Backend that receives an
exception learns nothing about the six images that were fine.

Every stage records three things: whether it ran, whether it succeeded, and
what it cost. The recommendation is then made from whatever evidence exists,
and `assessment_confidence` says how much of it there was.

This is tested by breaking stages outright rather than by feeding in bad
images — a bad image is a *finding*, not a failure. With all five optional
stages replaced by functions that raise, the pipeline still returns a
structured `MANUAL_REVIEW` with five failures named.

### Two optimisations, one taken and one refused

**Taken.** The pipeline detects each face once and hands the result to both the
quality/embedding path and Module 7, which would otherwise detect again —
roughly 300 ms per image for an identical answer.

**Refused.** Sharing the rectified CNIC between Modules 5 and 6. It sounds
obviously right, and Module 6 exposes `prepare_card` partly for it. Measured,
`rectify_document` costs 90 ms against a 4293 ms CNIC read — **2.1%** — and
buying that back means coupling two services that currently share nothing but
an image. Not worth it.

---

## 2. The decision engine

### Approve needs everything; reject needs anything

```
APPROVE   ⟸  EVIDENCE_SUFFICIENT
          ∧  IDENTITY_CONFIDENCE_SUFFICIENT
          ∧  FRAUD_RISK_ACCEPTABLE

REJECT    ⟸  IDENTITY_CONFIDENCE_TOO_LOW
          ∨  FRAUD_RISK_TOO_HIGH

otherwise →  MANUAL_REVIEW
```

Conjunctive for approval, disjunctive for rejection, and rejection wins when
both could fire. Manual review is the **default**, not a fallback: an automated
identity decision should abstain when the evidence does not clearly support
either answer.

The conditions are not averaged, and that matters. A near-perfect face match
with a fraud risk of 50 is not an approval — were the conditions additive, a
confident match would buy an attacker past a signal raised specifically about
their submission, which is the whole point of raising it.

### Thresholds

From `configs/thresholds.yaml`:

| rule | value |
|---|---|
| `approve.min_identity_confidence` | 75.0 |
| `approve.max_fraud_risk` | 30.0 |
| `approve.min_assessment_confidence` | 0.70 |
| `reject.max_identity_confidence` | 40.0 |
| `reject.min_fraud_risk` | 65.0 |

### The evidence floor, and the defect that produced it

`min_assessment_confidence` was not in the original design. It was added
because the pipeline **approved a request that supplied only a selfie**:
identity confidence 100, fraud risk 0, recommendation APPROVE.

Nothing was wrong with either number. Nothing contradicted the applicant
because almost nothing had been checked, and the engine was reading "no
evidence against" as "evidence for". Those are not the same proposition, and
conflating them means the cheapest way to pass verification is to submit as
little as possible.

That submission now scores 0% evidence and goes to manual review.

### An absent score is not a score of nought

A verification with no usable face has *no* identity confidence, and the field
is `None` rather than `0.0`. Substituting zero would reject the applicant for
impersonation when the actual finding is that their photograph was unusable —
a different answer, and one they can fix.

### Every outcome explains itself

Each reason carries a code, a message naming both the observed value and the
threshold it was compared against, and the detail behind it:

```
+ EVIDENCE_SUFFICIENT
    100% of the intended checks ran, meeting the 70% required before a
    verification can be approved automatically.
+ IDENTITY_CONFIDENCE_SUFFICIENT
    Identity confidence of 92.9 meets the automatic-approval threshold of 75.0.
+ FRAUD_RISK_ACCEPTABLE
    Fraud risk of 10.0 is within the automatic-approval ceiling of 30.0.
```

A reviewer can act on that without also holding the configuration file open,
which is the difference between an explanation and a label.

---

## 3. Measured end-to-end behaviour

Six scenarios, run through `scripts/demo_pipeline.py`. Every image is synthetic
or a print-degraded public-domain reference portrait.

| scenario | verdict | identity | fraud | evidence | decisive signal |
|---|---|---|---|---|---|
| coherent | APPROVE | 92.9 | 10.0 | 100% | — |
| impostor | REJECT | 45.0 | 72.8 | 100% | `CNIC_FACE_MISMATCH` |
| held-up card | REJECT | 45.0 | 88.4 | 100% | `CNIC_FOREIGN_FACE_PRESENT` + `CNIC_FACE_MISMATCH` |
| duplicate | REJECT | 92.9 | 73.0 | 100% | `DUPLICATE_FACE_DETECTED` |
| unreadable CNIC | MANUAL_REVIEW | 100.0 | 61.0 | 100% | `OCR_NOT_A_CNIC`, `CNIC_FACE_NOT_FOUND` |
| selfie only | MANUAL_REVIEW | not established | 0.0 | 0% | — (evidence floor) |

Three of these are worth reading closely.

**The impostor is caught by the document, not the face.** Identity confidence
is capped at 45 because the CNIC portrait disagrees with the selfie, even
though the selfie and profile agree with each other perfectly. Two photographs
of the same person are not evidence of identity when the document names
somebody else.

**The held-up card is caught twice over.** The frame contains two faces: the
small printed portrait and the large live one behind it. A "biggest face wins"
extractor picks the live face, after which the selfie is compared against
itself and passes whoever the card belongs to. Module 6's geometric
containment picks the printed portrait instead, and Module 9 additionally
raises `CNIC_FOREIGN_FACE_PRESENT` for the live face that has no business
being in a document photograph.

**The duplicate keeps a high identity confidence and is rejected anyway.**
Identity 92.9 is correct — that really is the same person as the card. The
problem is that the same face is already enrolled under a different account,
which no amount of biometric agreement addresses.

---

## 4. Defects found while building this module

Eight, all found by driving the whole thing rather than by reading it.

### `create_app(settings)` silently ignored its argument

Every authenticated endpoint returned **500 CONFIGURATION_ERROR** while
`/health`, `/ready` and `/metrics` all returned 200.

`lifespan` called `get_settings()` itself rather than using what
`create_app` was given, so a caller's API keys never reached the routes,
`verify_api_key` saw an empty key list, and it correctly refused to run
unauthenticated. The 500 named the symptom precisely and the cause not at all.

### `UploadFile` annotations could not be resolved

`FastAPIError: Invalid args for response field! ... ForwardRef('UploadFile | None')`.

The fastapi imports were function-local while `from __future__ import
annotations` had turned every annotation into a string, so FastAPI resolved
handler annotations against a module namespace that did not contain
`UploadFile`. The imports now stay at module level, with a note in
`register_routes` saying why they must.

### `warnings` was empty in exactly the case where the most had gone wrong

With five stages deliberately broken, `stages[].error` held five failures and
`warnings` was `[]`. Warnings were only ever collected from sub-results' own
warning lists, and a stage that raised has no sub-result to collect from.

`warnings` is the field a caller reads to find out what went wrong, and it was
silent precisely when the most had. Each failed stage now emits
`VERIFICATION_STAGE_FAILED`.

### `complete: true` beside `evidence available: 0%`

Two fields contradicting each other in one response. `complete` is documented
as "whether every intended stage ran", but the implementation considered only
stages that *had* run — so four skipped stages left it `true`. A stage exists
in that list only because the pipeline intended to run it, so a skipped one
now counts against completeness exactly as a failed one does.

### The same fraud signal reported twice

`CNIC_FACE_NOT_FOUND` and `CNIC_TOO_FEW_FIELDS` each appeared twice, because
two callers legitimately observe the same thing — the pipeline sees an absent
portrait directly, and `collect_cnic_face` sees it in the service's own
warnings.

The score was unaffected, since the aggregator takes the maximum within a
family rather than summing, and the risk stayed at 61.0 after the fix, which
confirms it. But a reviewer shown the same objection twice reasonably reads it
as two separate problems. `add_code` now collapses repeats, keeping the higher
confidence.

### A closed registry poisoned the whole process

The worst of the six, because its symptom pointed somewhere else entirely.

`ModelRegistry.close()` marks an instance dead permanently and shuts down its
thread pool, but it cannot clear the module-level global pointing at it — it
has no reference to it. Anything closing the registry directly rather than
through `reset_registry` therefore left every later caller holding a corpse,
and the next model load failed with:

> The face recogniser 'face_embedder_arcface' could not be loaded … Run
> `python scripts/download_models.py`

which names a missing weights file. The weights were present the whole time.

It surfaced because an in-process `TestClient` shutdown releases its services'
sessions, so **every integration test that ran after the API tests skipped
silently** — nine of them, reporting a plausible reason. Running the pipeline
tests before the API tests passed, which is what made it look like process
contention rather than a defect.

That is reachable in production, not only in tests: a second `create_app` in
one process, two apps mounted together, or an in-process restart would all find
nothing loadable. `get_registry` now rebuilds rather than returning a closed
instance, which is what a first call would have done anyway. Fixing it recovered
14 tests that had been passing by not running.

### `HQ_SECURITY__API_KEYS=my-key` refused to start

`SecurityConfig.api_keys` carried a validator whose docstring said it accepted a
comma-separated string "so `HQ_SECURITY__API_KEYS` works". It did not work.
pydantic-settings classifies `list[str]` as a complex field and calls
`json.loads` inside the **source**, before any field validator runs, so a
plainly-written key failed with:

> SettingsError: error parsing value for field "security"

naming the whole section rather than the field, and saying nothing about the
real requirement, which was JSON. Environment variables are strings and an
operator setting one API key will write it plainly, so that trap sat squarely in
the deployment path — and the compose stack passes keys exactly this way.

The field is now annotated `NoDecode`, the framework's own mechanism for
declining source-level parsing, so the validator finally sees the raw string.
A bare key, a comma-separated list and a JSON array all work; a truncated array
is refused rather than silently becoming a literal key named `["broken`.

An earlier attempt wrapped the environment source to retry a failed parse. It
made the bare string *stop erroring* and silently yield an empty key list —
strictly worse than the error, since a deployment would then start with no keys
at all. It was discarded rather than tuned: reaching for a framework's escape
hatch beat fighting its parsing order.

### `/metrics` served nothing

The endpoint returned 200 with only `prometheus_client`'s default process
collectors. No verification counter, no latency histogram, nothing about
decisions — a monitored service that could not answer "is it working", "is it
fast enough" or "is it deciding sensibly". Module 10 now defines eleven
collectors; a single verification populates 124 series.

Two further defects were found in the *tests* rather than the code, and both
turned out to be the code behaving better than the test assumed. A uniform
image is rejected at decode with "this image is a single flat colour" rather
than reaching the detector and answering the true-but-useless "no face
detected"; and `HmacConfig` refuses to validate when signing is enabled
without a secret, so the deployment fails at startup rather than at the first
signed request. Both tests were rewritten to assert the better behaviour, with
the runtime check kept as defence in depth for the path validation cannot
reach.

---

## 5. The HTTP surface

| endpoint | purpose | auth |
|---|---|---|
| `POST /v1/verify` | run a full verification | API key |
| `POST /v1/duplicate/enrol` | add a face to the gallery | API key |
| `DELETE /v1/duplicate/{reference}` | erase a face from the gallery | API key |
| `GET /health` | liveness | none |
| `GET /ready` | readiness | none |
| `GET /metrics` | Prometheus exposition | none |

### Liveness and readiness are different questions

`/health` answers "is this process running" and returns 200 as long as it can.
`/ready` answers "should traffic be sent here" and returns 503 while models are
loading or a required one failed.

Conflating them is a classic way to build a crash loop: the orchestrator kills
a pod that was merely still warming up, and it never finishes warming up. A
liveness probe also cannot carry an API key, which is why `/health` is
unauthenticated.

`/metrics` is unauthenticated deliberately too — a scrape must keep working
when key rotation goes wrong, which is exactly when the metrics matter most.
Expose the port only inside the cluster.

### Multipart, not base64

Seven photographs base64-encode to roughly a third more bytes than they need,
and the multipart parser streams rather than buffering the whole body as one
string. Every upload goes through `decode_image`, which enforces size, format,
dimension and decompression-bomb limits **before** anything is allocated at
full resolution — a decompression bomb is only a bomb if you decode it first.

### Errors carry codes, not stack traces

Every `HamqadamError` knows its HTTP status and its stable code, so the handler
maps them mechanically. An unexpected exception becomes a generic 500 with the
request id, and the detail goes to the log — an internal traceback in an API
response is an information leak with a debugging excuse.

Observed:

| request | status | code |
|---|---|---|
| no API key | 401 | `UNAUTHORIZED` |
| wrong API key | 401 | `UNAUTHORIZED` |
| no images | 400 | `VALIDATION_ERROR` |
| corrupt upload | 415 | `UNSUPPORTED_IMAGE_FORMAT` |
| uniform image | 415 | `IMAGE_DECODE_FAILED` |
| faceless enrolment | 4xx | `FACE_NOT_DETECTED` |

### Erasure is a first-class route

`DELETE /v1/duplicate/{reference}` exists because the service stores biometric
templates, and one that cannot delete them on request cannot lawfully be
deployed. It is idempotent — erasing an absent reference returns 200 with
`removed: false, erased: true`, not 404 — because a caller retrying an erasure
must not be told it failed the second time. That is how erasure requests get
abandoned half-done.

### Three security layers, and where failing closed is wrong

**The API key** says which caller this is, compared with `hmac.compare_digest`
rather than `==`: a naive comparison returns early on the first differing byte
and leaks the key one character at a time to anyone who can measure response
times. Every configured key is compared, and all of them are compared even
after a match, so the time taken reveals neither which key matched nor how
many exist.

**The HMAC signature** says the body has not been altered and is not a replay.
The timestamp is inside the signed payload, so an attacker cannot replay an old
body under a fresh timestamp without the secret. Absent, malformed, stale and
simply wrong all return the same response — distinguishing them would tell an
attacker which part of their forgery to fix next.

**The rate limit** stops one caller consuming the whole inference pool. A token
bucket rather than a fixed window, because a fixed window lets a caller send
its whole allowance in the last second of one window and again in the first
second of the next. Two buckets, because a health probe and a verification are
not the same load and Kubernetes polls liveness forever.

Authentication fails **closed**. Rate limiting fails **open** when its backing
store is unreachable, and that asymmetry is deliberate: refusing every request
because Redis is down converts a nice-to-have into a hard dependency and turns
a degraded service into no service.

The in-process limiter means each replica enforces the limit independently — N
replicas allow N times the configured rate. That is a real limitation, the
right fix is the Redis-backed limiter the configuration anticipates, and this
one is a bound rather than no bound in the meantime.

---

## 6. Asynchronous verification

`workers/consumer.py` runs the same pipeline from a RabbitMQ queue. A
verification is seconds of CPU across seven images; holding an HTTP connection
open for that works, but it couples the Backend's request timeout to this
service's worst case.

**Prefetch is one, deliberately.** A verification saturates the inference pool
on its own, so prefetching more would only build a queue inside the worker
where the broker cannot see it or redistribute it.

**Redelivery is the one thing that makes this more than a loop.** A broker
guarantees *at least once*, so a worker that finishes and dies before
acknowledging will see the message again. For a stateless analysis that is
harmless. For enrolment it is not: the naive handling would add a second
template for one person, and every future query would match them twice.
Enrolment is keyed by the Backend's reference and **replaces** rather than
appends, so a redelivery is a no-op — a property that lives in Module 8's
store, which is why this consumer can be as simple as it is.

**Acknowledge after publishing, never before.** The other order loses the
result if publishing fails, and the broker has no way to know it should
redeliver.

**Poison messages are rejected without requeue after three attempts, and the
reason is published** rather than dropped. Requeuing forever is how one
malformed payload takes down a worker pool; dropping silently leaves the
Backend waiting for a result that will never come.

---

## 7. Observability

Eleven collectors, answering four questions. A single verification populates
124 series.

| metric | question |
|---|---|
| `hamqadam_verifications_total{recommendation,fraud_level}` | is it working |
| `hamqadam_errors_total{code}` | " |
| `hamqadam_verification_seconds` | is it fast enough |
| `hamqadam_stage_seconds{stage}` | " |
| `hamqadam_stage_status_total{stage,status}` | is it degraded |
| `hamqadam_budget_exhausted_total` | " |
| `hamqadam_identity_confidence` | is it deciding sensibly |
| `hamqadam_fraud_risk` | " |
| `hamqadam_decision_reasons_total{reason}` | " |
| `hamqadam_fraud_signals_total{code,family}` | " |
| `hamqadam_gallery_vectors` | how large is the gallery |

**Aggregate and non-identifying.** No label carries a user reference, a CNIC
number, a filename or an image hash. A metric is scraped, stored for a month
and read by anyone with dashboard access, which makes it exactly the wrong
place for anything about a specific applicant.

**Bounded cardinality.** Prometheus keeps one time series per distinct label
combination, so a label with unbounded values — a request id, an error message
— multiplies the series count without bound until the server falls over. Every
label here draws from a small fixed set.

**Histograms, not gauges,** for latency. A mean hides the tail, and the tail is
what the caller's timeout actually hits.

**`identity_confidence` is observed only when it exists.** A request with no
usable face has no identity confidence; observing a placeholder zero would put
a spike at the bottom of the histogram and make a shortage of evidence look
like a wave of impostors.

**Three stage states from two booleans.** A stage that did not run is not the
same as one that ran and failed, and collapsing them would hide degradation
behind an unchanged failure count.

**Instrumentation never raises.** Every recording function swallows its own
errors. Instrumentation that can fail a request is worse than none, because it
converts an observability problem into an availability problem.

### Alerts

Nine rules in `deploy/monitoring/alerts.yml`, each alerting on a symptom a
person would act on. An alert nobody acts on trains the on-call to ignore the
channel, which is how a real outage gets missed.

Every ratio uses `clamp_min` on its denominator. Without it an idle service
divides by zero and pages someone at 3 a.m. about nothing.

Three are decision-*quality* alerts rather than availability ones, and they are
the ones most likely to earn their keep. `ManualReviewSurge` fires when over
40% of verifications need a human — usually because an input source changed, a
new app version compressing uploads harder for instance, rather than because
fraud rose. `ApprovalRateCollapse` catches legitimate applicants being turned
away. `DuplicateSignalSpike` compares against the same window yesterday, since
a gallery that has grown may simply need its threshold recalibrated.

---

## 8. Deployment

Two-stage Dockerfile: the builder compiles wheels, the runtime carries only the
installed environment and the source. A build toolchain inside a running
container is a gift to anyone who gets a shell in it.

Specific choices worth naming:

- **CPU-only torch, explicitly.** The default index pulls the CUDA build:
  roughly 2 GB of GPU runtime a CPU deployment will never execute.
- **`libgl1` and `libglib2.0-0` in both stages.** Without them `import cv2`
  fails with a confusing `ImportError` about libGL, which is among the most
  commonly hit problems in containerised computer vision.
- **Non-root, uid 10001.** The service handles identity documents; a container
  escape from a root process is a much larger incident.
- **Model weights on a volume, not in the image.** Several hundred megabytes
  baked into every layer would make each deploy slow and every rollback
  expensive, and weights change on a different cadence from code.
- **`--workers 1`.** Each uvicorn worker loads its own copy of every model, so
  four workers means four times the memory for models already releasing the GIL
  inside ONNX Runtime. Scale with replicas.
- **`tmpfs` for temporary image work.** An identity document written to a
  container's writable layer survives in that layer after the request that
  created it, which is exactly what the retention policy forbids.
- **Secrets via `${VAR:?}`.** Compose fails with a named variable rather than
  starting a service that silently accepts every request.
- **Qdrant and Redis publish no ports.** The gallery holds biometric templates;
  it is reachable from the compose network and nowhere else.
- **Healthcheck hits `/health`, not `/ready`,** with a 180 s start period —
  see §5 on why conflating them produces crash loops.

---

## 9. What is not validated

Carried forward, and still true:

- **Match thresholds are engineering defaults** from the ArcFace literature,
  not values derived from this deployment's data. Every response says so via
  `MATCHING_THRESHOLDS_UNVALIDATED` and `thresholds_validated: false`. Run
  `scripts/evaluate_matching.py` against a labelled corpus and re-pin them.
- **Fraud weights are reasoned, not fitted** (`weights_validated: false`).
  They encode a considered ordering of severity; they are not calibrated
  against outcome data, because there is no outcome data yet.
- **The duplicate threshold is calibrated against a gallery size, not against
  this gallery.** Re-run `scripts/calibrate_duplicate_threshold.py` as it
  grows.
- **The decision thresholds in §2 are policy, not measurement.** They are where
  the Client's risk appetite belongs, and they should be revisited once manual
  review outcomes exist to tune against.

The service is honest about all four in every response it returns. That is the
point: a number presented without its provenance invites a confidence it has
not earned.

---

## 10. Privacy and retention

Unchanged from the constraints set at the outset, and enforced here:

- **No image is ever written to disk** by the API. Uploads are decoded in
  memory and dropped when the request ends.
- **No user data in metrics or logs.** Structlog's redaction processor runs
  last, and every metric label draws from a fixed set.
- **The user reference is never echoed back** in a response. The Backend
  already knows which account it asked about; repeating the identifier only
  widens where it is written down. Asserted by test.
- **Erasure is a supported operation,** idempotently.
- **No user CNIC image, profile image, selfie or verification datum may be
  used for personal, commercial, research or model-training purposes** without
  written authorisation from the Client.
- **No real identity document appears in this repository.** Every fixture is
  synthetic or a print-degraded public-domain reference portrait.
