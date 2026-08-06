# Hamqadam AI Verification Service — Architecture

## 1. Position in the system

The AI service is a **stateless analysis engine**. It never talks to the Flutter
app, never owns the verification record, and never decides a user's account
status. It receives images, returns scores and a *recommendation*, and forgets
everything.

```
Flutter App
    │  captures CNIC + profile + secondary images + live selfie
    ▼
Backend API                    ← owns the verification record, the database,
    │                            the rules engine and the audit trail
    ▼
AI Verification Service        ← THIS PROJECT
    │  face / document / fraud analysis
    ▼
Backend Rules Engine           ← turns the recommendation into a decision
    │
    ▼
APPROVED / REJECTED / MANUAL_REVIEW → Database → Flutter
```

Section 25 of the requirements document fixes this boundary. The AI service
returns `APPROVE`, `REJECT` or `MANUAL_REVIEW` as a *recommendation*; the
Backend is free to override it with business rules the AI knows nothing about
(a manual allow-list, a regulatory hold, a VIP fast-track).

## 2. Layering

Clean Architecture, four rings. Dependencies point strictly inward — a detector
never imports a service, a service never imports FastAPI.

```
┌─ delivery ──────────────────────────────────────────────────┐
│  api/            FastAPI routers, middleware, auth          │
│  workers/        RabbitMQ consumers for async verification  │
├─ application ───────────────────────────────────────────────┤
│  services/       one per capability; orchestration + policy │
│  pipelines/      the end-to-end verification pipeline       │
│  decision/       MODULE 10  recommendation rules            │
├─ capability ────────────────────────────────────────────────┤
│  detectors/      MODULE 1   face detection                  │
│  quality/        MODULE 2   image + face quality            │
│  embeddings/     MODULE 3   ArcFace embeddings              │
│  matching/       MODULE 4   cosine matching                 │
│  ocr/            MODULE 5   CNIC text extraction            │
│  documents/      MODULE 6   CNIC portrait localisation      │
│  authenticity/   MODULE 7   is this a genuine capture?      │
│  duplicate_detection/  MODULE 8   the enrolled-face gallery │
│  fraud_detection/      MODULE 9   signal families + noisy-OR│
├─ infrastructure ────────────────────────────────────────────┤
│  models/         ONNX sessions, versioned registry          │
│  utils/          image codecs, geometry, timing, scratch    │
│  logging/        structlog + mandatory PII redaction        │
│  observability/  Prometheus collectors, no PII labels       │
├─ domain ────────────────────────────────────────────────────┤
│  core/           config, error taxonomy, context, retries   │
│  schemas/        Pydantic request/response contracts        │
└─────────────────────────────────────────────────────────────┘
```

Each capability package exposes an **abstract port** plus one or more concrete
**adapters**. Adapters are selected by configuration, so swapping SCRFD for
YOLO, or PaddleOCR for EasyOCR, is a YAML edit and a restart.

## 3. Cross-cutting decisions

### 3.1 Every threshold lives in `configs/thresholds.yaml`

No number that can change an accept/reject outcome is hard-coded in Python.
Operations retune the system by editing YAML (or setting an `HQ_*` environment
variable) and restarting — no code change, no image rebuild. The config is
validated at start-up by Pydantic, so a typo fails the pod rather than silently
disabling a check.

### 3.2 Degrade, never fail

A verification service returning HTTP 503 because a model file is missing is
strictly worse than one returning a correct answer from a weaker model and
saying so. Every capability has a fallback chain, every result carries the
adapter and version that produced it, and every degradation raises a warning
that reaches the Backend and the fraud engine.

### 3.3 Business outcomes are not exceptions

"No face detected" is a normal, expected result. It is returned as a populated
result object with `passed=False` and an error code — never raised — because the
pipeline must still analyse the other six images and hand the Backend a complete
picture. Exceptions are reserved for genuine faults.

### 3.4 Images are never persisted

Section 21 of the requirements document forbids permanent AI-side storage. The
pipeline works entirely in memory. Scratch space exists only for third-party
libraries that demand a file path, and it is:

- created per-request with `0700` permissions,
- deleted in a `finally` block,
- overwritten before unlinking when `storage.secure_delete` is on,
- swept by a background janitor for the crash case a `finally` cannot cover.

### 3.4a One module is stateful, and it is the exception that proves the rule

Module 8 stores face templates. It is the only component that persists
anything, and the privacy posture is built into its port rather than bolted on:
records are keyed by an opaque reference the Backend supplies, erasure is a
first-class idempotent operation, and neither the vector nor the matched
account reaches a response or a log line. See
[modules/08_duplicate_detection.md](modules/08_duplicate_detection.md).

### 3.5 PII cannot reach a log sink

The structlog pipeline ends with a redaction processor that runs *after* every
other processor. Three mechanisms: key dropping (image bytes, embeddings),
key masking (names, CNIC numbers), and regex scrubbing of every remaining
string. A CNIC number embedded in a free-text OCR dump is caught by the third
even though the first two cannot see it. `user_id` is stored in the request
context only as a salted BLAKE2b digest.

### 3.6 Model integrity is verified, not assumed

A tampered detector is a total, silent compromise of the verification decision.
Every artefact is SHA-256 verified against `configs/model_digests.lock.yaml`
before ONNX Runtime is allowed to open it. Because upstream does not sign its
releases, the lock file is generated by `scripts/download_models.py` on first
download and then committed and code-reviewed — trust-on-first-download
followed by pinning, the same model every language package manager uses.

Digests are deliberately **not** hand-written into `models.yaml`: publishing a
digest nobody computed looks like a verification and is not one.

### 3.7 Provenance in every response

`model_versions` reports the pinned version of every model that contributed to
a result, alongside the resolved device. A rejected verification can be
reproduced months later against the exact artefacts that produced it, which is
a hard requirement for disputing or auditing a decision.

## 4. Runtime model

- **Inference** is dispatched to a bounded `ThreadPoolExecutor`. ONNX Runtime
  releases the GIL inside its kernels, so this gives genuine parallelism while
  keeping the event loop responsive. The pool is bounded because each in-flight
  session holds an arena allocation and an unbounded pool OOMs a container long
  before it saturates the CPU.
- **Device selection** resolves `runtime.device` against the providers ONNX
  Runtime actually reports, and always terminates in `CPUExecutionProvider`. A
  requested-but-unavailable accelerator degrades with a warning and is flagged
  `degraded: true` on `/health`, so a silently CPU-bound production node is
  visible rather than merely slow.
- **Models load eagerly at start-up**, not lazily. Lazy loading makes the first
  verification on every cold pod seconds slower and turns a missing model into a
  500 on a real user's request instead of a failed readiness probe.

## 5. Time budget

One `Deadline` is created per request from `server.request_timeout_seconds` and
every stage asks it how much time is left. This prevents the failure mode where
each stage has a generous individual timeout and their sum quietly exceeds the
caller's own timeout.

A stage that would start past the budget is **skipped and marked so** rather
than started and abandoned, and the response carries
`VERIFICATION_BUDGET_EXHAUSTED` alongside whatever evidence did arrive. The
recommendation is then made from that partial evidence, which is why
`assessment_confidence` exists: it reports the share of intended checks that
actually ran, and the decision engine refuses to approve below 0.70.

Measured on a coherent submission — selfie, profile and CNIC — the stages sum
to 15.5 s of work and complete in **7.9 s** wall-clock. The CNIC read (7.9 s)
sets the floor; duplicate search and matching cost under 2 ms each.

## 6. Module status

| Module | Package | Notes | Status |
|---|---|---|---|
| 0 — Foundation | `core/`, `schemas/`, `utils/`, `models/`, `logging/` | — | Complete |
| 1 — Face Detection | `detectors/`, `services/face_detection_service.py` | [doc](modules/01_face_detection.md) | Complete |
| 2 — Face Quality | `quality/` | [doc](modules/02_face_quality.md) | Complete |
| 3 — Face Embeddings | `embeddings/` | [doc](modules/03_face_embeddings.md) | Complete |
| 4 — Face Matching | `matching/` | [doc](modules/04_face_matching.md) | Complete |
| 5 — CNIC OCR | `ocr/` | [doc](modules/05_cnic_ocr.md) | Complete |
| 6 — CNIC Face Matching | `documents/`, `services/cnic_face_service.py` | [doc](modules/06_cnic_face_matching.md) | Complete |
| 7 — Profile Image Analysis | `authenticity/`, `services/profile_service.py` | [doc](modules/07_profile_image_analysis.md) | Complete |
| 8 — Duplicate Detection | `duplicate_detection/`, `services/duplicate_service.py` | [doc](modules/08_duplicate_detection.md) | Complete |
| 9 — Fraud Risk Engine | `fraud_detection/`, `services/fraud_service.py` | [doc](modules/09_fraud_risk_engine.md) | Complete |
| 10 — Decision + Pipeline + API | `decision/`, `pipelines/`, `api/`, `workers/`, `observability/` | [doc](modules/10_decision_pipeline_api.md) | Complete |

## 7. Delivery surfaces

Two, over one pipeline.

**Synchronous HTTP** (`api/`) for the interactive path: the Backend posts a
multipart submission and holds the connection for the few seconds a
verification takes.

**A RabbitMQ consumer** (`workers/`) for bulk work — enrolment sweeps,
re-verification runs — where coupling the Backend's request timeout to this
service's worst case buys nothing.

Both call `VerificationPipeline.verify`. Neither contains policy: the recommendation
comes from `decision/`, and the thresholds come from `configs/thresholds.yaml`.

The Backend team's contract lives in two places: [integration.md](integration.md)
for the prose, worked examples and error semantics, and
[openapi.json](openapi.json) for the machine-readable schema. The latter is
committed rather than only served, so a field renamed between releases shows up
in a diff instead of at runtime on a real applicant's verification. Regenerate it
with `python scripts/export_openapi.py -o docs/openapi.json`.

| endpoint | purpose | auth |
|---|---|---|
| `POST /v1/verify` | verify, check the gallery, decide, enrol | API key |
| `DELETE /v1/duplicate/{reference}` | erase a face from the gallery | API key |
| `GET /health` | liveness | none |
| `GET /ready` | readiness | none |
| `GET /metrics` | Prometheus exposition | none |

`/health` and `/ready` answer different questions on purpose. Liveness returns
200 whenever the process is alive; readiness returns 503 while models load.
Conflating them kills a pod that was merely still warming up, so it never
finishes warming up — and a liveness probe cannot carry an API key anyway.
`/metrics` is unauthenticated for the same class of reason: a scrape must keep
working when key rotation goes wrong, which is exactly when it matters most.

Erasure is a route rather than an operational script because the service stores
biometric templates, and one that cannot delete them on request cannot lawfully
be deployed. It is idempotent — a caller retrying an erasure must not be told
it failed the second time.

## 8. Deployment

`deploy/` carries a two-stage Dockerfile, a compose stack (API, worker, Qdrant,
Redis, RabbitMQ, Prometheus, Grafana) and the monitoring configuration.

The choices that are not obvious:

- **CPU-only torch, explicitly.** The default index pulls the CUDA build:
  ~2 GB of GPU runtime a CPU deployment never executes.
- **`--workers 1`, scale with replicas.** Each uvicorn worker loads its own copy
  of every model, so four workers cost four times the memory for models already
  releasing the GIL inside ONNX Runtime.
- **Weights on a volume, not in the image.** They change on a different cadence
  from the code, and baking them in makes every rollback expensive.
- **`tmpfs` for scratch space.** An identity document written to a container's
  writable layer outlives the request that created it, which the retention
  policy forbids.
- **Qdrant and Redis publish no ports.** The gallery holds biometric templates.
- **Non-root, uid 10001, `no-new-privileges`.**
- **Secrets via `${VAR:?}`,** so compose fails with a named variable rather than
  starting a service that silently accepts every request.

Metrics are aggregate and non-identifying: no label carries a user reference, a
CNIC number, a filename or an image hash, and every label draws from a small
fixed set so Prometheus' series count stays bounded. Nine alert rules cover
availability, latency and **decision quality** — a surge in manual review or a
collapse in approvals usually means an input source changed, not that fraud
rose, and that is invisible to an availability check.
