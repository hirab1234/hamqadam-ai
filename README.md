# Hamqadam AI Identity Verification Service

Face, document and fraud analysis for identity verification in Pakistan. Given a
selfie, a profile photograph and a photograph of a CNIC, it returns a
recommendation — `APPROVE`, `REJECT` or `MANUAL_REVIEW` — with the evidence
behind it.

This repository is **the AI service only**. It does not contain the Flutter app
or the Backend business logic, by design:

```
Flutter App  →  Backend API  →  AI Verification Service  →  Backend  →  Flutter
                                    (this repository)
```

The service never talks to Flutter, never owns the verification record, and
never decides a user's account status. It analyses images, returns scores and a
*recommendation*, and forgets everything. The Backend owns the decision.

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate     # POSIX: source .venv/bin/activate
pip install -r requirements/ml.txt
python scripts/download_models.py
```

Then see what it actually does, on synthetic data:

```bash
python scripts/demo_pipeline.py --list
python scripts/demo_pipeline.py --scenario impostor
```

Serve it:

```bash
uvicorn hamqadam_ai.api.app:create_app --factory --port 8000
```

Or bring up the whole stack — API, worker, Qdrant, Redis, RabbitMQ, Prometheus,
Grafana:

```bash
HQ_API_KEYS=your-key RABBITMQ_PASSWORD=… GRAFANA_PASSWORD=… \
  docker compose -f deploy/docker-compose.yml up -d
```

---

## What it does

Ten modules, each independently testable, composed by one pipeline.

| # | Module | Package |
|---|---|---|
| 1 | Face detection (SCRFD, with fallbacks) | `detectors/` |
| 2 | Face and image quality | `quality/` |
| 3 | Face embeddings (ArcFace) | `embeddings/` |
| 4 | Face matching | `matching/` |
| 5 | CNIC OCR | `ocr/` |
| 6 | CNIC portrait localisation and matching | `documents/` |
| 7 | Profile image authenticity | `authenticity/` |
| 8 | Duplicate face detection | `duplicate_detection/` |
| 9 | Fraud risk aggregation | `fraud_detection/` |
| 10 | Decision engine, pipeline, API, workers | `decision/`, `pipelines/`, `api/`, `workers/` |

Measured end-to-end behaviour on synthetic fixtures:

| scenario | verdict | identity | fraud |
|---|---|---|---|
| coherent submission | APPROVE | 92.9 | 10.0 |
| impostor (stranger's card) | REJECT | 45.0 | 72.8 |
| card held up in front of a face | REJECT | 45.0 | 88.4 |
| face already enrolled elsewhere | REJECT | 92.9 | 73.0 |
| unreadable CNIC | MANUAL_REVIEW | 100.0 | 61.0 |
| selfie only | MANUAL_REVIEW | none | 0.0 |

A full submission takes **7–20 s** — the CNIC read dominates. Reproduce the
table with `scripts/demo_pipeline.py`.

---

## Documentation

| | |
|---|---|
| [architecture.md](docs/architecture.md) | Layering, runtime model, cross-cutting decisions |
| [integration.md](docs/integration.md) | **Start here if you are building the Backend** |
| [openapi.json](docs/openapi.json) | Machine-readable contract |
| [docs/modules/](docs/modules/) | One document per module: design, measurements, defects found |

The module documents are worth reading before changing anything. Each records
what was measured, what was refused, and which of my own assumptions the data
disproved — several thresholds exist because a plausible design failed against
real images.

---

## Development

```bash
pip install -r requirements/dev.txt

pytest tests/unit -q                 # fast, no model weights
pytest tests/ -q                     # everything (~8 min, needs weights)
pytest tests/ -m "not performance"   # skip the benchmarks

ruff check src tests scripts
mypy src
```

1544 tests. `ruff` and `mypy --strict` are clean across 117 source files.

Tests are marked `unit`, `integration`, `performance`, `accuracy` and `gpu`.
Anything needing model weights skips cleanly when they are absent rather than
failing.

---

## Configuration

Layered, and nothing is hard-coded:

```
configs/*.yaml  →  configs/app.<env>.yaml  →  .env  →  HQ_-prefixed env vars
```

Every accept/reject threshold lives in [`configs/thresholds.yaml`](configs/thresholds.yaml).
Nested values use `__`:

```bash
HQ_SECURITY__API_KEYS=key-a,key-b
HQ_DECISION__APPROVE__MIN_IDENTITY_CONFIDENCE=80
HQ_DUPLICATE__BACKEND=qdrant
```

Production mode refuses to start if it is not hardened — no API keys, debug
left on, wildcard CORS, redaction disabled, checksum verification off. A
misconfigured deployment fails loudly at startup rather than quietly accepting
every request.

---

## What is not validated

Stated plainly, because a number presented without its provenance invites a
confidence it has not earned. The service reports all of this in its own
responses.

- **Match thresholds are engineering defaults** from the ArcFace literature, not
  values fitted to this deployment's data. Every response carries
  `thresholds_validated: false`. Run `scripts/evaluate_matching.py` against a
  labelled corpus and re-pin them.
- **Fraud weights are reasoned, not fitted** (`weights_validated: false`). They
  encode a considered ordering of severity, not a calibration.
- **The duplicate threshold is calibrated against an assumed gallery size.**
  Re-run `scripts/calibrate_duplicate_threshold.py` as it grows.
- **The decision thresholds are policy, not measurement.** They are where the
  Client's risk appetite belongs, and should be revisited once manual-review
  outcomes exist to tune against.

---

## Privacy and data handling

- **No image is ever written to disk.** Uploads are decoded in memory and
  dropped when the request ends; scratch space is `tmpfs` in the container.
- **PII cannot reach a log sink.** Redaction is the final structlog processor,
  and user references are pseudonymised before they reach a log line.
- **The user reference is never echoed** in a response.
- **Erasure is a first-class route.** `DELETE /v1/duplicate/{reference}`, and it
  is idempotent. A service that stores biometric templates and cannot delete
  them on request cannot lawfully be deployed.
- **User CNIC images, profile images, selfies and verification data must not be
  used for personal, commercial, research or model-training purposes** without
  written authorisation from the Client.
- **No real identity document appears in this repository.** Every fixture is
  synthetic or a print-degraded public-domain reference portrait.
