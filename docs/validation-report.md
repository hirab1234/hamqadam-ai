# Validation Report

Every figure below was produced by running this system on this machine
(12 logical cores, 8 GB RAM, Windows, CPU-only ONNX Runtime). Nothing is
estimated unless labelled as such.

---

## 1. Authentication

### Why you still saw 401

The fix from the previous round was correct and present in the code. Your 401
came from **a stale server process**: when I tried to bind port 8000 the OS
refused with `WinError 10048 — only one usage of each socket address`, meaning
something was already listening there. That process was started before the auth
fix, so it served the old OpenAPI document and the old handler.

Stop every running uvicorn and start one fresh. There is nothing to change in
the code.

### The exact key

```
X-API-Key: dev-key-123
```

Loaded from `configs/app.development.yaml`, which merges over `app.yaml` only
when `app.environment == "development"`. **No environment variable is needed.**
Verified:

```
header: X-API-Key | keys: ['dev-key-123'] | required: True
```

### Verified on a live server

| check | result |
|---|---|
| `components.securitySchemes.ApiKeyAuth` | `{"type":"apiKey","in":"header","name":"X-API-Key"}` |
| `/v1/verify` security requirement | `[{"ApiKeyAuth": []}]` |
| Swagger lock icons | all three protected routes flip `unlocked` → `locked` |
| Generated cURL | `-H 'X-API-Key: dev-key-123'` present |
| No header | **401** |
| Wrong key | **401** |
| Correct key | **200** |

The `-H` line was read out of Swagger's own DOM after clicking Execute, not
inferred.

---

## 2. `secondary_images`

`array<(string | string)>` was a regression I introduced. To let Swagger's
"Send empty value" checkbox work I had widened the annotation to
`UploadFile | str`, and Swagger renders a union as a **text box** — trading the
upload control for the ability to submit.

Both are now fixed together. The route parses the multipart form itself and the
schema is declared explicitly via `openapi_extra`, so the fields are honest
`format: binary`:

```json
"secondary_images": {"type":"array","items":{"type":"string","format":"binary"}}
```

Measured in the live Swagger UI: **4 file inputs → 6 after two clicks** of
"Add item". Each secondary image is processed as its own pipeline stage:

```
secondary_analyses returned: 2
stages named secondary     : ['secondary[0]', 'secondary[1]']
```

---

## 3. End-to-end scenarios

All five over real HTTP against the running server.

| | scenario | verdict | identity | fraud | decisive signals |
|---|---|---|---|---|---|
| **A** | correct: selfie + profile + CNIC + 2 secondary | **APPROVE** | 94.3 | 10.0 LOW | — |
| **B** | wrong person's selfie vs Alice's CNIC | **MANUAL_REVIEW** | 43.8 | 55.9 MED | `CNIC_FACE_MISMATCH` (w 0.75) |
| **C** | invalid CNIC (noise) | **MANUAL_REVIEW** | 100.0 | 61.0 MED | `OCR_NOT_A_CNIC`, `CNIC_NOT_RECOGNISED`, `CNIC_FACE_NOT_FOUND` |
| **D** | blurry selfie | **MANUAL_REVIEW** | 67.1 | 15.0 LOW | `LOW_IMAGE_QUALITY`, quality `usable=False, limiting=blur` |
| **E** | bystander in frame (34% face area) | **APPROVE** | 92.7 | 10.0 LOW | warning `BACKGROUND_FACES_IGNORED` |
| **E2** | **two equal-sized faces** | **MANUAL_REVIEW** | none | 28.0 | `MULTIPLE_FACES_DETECTED` |

Scenario A extracted the full card: `AYESHA KHAN`, father `MUHAMMAD KHAN`,
`42101-8375926-4`, gender `F`, all three dates, province `Sindh` derived from
the ID prefix, gender cross-check passed.

**On E and E2.** My first multi-face fixture had a second face at 34% of the
first — one point under the `bystander_area_ratio` of 0.35 — so it was correctly
treated as a bystander and warned about. That did not test the dangerous case,
so I built one with a measured **0.99 area ratio**. That is E2, and it is
correctly refused. The distinction matters: ignoring a small bystander is right,
while silently picking one of two equal faces would be arbitrary.

---

## 4. Every module proven executing

Captured from a single verification at DEBUG. Actual log lines:

| Module | Log evidence |
|---|---|
| Face detection | `detection.completed detector=scrfd count=1 passed=True visibility=88.89` ×4 images |
| **Face alignment** | 5 landmarks `[(224,191),(309,189),(269,236),(234,278),(303,276)]` → `aligned=True`, `alignment_residual=0.04188` (normal band 0.034–0.061) |
| Face recognition/embedding | `embedder.arcface.ready dimension=512 input_size=[112,112]`, `embedding.batch_completed forward_passes=1 succeeded=1` |
| Image quality | `quality.completed blur=99.98 sharpness=99.87 overall=99.77 usable=True` ×4 |
| CNIC OCR | `ocr.completed engine=onnx_ppocr confidence=76.57 completeness=1.0 present=[6 fields] is_cnic=True` |
| CNIC portrait extraction | `cnic_face.portrait_ready found=True quality=66.93 foreign_faces=0 scale=1.186` |
| Face matching | `matching.completed cnic=87.64 profile=100.0 compared=3 strong=3 identity_confidence=94.24` |
| Duplicate detection | `duplicate.completed searched=True gallery_size=0 duplicate=False` |
| Fraud detection | `fraud.service_ready low_max=30 medium_max=65`; `fraud_risk=10.0` |
| Recommendation engine | `verification.completed recommendation=APPROVE automated=True` |
| Confidence calculation | `identity_confidence=94.24 assessment_confidence=1.0` |

Also visible: the detector fallback chain resolving —
`detector.chain_ready active=scrfd depth=3 unavailable=['yolo']`.

---

## 5. Models

| Stage | Model | Version | Source | Local | Auto-download | Net after setup |
|---|---|---|---|---|---|---|
| Face detection | SCRFD-10G-BNKPS `det_10g.onnx` (16.9 MB) | `scrfd_10g_bnkps-insightface-0.7` | InsightFace `buffalo_l` | ✅ | ✅ via `download_models.py` | ❌ |
| Face embedding | ArcFace R50 `w600k_r50.onnx` (174.4 MB) | `arcface_r100_glint360k-buffalo_l-0.7` | InsightFace `buffalo_l` | ✅ | ✅ | ❌ |
| CNIC OCR | PP-OCR det+cls+rec | `rapidocr-1.2.3` | RapidOCR (bundled) | ✅ | ships in the wheel | ❌ |
| Fallback detector | ResNet-10 SSD (10.7 MB) | `res10_300x300_ssd_iter_140000` | OpenCV 3rdparty | ✅ | ✅ | ❌ |
| Fallback detector | Haar cascade | `opencv-haarcascade-frontalface-default` | OpenCV (vendored) | ✅ | in-repo | ❌ |
| Configured, **not installed** | YOLOv8n-face | — | Ultralytics | — | — | — |

Licences: InsightFace **code** MIT, but the **`buffalo_l` weights are
non-commercial research only**. PP-OCR and OpenCV artefacts are Apache-2.0.
Details and remediation in [licensing.md](licensing.md) §3.

Matching thresholds were measured against *these* weights — swapping the
embedder requires re-deriving them and re-embedding the gallery.

---

## 6. No paid APIs — proven three ways

**1. SDK search.** Searched all source for `openai`, `anthropic`,
`google.cloud`, `googleapiclient`, `vision`, `gemini`, `azure`,
`cognitiveservices`, `boto3`, `rekognition`, `faceplusplus`, `clarifai`,
`deepface`, `replicate`, `huggingface_hub`, `transformers`. **Zero matches.**

**2. Outbound-call search.** `urlopen`, `urlretrieve`, `requests.get/post`,
`httpx`, `aiohttp`, `socket.connect` across `src/`: **zero matches.** The only
remote URL literal anywhere in `src/` is the InsightFace GitHub release URL,
used once at setup by `scripts/download_models.py`.

**3. Air-gap test — the decisive one.** I monkey-patched `socket.connect` to
raise on any non-loopback address, then ran a full verification:

```
recommendation   : APPROVE
identity         : 93.05
ocr name/number  : AYESHA KHAN / 42101-8375926-4
stages failed    : none
outbound attempts: ZERO - nothing tried to leave the machine
```

| Service | Used? |
|---|---|
| OpenAI, Anthropic, Google Vision, Gemini, Azure Face, AWS Rekognition, Face++, any SaaS AI | **No — none** |

**Internet is not required at inference.** It is required exactly once, to
download weights.

---

## 7. Performance

Measured across the whole process tree (uvicorn spawns a worker; measuring the
launcher alone reports a meaningless 4 MB).

| Metric | Value |
|---|---|
| Cold start to ready | **4.4 s** |
| RSS after load, idle | **510 MB** |
| RSS peak under load | **513 MB** |
| Threads | 69 |
| Full verification (n=5) | min 4.7 s / median 5.3 s / max 5.9 s |
| Selfie only | **0.58 s** |
| CPU, one verification | **941% of one core** (~9.4 cores) |
| 4 concurrent | 13.4 s wall, 0.30 verif/s, speed-up only **1.58×** |

**The governing fact:** a single verification already consumes ~9.4 cores, so
the service is CPU-saturated by one request. Concurrency buys little — 4× the
load returned 1.58× the throughput. Memory is not the constraint; CPU is.

Cost per verification ≈ **30 core-seconds** (from the concurrent run, the more
realistic figure).

### Capacity estimates — extrapolated, not measured

This machine has 12 cores. Scaling *down* to 4 and 8 vCPU is arithmetic on the
core-seconds figure, so treat it as an estimate and load-test before committing.

| | 4 vCPU / 8 GB | 8 vCPU / 16 GB |
|---|---|---|
| Latency, single request | ~10–15 s | ~6–8 s |
| Comfortable concurrency | 1–2 | 2–4 |
| Throughput @ 70% CPU | ~0.09 verif/s (~330/hour) | ~0.19 verif/s (~670/hour) |
| Daily, 10 busy hours | **~3,000** | **~6,500** |
| Daily, flat 24 h | ~8,000 | ~16,000 |
| Workers per box | 1 (510 MB each) | 1–2 |

RAM allows more workers than CPU does; do not add them. Scale horizontally with
replicas behind a queue.

**The CNIC read dominates** — 4–7 s of the ~5 s median. Verifications without a
card run in **0.58 s**, an order of magnitude faster. If throughput becomes the
constraint, OCR is the only thing worth optimising.

---

## 8. Bugs found and fixed this round

### 1. `secondary_images` had no file picker *(my regression)*
Widening to `UploadFile | str` made Swagger render a text box. Fixed with manual
form parsing plus an explicit `openapi_extra` schema.

### 2. Two-face selfie crashed the selfie stage
With two equally-sized faces the policy sets `passed=False` and selects **no
primary face**, but the code guarded on `face_detected` — which is `True`.
Alignment then received nothing and raised a bare
`ValueError: Alignment needs either landmarks or a bounding box`, which
`except HamqadamError` did not catch. Consequences: the stage died,
`selfie_detection` came back `null`, and the reviewer was told **"No usable live
selfie was supplied"** — false, and unactionable. Fixed by guarding on
`primary_face` and catching unexpected errors without discarding the detection.

### 3. `MULTIPLE_FACES_DETECTED` never reached the fraud engine
A rejected detection reports through `error_code`, not `warnings`, and the
collector only read warnings. A catalogued, weighted fraud signal was being
silently dropped. Now collected — fraud on a two-face selfie rose 10.0 → 28.0.

### 4. My own first fix caused a regression, caught by re-running
Guarding on `passed` was too strict: a badly-posed *single* face also fails the
policy. That turned scenario B from a correct "identity 43.8,
`CNIC_FACE_MISMATCH`" into "no identity established" — the system stopped
catching the impostor for the stated reason. Corrected to guard on
`primary_face`, and B is restored to 43.8.

### Files modified

| File | Change |
|---|---|
| `src/hamqadam_ai/api/routes.py` | Manual multipart parsing; `VERIFY_FORM_SCHEMA` with `format: binary` |
| `src/hamqadam_ai/pipelines/verification.py` | `_embed_or_none` guard; detection `error_code` → fraud collector |
| `docs/validation-report.md` | This document |

Earlier in the session: `api/security.py`, `api/app.py`,
`configs/app.development.yaml`, `models/registry.py`, `core/config.py`,
`core/exceptions.py`, `deploy/*`, `docs/licensing.md`, plus tests.

---

## 9. Verdict

**Fully functional — yes, demonstrated.** Server starts in 4.4 s; all six
endpoints return correct codes; authentication works from Swagger and curl;
multi-file upload works; all eleven AI modules execute with log proof; the
recommendation moves correctly across five adversarial scenarios;
**1562 tests pass**, ruff and mypy clean over 117 source files.

**Self-hosted only, no paid AI APIs — yes, proven** by SDK search, outbound-call
search, and an air-gap run with zero outbound attempts.

**Production-ready — with three conditions:**

1. **InsightFace weights are non-commercial.** A licence from
   `recognition-oss-pack@insightface.ai`, or swap the models. This is a
   commercial blocker, not a technical one.
2. **Switch the duplicate gallery to Qdrant.** The service warns about this
   itself at startup: `duplicate.using_in_process_gallery — the gallery is lost
   on restart and not shared between replicas`. `deploy/docker-compose.yml`
   already configures it; the default is in-memory.
3. **Re-derive the thresholds** against your labelled data. Every response says
   `thresholds_validated: false` because they are ArcFace-literature defaults.

**Enterprise-ready — not yet.** Missing: SSO/RBAC beyond a shared API key; a
distributed rate limiter (the current one is per-replica, so N replicas allow N×
the rate); an immutable audit trail; and the Docker image has never been built
here, so deployment is verified statically only.

### Known issues

- **YOLOv8-face sits in the fallback chain and is AGPL-3.0.** Not installed, so
  it never loads — but remove the entry rather than leave the trap.
- **`requirements/ml.txt` over-declares.** torch, paddlepaddle, paddleocr,
  easyocr and insightface have zero or near-zero import sites; the image carries
  ~3–4 GB it never executes.
- **One flaky test**, seen once, never reproduced across six subsequent full
  runs, still unidentified.
- **Order-independence untested** — `pytest-randomly` is not installed, so every
  run has used deterministic file order.
- **Scenario C reports identity 100.0** because its selfie and profile are the
  same image, so the comparison is a self-match. Correct arithmetic, misleading
  at a glance; a real submission would not do this.
