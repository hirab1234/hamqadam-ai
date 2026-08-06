# Backend Integration Guide

For the team building the Backend API. Everything here is verbatim from a
running service; nothing is illustrative.

The machine-readable contract is [`openapi.json`](openapi.json), regenerated with:

```bash
python scripts/export_openapi.py -o docs/openapi.json
```

---

## 1. Where this service sits

```
Flutter App  →  Backend API  →  AI Verification Service  →  Backend  →  Flutter
```

The AI service **never** talks to Flutter and never owns the verification
record. It receives images, returns scores and a *recommendation*, and forgets
everything. You own the decision, the database and the audit trail.

The recommendation is advice. You are expected to override it with business
rules this service knows nothing about — a manual allow-list, a regulatory
hold, a VIP fast-track.

---

## 2. The one call you need

One request verifies, checks the duplicate gallery, decides, and enrols.

```bash
curl -X POST https://ai.internal/v1/verify \
  -H "X-API-Key: $HQ_API_KEY" \
  -F "verification_id=ver_01HQ8XZ" \
  -F "user_reference=acct_88213" \
  -F "live_selfie=@selfie.jpg" \
  -F "profile_image=@profile.jpg" \
  -F "cnic_image=@cnic_front.jpg" \
  -F "secondary_images=@extra1.jpg" \
  -F "secondary_images=@extra2.jpg"
```

Multipart, not base64 — seven photographs base64-encode to roughly a third more
bytes than they need.

| field | required | notes |
|---|---|---|
| `verification_id` | yes | Yours. Echoed back, and used as the log correlation id. |
| `user_reference` | no | Your account identifier. Pseudonymised before it reaches any log, and **never echoed in the response**. |
| `enrol_on_success` | no | Override only. **Omit it** and `duplicate.enrol_policy` decides — see §5. `false` suppresses enrolment for this call. |
| `live_selfie` | no* | The biometric reference for the whole decision. |
| `profile_image` | no | The account's main photograph. |
| `cnic_image` | no | Front of the card. |
| `secondary_images` | no | Repeat the field for each. |

\* Every image is optional at the transport layer, deliberately: a missing one
is a *finding* the service reports rather than a request it refuses, because you
usually still want the analysis of whatever did arrive. A request with **no**
images at all is refused with `VALIDATION_ERROR`.

Expect **7–20 seconds**. Set your timeout accordingly; 30 s is comfortable.

---

## 3. What comes back

Abridged from an actual approval:

```json
{
  "verification_id": "demo-coherent",
  "recommendation": "APPROVE",
  "automated": true,
  "identity_confidence_score": 92.88,
  "fraud_risk_score": 10.0,
  "fraud_risk_level": "LOW",
  "assessment_confidence": 1.0,
  "complete": true,
  "requires_human_review": false,
  "thresholds_validated": false,
  "recommendation_reasons": [
    {
      "code": "IDENTITY_CONFIDENCE_SUFFICIENT",
      "message": "Identity confidence of 92.9 meets the automatic-approval threshold of 75.0.",
      "satisfied": true,
      "detail": { "identity_confidence": 92.88, "threshold": 75.0 }
    }
  ],
  "stages": [
    { "stage": "selfie", "ran": true, "succeeded": true, "duration_ms": 1865.86, "error": null }
  ],
  "warnings": [
    { "code": "OCR_CARD_NOT_ISOLATED", "stage": "ocr" },
    { "code": "MATCHING_THRESHOLDS_UNVALIDATED", "stage": "matching" }
  ],
  "processing_time": { "total": 6834.37, "unit": "ms" }
}
```

### The five fields to build on

| field | type | meaning |
|---|---|---|
| `recommendation` | `APPROVE` \| `REJECT` \| `MANUAL_REVIEW` | The advice. |
| `identity_confidence_score` | `number \| null` | 0–100. **Null is not zero** — see below. |
| `fraud_risk_score` | `number` | 0–100. |
| `assessment_confidence` | `number` | 0–1. How much of the intended evidence actually existed. |
| `recommendation_reasons` | `array` | Why, in a form you can show a reviewer. |

### Three things that will bite you if you skip them

**`identity_confidence_score` can be `null`, and null is not zero.** A
verification with no usable face has *no* identity confidence. Coercing null to
0.0 rejects the applicant for impersonation when the real finding is that their
photograph was unusable — a different answer, and one they can fix. Handle the
null branch explicitly.

**`assessment_confidence` is the field that stops a thin submission passing.** A
request supplying only a selfie once scored identity 100 / fraud 0 and was
approved: nothing contradicted the applicant because almost nothing had been
checked. "No evidence against" is not "evidence for". If you re-derive the
decision yourself, apply an evidence floor too — this service refuses to approve
below 0.70.

**`thresholds_validated: false` is not a bug, and you should surface it.** The
match thresholds are engineering defaults from the ArcFace literature, not
values fitted to your data. Until `scripts/evaluate_matching.py` is run against
a labelled corpus, treat the absolute scores as ordinal rather than calibrated.

### Reading `stages`

Three states from two booleans:

| `ran` | `succeeded` | meaning |
|---|---|---|
| true | true | ran, produced a result |
| true | false | ran, failed — `error` says why |
| false | false | skipped: no input for it, or the time budget ran out |

A stage that failed is not the same as a stage with nothing to find. `"no face
detected"` is a *successful* stage with a negative finding.

### Reading `warnings`

Advisory, and worth logging. Notable codes:

| code | meaning |
|---|---|
| `VERIFICATION_STAGE_FAILED` | A stage failed; its evidence is missing. |
| `VERIFICATION_BUDGET_EXHAUSTED` | The request ran out of time. The recommendation is from partial evidence. |
| `MATCHING_THRESHOLDS_UNVALIDATED` | See above. Present on every response until calibration. |
| `OCR_CARD_NOT_ISOLATED` | No card-shaped quadrilateral found; the whole frame was read. Fields may be missed. |

---

## 4. Errors

Every error has the same shape. There is never a stack trace in a response.

```json
{
  "error": {
    "code": "UNSUPPORTED_IMAGE_FORMAT",
    "message": "…",
    "details": {},
    "retryable": false
  }
}
```

| status | code | retryable | what to do |
|---|---|---|---|
| 400 | `VALIDATION_ERROR` | no | Malformed request, or no images at all. |
| 401 | `UNAUTHORIZED` | no | Fix the key. |
| 403 | `FORBIDDEN` | no | HMAC signature absent, stale or wrong. |
| 413 | `IMAGE_TOO_LARGE` | no | Downscale before sending. |
| 413 | `PAYLOAD_TOO_LARGE` | no | The whole request body is too big. |
| 413 | `DECOMPRESSION_BOMB` | no | Declared dimensions exceed the decode ceiling. |
| 415 | `UNSUPPORTED_IMAGE_FORMAT` | no | Not a format we decode. Re-encode as JPEG or PNG. |
| 422 | `IMAGE_DECODE_FAILED` | no | Corrupt, or a single flat colour with no detail. |
| 422 | `IMAGE_TOO_SMALL` | no | Shorter side below 64 px. |
| 429 | `RATE_LIMITED` | **yes** | Honour `Retry-After`. |
| 500 | `AI_SERVICE_ERROR` | **yes** | Retry with backoff; quote the request id. |
| 500 | `CONFIGURATION_ERROR` | no | Our misconfiguration. Do not retry; page us. |
| 500 | `INFERENCE_FAILED` | **yes** | Transient model failure. |
| 503 | `MODEL_NOT_LOADED` | **yes** | Still starting. Retry with backoff. |
| 503 | `DEPENDENCY_UNAVAILABLE` | **yes** | Qdrant, Redis or the broker is down. |
| 503 | `VECTOR_DB_ERROR` | **yes** | Gallery unreachable. |
| 504 | `PROCESSING_TIMEOUT` | **yes** | Exceeded the server-side budget. |

Read the `retryable` field rather than hard-coding this list. The status codes
alone will mislead you: **two different 500s disagree** — `AI_SERVICE_ERROR` is
worth retrying and `CONFIGURATION_ERROR` never is, and a client retrying the
latter just burns its budget against a fault only we can fix. Likewise not every
503 is retryable: `MODEL_CHECKSUM_MISMATCH` means an artefact failed integrity
verification, which will not resolve on its own.

Note the 422 family. Codes like `FACE_NOT_DETECTED` appear there because
enrolment genuinely cannot proceed without a face — but the **same
finding on `/v1/verify` returns 200**, as part of the analysis. Same code, two
meanings, depending on whether it prevented the operation or merely described
its outcome.

**Business outcomes are never errors.** A rejection, an unreadable card, a
duplicate face — all return **200** with a recommendation. A non-2xx means the
service could not analyse the request, not that the applicant failed.

### Correlation

Send `X-Request-ID`; it is echoed on the response and appears on every log line
the request produces. If you do not send one, a UUID is generated and returned.
`X-Response-Time-Ms` is on every response.

---

## 5. The duplicate gallery

**One call does everything.** `/v1/verify` verifies, searches the gallery,
decides, and enrols. There is no separate enrolment request — `POST
/v1/duplicate/enrol` was removed, because a Backend could verify without ever
calling it, leaving every duplicate search to run against an empty gallery and
find nothing. That failure was silent.

Enrolment is governed by `duplicate.enrol_policy`:

| policy | enrols on |
|---|---|
| `never` | nothing |
| `on_approve` | APPROVE only — **default** |
| `unless_rejected` | APPROVE or MANUAL_REVIEW |

`on_approve` leaves a real hole: an applicant sent to MANUAL_REVIEW is never
enrolled, so their second account has nothing to collide with — the
multi-account case the gallery exists to catch is the one it misses.
`unless_rejected` closes it, at the cost of storing a template for someone not
yet approved. That is a retention decision, not an engineering one.

**A REJECT is never enrolled, under any policy or override.** Storing a refused
applicant's template would make it collide with their next legitimate attempt —
the service would manufacture a duplicate out of its own earlier refusal.

You may override per request with `enrol_on_success`:

- **omit it** (normal) — policy decides
- `false` — suppress enrolment for this submission
- `true` — force it even where policy is `never`

Read the outcome from the response:

```json
"duplicate": {
  "duplicate_found": true,
  "best_similarity": 1.0,
  "gallery_size": 2,
  "store": "qdrant",
  "candidates": [{"reference": "acct-A", "...": "..."}]
}
```

If `store` says `"memory"`, the gallery is **not durable** — it is lost on
restart and not shared between replicas. Production must report `"qdrant"`.

### Erasure stays a route

```bash
curl -X DELETE https://ai.internal/v1/duplicate/acct_88213   -H "X-API-Key: $HQ_API_KEY"
```

Unchanged, and still idempotent: erasing an absent reference returns 200 with
`removed: false, erased: true`. A caller retrying must not be told it failed the
second time — that is how erasure requests get abandoned half-done. **You are
responsible for calling this when a user exercises a deletion right**; the AI
service cannot know they asked.

## 6. Operations

| endpoint | use |
|---|---|
| `GET /health` | Liveness. 200 whenever the process is alive. No key. |
| `GET /ready` | Readiness. 503 while models load. No key. |
| `GET /metrics` | Prometheus. No key — expose only inside the cluster. |

Point your liveness probe at `/health` and your readiness probe at `/ready`.
Pointing liveness at `/ready` kills pods that are merely still warming up, so
they never finish warming up. First start takes a few minutes.

---

## 7. Asynchronous alternative

For bulk work, publish to `hamqadam.verification.requests` and consume from
`hamqadam.verification.results`. Same pipeline, same response body.

```json
{
  "verification_id": "ver_01HQ8XZ",
  "user_reference": "acct_88213",
  "enrol_on_success": true,   // optional; omit to follow enrol_policy
  "live_selfie": "<base64>",
  "profile_image": "<base64>",
  "cnic_image": "<base64>",
  "secondary_images": ["<base64>"]
}
```

The broker guarantees at-least-once, so **expect a duplicate delivery
occasionally**. Verification is stateless and idempotent; enrolment replaces
rather than appends, so a redelivery is harmless. Make your result handler
idempotent on `verification_id` and you are covered.

A message that fails three times is rejected and a failure published:

```json
{ "verification_id": "ver_01HQ8XZ", "status": "FAILED", "reason": "…" }
```

Handle that — otherwise you wait forever for a result that will never come.

---

## 8. Your obligations

- **Send images over HTTPS only.** These are identity documents.
- **Do not log the response body wholesale.** It contains OCR'd CNIC fields.
- **Call the erasure route** when a user exercises a deletion right.
- **No user CNIC image, profile image, selfie or verification datum may be used
  for personal, commercial, research or model-training purposes** without
  written authorisation from the Client. This service stores no image; what you
  store is your responsibility.
- **Rotate API keys** by setting several — `HQ_API_KEYS=old,new` — so both work
  during the overlap and callers can move across without a coordinated cutover.
