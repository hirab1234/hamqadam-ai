# Hamqadam AI Verification API v1

Curated API documentation for the Backend that calls the AI Verification Service.

This service analyses images and returns a **recommendation**. It never owns the
verification record and never decides a user's account status — the Backend does.
Admin endpoints are development-only and refused in production.

```
Flutter App  →  Backend API  →  AI Verification Service  →  Backend  →  Flutter
                                    (this service)
```

Every example below uses synthetic data. No real identity document appears in
this documentation.

**Base URL:**

```
{{AI_SERVICE_URL}}
```

**Assets:**

```
Browser docs:   {{AI_SERVICE_URL}}/api-docs
Markdown docs:  {{AI_SERVICE_URL}}/api-docs.md
OpenAPI JSON:   {{AI_SERVICE_URL}}/openapi.json
Swagger UI:     {{AI_SERVICE_URL}}/docs        (development only)
```

**Authentication header:**

```
X-API-Key: {{api_key}}
Accept: application/json
```

Every endpoint requires it except `/health`, `/ready`, `/metrics`, `/api-docs`
and `/api-docs.md`. A missing or wrong key returns `401 UNAUTHORIZED`.

Several keys can be active at once (comma-separated in `HQ_SECURITY__API_KEYS`),
so an old and a new key overlap during rotation. Never hard-code a key in the
Flutter app — only the Backend holds it.

---

## Backend Endpoint Checklist

Use this checklist for Backend integration QA. These are the only endpoints the
Backend needs:

| Group | Required Endpoints |
|---|---|
| Verification | `/v1/verify` |
| Duplicate gallery | `/v1/duplicate/{reference}` |
| Operations | `/health`, `/ready`, `/metrics` |
| Documentation | `/api-docs`, `/api-docs.md`, `/openapi.json` |
| Admin (dev only) | `/admin/qdrant/count`, `/admin/qdrant/list`, `/admin/qdrant/{reference}` |

**Standard success shape.** There is no wrapper envelope — the verification
result is the response body:

```json
{
  "verification_id": "ver_01HQ8XZ",
  "recommendation": "APPROVE",
  "identity_confidence_score": 93.06,
  "fraud_risk_score": 10.0,
  "requires_human_review": false
}
```

**Standard error envelope:**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "A human-readable explanation.",
    "details": {},
    "retryable": false
  }
}
```

`retryable` tells you whether repeating the identical request could succeed.

**A negative result is not an error.** A face that does not match is
`200 OK` with `recommendation: "REJECT"`. Errors mean the question could not be
answered at all.

---

## Limits

| Setting | Value |
|---|---|
| Request timeout | 45 s |
| Max request size | 40 MB |
| Rate limit, general | 600 requests/minute |
| Rate limit, `/v1/verify` | 120 requests/minute |
| Burst | 60 |
| Typical `/v1/verify` latency | 7–20 s (the CNIC read dominates) |

Exceeding a limit returns `429 RATE_LIMITED`. **Call `/v1/verify`
asynchronously** — queue the work and notify the app when the answer arrives.
Never block a user-facing request on it.

If you put nginx in front, raise `client_max_body_size` to 40M and
`proxy_read_timeout` to 90s, or uploads are rejected at 1 MB and verifications
are cut off at 60 s.

---

## Verification

| Feature | Method | Endpoint | Payload |
|---|---|---|---|
| Run a full identity verification | `POST` | `/v1/verify` | `multipart/form-data` — `verification_id` (required), `live_selfie`, `profile_image`, `cnic_image`, `secondary_images[]`, `user_reference`, `enrol_on_success` |

### Payload fields

| Field | Type | Required | Description |
|---|---|---|---|
| `verification_id` | string | **Yes** | Your identifier for this attempt. Echoed back. |
| `live_selfie` | file | No\* | The live capture. The biometric reference everything else is compared against. |
| `profile_image` | file | No\* | The account's main photograph. |
| `cnic_image` | file | No\* | Front of the CNIC. |
| `secondary_images` | file[] | No | Additional photographs of the same person. Repeat the field once per file. |
| `user_reference` | string | No | Your stable account identifier. **Required for duplicate detection** — without it nothing is enrolled and nothing can ever collide. Never echoed back. |
| `enrol_on_success` | boolean | No | Overrides policy for this one call. Omit to use the configured policy. `false` suppresses enrolment; `true` forces it. A `REJECT` never enrols under any value. |

\* At least one image is required, but **send all three**. A missing mandatory
image prevents automatic approval — see *Why a verification is not approved*.

Accepted formats: JPEG, PNG, WEBP, BMP.

### Request

```bash
curl -X POST {{AI_SERVICE_URL}}/v1/verify \
  -H "X-API-Key: $HQ_API_KEY" \
  -F "verification_id=ver_01HQ8XZ" \
  -F "user_reference=acct_84213" \
  -F "live_selfie=@selfie.jpg;type=image/jpeg" \
  -F "profile_image=@profile.jpg;type=image/jpeg" \
  -F "cnic_image=@cnic-front.jpg;type=image/jpeg" \
  -F "secondary_images=@extra1.jpg;type=image/jpeg" \
  -F "secondary_images=@extra2.jpg;type=image/jpeg"
```

### Sample response — `200 OK`

Abridged. The full body also carries per-image detection, quality and
authenticity detail; the fields below are the ones a Backend acts on.

```json
{
  "verification_id": "ver_01HQ8XZ",
  "completed_at": "2026-01-14T09:22:11.482Z",
  "recommendation": "APPROVE",
  "automated": true,
  "identity_confidence_score": 93.06,
  "fraud_risk_score": 10.0,
  "fraud_risk_level": "LOW",
  "requires_human_review": false,
  "complete": true,
  "assessment_confidence": 1.0,
  "thresholds_validated": false,

  "recommendation_reasons": [
    {
      "code": "EVIDENCE_SUFFICIENT",
      "message": "100% of the intended checks ran, meeting the 70% required.",
      "satisfied": true,
      "detail": { "assessment_confidence": 1.0, "threshold": 0.7 }
    },
    {
      "code": "IDENTITY_CONFIDENCE_SUFFICIENT",
      "message": "Identity confidence of 93.1 meets the threshold of 75.0.",
      "satisfied": true,
      "detail": { "identity_confidence": 93.06, "threshold": 75.0 }
    },
    {
      "code": "FRAUD_RISK_ACCEPTABLE",
      "message": "Fraud risk of 10.0 is within the ceiling of 30.0.",
      "satisfied": true,
      "detail": { "fraud_risk": 10.0, "fraud_level": "LOW", "threshold": 30.0 }
    }
  ],

  "matching": {
    "identity_confidence_score": 93.06,
    "profile_face_match_score": 100.0,
    "cnic_face_match_score": 87.67,
    "secondary_face_match_scores": [96.4],
    "identity_available": true,
    "any_comparison_failed": false,
    "capped_by": null,
    "effective_weights": { "cnic": 0.45, "profile": 0.35, "secondary": 0.20 },
    "comparisons": [
      {
        "comparison": "profile",
        "compared": true,
        "decision": "STRONG_MATCH",
        "score": 100.0,
        "similarity": 0.9981,
        "confidence": 1.0,
        "strong_match_threshold": 0.62,
        "review_threshold": 0.45,
        "reason": null
      },
      {
        "comparison": "cnic",
        "compared": true,
        "decision": "STRONG_MATCH",
        "score": 87.67,
        "similarity": 0.71393,
        "confidence": 1.0,
        "strong_match_threshold": 0.42,
        "review_threshold": 0.30,
        "reason": null
      }
    ],
    "model_version": "arcface_r100_glint360k-buffalo_l-0.7",
    "thresholds_validated": false
  },

  "cnic_ocr": {
    "success": true,
    "is_cnic": true,
    "cnic_number": "00000-0000000-0",
    "name": "SPECIMEN NAME",
    "father_name": "SPECIMEN FATHER",
    "gender": "F",
    "date_of_birth": "2000-01-01",
    "issue_date": "2020-01-01",
    "expiry_date": "2030-01-01",
    "is_expired": false,
    "ocr_confidence_score": 88.4,
    "completeness": 1.0,
    "fields_missing": [],
    "engine": "onnx_ppocr",
    "engine_version": "rapidocr-1.2.3"
  },

  "duplicate": {
    "duplicate_found": false,
    "needs_review": false,
    "best_similarity": 0.1013,
    "best_match_score": 8.73,
    "candidates": [],
    "gallery_size": 42,
    "searched": true,
    "store": "qdrant",
    "duplicate_threshold": 0.68,
    "review_threshold": 0.58,
    "recommended_action": "proceed"
  },

  "fraud": {
    "fraud_risk_score": 10.0,
    "fraud_risk_level": "LOW",
    "top_factors": [],
    "assessment_confidence": 1.0,
    "weights_validated": false
  },

  "stages": [
    { "stage": "selfie", "ran": true, "succeeded": true, "duration_ms": 3127.1, "error": null },
    { "stage": "profile", "ran": true, "succeeded": true, "duration_ms": 3549.1, "error": null },
    { "stage": "cnic_ocr", "ran": true, "succeeded": true, "duration_ms": 6746.1, "error": null },
    { "stage": "cnic_portrait", "ran": true, "succeeded": true, "duration_ms": 3645.4, "error": null },
    { "stage": "duplicate", "ran": true, "succeeded": true, "duration_ms": 198.7, "error": null },
    { "stage": "matching", "ran": true, "succeeded": true, "duration_ms": 10.0, "error": null }
  ],

  "warnings": [],
  "processing_time": { "total": 12351.5, "unit": "ms" },
  "model_versions": {
    "versions": {
      "face_detector_scrfd": "scrfd_10g_bnkps-insightface-0.7",
      "face_embedder_arcface": "arcface_r100_glint360k-buffalo_l-0.7"
    },
    "service_version": "1.0.0"
  }
}
```

### Sample response — held for review

```json
{
  "verification_id": "ver_01HQ900",
  "recommendation": "MANUAL_REVIEW",
  "identity_confidence_score": 45.0,
  "fraud_risk_score": 22.0,
  "requires_human_review": true,
  "recommendation_reasons": [
    {
      "code": "SUPPLIED_IMAGE_NOT_COMPARED",
      "message": "A photograph was submitted that no face could be read from, so it was never compared against the live selfie.",
      "satisfied": false,
      "detail": {}
    }
  ],
  "matching": { "capped_by": "profile_failure", "identity_confidence_score": 45.0 }
}
```

### `gallery_size` is measured *before* enrolment

The duplicate search runs before this applicant is added, so a first
verification reports `gallery_size: 0` and then enrols. The count describes the
gallery the search ran against, not the gallery afterwards.

---

## Duplicate gallery

| Feature | Method | Endpoint | Payload |
|---|---|---|---|
| Erase a face from the duplicate gallery | `DELETE` | `/v1/duplicate/{reference}` | None. `reference` is the `user_reference` you enrolled with |

Idempotent — deleting a reference that is not there is a success.

```bash
curl -X DELETE {{AI_SERVICE_URL}}/v1/duplicate/acct_84213 \
  -H "X-API-Key: $HQ_API_KEY"
```

```json
{ "reference": "acct_84213", "removed": true, "erased": true }
```

Use this for erasure requests. A service that stores biometric templates and
cannot delete them on request cannot lawfully be deployed.

---

## Operations

No API key required.

| Feature | Method | Endpoint | Payload |
|---|---|---|---|
| Liveness — is the process alive? | `GET` | `/health` | None |
| Readiness — should traffic be sent here? | `GET` | `/ready` | None |
| Prometheus metrics | `GET` | `/metrics` | None |

Point your orchestrator's **liveness** probe at `/health` and its **readiness**
probe at `/ready`. Using `/health` for both produces a crash loop; using
`/ready` for both restarts a container that is merely still loading models.

```json
{ "status": "alive", "service": "hamqadam-ai-verification", "version": "1.0.0" }
```

`/ready` returns `503` with the reason when the service cannot serve:

```json
{ "status": "not_ready", "reason": "VectorStoreError: could not reach Qdrant" }
```

---

## Admin — development only

Disabled by default and **refused outright in production**. Enable locally with
`HQ_ADMIN__ENABLED=true`, then restart. They never return a vector — a 512-float
template is biometric data.

| Feature | Method | Endpoint | Payload |
|---|---|---|---|
| How many templates the gallery holds | `GET` | `/admin/qdrant/count` | None |
| Enumerate stored references | `GET` | `/admin/qdrant/list` | Query: `limit` (max 100), `offset` |
| Look up one reference | `GET` | `/admin/qdrant/{reference}` | None |
| Erase one reference | `DELETE` | `/admin/qdrant/{reference}` | None |

```json
{
  "store": "qdrant",
  "count": 1,
  "returned": 1,
  "limit": 50,
  "offset": 0,
  "records": [
    {
      "reference": "acct_84213",
      "model_version": "arcface_r100_glint360k-buffalo_l-0.7",
      "enrolled_at": "2026-01-14T09:22:13.104Z",
      "metadata": {}
    }
  ]
}
```

If these return `403 FORBIDDEN`, they are switched off — which is correct in
production.

---

## The three recommendations

| Value | Meaning | What the Backend should do |
|---|---|---|
| `APPROVE` | Every approval condition was satisfied | Approve, or apply your own extra rules |
| `REJECT` | A definite adverse finding | Refuse |
| `MANUAL_REVIEW` | Evidence is missing, degraded, or inconclusive | Queue for a human |

**`MANUAL_REVIEW` is not a soft `APPROVE`.** It is the service saying it cannot
answer safely. Treating it as a pass removes most of the fraud protection.

`recommendation` is advisory. The Backend owns the account outcome.

---

## Why a verification is not approved

`recommendation_reasons[].code` lists the conditions.

### Missing evidence — always `MANUAL_REVIEW`

| Code | Meaning |
|---|---|
| `NO_LIVE_SELFIE` | No usable selfie, so there is nothing to compare against |
| `NO_IDENTITY_COMPARISON` | No face comparison completed |
| `MANDATORY_STAGE_INCOMPLETE` | A required check did not run, or ran and failed |
| `SUPPLIED_IMAGE_NOT_COMPARED` | An image was submitted that no face could be read from. Supplying an unreadable photograph is not better than supplying none |
| `DUPLICATE_FACE_NEEDS_REVIEW` | This face is already enrolled under a different reference |

### Degraded evidence — `MANUAL_REVIEW` unless a rejection rule also fires

| Code | Meaning |
|---|---|
| `SELFIE_NOT_USABLE` | The selfie failed its own checks (pose, quality). Every comparison is measured against it, so its defects lower every score |

### Definite adverse findings — `REJECT`

| Code | Meaning |
|---|---|
| `IDENTITY_CONFIDENCE_TOO_LOW` | At or below 40.0 |
| `FRAUD_RISK_TOO_HIGH` | At or above 65.0 |
| `DUPLICATE_FACE_CONFIRMED` | Duplicate, and policy is to refuse outright |

### Identity confidence caps

When one comparison contradicts the others, identity confidence is **capped at
45.0** rather than averaged — below the 75.0 approval floor.
`matching.capped_by` says which:

| `capped_by` | Trigger |
|---|---|
| `cnic_failure` | The CNIC portrait does not match the selfie |
| `profile_failure` | The profile photograph does not match the selfie |
| `secondary_failure` | A secondary photograph does not match the selfie |

A mismatching image is a contradiction in the submission, not a low score to be
outvoted by the images that do match.

---

## Error codes

| HTTP | Code | Retryable | Cause |
|---|---|---|---|
| 400 | `VALIDATION_ERROR` | No | Malformed request |
| 400 | `INVALID_IMAGE` | No | Not a readable image |
| 400 | `UNSUPPORTED_IMAGE_FORMAT` | No | Format not accepted |
| 400 | `IMAGE_TOO_LARGE` / `IMAGE_TOO_SMALL` | No | Outside accepted dimensions |
| 401 | `UNAUTHORIZED` | No | Missing or wrong `X-API-Key` |
| 403 | `FORBIDDEN` | No | Route disabled (for example admin in production) |
| 413 | `PAYLOAD_TOO_LARGE` | No | Over 40 MB |
| 429 | `RATE_LIMITED` | Yes | Back off and retry |
| 503 | `MODEL_NOT_LOADED` | Yes | Still starting, or a backend was unreachable at startup |
| 503 | `DEPENDENCY_UNAVAILABLE` | Yes | A required backend is down |
| 504 | `PROCESSING_TIMEOUT` | Yes | Exceeded the 45 s budget |

---

## Integration notes

**Call it asynchronously.** A verification takes 7–20 seconds. Queue the work
and notify the app when the answer arrives.

**Always send `user_reference`** if you want duplicate detection. Without it
nothing is enrolled, and every duplicate search runs against a gallery that
never grows.

**Never echo `user_reference` to the client.** The service deliberately does not
return it.

**Store the whole response.** When a decision is questioned months later, the
per-stage scores are the only record of why.

**`thresholds_validated: false` is expected.** Match thresholds are engineering
defaults from the ArcFace literature, not values fitted to this deployment's
data. Re-pin them against a labelled corpus before treating the numbers as
calibrated probabilities.

---

## Privacy

- No image is written to disk. Uploads are decoded in memory and dropped when
  the request ends.
- The `user_reference` is pseudonymised before it reaches any log line, and is
  never echoed in a response.
- Erasure is a first-class route: `DELETE /v1/duplicate/{reference}`.
- User CNIC images, profile images, selfies and verification data must not be
  used for personal, commercial, research or model-training purposes without
  written authorisation from the Client.
