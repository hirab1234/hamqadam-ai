# Running on a VPS with persistent Qdrant

No managed or paid service is involved. Qdrant runs as your own container and
stores vectors on your own disk.

---

## 1. Requirements

| | Minimum | Comfortable |
|---|---|---|
| vCPU | 4 | 8 |
| RAM | 8 GB | 16 GB |
| Disk | 20 GB | 40 GB |

Measured: the API holds ~510 MB resident and one verification consumes ~9.4
cores for ~5 s, so **CPU is the binding constraint, not memory**. Four
concurrent verifications returned only 1.58× the throughput of one — the box is
already saturated by a single request.

Qdrant's own footprint is small: a 512-dimension float32 vector is 2 KB, so
100,000 enrolled faces is roughly 200 MB plus index overhead.

---

## 2. Bring it up

```bash
git clone <your-repo> && cd hamqadam-ai
export HQ_API_KEYS="$(openssl rand -hex 24)"
export RABBITMQ_PASSWORD="$(openssl rand -hex 16)"
export GRAFANA_PASSWORD="$(openssl rand -hex 16)"
docker compose -f deploy/docker-compose.yml up -d
```

Print your key once and store it in your Backend's secret manager:

```bash
echo "$HQ_API_KEYS"
```

First start downloads ~200 MB of model weights, so allow a few minutes before
`/ready` turns green. Watch it:

```bash
docker compose -f deploy/docker-compose.yml logs -f api
```

---

## 3. Confirm Qdrant is actually the store

The commonest silent failure is the service falling back to the in-memory
gallery. It says so at startup — check for it:

```bash
docker compose -f deploy/docker-compose.yml logs api | grep duplicate
```

**Correct:**

```
duplicate.qdrant_collection_created
duplicate.qdrant_store_ready
duplicate.service_ready   store=qdrant
```

**Wrong — stop and fix the config:**

```
duplicate.using_in_process_gallery
  note: the gallery is lost on restart and not shared between replicas
```

Every verification response also carries the answer:

```json
"duplicate": { "store": "qdrant", "gallery_size": 1 }
```

If that says `"store": "memory"`, duplicate detection is not durable.

---

## 4. Prove persistence before you trust it

This is the test that matters, and it takes a minute:

```bash
# enrol a face
curl -X POST http://localhost:8000/v1/verify \
  -H "X-API-Key: $HQ_API_KEYS" \
  -F verification_id=persist-1 -F user_reference=acct-A \
  -F enrol_on_success=true \
  -F live_selfie=@face.jpg -F profile_image=@face.jpg -F cnic_image=@card.jpg
```

```bash
docker compose -f deploy/docker-compose.yml restart api
```

```bash
# same face, different account - must report a duplicate
curl -X POST http://localhost:8000/v1/verify \
  -H "X-API-Key: $HQ_API_KEYS" \
  -F verification_id=persist-2 -F user_reference=acct-B \
  -F live_selfie=@face.jpg -F profile_image=@face.jpg -F cnic_image=@card.jpg
```

Expect `"duplicate_found": true` and `"gallery_size": 1`. If you get `false`
and `0`, the gallery did not survive — check §3.

I ran exactly this sequence against an on-disk Qdrant: enrolled under `acct-A`,
restarted, and the second account came back **REJECT** with
`duplicate_found: true`, `best_similarity: 1.0`, matched against `acct-A`.

---

## 5. Back it up

The named volume holds biometric templates. Losing it means losing every
enrolment; leaking it is a data-protection incident.

```bash
docker run --rm \
  -v hamqadam-ai_qdrant:/data:ro \
  -v "$PWD/backups:/backup" \
  alpine tar czf /backup/qdrant-$(date +%F).tar.gz -C /data .
```

Encrypt the archive at rest and keep it inside the same jurisdiction as the
verification data.

---

## 6. Two policy switches

Both default to the conservative option. Neither is a technical choice.

```bash
# on_approve       only an APPROVE enrols (default)
# unless_rejected  APPROVE or MANUAL_REVIEW enrol
export HQ_ENROL_POLICY=on_approve

# manual_review    a duplicate goes to a human (default)
# reject           a duplicate is refused outright
export HQ_ON_DUPLICATE=manual_review
```

**On `enrol_policy`.** `on_approve` leaves a real hole: an applicant sent to
MANUAL_REVIEW is never enrolled, so when they open a second account there is
nothing to collide with — the multi-account case the gallery exists to catch is
the one it misses. `unless_rejected` closes it, at the cost of storing a
template for someone not yet approved. Whether that is lawful depends on a
retention basis I cannot assess for you.

**On `on_duplicate`.** `reject` is decisive and does not depend on the fraud
score. `manual_review` is safer while your threshold is uncalibrated — identical
twins and siblings do score highly.

---

## 7. Hardening

The image refuses to start in production unless it is hardened — no API keys,
debug on, wildcard CORS, redaction off, or checksum verification off all fail
at boot rather than serving traffic.

Beyond that:

- **Terminate TLS in front of the API.** Nothing here serves HTTPS; put nginx
  or Caddy ahead of it. Identity documents must not cross a network in clear.
- **Only the API port should be reachable.** Qdrant, Redis and RabbitMQ publish
  no ports in the compose file; Prometheus and Grafana bind to `127.0.0.1`
  only. Keep it that way and reach the dashboards over an SSH tunnel.
- **The rate limiter is per-replica.** N replicas allow N× the configured rate.
  With one API container this is exact; if you scale out, enforce the real limit
  at your reverse proxy.
- **Scale with replicas, not `--workers`.** Each worker loads its own copy of
  every model — four workers means four times the memory for models that
  already release the GIL inside ONNX Runtime.

---

## 8. Before you go live

Two things this repository cannot do for you:

1. **License the InsightFace weights, or replace them.** The SCRFD detector and
   ArcFace embedder come from the `buffalo_l` pack, whose weights are
   *non-commercial research only*. See [licensing.md](licensing.md) §3.
2. **Calibrate the thresholds.** Every response carries
   `thresholds_validated: false`. Run `scripts/evaluate_matching.py` against
   labelled pairs from your own users, and re-run
   `scripts/calibrate_duplicate_threshold.py` as the gallery grows — the
   per-comparison false-match rate compounds with gallery size.
