# Dependency and Licensing Report

Audited by inspecting the installed environment, the source imports, the model
artefacts actually on disk, and every URL literal in the codebase. Package
versions and licences below were read from installed package metadata, not from
`requirements/*.txt` — the two differ, and §5 explains how.

---

## 1. The three direct answers

**Is any paid API used anywhere?**
**No.** There is no billing relationship anywhere in this codebase. No API keys
to a vendor, no SaaS client library, no metered service.

**Is any cloud AI service called?**
**No.** A search of the source for outbound HTTP, and for every known cloud-AI
SDK (`openai`, `anthropic`, `boto3`, `google.cloud`, `azure`, Rekognition),
returns nothing. All inference is local ONNX Runtime on CPU.

**Does inference need an internet connection?**
**No.** Every URL in the source is either `localhost`, an internal compose
service (`http://qdrant:6333`), or a model download URL used **once at setup**
by `scripts/download_models.py`. Once `model_store/` is populated the service
runs fully air-gapped. The container even ships weights on a mounted volume
rather than fetching them at boot.

> **But there is a licensing problem that costs money.** See §3. It is not an
> API charge, and it is material.

---

## 2. What actually runs

Verified against `model_store/` and the live process, not against configuration.

| Artefact | Role | Source | Licence | Commercial use |
|---|---|---|---|---|
| `buffalo_l/det_10g.onnx` (SCRFD) | Primary face detection | InsightFace | Model: **non-commercial research only** | ❌ **Not without a licence** |
| `buffalo_l/w600k_r50.onnx` (ArcFace R50) | Face embeddings — the core of matching | InsightFace | Model: **non-commercial research only** | ❌ **Not without a licence** |
| `opencv/res10_300x300_ssd` | Fallback detector | OpenCV 3rdparty | Apache-2.0 (BSD-3 pre-4.5) | ✅ Yes |
| Haar cascades | Last-resort detector | OpenCV | Apache-2.0 | ✅ Yes |
| PP-OCR v3/v4 ONNX (bundled in `rapidocr-onnxruntime`) | CNIC text extraction | PaddleOCR → RapidOCR | Apache-2.0 | ✅ Yes |
| `yolo/yolov8n-face.onnx` | Configured fallback, **not installed** | Ultralytics YOLOv8 | **AGPL-3.0** | ⚠️ See §4 |

---

## 3. The finding that matters: InsightFace

The primary detector and the **only** embedding model both come from
InsightFace's `buffalo_l` pack. Quoting the project's own README, retrieved and
verified during this audit:

> "The code of InsightFace is released under the MIT License. There is no
> limitation for both academic and commercial usage."
>
> "The training data containing the annotation (and the models trained with
> these data) are available for non-commercial research purposes only."

The **code** is MIT and unrestricted. The **pretrained weights are not.** The
README further directs commercial users of the open-sourced recognition models
to `recognition-oss-pack@insightface.ai` for a licence.

**What this means for you.** This is not a paid *API* — nothing phones home and
nothing is metered — but deploying these weights commercially requires a
licence agreement with InsightFace. So the answer "no paid AI APIs" is true and
also incomplete on its own: there is a commercial licensing obligation attached
to the two models the system depends on most.

I flag this rather than bury it because face recognition for paying customers is
squarely commercial use, and it is the primary detector *and* the embedding
model — not a peripheral component.

### Your options

1. **Licence it.** Contact `recognition-oss-pack@insightface.ai`. Least
   engineering work; the models stay as they are and the measured accuracy in
   the module docs still holds.
2. **Swap the embedding model** for a permissively-licensed ArcFace — for
   example a model trained on MS1M or Glint360K released under Apache-2.0/MIT.
   The architecture already supports this: `configs/models.yaml` selects the
   artefact, `HQ_EMBEDDING__MODEL` overrides it, and Module 3's adapter takes
   any 512-dimensional ONNX recogniser. **But** every threshold in
   `configs/thresholds.yaml` was measured against *this* model, so they must be
   re-derived with `scripts/evaluate_matching.py`, and the entire enrolled
   gallery must be re-embedded — vectors from a different model occupy a
   different space and cannot be compared.
3. **Swap the detector too**, if you want to drop InsightFace entirely. Lower
   risk than (2): detection feeds alignment, not the vector space, so
   thresholds are far less sensitive. The OpenCV SSD fallback is already wired
   and permissively licensed, at some accuracy cost.

Option 1 is the pragmatic choice unless the licence fee is prohibitive. Option 2
is the one that removes the dependency, and it is a re-calibration project, not
a config change.

---

## 4. YOLOv8-face and AGPL-3.0

`configs/models.yaml` lists `yolo/yolov8n-face.onnx` in the fallback chain. **It
is not on disk and therefore never loads** — the active chain resolves SCRFD →
OpenCV SSD → Haar.

If anyone does install it, Ultralytics YOLOv8 is **AGPL-3.0**. For a network
service that is the strongest copyleft in common use: it can be read to require
publishing the source of the whole service to its users. Either buy an
Ultralytics commercial licence or remove the entry from the chain. Given it is
unused, **removing it is the safer default.**

---

## 5. Runtime dependencies

Every one is permissively licensed and free. Versions read from the installed
environment.

| Package | Version | Purpose | Hosting | Cost | Licence | Commercial |
|---|---|---|---|---|---|---|
| `onnxruntime` | 1.28.0 | All model inference | Self-hosted | Free | MIT | ✅ |
| `opencv-python-headless` | 4.13.0.92 | Image processing, Haar/SSD detectors | Self-hosted | Free | Apache-2.0 | ✅ |
| `rapidocr-onnxruntime` | 1.2.3 | CNIC OCR (PP-OCR on ONNX) | Self-hosted | Free | Apache-2.0 | ✅ |
| `numpy` | 2.5.1 | Numerics | — | Free | BSD-3-Clause | ✅ |
| `scipy` | 1.18.0 | Signal/geometry maths | — | Free | BSD-3-Clause | ✅ |
| `scikit-image` | 0.26.0 | Image metrics | — | Free | BSD-3-Clause | ✅ |
| `pillow` | 12.3.0 | Image decoding | — | Free | MIT-CMU | ✅ |
| `shapely` / `pyclipper` | 2.1.2 / 1.4.0 | OCR polygon geometry | — | Free | BSD-3 / BSD | ✅ |
| `fastapi` | 0.116.2 | HTTP API | Self-hosted | Free | MIT | ✅ |
| `uvicorn` | 0.35.0 | ASGI server | Self-hosted | Free | BSD-3-Clause | ✅ |
| `pydantic` / `pydantic-settings` | 2.13.4 / 2.14.2 | Schemas, config | — | Free | MIT | ✅ |
| `python-multipart` | 0.0.32 | Upload parsing | — | Free | Apache-2.0 | ✅ |
| `orjson` | 3.11.9 | JSON | — | Free | Apache-2.0 / MIT | ✅ |
| `structlog` | 26.1.0 | Logging + PII redaction | — | Free | Apache-2.0 / MIT | ✅ |
| `prometheus-client` | 0.26.0 | Metrics | Self-hosted | Free | Apache-2.0 | ✅ |
| `qdrant-client` | 1.18.0 | Vector gallery client | Self-hosted | Free | Apache-2.0 | ✅ |
| `httpx` / `anyio` | 0.28.1 / 4.14.2 | Async plumbing | — | Free | BSD-3 / MIT | ✅ |
| `tenacity` | 9.1.4 | Retries | — | Free | Apache-2.0 | ✅ |
| `PyYAML` | 6.0.3 | Config parsing | — | Free | MIT | ✅ |

### Optional, and not currently installed

| Package | Purpose | Licence | Commercial |
|---|---|---|---|
| `redis` | Shared embedding cache | MIT | ✅ |
| `aio-pika` | Queue worker | Apache-2.0 | ✅ |
| `qdrant` (server) | Persistent gallery | Apache-2.0 | ✅ |
| `rabbitmq` (server) | Broker | MPL-2.0 | ✅ |
| `prometheus` / `grafana` | Monitoring | Apache-2.0 / AGPL-3.0¹ | ✅ |

¹ Grafana is AGPL-3.0 but is a **separate, unmodified process** you deploy
alongside — it does not link into your code and imposes nothing on it. Standard
practice; only relevant if you modify and redistribute Grafana itself.

### Declared but unused — an image-size problem

`requirements/ml.txt` pins **`torch`, `torchvision`, `paddlepaddle`,
`paddleocr`, `easyocr` and `insightface`**, and `deploy/Dockerfile` installs
that file. Checked against the source: **zero import sites** for `torch`,
`torchvision`, `paddlepaddle`, `insightface` and `mlflow`. `paddleocr` and
`easyocr` appear once each, inside optional fallback adapters that never run
because RapidOCR is the default and succeeds.

The container therefore carries roughly **3–4 GB** of unused frameworks. All are
permissively licensed (BSD-3 / Apache-2.0), so this is a size and attack-surface
issue rather than a legal one — but it is worth trimming, and doing so also
removes the `insightface` *package* from the image (though not the weights
question in §3, which is about the artefacts, not the library).

---

## 6. Data handling

Relevant to due diligence, and unchanged by anything above:

- **No image leaves the process.** Uploads are decoded in memory and dropped
  when the request ends; container scratch space is `tmpfs`.
- **Nothing is sent to any third party.** No telemetry, no analytics, no
  crash reporting.
- **No image is written to disk** at any point in the request path.
- **User CNIC images, profile images, selfies and verification data must not be
  used for personal, commercial, research or model-training purposes** without
  written authorisation from the Client.

---

## 7. Verification method and its limits

Package licences were read from installed metadata. The InsightFace terms in §3
were retrieved from the project README during this audit and are quoted
verbatim. Model licences for OpenCV and PP-OCR artefacts are stated from their
upstream projects' terms.

Licence terms change, and this is a summary rather than legal advice. Before
commercial launch, have counsel confirm §3 in particular — it is the one item
that could require either a payment or an engineering change.
