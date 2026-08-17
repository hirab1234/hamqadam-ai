# Hamqadam AI Identity Verification Service.
#
# Two stages. The builder compiles wheels; the runtime carries only the
# installed environment and the source. It keeps the deployed image free of
# compilers and headers, which is both smaller and a smaller attack surface -
# a build toolchain inside a running container is a gift to anyone who gets a
# shell in it.

# --------------------------------------------------------------------------- #
# Stage 1 - build
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential for any package without a manylinux wheel; libgl1 and
# libglib2.0-0 are OpenCV's runtime shared libraries. `import cv2` fails with
# a confusing ImportError about libGL if they are missing, which is one of the
# most commonly hit problems in containerised computer vision.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Dependency files first, source second. Docker caches per layer, so this
# ordering means editing a Python file does not reinstall PyTorch.
COPY pyproject.toml README.md ./
COPY requirements/ requirements/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch, explicitly and *first*. The default index pulls the CUDA
# build: roughly 2 GB of GPU runtime a CPU deployment will never execute.
#
# The versions are pinned to exactly what requirements/ml.txt asks for, which is
# what makes the ordering work: PEP 440 treats `torch==2.5.1` as satisfied by
# `2.5.1+cpu`, so the second command sees the requirement already met and leaves
# the CPU wheel alone. Pinning loosely here would let pip decide the pinned
# version was unsatisfied and quietly reinstall the CUDA build over it.
RUN pip install --upgrade pip setuptools wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1

# ml.txt, not base.txt. base.txt covers the API and the ONNX runtime but none of
# the backends the compose stack configures - Qdrant for the gallery, Redis for
# the cache, aio-pika for the worker, PaddleOCR for the CNIC read. An image
# built from base.txt starts, serves /health, and then fails every verification
# that touches a card and kills the worker container on startup.
RUN pip install -r requirements/ml.txt

# --------------------------------------------------------------------------- #
# Stage 2 - runtime
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="Hamqadam AI Identity Verification" \
      org.opencontainers.image.description="Face, document and fraud analysis" \
      org.opencontainers.image.licenses="Proprietary"

# The package is imported from /app/src rather than installed into
# site-packages, and that is load-bearing rather than lazy. `core.config`
# resolves the configuration directory as `Path(__file__).parents[3]/configs`,
# which is /app/configs only while the source sits at /app/src.
#
# An editable install would be worse still: `pip install -e .` in the builder
# writes a pointer to /build/src, a path that does not exist in this stage, so
# every import would fail at runtime having succeeded at build time.
#
# HQ_CONFIG_DIR is set anyway, so the container does not silently depend on that
# path arithmetic surviving a future directory move.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    HQ_APP__ENVIRONMENT=production \
    HQ_CONFIG_DIR=/app/configs \
    HQ_STORAGE__MODEL_DIR=/models \
    OMP_NUM_THREADS=4

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root. The service handles identity documents; a container escape from a
# root process is a much larger incident than one from an unprivileged user.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin hamqadam

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=hamqadam:hamqadam src/ src/
COPY --chown=hamqadam:hamqadam configs/ configs/
COPY --chown=hamqadam:hamqadam pyproject.toml README.md ./

# Operational scripts. `download_models.py` is the one that matters: the compose
# stack runs it as a one-shot `model-init` service to populate the shared
# `models` volume, and without it here that service dies with
# "can't open file '/app/scripts/download_models.py'" - after which api and
# worker never start, because they gate on it completing.
#
# Copied after the dependency layers so editing a script does not invalidate the
# pip cache and force a rebuild of PyTorch.
COPY --chown=hamqadam:hamqadam scripts/ scripts/

# The API reference the /api-docs route renders. Without it that route is the
# only thing in the image that works in development and 500s in production -
# the same shape of defect as the missing scripts/ directory above, which killed
# model-init, and the undeclared OCR engine, which silently swapped the model.
COPY --chown=hamqadam:hamqadam docs/ docs/

# Model weights live on a mounted volume, not in the image. Several hundred
# megabytes of ONNX baked into every layer would make each deploy slow and
# every rollback expensive, and the weights change on a different cadence from
# the code.
RUN mkdir -p /models /tmp/hamqadam \
    && chown -R hamqadam:hamqadam /models /tmp/hamqadam

USER hamqadam

EXPOSE 8000

# Liveness, not readiness. Readiness is /ready, and the orchestrator should
# poll that separately - see the note in api/app.py about why conflating the
# two produces crash loops.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health || exit 1

CMD ["uvicorn", "hamqadam_ai.api.app:create_app", \
     "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "75"]

# --workers 1 is deliberate. Each worker loads its own copy of every model, so
# four uvicorn workers means four times the memory for models that are already
# releasing the GIL inside ONNX Runtime. Scale with replicas, not with workers.
