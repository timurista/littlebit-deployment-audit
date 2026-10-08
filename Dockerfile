# Reproducible CPU-only environment for the LittleBit deployment audit.
#
# Status: built and tested by the project owner. Image
# sha256:e2001739838437f9e33226e856b2dc4caa47aae1269791455b3699773c818588 (arm64,
# evidence/container-image.txt) ran the 95-test suite of that revision offline with result OK
# (evidence/container-tests-quality.log). The tests added later (test_public_release.py,
# test_report_results.py) have not been run in the container yet. The quality pilot itself ran on
# the host, not in this image (the image has no transformers). A separate earlier standard-library
# audit ran in a plain container (evidence/container-audit.log).
#
# Design rules:
#   * CPU only, pinned versions matching evidence/environment-freeze.txt (Python 3.11, torch 2.6.0).
#   * Network is needed only at build time to fetch pinned wheels. Run with --network=none.
#   * No credentials: no tokens, no ssh keys, no Hugging Face cache, no cloud SDKs.
#   * Upstream source is NOT copied into the image; the workspace is bind mounted read-only.
#   * Non-root user, bounded commands via `timeout`.
#
# Pin the base image by digest after the first local pull (docker inspect --format '{{index .RepoDigests 0}}').
ARG PYTHON_IMAGE=python:3.11.15-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Pinned CPU wheels. The CPU index avoids CUDA wheels; numpy comes from PyPI.
# Follow-up: generate a hash-locked requirements file and switch to --require-hashes.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
 && python -m pip install numpy==2.2.4 \
 && python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"

RUN useradd --create-home --uid 10001 audit
USER audit
WORKDIR /work

# Default: bounded unit test run. Writes nothing to the workspace.
CMD ["timeout", "600", "python", "-m", "unittest", "discover", "-s", "tests", "-v"]

# Bounded run commands (host side), workspace read-only, results to a separate writable dir:
#
#   docker build -t littlebit-audit:cpu .
#   docker run --rm --network=none --cpus=1 --memory=1g --pids-limit=256 --read-only \
#     --tmpfs /tmp:rw,size=256m --security-opt=no-new-privileges --cap-drop=ALL \
#     -v "$PWD":/work:ro littlebit-audit:cpu
#   docker run --rm --network=none --cpus=1 --memory=2g --pids-limit=256 --read-only \
#     --tmpfs /tmp:rw,size=256m --security-opt=no-new-privileges --cap-drop=ALL \
#     -v "$PWD":/work:ro -v "$PWD/results-container":/out \
#     littlebit-audit:cpu sh -c 'timeout 900 python src/audit.py --out /out && \
#       timeout 1800 python src/bench.py --out /out --iters 20 --warmup 3 --threads 1 && \
#       timeout 900 python src/patched_regression.py --out /out'
