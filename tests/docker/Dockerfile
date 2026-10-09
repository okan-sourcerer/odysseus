# Test runner image: the app image plus what the full suite needs beyond it,
# mirroring the setup steps in .github/workflows/ci.yml.
#
#   docker build -f tests/docker/Dockerfile -t odysseus-test:dev .
#   docker run --rm --cap-add SYS_ADMIN --security-opt seccomp=unconfined \
#     --security-opt apparmor=unconfined -v "$PWD:/src" -w /src odysseus-test:dev \
#     python -m pytest -q -p no:cacheprovider -n auto --max-worker-restart=0 tests
#
# The source tree is bind-mounted at run time, so the image only needs
# rebuilding when dependencies change. SYS_ADMIN and the seccomp/apparmor
# options let bubblewrap create the namespaces the containment tests need.
ARG BASE_IMAGE=odysseus-local:dev
FROM ${BASE_IMAGE}

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg bubblewrap \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-dev.txt requirements.txt /tmp/req/
RUN pip install --no-cache-dir "$(grep -E '^pytest-xdist' /tmp/req/requirements-dev.txt)"

# Node packages live at /node_modules: ES-module resolution walks up from the
# importing file, so tests under the bind-mounted /src still find them.
COPY package.json package-lock.json /
RUN cd / && npm ci --no-audit --no-fund \
 && npx playwright install --with-deps chromium
