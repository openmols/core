# pyKVFinder (GPL-3.0) cavity detection, run by openmols as an external process.
FROM python:3.11-slim
LABEL org.opencontainers.image.source="https://github.com/openmols/core" \
      org.opencontainers.image.description="pyKVFinder cavity detection for openmols" \
      org.opencontainers.image.licenses="GPL-3.0"
RUN pip install --no-cache-dir "pyKVFinder==0.9.*" \
    && pyKVFinder --version
WORKDIR /work
