# pyKVFinder (GPL-3.0) cavity detection, run by openmols as an external process.
FROM python:3.11-slim
RUN pip install --no-cache-dir "pyKVFinder==0.9.*" \
    && pyKVFinder --version
WORKDIR /work
