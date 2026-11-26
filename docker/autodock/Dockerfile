# AutoDock Vina, AutoDock4 (autodock + autogrid), Meeko preparation and RDKit.
# Everything comes from conda-forge and bioconda, so nothing needs to be supplied by hand.
# Cite: Eberhardt et al. 2021 J Chem Inf Model 61:3891 (Vina 1.2);
#       Morris et al. 2009 J Comput Chem 30:2785 (AutoDock4).
FROM mambaorg/micromamba:1.5.10
LABEL org.opencontainers.image.source="https://github.com/openmols/core" \
      org.opencontainers.image.description="AutoDock Vina, AutoDock4, Meeko, RDKit and Open Babel for openmols" \
      org.opencontainers.image.licenses="Apache-2.0 AND GPL-2.0 AND LGPL-2.1 AND BSD-3-Clause"
USER root
RUN micromamba install -y -n base -c conda-forge -c bioconda python=3.11 vina autodock autogrid meeko rdkit openbabel numpy scipy \
    && micromamba clean -a -y
ENV PATH="/opt/conda/bin:${PATH}"
WORKDIR /work
