# FRODOCK protein-protein docking (chaconlab/FRODOCK).
#
# Built locally from a copy of the FRODOCK tree supplied as the build context.
# openmols neither ships nor redistributes FRODOCK. The repository carries no
# licence file, so check the terms with its authors before sharing this image
# or using it commercially.
#
# Cite: Garzon et al. 2009 Bioinformatics 25:2544-2551;
#       Ramirez-Aportela et al. 2016 Bioinformatics 32:2386-2388.
FROM debian:bookworm-slim
LABEL org.opencontainers.image.source="https://github.com/chaconlab/FRODOCK" \
      org.opencontainers.image.description="FRODOCK protein-protein docking, packaged for openmols" \
      org.opencontainers.image.licenses="NOASSERTION"

# frodock links against single-precision FFTW (-lfftw3f).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         build-essential gfortran libgomp1 libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/frodock
WORKDIR /opt/frodock

# A Windows checkout leaves CRLF endings, which bash and make both reject.
RUN find . -type f \( -name '*.sh' -o -name 'Makefile*' -o -name 'makefile' -o -name '*.mk' \) -print0 \
    | xargs -0 sed -i 's/\r$//'

# Only frodock_gcc ships prebuilt; the rest are compiled here.
#
# compile_frodock.sh walks every build variant, including the Intel (icpc) and
# MPI (mpiCC) ones, which are not installed and abort the run. Build just the
# GCC targets, in that script's dependency order. Note the upstream quirks:
# libpdb has a second target (Release2_gcc -> libpdb2_gcc.a) that the cluster
# tools link against, and libfrodockcluster's directory is spelled Relase_gcc.
RUN set -e; \
    for part in libtools libpdb libvolume libfrm libbessel libfrodock \
                libfrodockcluster libprotprot libsas \
                frodock frodockgrid frodockonstraint frodockcluster \
                frodockview frodockcheck; do \
      built=0; \
      for variant in Release_gcc Release2_gcc Relase_gcc; do \
        dir="src/${part}/${variant}"; \
        [ -d "$dir" ] || continue; \
        echo "=== building ${part}/${variant}"; \
        if make -C "$dir" -j"$(nproc)" > "/tmp/build-${part}-${variant}.log" 2>&1; then \
          built=1; continue; \
        fi; \
        case "$part" in \
          frodockonstraint|frodockcheck) \
            echo "--- ${part} did not build (optional)"; \
            grep -iE 'error:|Error [0-9]|cannot find' "/tmp/build-${part}-${variant}.log" | head -5 ;; \
          *) \
            echo "--- ${part}/${variant} failed:"; \
            grep -iE 'error:|Error [0-9]|cannot find' "/tmp/build-${part}-${variant}.log" | head -10; \
            exit 1 ;; \
        esac; \
      done; \
      [ "$built" = 1 ] || echo "=== ${part}: nothing built"; \
    done

# The tools are built with a compiler suffix (_gcc); add plain aliases.
RUN cd bin \
    && (chmod +x frodock* || true) \
    && for t in frodock frodockgrid frodockcluster frodockview frodockonstraints frodockcheck; do \
         if [ ! -e "$t" ] && [ -e "${t}_gcc" ]; then ln -sf "${t}_gcc" "$t"; fi; \
       done \
    && ls -l \
    && for t in frodock frodockgrid frodockcluster frodockview; do \
         test -x "$t" || { echo "missing: $t"; exit 1; }; \
       done

ENV PATH="/opt/frodock/bin:${PATH}"
# frodock needs the SOAP potential passed explicitly with -s.
ENV FRODOCK_SOAP="/opt/frodock/bin/soap.bin"
WORKDIR /work
