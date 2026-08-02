# A flux view with flux-secretary already in it.
ARG view=ghcr.io/converged-computing/flux-view-ubuntu:tag-noble
FROM ${view}
ARG spec="[all]"
COPY . /opt/flux-secretary-src

# spaaaack (noises of anguish)
RUN . /opt/spack-environment/activate.sh && \
    python3 -m ensurepip --default-pip 2>/dev/null || true && \
    . /opt/spack-environment/activate.sh && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir "/opt/flux-secretary-src${spec}" && \
    python3 -c "import flux, fluxsecretary, fluxsecretary.cli; print('view python:', __import__('sys').version); print('flux-secretary:', fluxsecretary.__file__)" && \
    python3 -c "import strands; print('backend:', strands.__file__)" && \
    rm -rf /opt/flux-secretary-src
