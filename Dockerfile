FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps: Rust build tooling, HPC libs, Python 3.10, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    pkg-config \
    build-essential \
    cmake \
    clang \
    lld \
    gfortran \
    libssl-dev \
    openmpi-bin \
    libopenmpi-dev \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    libfftw3-dev \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    patchelf \
    libfontconfig1-dev \
    libfreetype6-dev \
    gmsh \
    libgl1 \
    libglu1-mesa \
    libx11-6 \
    libxext6 \
    libxi6 \
    libxrender1 \
    libxfixes3 \
    libxrandr2 \
    libxcursor1 \
    libxinerama1 \
    libsm6 \
    libice6 \
 && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"
ENV WORKSPACE=/workspace
ENV DEPS_DIR=/deps
ENV LOCAL_EXAFMM_DIR=/workspace/external/exafmm-t
ENV LOCAL_BEMPP_DIR=/workspace/external/bempp-cl
ENV LOCAL_KIFMM_DIR=/workspace/external/kifmm-for-rsrs
ENV BEMPP_KIFMM_ROOT=/workspace/external/kifmm-for-rsrs

WORKDIR /workspace

# Copy the project into the image so plain `docker build` / `docker run`
# also has access to the checked-out exafmm-t, bempp-cl, and kifmm-for-rsrs submodules.
COPY . /workspace

RUN install -m 0755 /workspace/scripts/setup_deps.sh /usr/local/bin/setup_deps.sh

# Run setup on container start, then exec the CMD (bash by default)
ENTRYPOINT ["/usr/local/bin/setup_deps.sh"]
CMD ["bash"]
