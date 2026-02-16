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

WORKDIR /workspace

# Copy the bootstrap script into the image
COPY scripts/setup_deps.sh /usr/local/bin/setup_deps.sh
RUN chmod +x /usr/local/bin/setup_deps.sh

# Run setup on container start, then exec the CMD (bash by default)
ENTRYPOINT ["/usr/local/bin/setup_deps.sh"]
CMD ["bash"]
