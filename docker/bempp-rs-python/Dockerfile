FROM --platform=linux/amd64 ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG RUST_TOOLCHAIN=1.99.1
ARG RUST_NIGHTLY=nightly-2026-10-15

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        git \
        libclang-dev \
        libffi-dev \
        libfftw3-dev \
        liblapack-dev \
        libopenblas-dev \
        libopenmpi-dev \
        libssl-dev \
        ninja-build \
        openmpi-bin \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
    && /root/.cargo/bin/rustup toolchain install "${RUST_NIGHTLY}" --profile minimal

ENV PATH="/root/.cargo/bin:/opt/rsrs-venv/bin:${PATH}"
ENV VIRTUAL_ENV=/opt/rsrs-venv
ENV CARGO_TARGET_DIR=/opt/cargo-target
ENV BEMPP_CFFI_TOOLCHAIN=nightly-2026-10-15

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && pip install --no-cache-dir --upgrade \
        pip \
        maturin \
        numpy \
        pytest \
        scipy

WORKDIR /workspace

CMD ["bash"]
