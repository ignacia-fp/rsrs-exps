#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="rsrs-bempp-python-dev"

docker build \
    --platform linux/amd64 \
    --tag "${image}" \
    --file "${repo_root}/docker/bempp-rs-python/Dockerfile" \
    "${repo_root}"

docker run --rm -it \
    --platform linux/amd64 \
    --volume "${repo_root}:/workspace" \
    --workdir /workspace \
    "${image}"
