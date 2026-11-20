#!/usr/bin/env bash
# Build a vLLM Docker image for AMD gfx1030 (Radeon Pro V620 / RX 6800-6900).
#
# Upstream vLLM images (rocm/vllm-dev, vllm/vllm-openai-rocm) ship no gfx1030
# kernels and fail with "HIP error: invalid device function". This builds
# vLLM's own ROCm Dockerfiles from vllm-project/vllm PR #52391
# ("enable gfx1030 (RDNA2) as a recognized RDNA arch"), with PyTorch/Triton
# compiled for gfx1030.
#
#   ./build.sh            -> image vllm-navi21:<commit>  (+ :latest)
#
# Environment:
#   VLLM_REF    commit/ref to build  (default: PR #52391 head, pinned)
#   ARCHS       PYTORCH_ROCM_ARCH for the base image (default "gfx942;gfx1030", see below)
#   MAX_JOBS    parallel jobs for the vLLM stage (default: nproc; the PyTorch stage uses 16)
#   WORK        working directory    (default ./work)
#
# Takes several hours the first time: the base stage compiles PyTorch, Triton,
# flash-attention and AITER from source. Needs ~150 GB free disk and >= 32 GB RAM.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/work}"
# PR #52391 head as of 2026-09-23. Override with VLLM_REF=main once it is merged.
VLLM_REF="${VLLM_REF:-d48e89dc1f6561cb92c856aa32be4673bae2df91}"
# gfx942 is only there so the flash-attention (CK) stage has a target: that stage
# strips ";gfx1xxx" from the list, and CK flash-attention can't build for RDNA2.
ARCHS="${ARCHS:-gfx942;gfx1030}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

log() { printf '\n==> %s\n' "$*"; }

mkdir -p "$WORK"
if [[ ! -d "$WORK/vllm/.git" ]]; then
    log "Cloning vllm"
    git clone --filter=blob:none https://github.com/vllm-project/vllm "$WORK/vllm"
fi
cd "$WORK/vllm"
git fetch -q origin "$VLLM_REF" 2>/dev/null || git fetch -q origin "refs/pull/52391/head"
git -c advice.detachedHead=false checkout -q FETCH_HEAD
SHA="$(git rev-parse --short=9 HEAD)"

grep -q "on_gfx10x" vllm/platforms/rocm.py \
    || { echo "!! $SHA has no gfx1030 support (on_gfx10x missing in vllm/platforms/rocm.py)" >&2; exit 1; }
log "vLLM $SHA ($(git log -1 --format='%ad %s' --date=short))"

log "Stage 1/2: ROCm base (PyTorch, Triton, FA, AITER) for $ARCHS"
DOCKER_BUILDKIT=1 docker build \
    -f docker/Dockerfile.rocm_base \
    --build-arg "PYTORCH_ROCM_ARCH=$ARCHS" \
    -t "vllm-rocm-base:navi21-$SHA" .

log "Stage 2/2: vLLM for gfx1030"
DOCKER_BUILDKIT=1 docker build \
    -f docker/Dockerfile.rocm \
    --build-arg "BASE_IMAGE=vllm-rocm-base:navi21-$SHA" \
    --build-arg "ARG_PYTORCH_ROCM_ARCH=gfx1030" \
    --build-arg "max_jobs=$MAX_JOBS" \
    --build-arg REMOTE_VLLM=0 \
    --target vllm-openai \
    -t "vllm-navi21:$SHA" -t vllm-navi21:latest .

log "Done: vllm-navi21:$SHA  — start with: docker compose -f $HERE/docker-compose.yml up -d"
