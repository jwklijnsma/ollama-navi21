#!/usr/bin/env bash
# Build Ollama with the Navi21 / Radeon Pro V620 (gfx1030) llama.cpp fixes.
#
#   ./build.sh tarball   -> dist/bin/ollama + dist/lib/ollama   (for an LXC / bare metal, needs Docker on the build host)
#   ./build.sh image     -> docker image ollama-navi21:<ref>
#   ./build.sh native    -> build/ + ./ollama in the checkout     (needs ROCm 7.x, Go, CMake >= 3.24, Ninja on this host)
#
# Environment:
#   OLLAMA_REF  Ollama tag/branch to build      (default v0.40.16)
#   GFX         HIP targets, ';'-separated     (default gfx1030)
#   WORK        working directory              (default ./work)
set -euo pipefail

MODE="${1:-tarball}"
OLLAMA_REF="${OLLAMA_REF:-v0.40.16}"
GFX="${GFX:-gfx1030}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/work}"
SRC="$WORK/ollama-$OLLAMA_REF"

log() { printf '\n==> %s\n' "$*"; }

# 1. Fresh Ollama checkout at the requested ref
if [[ ! -d "$SRC/.git" ]]; then
    log "Cloning ollama $OLLAMA_REF"
    git -c advice.detachedHead=false clone -q --depth 1 --branch "$OLLAMA_REF" https://github.com/ollama/ollama "$SRC"
fi
log "Ollama $OLLAMA_REF pins llama.cpp $(cat "$SRC/LLAMA_CPP_VERSION")"

# 2. Drop our patch next to Ollama's own compat patch. Ollama's patch applier
#    (cmake/apply-git-patches.cmake) applies every *.patch in llama/compat in
#    filename order, so 100-* runs after 001-llama-cpp-hooks.patch.
cp "$HERE"/patches/*.patch "$SRC/llama/compat/"

# 3. Check the patch still fits this llama.cpp version before spending an hour compiling
log "Checking that the patch applies to llama.cpp $(cat "$SRC/LLAMA_CPP_VERSION")"
CHECK="$WORK/llama-check-$(cat "$SRC/LLAMA_CPP_VERSION")"
if [[ ! -d "$CHECK/.git" ]]; then
    git -c advice.detachedHead=false clone -q --depth 1 --branch "$(cat "$SRC/LLAMA_CPP_VERSION")" https://github.com/ggml-org/llama.cpp "$CHECK"
fi
git -C "$CHECK" checkout -q -- .
git -C "$CHECK" apply "$SRC/llama/compat/001-llama-cpp-hooks.patch"
for p in "$HERE"/patches/*.patch; do
    git -C "$CHECK" apply --check "$p" || { echo "!! $(basename "$p") does not apply to this llama.cpp version" >&2; exit 1; }
done
git -C "$CHECK" checkout -q -- .
echo "patches apply cleanly"

# 4. Only build for our GPU: the stock preset compiles 13 AMD targets
sed -i -E "s/(\"AMDGPU_TARGETS\": )\"gfx908[^\"]*\"/\1\"${GFX}\"/" "$SRC/llama/server/CMakePresets.json"
grep -q "\"AMDGPU_TARGETS\": \"${GFX}\"" "$SRC/llama/server/CMakePresets.json" \
    || { echo "!! could not set AMDGPU_TARGETS in CMakePresets.json" >&2; exit 1; }

cd "$SRC"
case "$MODE" in
    tarball)
        log "Docker build (FLAVOR=rocm, $GFX) -> $HERE/dist"
        rm -rf "$HERE/dist"
        DOCKER_BUILDKIT=1 docker build \
            --build-arg FLAVOR=rocm \
            --target image-archive \
            --output "type=local,dest=$HERE/dist" .
        log "Done: $HERE/dist/bin/ollama and $HERE/dist/lib/ollama"
        ;;
    image)
        log "Docker build (FLAVOR=rocm, $GFX) -> ollama-navi21:$OLLAMA_REF"
        DOCKER_BUILDKIT=1 docker build \
            --build-arg FLAVOR=rocm \
            -t "ollama-navi21:$OLLAMA_REF" .
        log "Done: docker run -d --device /dev/kfd --device /dev/dri -p 11434:11434 -v ollama:/root/.ollama ollama-navi21:$OLLAMA_REF"
        ;;
    native)
        log "Native build (rocm_v7_2, $GFX)"
        cmake -B build . -G Ninja \
            -DOLLAMA_LLAMA_BACKENDS=rocm_v7_2 \
            -DCMAKE_HIP_ARCHITECTURES="$GFX" \
            -DAMDGPU_TARGETS="$GFX"
        cmake --build build --parallel "$(nproc)"
        log "Done: $SRC/ollama serve   (or: cmake --install build --prefix /usr/local)"
        ;;
    *)
        echo "usage: $0 [tarball|image|native]" >&2; exit 2 ;;
esac
