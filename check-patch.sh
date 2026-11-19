#!/usr/bin/env bash
# Check that patches/*.patch still apply to the llama.cpp pinned by an Ollama ref.
#   ./check-patch.sh                 # latest stable Ollama release
#   ./check-patch.sh main
#   ./check-patch.sh v0.40.17-rc1
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REF="${1:-latest}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [[ "$REF" == latest ]]; then
    REF="$(git ls-remote --tags --refs --sort=-v:refname https://github.com/ollama/ollama 'v*' \
        | awk -F/ '{print $3}' | grep -v -- '-rc' | head -1)"
fi

git -c advice.detachedHead=false clone -q --depth 1 --branch "$REF" https://github.com/ollama/ollama "$WORK/ollama"
LLAMA="$(tr -d '[:space:]' < "$WORK/ollama/LLAMA_CPP_VERSION")"
echo "Ollama $REF -> llama.cpp $LLAMA"

git -c advice.detachedHead=false clone -q --depth 1 --branch "$LLAMA" https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp"
cd "$WORK/llama.cpp"

# Same order as Ollama's cmake/apply-git-patches.cmake: all llama/compat patches by filename.
cp "$HERE"/patches/*.patch "$WORK/ollama/llama/compat/"
status=0
while IFS= read -r p; do
    name="$(basename "$p")"
    if git apply --whitespace=nowarn "$p" 2>"$WORK/err"; then
        echo "  OK    $name"
    else
        echo "  FAIL  $name"; sed 's/^/        /' "$WORK/err"; status=1
    fi
done < <(find "$WORK/ollama/llama/compat" -name '*.patch' | sort)

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    { [[ $status -eq 0 ]] && echo "✅" || echo "❌"; echo " Ollama \`$REF\` (llama.cpp \`$LLAMA\`)"; } | tr -d '\n' >> "$GITHUB_STEP_SUMMARY"
    echo >> "$GITHUB_STEP_SUMMARY"
fi
exit $status
