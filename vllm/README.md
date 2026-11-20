# vLLM voor de Radeon Pro V620 (gfx1030) — met DeepSeek

Naast Ollama: een vLLM-server (OpenAI-compatibele API) die beide V620's met tensor parallelism gebruikt.

## Waarom zelf bouwen

De officiële vLLM-images (`rocm/vllm-dev`, `vllm/vllm-openai-rocm`) bevatten geen gfx1030-kernels en stoppen op een V620 met `HIP error: invalid device function`. vLLM PR [#52391](https://github.com/vllm-project/vllm/pull/52391) voegt gfx1030 toe als RDNA-architectuur. Die PR is nog niet gemerged. `build.sh` bouwt vLLM's eigen ROCm-Dockerfiles vanaf die PR (gepind op `d48e89dc`), met PyTorch en Triton gecompileerd voor gfx1030.

```bash
./build.sh                          # -> vllm-navi21:<commit> + :latest
VLLM_REF=main ./build.sh            # zodra de PR gemerged is
```

De eerste build duurt uren: de base-stage compileert PyTorch, Triton, flash-attention en AITER vanaf broncode. Reken op ~150 GB vrije schijfruimte en minstens 32 GB RAM.

## DeepSeek draaien

```bash
docker compose up -d deepseek-r1
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-r1","messages":[{"role":"user","content":"Hoeveel is 17*23?"}]}'
```

| Service | Model | Status op gfx1030 |
|---|---|---|
| `deepseek-r1` (standaard) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`, fp16, TP=2, fp8 KV, 32k context | Standaard attention (`ROCM_ATTN`), de veilige route. Met `--reasoning-parser` komt het denkwerk in `reasoning_content`. |
| `deepseek-coder-v2-lite` (`--profile mla`) | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`, MoE + MLA, fp16, TP=2 | **Experimenteel.** Het snelle MLA-pad (AITER) werkt alleen op Instinct, dus vLLM valt terug op `TRITON_MLA`. Dat is niet getest op RDNA2 en kan tegen de LDS-limiet van 64 KB aanlopen. |

Draai één service tegelijk, want elke service gebruikt beide GPU's.

## Instellingen voor gfx1030

- **`VLLM_USE_BREAKABLE_CUDAGRAPH=1`**: torch.compile blijft hangen op gfx1030. Hangt het daarna nog steeds, voeg dan `--enforce-eager` toe.
- **`--attention-backend=ROCM_ATTN`**: `TRITON_ATTN` gaat over de LDS-limiet per workgroup van RDNA2.
- **`--dtype=float16`**: RDNA2 heeft geen snelle bf16.
- **`--kv-cache-dtype=fp8`**: geeft ongeveer 2× zoveel KV-capaciteit, voor zo'n 10% minder tokens per seconde.

Deze instellingen komen uit de PR (getest op 4× V620 met `--enforce-eager`) en uit het community-image `exaroot/gfx1030-vllm-rdna2-extras`.

## vLLM of Ollama

- **vLLM**: hogere doorvoer bij meerdere gelijktijdige requests (continuous batching), echte tensor parallelism over beide kaarten, en de API is OpenAI-compatibel.
- **Ollama (navi21-patch)**: makkelijker, GGUF-quantisaties (Q4/Q5/Q8), sneller voor één gebruiker met grote modellen, en werkt direct met ansreview, helmreview en docpipe.

Ollama en vLLM kunnen niet tegelijk dezelfde V620's volledig gebruiken. Kies per kaart, of stop de ene dienst voordat je de andere start.
