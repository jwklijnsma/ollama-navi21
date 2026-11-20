# ollama-navi21

Ollama met de Navi21-fixes uit [sixvolts/llama-navi21-furnace](https://github.com/sixvolts/llama-navi21-furnace), voor de **AMD Radeon Pro V620** (en andere gfx1030 / RDNA2-kaarten zoals RX 6800/6900).

## Waarom

Ollama draait sinds de 0.40-reeks `llama-server` uit een vastgepinde llama.cpp (`LLAMA_CPP_VERSION`). In die upstream-code:

- crasht flash attention op gfx1030 voor head_dim 128/256/512/576 (`GGML_ASSERT(max_blocks_per_sm > 0)`): Qwen3.5/3.6, Gemma 4, DeepSeek-MLA;
- valt MMVQ (decode-matmul) op RDNA2 terug op 1 wave per workgroup, omdat RDNA2 in de config-tabellen ontbreekt.

`patches/100-navi21-rdna2.patch` zet 11 commits uit de furnace-fork over naar Ollama's llama.cpp:

| Onderdeel | Wat |
|---|---|
| `fattn-common.cuh` | eigen occupancy-berekening voor RDNA2 (HIP onderschat het registerbestand) |
| `fattn-tile.cuh` | aparte RDNA2-tile-tabel + caps op cols_per_block/ncols2 → geen aborts meer |
| `mmvq.cu` | RDNA2-entry in de MMVQ-tabellen (nwarps/rows_per_block) |
| `vecdotq.cuh` | K-quant sub-block-schaal in float op AMD (MMQ, prefill) |

Alles staat achter `RDNA2`/`GGML_CUDA_CC_IS_RDNA2`-checks; andere GPU's veranderen niet. Bewust **niet** meegenomen: de speculative-decoding-, MTP- en fusion-commits van de fork (te groot/riskant om op een nieuwere llama.cpp te zetten).

De patch is getest op `git apply` tegen llama.cpp **b12177** (Ollama v0.40.16) en **b12257** (Ollama v0.40.17-rc1 / main), telkens na Ollama's eigen `001-llama-cpp-hooks.patch`.

## Bouwen

Het script haalt Ollama op, legt de patch in `llama/compat/` (Ollama's eigen patch-applier pakt hem op), controleert of hij past en zet de HIP-targets op alleen `gfx1030` (scheelt veel compileertijd).

```bash
# Tarball voor LXC / bare metal (Docker op de build-host nodig, geen ROCm)
./build.sh tarball            # -> dist/bin/ollama + dist/lib/ollama

# Docker-image
./build.sh image              # -> ollama-navi21:v0.40.16

# Native (ROCm 7.x, Go, CMake >= 3.24, Ninja op deze host)
./build.sh native

# Andere Ollama-versie / extra targets
OLLAMA_REF=v0.40.17-rc1 GFX="gfx1030;gfx1100" ./build.sh tarball
```

## Installeren in een Proxmox-LXC

LXC-config (`/etc/pve/lxc/<id>.conf`):

```
lxc.cgroup2.devices.allow: c 226:* rwm
lxc.cgroup2.devices.allow: c 234:* rwm
lxc.mount.entry: /dev/dri dev/dri none bind,optional,create=dir
lxc.mount.entry: /dev/kfd dev/kfd none bind,optional,create=file
```

(controleer het major-nummer van `/dev/kfd` met `ls -l /dev/kfd`; dat is niet altijd 234.)

In de container:

```bash
tar -C dist -cf - bin lib | tar -C /usr -xf -     # -> /usr/bin/ollama, /usr/lib/ollama
useradd -r -s /bin/false -m -d /usr/share/ollama ollama
usermod -aG render,video ollama
cat >/etc/systemd/system/ollama.service <<'EOF'
[Unit]
Description=Ollama (navi21)
After=network-online.target
[Service]
User=ollama
Group=ollama
Environment=OLLAMA_HOST=0.0.0.0:11434
Environment=OLLAMA_FLASH_ATTENTION=1
Environment=OLLAMA_KV_CACHE_TYPE=q8_0
ExecStart=/usr/bin/ollama serve
Restart=always
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload && systemctl enable --now ollama
```

Docker:

```bash
docker run -d --device /dev/kfd --device /dev/dri \
  -e OLLAMA_FLASH_ATTENTION=1 -e OLLAMA_KV_CACHE_TYPE=q8_0 \
  -p 11434:11434 -v ollama:/root/.ollama ollama-navi21:v0.40.16
```

## Controleren

```bash
journalctl -u ollama -f          # moet beide V620's tonen (ROCm, gfx1030, 32 GiB)
ollama run qwen3.5:27b --verbose "schrijf een haiku over Friesland"
```

- Geen `GGML_ASSERT(max_blocks_per_sm > 0)` meer met flash attention aan → patch zit erin.
- Vergelijk `eval rate` / `prompt eval rate` met stock Ollama (`OLLAMA_FLASH_ATTENTION=0`) om de winst te zien.

## Kanttekeningen

- In mijn bouwomgeving was geen ROCm-toolchain beschikbaar: de patch is op toepasbaarheid en code-context gecontroleerd, **niet gecompileerd**. De eerste build op jouw machine is de echte test.
- Upstream heeft tussen de basis van de fork (sept. 2026) en b12257 ~1200 commits; de FA-launchlogica is daarin ook veranderd. Draai bij twijfel `test-backend-ops -o FLASH_ATTN_EXT` en `-o MUL_MAT` uit een llama.cpp-build met dezelfde patch.
- Bij een nieuwe Ollama-release faalt `build.sh` netjes in stap 3 als de patch niet meer past.

## CI

`.github/workflows/patch-check.yml` draait wekelijks (en bij elke push) `check-patch.sh` tegen de laatste Ollama-release en Ollama `main`. Een rode run betekent dat een nieuwe Ollama-versie een andere llama.cpp pint waar de patch niet meer op past.

```bash
./check-patch.sh            # laatste release
./check-patch.sh main
```

## Releases

`.github/workflows/release.yml` maakt automatisch een GitHub-release (tag + patch + scripts, geen binaries):

- **na elke gemergede pull request** naar `main`;
- **wekelijks** (maandagochtend), maar alleen als er iets veranderd is: nieuwe commits sinds de vorige release, of een nieuwe Ollama-release;
- **handmatig** via *Actions → release → Run workflow* (met `force` voor een release zonder wijzigingen).

Tags heten `v<ollama-versie>-navi21.<n>`, bijv. `v0.40.16-navi21.1`. Er komt alleen een release als de patch op die Ollama-versie past; `build.sh` in de release-tarball staat standaard op die versie.

## Licentie

De patch is afgeleid van [sixvolts/llama-navi21-furnace](https://github.com/sixvolts/llama-navi21-furnace) en [llama.cpp](https://github.com/ggml-org/llama.cpp), beide MIT. Ollama is MIT.
