<h1 align="center">Qwen3.8-27B on TensorFold for RTX 5060 Ti (16 GB)</h1>

<p align="center">
  <sub>by <a href="https://screwedup.tech">Screwed Up Tech</a></sub>
  <br><br>
  <a href="https://x.com/ScrewedUpTech" target="_blank"><img src="https://img.shields.io/badge/Follow%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow on X" height="28" /></a>
  <a href="https://screwedup.tech/chop-shop" target="_blank"><img src="https://img.shields.io/badge/Chop%20Shop-FF4500?style=for-the-badge&logo=data:image/svg+xml;base64,&logoColor=white" alt="Chop Shop" height="28" /></a>
</p>

Serve **Qwen3.8-27B** through an OpenAI-compatible API on a single consumer NVIDIA GPU
with 16 GB VRAM, using [TensorFold](https://github.com/ashhart/TensorFold) v0.6.1. Windows and Linux,
one command.

| Model | Type | Size on disk | VRAM | Decode | Spec decode |
| --- | --- | ---: | ---: | ---: | --- |
| **Qwen3.8-27B** EXL3 2.0bpw | Dense | 6.8 GB | ~7.2 GB | **27.7 tok/s** | Not available on 16 GB |

> **Flash Next note:** Qwen3.8 Flash Next (512-expert MoE, 62.8 GB) does **not** fit on 16 GB even
> with `--ssd-experts` maximum offload. The shared attention/embedding weights alone exceed available
> VRAM after CUDA's 2 GB reserve. Flash Next needs 24 GB+ VRAM, or a DGX Spark (128 GB unified).
> We tested and confirmed this — see [Speed tuning guide](#speed-tuning-guide).

The dense model is proven and fast. Flash Next is the upgrade path — a 512-expert MoE with built-in
multi-token prediction (MTP) that TensorFold uses for speculative decoding at zero extra VRAM, with
expert weights streaming from SSD.

> **This is the first published TensorFold benchmark on a consumer NVIDIA GPU.**

---

## Performance

**RTX 5060 Ti 16 GB** (SM 12.0 Blackwell, GDDR7 576 GB/s), Windows 11, TensorFold v0.6.1.
Measured through the OpenAI API from another machine on the LAN.

### Qwen3.8-27B Dense (EXL3 2.0bpw, no drafts, thinking off)

| Metric | Value |
| --- | ---: |
| Decode speed | **27.7 tok/s** |
| Prefill speed | **162 tok/s** |
| Context window | 3,072 tokens |
| VRAM used | ~7.2 GB / 16 GB |
| Time to first token | ~250 ms |

The 5060 Ti's GDDR7 bandwidth (576 GB/s) gives it a real advantage over memory-bandwidth-bound
architectures. For comparison, a DGX Spark (128 GB LPDDR5X, 273 GB/s) runs the same model at
~30 tok/s — the 5060 Ti is within 8% on half the memory bandwidth, because decode is
bandwidth-bound and GDDR7 delivers.

### Qwen3.8 Flash Next (EXL3 2.05bpw, MTP, SSD offload)

**Does not fit on 16 GB.** Tested with `--ssd-experts 62 --context 256 --mtp-drafts 0` —
TensorFold reports "0 tokens across the ranks." The shared attention and embedding weights
(non-expert layers that cannot be offloaded) exceed available VRAM after the mandatory
2 GB CUDA reserve. Flash Next needs 24 GB+ VRAM for TensorFold, or runs on DGX Spark
(128 GB unified) at ~63.6 tok/s.

---

## What the kit picks for your GPU

| VRAM | Dense model | Flash Next | Notes |
| ---: | --- | --- | --- |
| 12 GB | 2.0 bpw @ 2k context | Does not fit | Floor: tight fit |
| **16 GB** | **2.0 bpw @ 3k context** | **Does not fit** | **This kit's target (dense)** |
| 24 GB | 4.0 bpw @ 8k context | 2.05 bpw + SSD offload | Flash Next starts here |
| 32 GB+ | 6.0 bpw @ 16k+ context | 3.05 bpw, full in VRAM | Near-lossless |

### Why not higher bpw on 16 GB?

| Quant | Model size | + KV cache | Fits 16 GB? |
| --- | ---: | ---: | --- |
| 2.0 bpw | ~6.8 GB | ~7.2 GB | Yes, 9 GB headroom |
| 2.5 bpw | ~8.5 GB | ~9.2 GB | Tight |
| 3.0 bpw | ~10.2 GB | ~11 GB | Very tight, <5k context |
| 3.5 bpw | ~11.9 GB | ~13 GB | Only ~2k context |
| 4.0 bpw | ~13.5 GB | ~14.5 GB | No headroom for KV |

---

## Requirements

| | |
| --- | --- |
| GPU | NVIDIA, 16 GB VRAM, compute capability 7.5+ (Turing and newer). Blackwell (SM 12.0) tested. |
| Driver | 570 or newer |
| Python | 3.11 or newer, 64-bit |
| Disk | 7 GB (dense) or 63 GB (Flash Next) for model weights |
| SSD | NVMe recommended for Flash Next expert offload |

**Windows only:** Visual Studio Build Tools (MSVC `cl.exe`) is needed for CUDA kernel JIT
compilation on first inference. The setup script checks for it. TensorFold ships prebuilt
Python wheels but compiles GPU kernels at runtime.

**Not needed:** CUDA Toolkit (separate from driver), Git.

---

## Quick start

### Windows

```
windows\setup.bat          first run: creates venv, downloads model
windows\start.bat          start the server
windows\stop.bat           stop the server
```

### Linux

```bash
chmod +x linux/*.sh
./linux/setup.sh           # first run: creates venv, downloads model
./linux/start.sh           # start the server (Ctrl-C to stop)
./linux/start.sh -b        # start in background
./linux/stop.sh            # stop the server
```

### Use it

```bash
# Check health
curl http://localhost:8090/health

# Chat completion
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-exl3-2bpw-tensorfold",
    "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
    "max_tokens": 500
  }'
```

Any OpenAI-compatible client works: `base_url = "http://<address>:8090/v1"`.

---

## Configuration

Everything lives in `.env` (copy `.env.example`). The important settings:

| Variable | Default | What it does |
| --- | --- | --- |
| `MODEL` | `dense` | `dense` or `flashnext` |
| `THINKING` | `off` | `off`: 14x faster, no reasoning chain. `on`: think first |
| `DRAFT` | `none` | Dense model drafting. `none` on 16 GB (DFlash2 doesn't fit) |
| `CONTEXT_SIZE` | `3072` | Token window. Dense: 2048-4096. Flash Next: up to 32768 |
| `CACHE_QUANT` | `4` | KV cache: `4` (int4, lossless), `8,4`, `8`, or `none` (fp16) |
| `GPU_MEM_GB` | `14.7` | VRAM budget. 16 GB cards have ~14.7 GB usable |
| `SSD_EXPERTS_GIB` | `50` | Flash Next: GB of expert weights on SSD (0 = all in VRAM) |
| `MTP_DRAFTS` | `6` | Flash Next: MTP speculative drafts per round |
| `VISION` | `off` | Image input. `auto` loads if it fits |
| `PORT` | `8090` | API port |
| `HOST` | `0.0.0.0` | Bind address. `127.0.0.1` for local only |

---

## Speed tuning guide

### What we tried (so you don't have to)

| Optimization | Result on 16 GB |
| --- | --- |
| `--no-thinking` | **14x speedup** (1.9 → 27.7 tok/s). The single biggest win. |
| `--no-drafts` | Required — DFlash2 (3.8 GB) OOMs with any quant >= 1.8 bpw |
| `--drafter none` | TensorFold CUDA engine doesn't support MTP for dense Qwen3.8 |
| `--kv-dtype int4` | Rejected for dense Qwen3.8 on CUDA (Flash Next only) |
| `--prefill-fp8` | Rejected for EXL3 packs (NVFP4 checkpoints only) |
| `--lane-kernels auto` | On by default for tensor-unit GPUs (5060 Ti qualifies) |
| `--ple-on-ssd` | MLX checkpoints only — EXL3 packs map n-gram table automatically |
| Context 4096 | OOM at 4096, stable at 3072 |
| **Flash Next + SSD offload** | **Does not fit on 16 GB.** Even `--ssd-experts 62 --context 256 --mtp-drafts 0` = 0 tokens. Shared weights exceed VRAM after 2 GB CUDA reserve. Needs 24 GB+. |

### What works

1. **Disable thinking** (`THINKING=off`): 1.9 → 27.7 tok/s. Always do this unless you need chain-of-thought.
2. **Kill other GPU processes** before starting: browsers, Discord, LM Studio, Ollama eat VRAM silently.
3. **Dense 2.0 bpw is king on 16 GB** — every other optimization path hits a wall (see table above).

### Windows gotchas

| Issue | Fix |
| --- | --- |
| `vcvarsall.bat` hangs in headless/SSH sessions | Don't call it. Add `cl.exe` to PATH directly (the start script does this automatically). |
| `TENSORFOLD_MEMORY_RESERVE_GIB=1` rejected | Minimum is 2. Set `TENSORFOLD_MEMORY_RESERVE_GIB=2`. |
| First inference takes 60-120s | CUDA kernel JIT compilation. Subsequent queries are instant. |
| `--checkpoint-slots` crash on dense model | Invalid flag for dense Qwen3.8 — don't use it. |
| Process dies when SSH disconnects | The process needs an active session. Use a persistent terminal, `screen`, or a Windows scheduled task. |

### MTP and Flash Next: tested, doesn't fit

The dense Qwen3.8-27B has an MTP head in the checkpoint (50 MB, `mtp_num_hidden_layers: 1`,
38 tensors at 2-bit). ExLlamaV3 uses it (`DRAFT=mtp` in Mia's kit). But TensorFold's CUDA
engine requires DFlash2 for dense models — the MTP code path is Flash Next only.

Flash Next is a 512-expert MoE where MTP is native. We downloaded it (62.8 GB) and tested
every combination of `--ssd-experts` (up to 62 GB), context (down to 256 tokens), and
`--mtp-drafts 0`. Result: **"0 tokens across the ranks"** every time. The shared
attention/embedding weights that cannot be offloaded to SSD exceed the 14 GB available
after TensorFold's mandatory 2 GB CUDA reserve on a 16 GB card.

Flash Next needs 24 GB+ VRAM. On a DGX Spark (128 GB unified memory) it runs at ~63.6 tok/s.

---

## Architecture notes

### Dense model memory layout (16 GB)

```
 RTX 5060 Ti — 16 GB GDDR7

  Windows/driver baseline     ~1.5 GB
  Qwen3.8-27B EXL3 2.0bpw    ~6.8 GB
  KV cache (3072 tokens)     ~0.05 GB
  CUDA kernels + workspace    ~0.5 GB
  ─────────────────────────────────
  Free                        ~7.2 GB

  Total: ~8.8 GB used / 16 GB
```

### Flash Next: why it doesn't fit (16 GB)

```
 RTX 5060 Ti — 16 GB GDDR7

  TensorFold CUDA reserve      2.0 GB  (mandatory minimum)
  Windows/driver baseline     ~1.5 GB
  ─────────────────────────────────
  Available for model          12.5 GB

  Flash Next shared weights   ~14+ GB  (attention, embeddings,
                                         n-gram table — not offloadable)
  ─────────────────────────────────
  Result: does not fit. 0 tokens.
```

Flash Next needs 24 GB+ VRAM where the shared weights fit with room for KV cache.

---

## What's next

- [x] Flash Next on 16 GB — tested, does not fit (shared weights > 14 GB)
- [ ] Flash Next recipe for 24 GB cards (RTX 4090 / 5080)
- [ ] AMD ROCm recipe (RX 7900 XTX / Strix Halo)
- [ ] Qualcomm GenieX NPU recipe (Snapdragon 8 Elite)
- [ ] Multi-engine comparison: TensorFold vs ExLlamaV3 vs vLLM
- [ ] Higher bpw quants (2.5, 3.0, 3.5) with tighter context
- [ ] Offsec-tuned model variant

---

## Credits

- [TensorFold](https://github.com/ashhart/TensorFold) by @ashxhart — the engine
- [turboderp](https://github.com/turboderp-org/exllamav3) — EXL3 quantization
- [MiaAI-Lab](https://github.com/MiaAI-Lab) — recipe patterns and DGX Spark benchmarks
- [Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw) — the 2.0 bpw checkpoint

## License

MIT
