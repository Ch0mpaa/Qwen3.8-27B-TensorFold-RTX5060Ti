<h1 align="center">Qwen3.8-27B on TensorFold for RTX 5060 Ti (16 GB)</h1>

<p align="center">
  <sub>by <a href="https://screwedup.tech">Screwed Up Tech</a></sub>
  <br><br>
  <a href="https://x.com/ScrewedUpTech" target="_blank"><img src="https://img.shields.io/badge/Follow%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow on X" height="28" /></a>
  <a href="https://screwedup.tech/chop-shop" target="_blank"><img src="https://img.shields.io/badge/Chop%20Shop-FF4500?style=for-the-badge&logo=data:image/svg+xml;base64,&logoColor=white" alt="Chop Shop" height="28" /></a>
</p>

Serve **Qwen3.8-27B** and **Qwen3.8 Flash Next** through an OpenAI-compatible API on a single consumer NVIDIA GPU
with 16 GB VRAM, using [TensorFold](https://github.com/ashhart/TensorFold) v0.6.1. Windows and Linux,
one command.

Two models, one kit:

| Model | Type | Size on disk | VRAM | Decode | Spec decode |
| --- | --- | ---: | ---: | ---: | --- |
| **Qwen3.8-27B** EXL3 2.0bpw | Dense | 6.8 GB | ~7.2 GB | **27.7 tok/s** | Not available on 16 GB |
| **Qwen3.8 Flash Next** EXL3 2.05bpw | 512-expert MoE | 62.8 GB | ~8-10 GB + SSD | **TBD** | Native MTP (built-in) |

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

*Benchmarks in progress.* Target: 40-60+ tok/s with native MTP speculative decoding and
SSD expert offload on NVMe.

---

## What the kit picks for your GPU

| VRAM | Dense model | Flash Next | Notes |
| ---: | --- | --- | --- |
| 12 GB | 2.0 bpw @ 2k context | Not recommended | Floor: tight fit |
| **16 GB** | **2.0 bpw @ 3k context** | **2.05 bpw + SSD offload** | **This kit's target** |
| 24 GB | 4.0 bpw @ 8k context | 2.05 bpw, more in VRAM | Better quality |
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

**Not needed:** CUDA Toolkit, Visual Studio Build Tools, Git.
TensorFold ships prebuilt wheels for most configurations.

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
| Context 4096 | OOM at 4096, stable at 3072 |

### What works

1. **Disable thinking** (`THINKING=off`): 1.9 → 27.7 tok/s. Always do this unless you need chain-of-thought.
2. **Kill other GPU processes** before starting: browsers, Discord, LM Studio eat VRAM silently.
3. **Flash Next + MTP**: The upgrade path. MoE uses only active experts per token, SSD offload handles the rest, and built-in MTP gives speculative decode at zero extra VRAM cost.

### MTP: why Flash Next is the endgame

The dense Qwen3.8-27B has an MTP head in the checkpoint (50 MB, `mtp_num_hidden_layers: 1`,
38 tensors at 2-bit). ExLlamaV3 uses it (`DRAFT=mtp` in Mia's kit). But TensorFold's CUDA
engine requires DFlash2 for dense models — the MTP code path is Flash Next only.

Flash Next is a 512-expert MoE where MTP is native. TensorFold loads it automatically,
giving speculative decode without a separate 3.8 GB drafter. Combined with SSD expert
offload (`--ssd-experts`), Flash Next fits on 16 GB and should decode significantly faster.

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

### Flash Next memory layout (16 GB + SSD)

```
 RTX 5060 Ti — 16 GB GDDR7

  Windows/driver baseline     ~1.5 GB
  Active expert weights       ~4-6 GB
  Attention + shared layers   ~2-3 GB
  KV cache (int4)            ~0.5 GB
  MTP head                    ~0.05 GB
  ─────────────────────────────────
  Free for context / cache    ~4-8 GB

          | SSD offload (NVMe)
          v
  Inactive expert weights     ~50 GB
  N-gram embedding table      ~26 GB
```

---

## What's next

- [ ] Flash Next benchmark numbers on RTX 5060 Ti
- [ ] AMD ROCm recipe (RX 7900 XTX / Strix Halo)
- [ ] Qualcomm GenieX NPU recipe (Snapdragon 8 Elite)
- [ ] Multi-engine comparison: TensorFold vs ExLlamaV3 vs vLLM
- [ ] Offsec-tuned model variant

---

## Credits

- [TensorFold](https://github.com/ashhart/TensorFold) by @ashxhart — the engine
- [turboderp](https://github.com/turboderp-org/exllamav3) — EXL3 quantization
- [MiaAI-Lab](https://github.com/MiaAI-Lab) — recipe patterns and DGX Spark benchmarks
- [Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw) — the 2.0 bpw checkpoint

## License

MIT
