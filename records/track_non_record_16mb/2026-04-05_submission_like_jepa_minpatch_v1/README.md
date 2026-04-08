## Record: 11L + Depth Recurrence + MuonEq-R (WIP, targeting < 1.10 BPB)

**Baseline ref:** val_bpb 1.1228 (seed 1337, 8xH100 SXM, 600s) | **15.55 MB**
**Target:** beat PR #1435 (1.098 BPB, depth recurrence architecture)

### Changes vs Previous Record (1.1233 BPB)

| Change | Before | Now | Rationale |
|--------|--------|-----|-----------|
| **Depth Recurrence** | — | Layers 4,5 repeated → 13 virtual from 11 physical | PR #1435's core innovation, ~18% throughput cost |
| **MuonEq-R** | Standard Muon | Row-norm before Newton-Schulz | Stabilizes recurrent layer gradients |
| **XSA scope** | Last 4 layers | Last 11 (all) | Broader self-attention subtraction |
| **MLP expansion** | 3x (1536) | 4x (2048) | More capacity per layer |
| **qk_gain_init** | 1.5 | 5.0 | Sharper attention post-recurrence |
| **Warmdown** | 3500 iters | 5000 iters (0.667 frac) | Longer cooldown for depth recurrence |
| **Muon WD** | 0.04 | 0.09 | Stronger regularization |
| **Adam WD** | 0.04 | 0.02 | Lighter embedding regularization |
| **Embed WD** | (same as Adam) | 0.09 (separate) | Independent embed regularization |
| **EMA decay** | 0.997 | 0.9965 | Faster EMA tracking |
| **Matrix LR** | 0.025 | 0.02 | Slightly lower for recurrence stability |
| **Scalar LR** | 0.025 | 0.02 | Matched to matrix LR |
| **Embed LR** | 0.035 | 0.03 | Slightly lower |

### Runtime Optimizations

- `torch.compile(fullgraph=True)` with `torch.set_float32_matmul_precision("high")`
- SmearGate: `F.pad` instead of `torch.cat + zeros_like`
- Partial RoPE: single `torch.cat` instead of two
- Muon: pre-allocated flat buffer for updates
- EMA: `torch._foreach_mul_` / `torch._foreach_add_` vectorized ops
- TokenLoader: skip-based instead of full take+discard

### JEPA Evaluation (2026-04-07)

A/B comparison on spark1 (1xGPU, compile=True, 300s wallclock, depth recurrence at step 10):

| Step | Baseline Loss | JEPA Loss | Delta |
|------|-------------|-----------|-------|
| 10 | 7.700 | 7.669 | -0.030 |
| 15 | 7.456 | 7.449 | -0.007 |
| 20 | 7.388 | 7.410 | +0.023 |
| 25 | 7.323 | 7.363 | +0.040 |
| 30 | 7.225 | 7.296 | +0.071 |
| 37 | 7.250 | 7.329 | +0.080 |

**Conclusion:** With depth recurrence active, JEPA adds no benefit and is slightly worse (~0.08 higher loss at step 37). Depth recurrence subsumes JEPA's context compression gains. Proceeding without JEPA.

Throughput (spark1, 1xGPU, compile=True):
- Pre-recurrence: ~6.78s/step
- Post-recurrence: ~7.86-7.89s/step (~16% overhead from virtual layers)

### Architecture

- 11 physical transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- Depth recurrence: layers 4,5 → 13 virtual layers (activated at step 3000 in full runs)
- 4x MLP expansion (2048 hidden), relu-squared activation
- U-Net skip connections with learnable skip_weights
- XSA on all 11 layers (GQA-aware, zero-alloc)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- Shared Value Embedding (dim=128, layers 9,10) with per-layer learned scales
- SmearGate + BigramHash (2048 buckets, dim=128)
- Tied embeddings, logit softcap=30.0

### Training

- FlashAttention 3 (Hopper-optimized)
- MuonEq-R optimizer (matrices): lr=0.02, momentum=0.99, WD=0.09, row_norm=True
- AdamW (embeddings): lr=0.03, WD=0.09 | (scalars): lr=0.02, WD=0.02
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 5000 iterations (frac=0.667, wallclock-based)
- Depth recurrence start: step 3000 (recompiles model graph)
- EMA: decay=0.9965, every step
- Tight SWA: every 50 steps when scale<0.2
- Late QAT: STE int6 fake-quantization when LR scale<0.15
- OrthoInit + muP-scaled output projections

### Quantization

- GPTQ-lite: Per-row optimal clip percentile search (5 candidates) for int6
- Int6 per-row for MLP + attention weights
- Int8 per-row for embeddings
- Control tensors in fp32
- zstd level 22 compression

### Next Steps

- Full 600s run on 8xH100 RunPod with depth recurrence
- Tune depth_recur_start timing and recurrence layer selection
- Consider GPTQ int6 (PR #1435 approach) vs current GPTQ-lite
- Evaluate if more virtual layers (3 repeated instead of 2) improve BPB within budget
