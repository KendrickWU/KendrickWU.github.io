# Shadow Price in MoE Routing: From Queueing Theory to a Minimal vLLM Implementation

## Last updated

2026-03-01

> This post documents a minimal, **kernel-free** implementation of PriceMoE (shadow-price / queue-aware routing) in vLLM, plus the key experimental evidence we used to validate it.
>
> Scope: MVP focusing on the **congestion penalty** term only: $\tilde{s}_{t,e}=s_{t,e}-\alpha\,q_e(t)$. We intentionally do **not** modify CUDA kernels, and we do **not** touch the vLLM request scheduler.

---

## 0. Why this matters (systems view)

Mixture-of-Experts (MoE) inference is often limited by **load imbalance** rather than average FLOPs: a subset of experts become hotspots, which stretches the critical path of dispatch/expert compute/combine. This shows up as degraded throughput and worse tail latency.

A practical router should therefore be:

- **Congestion-aware**: avoid routing into already-hot experts.
- **Cheap**: zero/near-zero overhead when disabled.
- **Compatible** with existing kernels (so it’s easy to land in an inference stack).

---

## 1. From queueing theory to “shadow price” routing

### 1.1 Virtual queue as a dual signal

Following a queueing/dual interpretation, each expert $e$ maintains a virtual queue length $q_e(t)$ that summarizes how congested the expert is. In a fluid picture you can write:

$$
\dot{q}_e(t)=\lambda_e(t)-\mu_e\,u_e(t)
$$

In vLLM there is no literal “token waiting queue per expert” exposed at the Python side. So in the MVP we use a **virtual queue**: a smoothed counter of how often each expert has been selected recently.

### 1.2 Shadow price corrected logits

Let $s_{t,e}$ be the pre-softmax router logits for token $t$. PriceMoE applies a shadow-price correction:

$$
\tilde{s}_{t,e}(t)=s_{t,e}-\alpha\,q_e(t)
$$

and then routes by top-$k$ on $\tilde{s}$.

- $\alpha=0$ recovers the baseline router.
- Larger $\alpha$ spreads traffic more aggressively.

---

## 2. Implementation in vLLM (MVP, kernel-free)

### 2.1 Where to hook: `FusedMoE._select_experts()`

In vLLM, routing selection happens in:

- `external/vllm/vllm/model_executor/layers/fused_moe/layer.py`
- method: `FusedMoE._select_experts(hidden_states, router_logits)`

The goal is to inject a per-expert bias *before* vLLM’s fused top-k path runs, so we keep the performance-critical fused kernels intact.

### 2.2 The core design: a thin router with two hooks

We implemented `PriceMoERouter` in:

- `external/vllm/vllm/model_executor/layers/fused_moe/price_moe_router.py`

It exposes two inexpensive hooks:

- `get_expert_load_bias(device) -> Optional[Tensor(E,)]`

- Returns **bias** $b_e=-\alpha q_e$ as a float32 vector (shape `(E,)`).
- Returns `None` when $\alpha=0$ so the caller can skip any extra logic.

- `update_queue(topk_ids)`

- Updates the EMA virtual queue using the selected expert IDs.
- Uses `scatter_add_` + in-place EMA to avoid expensive ops.

### 2.3 What changes in `_select_experts()`

Inside `_select_experts()`, when the PriceMoE router is present:

- Read bias `b` from router.
- If `b is not None`: apply `router_logits = router_logits + b.unsqueeze(0)`.
- Call the existing fused top-k routine.
- After routing, call `update_queue(topk_ids)`.

This means:

- **No CUDA kernel changes**.
- When disabled, routing is identical to baseline (zero overhead path).

### 2.4 Control knobs (env vars)

The MVP is gated by environment variables (safe for A/B testing):

- `VLLM_PRICE_MOE_ENABLED=1`
- `VLLM_PRICE_MOE_ALPHA=<float>`
- `VLLM_PRICE_MOE_EMA_DECAY=<float>`
- `VLLM_PRICE_MOE_USE_ORIGINAL_WEIGHTS=1`

---

## 3. Experiment: alpha sweep & load-balance metrics

### 3.1 Metrics we report

We evaluate the router with two categories of metrics:

### Routing distribution / load balance

- `Imb` (expert imbalance): avg over layers of $(\max_e \text{load}_e)/(\text{mean}_e \text{load}_e)$ (lower is better)
- `Gini`: inequality of expert loads (lower is better)
- `EPimb` / `EPgini`: the same metrics after aggregating experts per EP rank (GPU-level imbalance)

### Throughput

- `Req/s` and `Tok/s`

Routing distribution stats are collected via:

- `external/vllm/vllm/moe_stats.py`

and analyzed by:

- `Code/Verification/E1.3/src/analyze_load_balance.py`

### 3.2 Results snapshot A (ShareGPT workload, EP-aware metrics)

Below is a snapshot from an alpha sweep on a ShareGPT-like workload (EMA decay = 0.9). Lower is better for imbalance metrics.


| Run | Mode | Imb ↓ | Gini ↓ | EPimb ↓ | EPgini ↓ | Req/s | Tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | baseline | 3.832 | 0.0926 | 1.129 | 0.0520 | 5.22 | 2376.1 |
| alpha=0.001 | price_moe | 3.692 | 0.0903 | 1.126 | 0.0510 | 4.93 | 2241.5 |
| alpha=0.01 | price_moe | 2.955 | 0.0751 | 1.101 | 0.0412 | 4.98 | 2264.5 |
| alpha=0.02 | price_moe | 2.559 | 0.0652 | 1.086 | 0.0353 | 4.91 | 2233.3 |
| alpha=0.05 | price_moe | 1.975 | 0.0457 | 1.060 | 0.0243 | 4.99 | 2271.1 |
| alpha=0.10 | price_moe | 1.614 | 0.0314 | 1.042 | 0.0166 | 5.01 | 2277.6 |

### 3.3 Results snapshot B (ShareGPT workload, simpler metrics)

We also observed a consistent reduction in expert imbalance and Gini under a separate ShareGPT run group (without EP breakdown in the summary table). In that snapshot, the best run (α=0.05) achieved:

- Load imbalance: 3.826 → 1.957 (−48.8%)
- Gini: 0.0927 → 0.0464 (−49.9%)

while throughput decreased by ~8.7% in Req/s.

| Run (timestamped) | Mode | Imbalance ↓ | Gini ↓ | Req/s | Tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 20260129_213151_sharegpt_baseline | baseline | 3.826 | 0.0927 | 5.21 | 2368.0 |
| 20260129_214106_sharegpt_price_moe_alpha0.05 | price_moe | 1.957 | 0.0464 | 4.75 | 2162.3 |
| 20260129_224741_sharegpt_price_moe_alpha0.02 | price_moe | 2.521 | 0.0650 | 4.65 | 2115.0 |

This reinforces the core claim of the MVP: **shadow-price routing reliably flattens expert usage**. Whether that translates to throughput depends on where the true bottleneck is (more in §4).

### 3.4 Quality vs balance: a concrete trade-off (Qwen3-30B)

Load balancing is only useful if we don’t destroy quality. In a Qwen3-30B setup, we tracked a simple quality proxy (PPL) alongside imbalance metrics across α.

The key pattern is a classic trade-off curve: larger α improves balance but can hurt PPL.

| α | Imbalance ↓ | Gini ↓ | PPL | PPL Δ% |
| ---: | ---: | ---: | ---: | ---: |
| baseline | 4.89 | 0.120 | 8.62 | – |
| 0.002 | 3.65 | 0.095 | 8.68 | +0.6% |
| 0.005 | 2.81 | 0.075 | 8.91 | +3.3% |
| 0.01 | 2.23 | 0.058 | 9.32 | +8.1% |
| 0.05 | 1.37 | 0.022 | 12.71 | +47% |
| 0.1 | 1.25 | 0.014 | 16.96 | +97% |

**Practical takeaway:** there is typically a "sweet spot" at small α.

- Conservative: α≈0.002 (≈25% imbalance improvement with ~0.6% PPL change)
- Balanced: α≈0.005 (~42% improvement with ~3.3% PPL increase)
- Aggressive: α≈0.01 (~55% improvement with ~8.1% PPL increase)

This is why the MVP is wired with environment variables: α should be tuned per model/workload/quality target.
| alpha=0.05 | price_moe | 1.975 | 0.0457 | 1.060 | 0.0243 | 4.99 | 2271.1 |
| alpha=0.10 | price_moe | 1.614 | 0.0314 | 1.042 | 0.0166 | 5.01 | 2277.6 |

**Relative to baseline (alpha=0.10):**

- Imbalance: 3.832 → 1.614 (**−57.9%**)
- Gini: 0.0926 → 0.0314 (**−66.1%**)
- EP-imbalance: 1.129 → 1.042 (**−7.8%**)
- EP-Gini: 0.0520 → 0.0166 (**−68.1%**)

Throughput in this sweep is roughly flat/slightly lower (Tok/s 2376.1 → 2277.6, about −4.1%). This is consistent with the MVP being optimized for **load balance first**, and suggests follow-up work is needed to translate better balance into end-to-end throughput under the tested setup.

---

## 4. Discussion: what we learned

1) **Shadow price is a strong signal:** Even a simple EMA queue proxy can drastically reduce expert-level inequality.

2) **GPU-level balance matters too:** EP-Gini improves significantly, but EP-imbalance reduces only modestly in this snapshot. That hints that (i) the hottest experts may already be distributed across ranks, and/or (ii) end-to-end bottlenecks include comm/dispatch effects not captured by the current penalty.

3) **Throughput isn’t guaranteed:** Better balance does not automatically imply higher throughput.

Likely reasons include:

- The virtual queue is not a perfect proxy for real critical-path delay.
- The penalty may cause extra routing “movement” that hurts locality/cache.
- Communication overhead and EP dispatch patterns can dominate.

---

## 5. Limitations & next steps

- **Queue definition:** current $q_e$ is EMA of routed counts, not measured queue time / execution time.
- **No explicit quality guard:** MVP doesn’t enforce a “quality loss ≤ ε” constraint; any quality regression must be measured.
- **No topology/comm penalty yet:** only the $\alpha q_e$ term is implemented.

Next steps (low-risk upgrades):

1. Use real runtime signals (per-expert time, comm time, or critical-path proxies) to define $q_e$.
2. Add a quality-aware constraint or conservative gating (e.g., only penalize within top-m candidate experts).
3. Explore per-layer / per-phase $\alpha$ and adaptive $\alpha(t)$.

---

## Appendix A — Minimal reproducibility notes

The repo includes an executable pipeline in:

- `Code/Verification/E1.3/`

Key scripts:

- `src/run_experiment.py` (runner)
- `src/analyze_load_balance.py` (aggregates `.npz` stats)

Key vLLM files:

- `external/vllm/vllm/model_executor/layers/fused_moe/layer.py`
- `external/vllm/vllm/model_executor/layers/fused_moe/price_moe_router.py`
- `external/vllm/vllm/moe_stats.py`
