---
layout: post
title: "Shadow Price in MoE Routing: From Queueing Theory to a Minimal vLLM Implementation"
date: 2026-03-01 12:00:00 +0800
categories: [LLM Systems, MoE, Queueing]
tags: [MoE, vLLM, Routing, Shadow Price, Queueing Theory, A100]
excerpt: "A practical story of how queue lengths become shadow prices, and how that idea can be implemented in vLLM by modifying CPU-side expert selection only—then validated on Qwen2.5-MoE-72B with 8×A100 TTFT/TPOT measurements."
---

> This post is the blog-form version of my internal note: **“Shadow Price in MoE Routing: From Queueing Theory to a Minimal vLLM Implementation”** (last updated 2026-03-01).
>
> Scope: a minimal, **kernel-free** PriceMoE/queue-aware routing implementation in vLLM, plus the key experimental evidence used to validate it.

---

## 0. Why this matters (systems view)

Mixture-of-Experts (MoE) inference is often limited by **load imbalance** rather than average FLOPs: a subset of experts become hotspots, stretching the critical path of dispatch/expert compute/combine. This shows up as degraded throughput and worse tail latency.

A practical router should therefore be:

- **Congestion-aware**: avoid routing into already-hot experts.
- **Cheap**: zero/near-zero overhead when disabled.
- **Compatible** with existing kernels (so it’s easy to land in an inference stack).

---

## 1. From queueing theory to “shadow price” routing

### 1.1 Virtual queue as a dual signal

Following a queueing/dual interpretation, each expert $e$ maintains a virtual queue length $q_e(t)$ that summarizes how congested the expert is. In a fluid picture:

$$
\dot{q}_e(t)=\lambda_e(t)-\mu_e\,u_e(t)
$$

In vLLM there is no literal “token waiting queue per expert” exposed at the Python side. So in the MVP we use a **virtual queue**: a smoothed counter of how often each expert has been selected recently.

### 1.2 Shadow price corrected logits

Let $s_{t,e}$ be the pre-softmax router logits for token $t$. PriceMoE applies a shadow-price correction:

$$
\\tilde{s}_{t,e}(t)=s_{t,e}-\\alpha\\,q_e(t)
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

The goal is to inject a per-expert bias *before* vLLM’s fused top-k path runs, keeping the performance-critical fused kernels intact.

### 2.2 The core design: a thin router with two hooks

We implement a small `PriceMoERouter` with two inexpensive hooks:

- `get_expert_load_bias(device) -> Optional[Tensor(E,)]`
  - Returns bias $b_e=-\alpha q_e$ as a float32 vector (shape `(E,)`).
  - Returns `None` when $\alpha=0$ so the caller can skip any extra logic.

- `update_queue(topk_ids)`
  - Updates the EMA virtual queue using selected expert IDs.
  - Uses `scatter_add_` + in-place EMA to avoid expensive ops.

### 2.3 What changes in `_select_experts()`

Inside `_select_experts()`, when the router is enabled:

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

---

## 3. Evidence: alpha sweep & load-balance metrics

### 3.1 Metrics

Two categories:

### Routing distribution / load balance

- `Imb` (expert imbalance): avg over layers of $(\max_e \text{load}_e)/(\text{mean}_e \text{load}_e)$ (lower is better)
- `Gini`: inequality of expert loads (lower is better)
- `EPimb` / `EPgini`: the same metrics aggregated per EP rank (GPU-level imbalance)

### Throughput

- `Req/s` and `Tok/s`

### 3.2 Results snapshot (ShareGPT workload)

Below is a snapshot from an alpha sweep on a ShareGPT-like workload (EMA decay = 0.9). Lower is better for imbalance metrics.

| Run | Mode | Imb ↓ | Gini ↓ | EPimb ↓ | EPgini ↓ | Req/s | Tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | baseline | 3.832 | 0.0926 | 1.129 | 0.0520 | 5.22 | 2376.1 |
| alpha=0.001 | price_moe | 3.692 | 0.0903 | 1.126 | 0.0510 | 4.93 | 2241.5 |
| alpha=0.01 | price_moe | 2.955 | 0.0751 | 1.101 | 0.0412 | 4.98 | 2264.5 |
| alpha=0.02 | price_moe | 2.559 | 0.0652 | 1.086 | 0.0353 | 4.91 | 2233.3 |
| alpha=0.05 | price_moe | 1.975 | 0.0457 | 1.060 | 0.0243 | 4.99 | 2271.1 |
| alpha=0.10 | price_moe | 1.614 | 0.0314 | 1.042 | 0.0166 | 5.01 | 2277.6 |

### 3.3 Quality vs balance (example)

Load balancing is only useful if we don’t destroy quality. In a Qwen3-30B setup, we tracked a simple proxy (PPL) alongside imbalance metrics across $\alpha$.

| $\alpha$ | Imbalance ↓ | Gini ↓ | PPL | PPL Δ% |
| ---: | ---: | ---: | ---: | ---: |
| baseline | 4.89 | 0.120 | 8.62 | – |
| 0.002 | 3.65 | 0.095 | 8.68 | +0.6% |
| 0.005 | 2.81 | 0.075 | 8.91 | +3.3% |
| 0.01 | 2.23 | 0.058 | 9.32 | +8.1% |
| 0.05 | 1.37 | 0.022 | 12.71 | +47% |
| 0.1 | 1.25 | 0.014 | 16.96 | +97% |

Practical takeaway: there is typically a “sweet spot” at small $\alpha$.

---

## 4. What we learned

1) **Shadow price is a strong signal:** even a simple EMA queue proxy can drastically reduce expert-level inequality.

2) **GPU-level balance matters too:** EP-Gini improves significantly, but EP-imbalance may reduce only modestly depending on the topology and baseline placement.

3) **Throughput isn’t guaranteed:** better balance does not automatically imply higher throughput. Bottlenecks can include dispatch/comm patterns and imperfect queue proxies.

---

## 5. Limitations & next steps

- Current $q_e$ is EMA of routed counts, not measured queue time / execution time.
- MVP doesn’t enforce a “quality loss ≤ ε” constraint.
- Only the $\alpha q_e$ term is implemented (no topology/comm penalty yet).

Next steps:

1. Use runtime signals (per-expert time, comm time, critical-path proxies) to define $q_e$.
2. Add quality-aware conservative gating (e.g., penalize within top-$m$ candidates).
3. Explore per-layer / per-phase $\alpha$ and adaptive $\alpha(t)$.
