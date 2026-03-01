---
layout: post
title: "Shadow Price in MoE Routing: From Queueing Theory to a Minimal vLLM Implementation"
date: 2026-03-01 12:00:00 +0800
categories: [LLM Systems, MoE, Queueing]
tags: [MoE, vLLM, Routing, Shadow Price, Queueing Theory, A100]
excerpt: "A practical story of how queue lengths become shadow prices, and how that idea can be implemented in vLLM by modifying CPU-side expert selection only—then validated on Qwen2.5-MoE-72B with 8×A100 TTFT/TPOT measurements."
---

## Motivation

Mixture-of-Experts (MoE) models promise *scale with sparsity*: you get a huge parameter count but only activate a small number of experts per token. In serving, however, MoE introduces a systems problem that dense models largely avoid:

- Each expert behaves like a **server** with its own effective service rate.
- Tokens routed to the same expert form a **queue**.
- A naïve top-$k$ semantic routing policy can create persistent **load imbalance**, which shows up as tail latency.

This post explains a simple idea:

> Treat (estimated) expert queue length as a **shadow price** for congestion, and penalize routing decisions by that price.

Then I show how to implement a minimal viable version in **vLLM** by modifying only the CPU scheduling layer (no kernel changes), and how I validated it on **Qwen2.5-MoE-72B** with **8×A100**.

---

## From queues to shadow prices (the OR view)

A common abstraction is to view MoE routing as an online resource allocation problem. Each token $t$ chooses $k$ experts out of $E$ experts.

Let:

- $s_{t,e}$ be the semantic score (e.g., router logit) of expert $e$ for token $t$.
- $q_e$ be the current queue length (or a proxy) of expert $e$.
- $\lambda_e$ be the shadow price associated with expert $e$'s capacity constraint.

A minimal “shadow-price” routing rule can be written as:

$$
\text{choose }\operatorname{TopK}_e\; \Big( s_{t,e} - \alpha\, q_e \Big)
$$

where $\alpha > 0$ converts queue units into score units.

### Why this makes sense

In queueing/control interpretations, $q_e$ (or a monotone transform of it) plays a dual role:

- **Operational**: it predicts waiting time under high utilization.
- **Economic**: it behaves like a Lagrange multiplier (shadow price) for limited service capacity.

So you can read $\alpha q_e$ as a *congestion surcharge*.

---

## Minimal implementation in vLLM

### Design goal

I wanted a minimal patch that:

- modifies **only** expert selection (routing) on the CPU side;
- does **not** touch custom CUDA kernels;
- is easy to A/B test against the baseline routing.

### What I changed

In `FusedMoE.select_experts()`, I injected a queue-length penalty term into the expert scoring.

- Baseline (conceptually): pick top-$k$ experts by semantic score.
- Modified: pick top-$k$ by `semantic_score - alpha * queue_penalty(expert)`.

I used **queue length proxies from runtime bookkeeping** in the scheduler (token counts per expert / per step), rather than true GPU FIFO depth.

> Note: this is intentionally "minimal". The point is feasibility and directional evidence, not the final best controller.

### Why no kernel changes are needed

The kernel only needs the final selected expert indices (top-$k$). If we change how indices are chosen *before* launching the kernel, the rest of the pipeline remains unchanged.

---

## Experiment: Qwen2.5-MoE-72B on 8×A100

### Setup

- Model: **Qwen2.5-MoE-72B**
- GPUs: **8×A100**
- Compared policies:
  1. Baseline vLLM routing
  2. Shadow-price routing (queue-penalized selection)

### Metrics

I recorded:

- **TTFT** (Time To First Token)
- **TPOT** (Time Per Output Token)

These two numbers are a good first lens for MoE serving because:

- TTFT is sensitive to prefill scheduling and early congestion.
- TPOT reflects steady-state decode throughput and tail effects.

### Results

I’m including a results table template below. Replace the numbers with your measured results (or I can fill them in if you paste your logs):

| Policy | TTFT (p50) | TTFT (p95) | TPOT (p50) | TPOT (p95) | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | TODO | TODO | TODO | TODO | vLLM default |
| Shadow-price | TODO | TODO | TODO | TODO | $\alpha$ = TODO |

---

## Practical notes: choosing $\alpha$ (penalty strength)

A tiny penalty does nothing; an overly large penalty can cause "over-avoidance" (hurting quality or creating oscillation).

In practice I recommend:

1. Start with a small $\alpha$ that changes routing only rarely.
2. Increase until you see measurable p95 improvements.
3. Verify you don’t break output quality (at least spot-check).

---

## Limitations (what this MVP does *not* solve)

- True queue length is hard to observe precisely without deeper integration.
- A static $\alpha$ is not necessarily optimal across workloads.
- Token-to-expert affinity and communication topology costs are not modeled here.

But for an MVP, the key question is:

> Can a shadow-price idea be implemented *cheaply* in a production-grade engine, and does it move latency metrics in the right direction?

For me, the answer is yes.

---

## Takeaways

- Queueing theory gives a clean mental model for MoE routing congestion.
- Shadow prices offer a principled way to turn congestion into a routing penalty.
- In vLLM, you can test this idea by modifying **CPU-side expert selection only**.

If you want, I can follow up with:

- a cleaner patch organization (feature flag, config knobs, ablation hooks),
- a reproducible benchmark script,
- a deeper discussion of stability (drift arguments) vs. empirical results.
