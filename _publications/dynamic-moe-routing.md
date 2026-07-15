---
title: "Dynamic MoE Routing and Precision Allocation for Quantized MoE Serving"
authors:
  - "Zhenghong Huang"
  - "Hongfan Wu"
  - "Jiheng Zhang"
publication_status: "NeurIPS 2026 version submitted; Operations Research version in preparation"
order: 3
excerpt: "Adaptive request routing across pre-quantized MoE instances, balancing throughput, congestion, and request-specific quality risk."
redirect_from:
  - /publications/pricemoe/
  - /publications/fluid-affinity/
  - /publications/topology-moe/
---

**Authors:** Zhenghong Huang, Hongfan Wu, and Jiheng Zhang<br>
**Status:** NeurIPS 2026 version submitted; *Operations Research* version in preparation

This research program studies how an inference service should route heterogeneous requests across pre-quantized copies of the same Mixture-of-Experts model.

## NeurIPS Version

*Adaptive Routing for Quantized Mixture-of-Experts Serving with Theoretical Guarantee* introduces Fragility-Weighted Perplexity (FWP), a request-level quality-risk estimator built from expert fragility and request-dependent expert affinity. A window-level linear program leads to a KKT-consistent greedy routing policy.

Experiments on Qwen3-30B-A3B and DeepSeek-V2-Lite show up to 1.38x decode throughput relative to the highest-bit instance baseline without measured response-quality loss.

## Operations Research Version

*Dynamic Precision Allocation for Mixture-of-Experts Inference Services* formulates the platform as a stochastic service-control problem. The model uses class-level quality-risk calibration, a fluid relaxation, and endogenous capacity and quality-risk shadow prices to make congestion and quality trade-offs explicit.

Earlier queue-aware and topology-aware MoE routing studies informed this program and are now consolidated here rather than listed as separate manuscripts.
