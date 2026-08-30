---
title: "Dynamic MoE Routing and Precision Allocation for Quantized MoE Serving"
authors:
  - "Zhenghong Huang"
  - "Hongfan Wu"
  - "Jiheng Zhang"
publication_status: "Under review at NeurIPS 2026; Operations Research version nearing completion; slow-placement extension in development"
order: 3
excerpt: "Adaptive request routing across pre-quantized MoE instances, balancing throughput, congestion, and request-specific quality risk."
redirect_from:
  - /publications/pricemoe/
  - /publications/fluid-affinity/
  - /publications/topology-moe/
---

**Authors:** Zhenghong Huang, Hongfan Wu, and Jiheng Zhang<br>
**Status:** Under review at NeurIPS 2026; *Operations Research* version nearing completion; slow-placement extension in development

This research program studies how an inference service should route heterogeneous requests across pre-quantized copies of the same Mixture-of-Experts model.

## NeurIPS 2026 Submission

The conference paper, *Adaptive Routing for Quantized Mixture-of-Experts Serving with Theoretical Guarantee*, is under review at NeurIPS 2026. It studies a fixed resident pool of pre-quantized MoE instances and combines calibrated request-level quality estimates, a window-level linear program, and a KKT-consistent routing policy.

The paper distinguishes offline quality estimators from signals available at routing time. Evaluation on Qwen3-30B-A3B and DeepSeek-V2-Lite remains the basis for the empirical audit; result-level claims remain tied to completed evidence checks.

## Operations Research Version

*Dynamic Precision Allocation for Mixture-of-Experts Inference Services* formulates the platform as a stochastic service-control problem. The model uses class-level quality-risk calibration, a fluid relaxation, and endogenous capacity and quality-risk shadow prices to make congestion and quality trade-offs explicit. The model and theory draft is near complete; final result-dependent claims remain gated on empirical validation.

Earlier queue-aware and topology-aware MoE routing studies informed this program and are now consolidated here rather than listed as separate manuscripts.

## Slow-Placement Extension

The next extension studies two timescales: a slower resident-pool placement decision and faster request routing. The formulation explicitly separates model-weight memory, service rates, calibrated quality, and reconfiguration cost. This is a working idea rather than a completed manuscript or validated result.
