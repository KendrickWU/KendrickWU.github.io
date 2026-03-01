---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

## Contact Information

- **Email**: wuhongfan0612@foxmail.com
- **Phone**: +86-13268672769
- **Location**: Hong Kong

## Research Interests

Distributed LLM/MoE inference systems; online stochastic control and queueing; topology-aware routing and communication optimization; spatial matching and marketplace design.

## Education

### Hong Kong University of Science and Technology (HKUST)
**Ph.D. in Industrial Engineering and Decision Analytics**  
*Aug 2023 – Jun 2027 (expected)*

### Nanyang Technological University (NTU)
**M.Eng. in Computer Control and Automation**  
*Aug 2022 – Jun 2023*  
GPA: 4.36/5.00

### Huazhong University of Science and Technology (HUST)
**B.Eng. in Electrical Engineering and Automation**  
*Sep 2018 – Jun 2022*  
GPA: 3.81/4.00

## Publications & Working Papers

### Spatial Matching with Heterogeneous Demand: Coordinated Key Matching Indices for Ride-Hailing
*submitted to Operations Research (with Sirui Wang, Jiheng Zhang)*

- **Model**: Two demand classes sharing drivers; endogenous pickup times via Cobb–Douglas spatial matching; two-stage customer loss (pre-match abandonment and post-match cancellation)
- **Theory**: Derived per-class Key Matching Indices (KMI) with decomposition into abandonment reduction, direct opportunity cost, and cross-market externality; optimal thresholds characterized by (ζ₁, ζ₂) = (1, 1)
- **Algorithm**: Jacobian-based coordinated Newton controller; robust convergence from arbitrary initial thresholds in numerical experiments; quantify welfare–revenue divergence under congestion

### Airline Cargo Transport Recovery
*working paper*

- Model disrupted cargo as a backlog-clearing problem with stochastic residual bellyhold capacities; allow direct vs. transfer routing with random transfer delay
- Developed time-threshold segmentation policies that tradeoff waiting cost against transfer time; analyze static vs. dynamic decision gaps

### PriceMoE: Shadow-Price-Based Dynamic Routing for MoE Serving
*working paper*

- Formulate MoE routing as network utility maximization with capacity constraints; use fluid limits for tractable analysis
- Interpret queue lengths/dual variables as shadow prices; resulting top-k routing trades off semantic score vs. congestion price

### Fluid-Affinity: Locality-Aware Token Routing for Distributed MoE Inference via Online Stochastic Control
*working paper*

- Online control for token routing across GPUs with transition (communication) costs; incorporate temporal locality via an affinity reward
- Derive Lyapunov drift-plus-penalty controller with [O(1/V), O(V)] optimality gap vs. average queue length tradeoff

### Topology-aware MoE Routing via Two-stage Re-rank
*paper-ready draft*

- Two-stage routing with a semantic candidate constraint (top-M) and topology-aware within-candidate re-ranking (communication cost + load guard)
- Targets multi-node prefill-heavy regimes to reduce cross-node bytes and stabilize tail latency; connects to OR online control viewpoints

## Research Summary (Selected Themes)

- **Online control in coupled systems**: Index/threshold policies, Jacobian-based coordination, and Lyapunov drift analysis
- **Fluid/steady-state approximations**: Tractable limits for optimization/design in spatial matching and distributed inference
- **Interpretable decision rules**: Shadow prices and decompositions that expose externalities and congestion costs

## Research & Project Experience

### CARTIN Lab, NTU
*Student Assistant — Shuttle bus data visualization system (Sentosa)*  
*Dec 2022 – Apr 2023*

- Built backend/database for real-time visualization of shuttle bus location and passenger boarding/alighting statistics

### IoT Distributed Home Monitoring & Security System
*Team Lead*  
*May 2020 – May 2021*

- Designed end-to-end IoT architecture and implemented prototypes with microcontroller-based sensing, firmware, and hardware assembly

## Technical Skills

- **Modeling/OR**: Queueing networks, spatial matching, stochastic control, constrained optimization, fluid limits
- **Programming**: Python, C/C++, MATLAB, SQL (MySQL)
- **Systems/Tools**: ARM Keil, PSIM, COMSOL

## Languages

- **Chinese**: Native (PSC Grade 2A)
- **English**: IELTS 7; CET-6 572

## Honors & Awards

- Outstanding Graduate, HUST
- Innovation of Science and Technology Scholarship
- Study Scholarship
- Interdisciplinary Contest in Modeling (ICM): Honorable Mention (H Prize)
- Excellence Award, Qiushi Cup Entrepreneurship Contest ("The Flash" EV charging service app)
- Outstanding Volunteer, Youth Reading Challenges (Forbes World Record, Asia & Pacific Area)

## Service & Additional Activities

- Student assistantship and project delivery in applied mobility analytics (NTU CARTIN Lab)
- Prior hardware/embedded prototyping experience (STM32/MCU-based control; IoT sensing)