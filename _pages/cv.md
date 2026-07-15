---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

## Profile

HKUST IEDA Ph.D. student and Quantitative Research Intern at NewDaq. My work combines operations research with executable systems and experiments: stochastic service models, LLM/MoE inference, accelerator scheduling, spatial matching, EV charging operations, and factor-family research.

I am interested in roles spanning LLM systems and AI infrastructure, MaaS and product-facing AI, technical solutions, and quantitative research.

## Contact

- **Email**: [hwucn@connect.ust.hk](mailto:hwucn@connect.ust.hk)
- **GitHub**: [github.com/KendrickWU](https://github.com/KendrickWU)
- **Website**: [kendrickwu.github.io](https://kendrickwu.github.io/)
- **Location**: Hong Kong / Shenzhen

## Experience

### NewDaq

**Quantitative Research Intern** · Shenzhen · *2026–Present (long-term)*

- Conduct factor-family alpha research through batch experiments inside an integrated factor library, testing, backtesting, and deployment workflow
- Evaluate linear and nonlinear predictive power, then use successful batch logic to guide exploration of structurally related factors
- Help maintain and reconstruct a multidimensional parallel research framework combining statistical search coverage with economically and physically interpretable hypotheses

### CARTIN Lab, Nanyang Technological University

**Student Assistant** · Singapore · *Dec 2022–Apr 2023*

- Built backend and database support for real-time visualization of Sentosa shuttle-bus locations and passenger boarding and alighting statistics

## Education

### Hong Kong University of Science and Technology (HKUST)

**Ph.D. in Industrial Engineering and Decision Analytics**<br>
*Aug 2023–Jun 2027 (expected)*<br>
Advisor: Prof. Jiheng Zhang

### Nanyang Technological University (NTU)

**M.Eng. in Computer Control and Automation**<br>
*Aug 2022–Jun 2023* · GPA: 4.36/5.00

### Huazhong University of Science and Technology (HUST)

**B.Eng. in Electrical Engineering and Automation**<br>
*Sep 2018–Jun 2022* · GPA: 3.81/4.00

## Publications & Working Papers

### Spatial Matching with Heterogeneous Demand: Coordinated Key Matching Indices for Ride-Hailing

**Submitted to *Operations Research*** · with Sirui Wang and Jiheng Zhang

- Model two demand classes sharing drivers, with endogenous pickup times, pre-match abandonment, and post-match cancellation
- Derive per-class Key Matching Indices that expose abandonment reduction, direct opportunity cost, and cross-market externalities
- Design a Jacobian-based coordinated controller; congested benchmarks show a 19.4% welfare improvement and an 8.7% revenue improvement over static priority

### COMPASS-ABS: Reducing Fragmentation in Shared GPU Clusters for Deep Learning Training Workloads

**Submitted to ACM SIGOPS Annual Technical Conference (ATC '26); *Operations Research* version in preparation** · lead / idea initiator

- Introduce Scheduler-Induced Fragmentation (SIF), a workload-history-independent metric that isolates fragmentation caused by scheduling decisions
- Design COMPASS-ABS around the power-of-two structure of dominant GPU demands and eight-GPU node capacity; the COMPASS operators maintain a compact Anchor-Based Space throughout online scheduling
- Prove an SIF bound of 2/N under the stated workload-composition condition and evaluate the scheduler using production traces, large-scale simulation, and a physical GPU cluster

### Dynamic MoE Routing and Precision Allocation for Quantized MoE Serving

**NeurIPS 2026 version submitted; *Operations Research* version in preparation** · with Zhenghong Huang and Jiheng Zhang

- **NeurIPS version**: *Adaptive Routing for Quantized Mixture-of-Experts Serving with Theoretical Guarantee* routes requests across pre-quantized MoE instances using Fragility-Weighted Perplexity, a window-level LP, and a KKT-consistent greedy policy
- **Operations Research version**: *Dynamic Precision Allocation for Mixture-of-Experts Inference Services* models routing as stochastic service control with class-level quality-risk calibration, fluid relaxation, and capacity and quality-risk shadow prices
- Experiments on Qwen3-30B-A3B and DeepSeek-V2-Lite show up to 1.38x decode throughput against the highest-bit instance baseline without measured response-quality loss

### Grid-Compliant Service-Time Scheduling for Mixed Single- and Three-Phase AC EV Charging

**Manuscript in preparation for *IEEE Transactions on Smart Grid*** · industry collaboration

- Work with a real EV charging operator serving European customers
- Calibrate each EV's natural AC charging envelope and schedule mixed single- and three-phase sessions under aggregate and per-phase grid constraints
- Minimize congestion-induced service stretch while preserving grid compliance and work-conserving operation

### Joint Pricing and Power Scheduling for EV Charging

**Manuscript in preparation for *Operations Research*** · industry collaboration

- Study joint pricing or admission and power-allocation control for capacity-constrained EV charging operations
- Examine when commercial decisions can be separated from physical scheduling and when they must be optimized jointly
- Develop the stochastic scheduling and fluid-control extension using operational requirements and charging-session structure from the industry collaboration

## Research Themes

- **AI systems and resource allocation**: Quantized MoE routing, GPU-cluster fragmentation and scheduling, congestion pricing, and quality-capacity trade-offs
- **Stochastic service systems**: Queueing, fluid approximations, Lyapunov control, LP/KKT analysis, dynamic programming, and threshold or index policies
- **Platforms and energy operations**: Spatial matching, customer impatience, EV charging schedules, pricing, and multi-resource grid constraints
- **Quantitative experimentation**: Factor-family hypothesis generation, linear and nonlinear signal evaluation, backtesting, and scalable research workflows

## Selected Engineering Projects

### C++ Limit Order Book Matching Engine

- Built an L2 order book with price-time priority, O(1) cancellation through order-ID indexing, unit tests, and a micro-benchmark harness

### Queue-Aware MoE Routing Prototypes

- Implemented vLLM routing prototypes that inject per-expert virtual-queue penalties into expert selection and connect congestion signals with online control

### IoT Distributed Home Monitoring and Security System

- Led end-to-end architecture and prototype development across microcontroller sensing, firmware, and hardware integration

## Technical Skills

- **Programming and data**: Python, C/C++, MATLAB, SQL, NumPy, Pandas, Matplotlib, simulation, and backtesting
- **AI systems**: MoE routing, quantized inference, vLLM prototyping, accelerator scheduling, and performance evaluation
- **Modeling and optimization**: Stochastic processes, queueing, fluid approximations, constrained optimization, LP/KKT, Lyapunov control, dynamic programming, and threshold/index policies
- **Languages**: Chinese (native); English (professional working proficiency; IELTS 7)

## Honors & Awards

- Outstanding Graduate, HUST
- Interdisciplinary Contest in Modeling: Honorable Mention
- Innovation of Science and Technology Scholarship; Study Scholarship
- Excellence Award, Qiushi Cup Entrepreneurship Contest ("The Flash" EV charging service app)
