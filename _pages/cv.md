---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

## Profile

HKUST IEDA Ph.D. student and Quantitative Research Intern at NewDaq. My work connects operations research with executable systems and experiments across high-frequency market microstructure, LLM/MoE inference, accelerator scheduling, spatial matching, and EV charging. I design research systems that turn domain hypotheses into reproducible candidates and layered, decision-ready evidence.

I focus on decision-making under uncertainty and resource constraints: building interpretable models, scalable evaluations, explicit acceptance gates, and evidence that remains useful when a hypothesis fails.

## Contact

- **Email**: [hwucn@connect.ust.hk](mailto:hwucn@connect.ust.hk)
- **GitHub**: [github.com/KendrickWU](https://github.com/KendrickWU)
- **Website**: [kendrickwu.github.io](https://kendrickwu.github.io/)
- **Location**: Hong Kong / Shenzhen

## Experience

### NewDaq

**Quantitative Research Intern** · Shenzhen · *Jun 2026–Present (long-term)*

- Designed a Feature Sheet–driven research compiler that freezes the research question, data capabilities, causal timing, labels, search spaces, evaluation protocols, and admission boundaries before execution
- Separated creative, interpretable market-mechanism traversal from deterministic compilation of parameters, windows, and statistical operators, avoiding outcome-driven factor redefinition
- Built dual statistical representations: vector-preserving robustification, neutralization, and orthogonalization, and collection-to-scalar summaries of event location, dispersion, tails, concentration, and temporal path
- Implemented an Evidence Router that distinguishes capability or quality failure, linear insufficiency, and hypothesis failure, with strict Forward OOF controls-only versus controls-plus-candidate nonlinear incremental tests
- Integrated family and library redundancy, nearest-neighbor residual re-evaluation, frozen lineage, content-addressed provenance, fail-closed cache reuse, Evidence Records, and human approval
- Reused the framework across active order flow, order-book liquidity, and large-trade/impact research, preserving authorized representative signals, reviewable evidence, and auditable negative results under one discipline

[Read the public framework overview →](/quant-research/systematic-alpha-research-framework/)

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

## Publications & Continuing Working Papers

### Spatial Matching with Heterogeneous Demand: Coordinated Key Matching Indices for Ride-Hailing

**Submitted to *Operations Research*** · with Sirui Wang and Jiheng Zhang

- Model two demand classes sharing drivers, with endogenous pickup times, pre-match abandonment, and post-match cancellation
- Derive per-class Key Matching Indices that expose abandonment reduction, direct opportunity cost, and cross-market externalities
- Design a Jacobian-based coordinated controller for coupled matching thresholds

### COMPASS-ABS: Reducing Fragmentation in Shared Accelerator Clusters

**Submitted to ACM ATC 2026; *Operations Research* extension in preparation** · lead / idea initiator

- Introduce Scheduler-Induced Fragmentation, a workload-history-independent metric that isolates fragmentation caused by scheduling decisions
- Maintain a compact Anchor-Based Space through online placement and compaction and prove a conditional fragmentation bound
- Evaluate with production traces, large-scale simulation, and a physical accelerator cluster

### Dynamic MoE Routing and Precision Allocation for Quantized MoE Serving

**Under review at NeurIPS 2026; *Operations Research* version nearing completion** · with Zhenghong Huang and Jiheng Zhang

- **NeurIPS 2026 version**: Route heterogeneous requests across a fixed resident pool of pre-quantized MoE instances using calibrated request-level quality estimates, a window-level LP, and a KKT-consistent routing policy
- **Operations Research version**: Formulate quantized MoE serving as stochastic service control with fluid relaxation and endogenous capacity and quality-risk shadow prices; model/theory work is near complete and result-dependent claims remain gated on empirical validation
- **Slow-placement extension**: Study resident instance mix on a slower timescale and request routing on a faster timescale under memory, service-rate, quality, and reconfiguration-cost constraints

### Grid-Compliant Service-Time Scheduling for Mixed Single- and Three-Phase AC EV Charging

**Manuscript in preparation for *Applied Energy*** · industry collaboration

- Work with an EV charging operator serving European customers and calibrate each EV's natural AC charging envelope
- Schedule mixed single- and three-phase sessions under aggregate and per-phase grid constraints, using congestion-induced service stretch as the main operational objective
- Reconcile the available data, algorithm design, comparator set, and manuscript claims before submission; claims not supported by the present evidence remain out of scope

### Joint Pricing and Power Scheduling for EV Charging

***Operations Research* working paper** · industry collaboration

- Study pricing or admission and power-allocation control for capacity-constrained EV charging operations
- Examine when commercial decisions can be separated from physical scheduling and when congestion, service quality, and grid constraints require joint optimization
- Develop two candidate extensions beyond the current service-time model; both remain at the formulation and evidence-design stage

## Exploratory Research Directions

These are current working ideas rather than completed manuscripts:

- **Capability, inference latency, and agent interaction**: characterize when faster inference changes the value of model capability in multi-round planning, tool use, and human-agent interaction
- **Calibrate Decisions, Not Predictions: Online Conformal Filtering under Delayed and Nonstationary Feedback**: accept, reject, shrink, or enhance existing alpha signals using uncertainty, cost, market state, and recent reliability; this is a falsifiable research design, not a validated return result

## Selected Engineering Projects

### C++ Limit Order Book Matching Engine

- Built an L2 order book with price-time priority, O(1) cancellation through order-ID indexing, unit tests, and a micro-benchmark harness

### Queue-Aware MoE Routing Prototypes

- Implemented vLLM routing prototypes that inject per-expert virtual-queue penalties into expert selection and connect measurable congestion signals with online control

### IoT Distributed Home Monitoring and Security System

- Led end-to-end architecture and prototype development across microcontroller sensing, firmware, and hardware integration

## Technical Skills

- **Programming and data**: Python, C/C++, MATLAB, SQL, NumPy, Pandas, scikit-learn, PyArrow/Parquet, simulation, and backtesting
- **Research systems**: Linux, Slurm, Git/GitLab, pytest, parallel and resumable workflows, provenance, and audit design
- **AI systems**: MoE routing, quantized inference, vLLM prototyping, accelerator placement and scheduling, and performance evaluation
- **Modeling and optimization**: Stochastic processes, queueing, fluid approximations, constrained optimization, LP/KKT, Lyapunov control, dynamic programming, and threshold/index policies
- **Languages**: Chinese (native); English (professional working proficiency; IELTS 7)

## Honors & Awards

- Outstanding Graduate, HUST
- Interdisciplinary Contest in Modeling: Honorable Mention
- Innovation of Science and Technology Scholarship; Study Scholarship
- Excellence Award, Qiushi Cup Entrepreneurship Contest ("The Flash" EV charging service app)
