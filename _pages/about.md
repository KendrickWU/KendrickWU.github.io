---
permalink: /
title: "About Me"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

<section class="home-intro">
  <div class="home-intro__eyebrow">Operations Research · AI Systems · Quantitative Research</div>
  <h2>Turning complex systems into practical decisions.</h2>
  <p class="home-intro__lead">I am Wu Hongfan, a Ph.D. student at HKUST and a Quantitative Research Intern at NewDaq. I study decision-making in high-frequency markets and resource-constrained AI, mobility, and energy systems.</p>
  <div class="home-intro__links">
    <a href="/publications/">Research</a>
    <a href="/cv/">Curriculum Vitae</a>
    <a href="/writing/">Personal Writing</a>
  </div>
</section>

I am a Ph.D. student in the Department of Industrial Engineering and Decision Analytics at **Hong Kong University of Science and Technology (HKUST)**, supervised by [Prof. Jiheng Zhang](https://reijz.github.io/), Head of IEDA and Full Professor in IEDA and MATH. I received my Master of Engineering degree from Nanyang Technological University (NTU) in 2023 and my Bachelor of Engineering degree from Huazhong University of Science and Technology (HUST) in 2022.

My work sits at the intersection of **operations research, AI systems, quantitative market microstructure, and data-intensive experimentation**. Across these settings, I build decision models together with scalable experiments, explicit acceptance gates, and reproducible evidence.

<div class="home-note">
  <strong>Two modes</strong>
  <p><a href="/blog/">Blog</a> collects technical notes and project write-ups. <a href="/writing/">Writing</a> is a quieter space for travel, films, cities, and personal essays.</p>
</div>

## Current Role

### Quantitative Research Intern, NewDaq

*Shenzhen · Jun 2026–Present · Long-term internship*

My work is organized around **factor families rather than isolated signals** and focuses on upstream discovery and evidence:

- Translate market-microstructure mechanisms in active order flow, order-book liquidity, large trades, impact, reversal, and recovery into reproducible candidate families
- Evaluate candidates with Rank IC/ICIR, multiple-testing control, cross-window robustness, redundancy gates, forward out-of-fold tests, and explicit spread and execution-cost checks
- Treat negative results as research output: statistically predictive signals that fail a tradability gate remain diagnostic evidence rather than being relabeled as Alpha
- Build auditable, resumable research workflows across large high-frequency datasets using Python, Parquet, Slurm, versioned provenance, automated checks, and human approval gates

AI agents assist task orchestration, monitoring, recovery, and audit preparation. Market mechanisms, evaluation definitions, admission thresholds, and final research responsibility remain human-defined.

## Research Interests

- **LLM and MoE inference systems**: Request-level routing across quantized MoE instances, slow-timescale resident-pool placement, quality-risk estimation, capacity allocation, and interpretable shadow-price policies
- **Accelerator-cluster scheduling**: Fragmentation measurement, topology-aware placement, and continuous compaction for deep learning training jobs in shared GPU clusters
- **Stochastic service systems**: Queueing, fluid approximations, Lyapunov control, LP/KKT analysis, dynamic programming, and threshold or index policies
- **Quantitative market microstructure**: High-frequency factor families, cost-aware evidence gates, and online calibration of decisions under delayed and nonstationary feedback
- **Agentic AI evaluation**: How model capability, inference latency, tool use, and multi-round interaction jointly determine task-level value
- **Mobility and marketplace design**: Spatial matching with heterogeneous demand, customer impatience, and coordinated real-time control
- **EV charging operations**: Grid-compliant charging schedules and joint pricing-power control, informed by collaboration with a charging operator serving European customers

## Publications & Working Papers

The [Publications page](/publications/) lists five established research programs and their current manuscript status.

1. **[Spatial Matching with Heterogeneous Demand: Coordinated Key Matching Indices for Ride-Hailing](/publications/spatial-matching/)**<br>
   Submitted to *Operations Research*.

2. **[COMPASS-ABS: Reducing Fragmentation in Shared GPU Clusters for Deep Learning Training Workloads](/publications/compass-abs/)**<br>
   Submitted to ACM SIGOPS Annual Technical Conference (ATC '26); an Operations Research version is in preparation.

3. **[Dynamic MoE Routing and Precision Allocation for Quantized MoE Serving](/publications/dynamic-moe-routing/)**<br>
   NeurIPS 2026 version under revision for ICLR resubmission; an Operations Research version is nearing completion, with empirical validation ongoing.

4. **[Grid-Compliant Service-Time Scheduling for Mixed Single- and Three-Phase AC EV Charging](/publications/ev-charging-applied-energy/)**<br>
   Manuscript in preparation for *Applied Energy*; the data, algorithm, and claim boundary are being reconciled before submission.

5. **[Joint Pricing and Power Scheduling for EV Charging](/publications/ev-charging-or/)**<br>
   *Operations Research* working paper; two candidate extensions are under development.

## Current Working Ideas

These directions are research plans, not completed papers or validated performance claims:

- **Slow placement, fast routing for quantized MoE serving**: Extend the fixed-resident-pool routing study to two timescales, with slower placement decisions and faster request routing under memory, service-rate, quality, and reconfiguration-cost constraints
- **Capability, inference latency, and agent interaction**: Study when faster inference changes the value of model capability once agents plan, call tools, and interact over multiple rounds
- **Calibrate Decisions, Not Predictions**: Develop online conformal filtering that accepts, rejects, shrinks, or enhances existing alpha signals under delayed, nonstationary feedback and transaction costs

## Education

| Degree | Institution | Period | Details |
| ------ | ----------- | ------ | ------- |
| Ph.D. in Industrial Engineering and Decision Analytics | Hong Kong University of Science and Technology (HKUST) | Aug 2023 – Jun 2027 (expected) | |
| M.Eng. in Computer Control and Automation | Nanyang Technological University (NTU) | Aug 2022 – Jun 2023 | GPA: 4.36/5.00 |
| B.Eng. in Electrical Engineering and Automation | Huazhong University of Science and Technology (HUST) | Sep 2018 – Jun 2022 | GPA: 3.81/4.00 |

## Technical Skills

- **Programming and data**: Python, C/C++, MATLAB, SQL, NumPy, Pandas, scikit-learn, PyArrow/Parquet, simulation, and backtesting
- **Research systems**: Linux, Slurm, Git/GitLab, pytest, parallel and resumable workflows, provenance, and audit design
- **AI systems**: MoE routing, quantized inference, vLLM prototyping, accelerator placement and scheduling, and performance evaluation
- **Modeling and optimization**: Queueing, stochastic control, fluid approximations, constrained optimization, LP/KKT, dynamic programming, and threshold/index policies

## Contact

- **Email**: [hwucn@connect.ust.hk](mailto:hwucn@connect.ust.hk)
- **GitHub**: [github.com/KendrickWU](https://github.com/KendrickWU)
- **Location**: Hong Kong / Shenzhen

## Honors & Awards

- Outstanding Graduate, HUST
- Innovation of Science and Technology Scholarship
- Study Scholarship
- Interdisciplinary Contest in Modeling (ICM): Honorable Mention (H Prize)
- Excellence Award, Qiushi Cup Entrepreneurship Contest ("The Flash" EV charging service app)
- Outstanding Volunteer, Youth Reading Challenges (Forbes World Record, Asia & Pacific Area)

## Languages

- **Chinese**: Native (PSC Grade 2A)
- **English**: IELTS 7; CET-6 572; CET-4 598
