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

- **Email**: [hwucn@connect.ust.hk](mailto:hwucn@connect.ust.hk)
- **GitHub**: [https://github.com/KendrickWU](https://github.com/KendrickWU)
- **Location**: Hong Kong

## Research Interests

Distributed LLM/MoE inference systems; online stochastic control and queueing; system-constrained routing and load balancing; spatial matching and marketplace design.

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

Submitted to *Operations Research* (with Sirui Wang, Jiheng Zhang)

- **Model**: Two demand classes sharing drivers; endogenous pickup times via Cobb–Douglas spatial matching; two-stage customer loss (pre-match abandonment and post-match cancellation)
- **Theory**: Derived per-class Key Matching Indices (KMI) with decomposition into abandonment reduction, direct opportunity cost, and cross-market externality; optimal thresholds characterized by (ζ₁, ζ₂) = (1, 1)
- **Algorithm**: Jacobian-based coordinated Newton controller; robust convergence from arbitrary initial thresholds in numerical experiments; quantify welfare–revenue divergence under congestion

### Airline Cargo Transport Recovery

Working Paper

- Model disrupted cargo as a backlog-clearing problem with stochastic residual bellyhold capacities; allow direct vs. transfer routing with random transfer delay
- Developed time-threshold segmentation policies that tradeoff waiting cost against transfer time; analyze static vs. dynamic decision gaps

### PriceMoE: Shadow-Price-Based Dynamic Routing for MoE Serving

Working Paper

- Formulate MoE routing as network utility maximization with capacity constraints; use fluid limits for tractable analysis
- Interpret queue lengths/dual variables as shadow prices; resulting top-k routing trades off semantic score vs. congestion price

### Fluid-Affinity: Locality-Aware Token Routing for Distributed MoE Inference via Online Stochastic Control

Working Paper

- Online control for token routing across GPUs with transition (communication) costs; incorporate temporal locality via an affinity reward
- Derive Lyapunov drift-plus-penalty controller with [O(1/V), O(V)] optimality gap vs. average queue length tradeoff

### System-constrained MoE Routing (private details)

Working Paper

- Routing policies under hardware/deployment constraints with a focus on robustness, stability, and tail-latency
- Details (method and evaluation) are available upon request

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

### Inverted Pendulum Research on Signal and Control Integrated Experiment

Sept 2020

- Completed one-order inverted pendulum control algorithm based on STM32, and further carried out the inverted pendulum heavy load research, which adopted PI control and had good robustness

### Intelligent Face Recognition and Face Changing System

*Shinetech Software & HUST Summer Camp*  
*Aug 2020 – Sept 2020*

- Responsible for the development of face recognition system and image cropping, which were based on a dilb database of Python

## Internship Experience

### Guangdong Shangsheng New Energy Technology Co., Ltd

*Assistant to Electric Vehicle Charging Station Engineer*  
*Jul 2020 – Aug 2020*

- Familiar with the knowledge of the electric vehicle charging station and participated in problem solving work
- Tried to develop relevant APP projects to solve the problems of irregularity and dispersion in the current charging APP market, such as the project "The Flash" APP for New Energy Car Charging Service

## Technical Skills

- **Modeling/OR**: Queueing networks, spatial matching, stochastic control, constrained optimization, fluid limits
- **Programming**: Python, C/C++, MATLAB, SQL (MySQL)
- **Systems/Tools**: ARM Keil, PSIM, COMSOL, AI designer

## Languages

- **Chinese**: Native (PSC Grade 2A)
- **English**: IELTS 7; CET-6 572; CET-4 598

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
- Class Cadre in Charge of Finance Issues (Dec 2020 – present)
- Class Cadre in Charge of Students' Daily Life Affairs (Aug 2018 – present)
- Member of the Public Relations Department, HUST (Oct 2019 – Dec 2020)
