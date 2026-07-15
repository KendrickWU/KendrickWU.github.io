---
title: "NPU Cluster Scheduling with Net-Gain Preemption"
publication_status: "Submitted to ACM ATC; Operations Research version in preparation"
order: 2
excerpt: "Placement and preemption policies for multi-pod accelerator jobs that act only when expected waiting-time reduction exceeds migration overhead."
---

**Status:** Submitted to ACM ATC; *Operations Research* version in preparation<br>
**Role:** Lead / idea initiator

This work studies cluster-level placement and migration or preemption for multi-pod jobs under resource fragmentation, workload churn, and heterogeneous pod-size requirements.

The proposed **Net-Gain** principle estimates expected waiting time with bitmask dynamic programming and triggers migration only when the expected reduction in delay exceeds cool-down and recovery overhead. Degree-of-freedom and expected-capacity objectives then select among feasible placement and migration plans.

Experiments span 99 random seeds, six placement policies, and four preemption policies. Net-Gain reduces average job delay by about 26–32% relative to no-preemption baselines, depending on the placement policy.
