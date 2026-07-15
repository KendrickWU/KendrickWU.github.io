---
title: "COMPASS-ABS: Reducing Fragmentation in Shared GPU Clusters for Deep Learning Training Workloads"
publication_status: "Submitted to ACM SIGOPS Annual Technical Conference (ATC '26); Operations Research version in preparation"
order: 2
excerpt: "A workload-history-independent fragmentation metric and online scheduler that continuously maintains compact resource states in shared GPU clusters."
redirect_from:
  - /publications/npu-scheduling/
---

**Status:** Submitted to ACM SIGOPS Annual Technical Conference (ATC '26); *Operations Research* version in preparation<br>
**Role:** Lead / idea initiator

Shared GPU clusters can remain underutilized even when jobs are waiting because each deep learning training instance must fit within a single node. This placement constraint fragments otherwise sufficient aggregate GPU capacity and increases job turnaround time.

We introduce **Scheduler-Induced Fragmentation (SIF)**, a partial-node-based metric that isolates fragmentation caused by previous scheduling decisions without requiring a historical workload distribution.

The **COMPASS-ABS** scheduler uses the alignment between dominant power-of-two GPU demands and eight-GPU node capacity. Its Place, Remove, and Compact operators confine online cluster states to a compact Anchor-Based Space. Under the paper's workload-composition condition, COMPASS-ABS guarantees that SIF is bounded by 2/N at all times.

Evaluation combines large-scale simulations on production traces with a physical-cluster deployment. The experiments assess fragmentation, GPU utilization, and deep learning training job completion time under varying cluster load, workload composition, and migration cost.
