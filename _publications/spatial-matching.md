---
title: "Spatial Matching with Heterogeneous Demand: Coordinated Key Matching Indices for Ride-Hailing"
authors:
  - "Hongfan Wu"
  - "Sirui Wang"
  - "Jiheng Zhang"
publication_status: "Submitted to Operations Research"
order: 1
excerpt: "A coordinated control framework for two demand classes sharing drivers under abandonment, cancellation, and endogenous pickup times."
---

**Authors:** Hongfan Wu, Sirui Wang, and Jiheng Zhang<br>
**Status:** Submitted to *Operations Research*

Ride-hailing platforms often serve heterogeneous demand classes from a shared driver pool. Waiting customers may abandon before matching, while long pickup times can cause cancellation after a match. Serving one class also consumes idle drivers and changes pickup efficiency for the other, so the two matching decisions cannot be optimized independently.

We formulate a joint steady-state optimization problem and derive one **Key Matching Index (KMI)** for each class. Each index separates abandonment reduction, the direct opportunity cost of consuming idle drivers, and the cross-market effect on the other class. Optimal thresholds are characterized by driving both indices to one simultaneously.

A Jacobian-based coordinated controller adjusts the two thresholds dynamically. In congested numerical benchmarks, the controller improves welfare by 19.4% and revenue by 8.7% relative to static priority while exposing the trade-off between the two customer classes.
