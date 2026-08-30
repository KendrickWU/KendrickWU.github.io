---
layout: post
title: "From Market Hypotheses to Auditable Evidence: Building a Systematic Alpha Research Framework"
date: 2026-08-24 00:00:00 +0800
permalink: /quant-research/systematic-alpha-research-framework/
categories: [Quantitative Research, Market Microstructure, Research Systems]
tags: [Systematic Alpha, Feature Sheet, Evidence Router, Forward OOF, Research Governance]
excerpt: "A Feature Sheet–driven research compiler that turns interpretable market-microstructure hypotheses into frozen candidates, routes them through classical and nonlinear incremental evidence, and preserves provenance and human admission gates."
---

> **Public boundary.** This overview describes the methodological architecture only. It omits proprietary factor definitions, parameter choices, admission thresholds, internal counts, and performance results.

The most important part of my quantitative-research internship was not a single factor or a larger search. It was turning factor research from a sequence of researcher-specific manual decisions into an executable, auditable system that can be reused across market-microstructure themes.

I think of the result as a **factor-research compiler**. A researcher declares the question and its admissible evidence; the framework compiles that declaration into interpretable physical hypotheses, parameterized statistical representations, frozen executable candidates, and finally decision-ready Evidence Records. No individual IC or model result is allowed to decide admission by itself.

```text
Feature Sheet / research contract
        ↓
Interpretable physical hypothesis space
        ↓
Parameter and window compilation
        ↓
Dual statistical representation
├─ Vector-preserving: robustification / neutralization / orthogonalization
└─ Collection-to-scalar: location / dispersion / tails / concentration / path
        ↓
Candidate freeze, lineage, and causal checks
        ↓
Evidence Router
├─ Capability or quality failure  → diagnose / repair
├─ Classical evidence passes     → redundancy review
└─ Quality passes, linear fails   → nonlinear incremental rescue
        ↓
Within-family compression + library redundancy + residual re-evaluation
        ↓
Evidence Record + human admission
```

## Feature Sheet as an executable research contract

The Feature Sheet is not a form to complete after a result is found. It is a declaration language for the research process.

Before computation begins, it freezes the research question, available data capabilities, economic mechanism, causal timestamp, target and alternate labels, admissible search space, evaluation protocol, admission criteria, and delivery boundary. This makes later results interpretable: a candidate can always be traced back to the exact question and evidence standard that produced it.

Freezing the contract also prevents a common source of research bias. A definition cannot be revised merely because an early backtest is disappointing, and a convenient transform cannot be retroactively described as a new economic mechanism.

## Mechanism-first search, not transform-first search

The first traversal is physical. It expands interpretable ideas about market behavior: absorption and support, liquidity replenishment and depletion, bursts and impacts, reversals and recovery, resilience, counterparty response, and state-dependent price paths.

Creative models and researchers are useful here, but their freedom is deliberately bounded. A proposed mechanism must pass checks for field availability, causal timing, semantic duplication, and information leakage, and it enters computation only after human confirmation.

Once the physical hypothesis is accepted, parameters, windows, and statistical operators are compiled from predeclared, versioned configuration. The creative layer proposes meaning; deterministic code produces the search space. This division preserves research intelligence without turning the process into outcome-guided permutation.

## Dual statistical traversal: searching representations

A single physical mechanism can have multiple statistically meaningful representations. The framework explores them through two complementary paths.

### Vector-preserving traversal

This path preserves the original observation structure and physical interpretation. Robustification, cross-sectional ranking, neutralization, and orthogonalization test whether the same mechanism remains present after nuisance variation or nearby exposures are removed.

The question is not “which transform wins?” It is whether the underlying mechanism survives reasonable alternative expressions.

### Collection-to-scalar traversal

Many high-frequency objects are event collections rather than fixed-length vectors: large trades, trade clusters, impact episodes, and recovery paths. This path compiles a variable-length event set into comparable scalar summaries from several views:

- location and relative position;
- dispersion and robust shape;
- tails and extremes;
- concentration and dominance;
- temporal ordering and path structure.

Every operator has explicit capability requirements, parameters, versioning, and tests. Statistical traversal therefore expands representation while remaining deterministic and auditable.

## Evidence Router: separating different kinds of failure

Weak linear evidence does not necessarily mean that a hypothesis is meaningless. Conversely, a statistically attractive result is not useful if the data or causal definition is invalid.

The Evidence Router separates three questions:

1. **Can the candidate be evaluated?** Data capability, coverage, missing-value semantics, causal timing, and computation quality must pass first.
2. **Does it have classical predictive evidence?** Staged samples, alternate labels, direction stability, robustness, and multiple-testing control form the classical evidence record.
3. **Could the candidate contain incremental nonlinear information?** Only quality-qualified candidates that fail the linear screen can enter nonlinear rescue.

This separation changes how negative results are interpreted. Some candidates need repair, some are evaluable but linearly insufficient, and some genuinely fail the predictive hypothesis. Those are different research outcomes and should not be collapsed into one pass/fail flag.

## Nonlinear rescue as a controlled incremental-information test

The nonlinear module is not permission to switch to a more expressive model until something passes.

For a quality-qualified linear reject, the framework compares a fixed controls-only model, a Ridge linear baseline, and a controls-plus-candidate shallow gradient-boosting model under strict time-forward out-of-fold prediction. The candidate is judged by its incremental information beyond the controls, using incremental Rank IC, permutation testing, multiple-testing correction, and cross-stage consistency.

The question is therefore not “can a tree model fit the label?” It is “does this candidate add independent information that the controls and linear baseline cannot express?” The purpose of rescue is diagnosis, not manufacturing additional survivors.

## Evidence beyond a single metric

An Evidence Record combines:

- the physical interpretation and exact computation rule;
- data and causal-quality checks;
- staged classical evaluation and alternate labels;
- nonlinear incremental evidence where applicable;
- within-family and full-library redundancy;
- nearest-neighbor residual retention;
- stability, limitations, and negative findings.

This evidence can support different outcomes: independent alpha, redundant alpha, an event-state descriptor, a nonlinear candidate, a stage-specific signal, a feature requiring repair, or a hypothesis that should be abandoned. Negative results and capability boundaries remain part of the record rather than disappearing from the final report.

## Redundancy is not incremental information

Low pairwise correlation is not proof of unique information, and high correlation is not an automatic reason to discard a candidate.

The framework first selects representatives within a mechanism family, then checks redundancy against the broader library. A highly related candidate can be residualized against its nearest library neighbor and rerun through the relevant classical and nonlinear evidence. Incremental information must survive after the nearby exposure is removed; correlation alone cannot substitute for that test.

## Lineage, provenance, and human admission

Every frozen candidate carries lineage back to the Feature Sheet, physical mechanism, configuration, operator versions, and source fields. Inputs, code, dependencies, upstream results, and artifacts are bound through content hashes. Cache reuse fails closed when those bindings do not match.

Execution is parallel and resumable, and AI agents assist hypothesis organization, task orchestration, monitoring, failure recovery, and audit preparation. They do not define the economic mechanism, change evaluation rules after seeing results, or approve a factor for admission.

The final boundary remains human. Evidence informs the decision; it does not silently replace research responsibility.

## From framework to systematic research output

Later studies of active order flow, order-book liquidity, large trades, impact, reversal, and recovery are best understood as transfer tests of the same framework across different data views and market mechanisms.

They demonstrate that a new theme can be converted into a structured candidate space and evaluated under one causal, statistical, and governance discipline. Representative signals, authorized outcomes, reviewable evidence, and well-explained negative results are outputs of the framework—not its only value.

The scarce capability is therefore not producing a longer factor list. It is building a research system that knows what may be searched, what counts as evidence, what must be rejected, and where human judgment must remain.
