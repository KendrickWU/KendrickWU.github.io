---
layout: post
title: "PriceMoE Presentation: AI Traffic Control"
date: 2026-03-23 10:00:00 +0800
categories: [Presentation, MoE, AI Systems]
tags: [PriceMoE, Queueing Theory, MoE Routing]
excerpt: "Interactive class presentation page for PriceMoE: how queue-aware routing reduces MoE hotspots."
---

This post is the blog entry for my PriceMoE class presentation page.

The full interactive page is here:

<p>
  <a href="{{ '/price-moe-presentation.html' | relative_url }}" style="display:inline-block;padding:10px 16px;border-radius:999px;border:1px solid #ddd;text-decoration:none;">
    Click to open the full PriceMoE presentation
  </a>
</p>

Quick summary:

- Problem: one hotspot expert can slow the whole MoE batch.
- Idea: add a queue-based congestion price during routing.
- Outcome: better load balance with small throughput trade-off.
