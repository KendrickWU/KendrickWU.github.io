---
layout: post
title: "Cross-Dataset Retinal Vessel Segmentation Presentation"
date: 2026-04-22 10:00:00 +0800
permalink: /presentation/retinal-vessel-segmentation/
categories: [Presentation, Medical Imaging, Domain Generalization]
tags: [Retinal Vessel Segmentation, U-Net, STARE, HRF]
excerpt: "Interactive class presentation on cross-dataset retinal vessel segmentation, showing that retinal-specific preprocessing drives most of the transfer gain."
---

This post is the blog entry for my COMP5423 final project presentation on cross-dataset retinal vessel segmentation.

The full interactive page is here:

[Click to open the full presentation]({{ '/slides/PPT.html' | relative_url }})

Quick summary:

- Problem: retinal vessel segmentation performance drops sharply when training and testing come from different datasets.
- Main result: green-channel extraction plus histogram equalization explains most of the transfer improvement.
- Takeaway: before changing the backbone, align the retinal input.
