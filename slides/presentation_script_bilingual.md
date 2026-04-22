# Bilingual Presentation Script

## How To Use
This version is intentionally more conversational and easier to memorize. Each slide gives you:

1. A short Chinese script you can speak directly.
2. A short English script you can speak directly.
3. A memory cue so you can remember the main point quickly.

---

## Slide 1 — Title / 标题页

### 记忆关键词
问题，方法，结论。

### 中文讲稿
大家好，今天我要汇报的题目是 Cross-Dataset Retinal Vessel Segmentation with Lightweight Retinal-Specific Priors。

这篇论文主要想回答一个很实际的问题：如果训练集和测试集来自不同的眼底数据集，一个简单的 U-Net 怎么做才能更稳。

这篇论文最重要的结论很清楚，就是在这个任务里，最有效的改进不是换更大的模型，而是做对预处理。

### English Script
Hello everyone.

Today I will present our paper, Cross-Dataset Retinal Vessel Segmentation with Lightweight Retinal-Specific Priors.

The paper asks a practical question: if training and testing come from different retinal datasets, how can a simple U-Net generalize better?

The main conclusion is very clear: the biggest gain does not come from a larger model. It comes from the right preprocessing.

---

## Slide 2 — Why This Paper Matters / 为什么这个问题重要

### 记忆关键词
任务重要，难点是域偏移。

### 中文讲稿
这篇论文做的是视网膜血管分割，也就是输入一张眼底图像，输出血管的像素级掩码。

这个任务很重要，因为血管形态和很多眼科分析任务直接相关。如果分割不准，后面的自动分析就会受影响。

真正的难点是 domain shift，也就是域偏移。不同数据集的亮度、对比度、分辨率和图像风格都不一样，所以在一个数据集上效果好，不代表换一个数据集还会好。

### English Script
This paper studies retinal vessel segmentation, which means the input is a fundus image and the output is a pixel-level vessel mask.

This task matters because vessel morphology supports many downstream ophthalmic analyses.

The real difficulty is domain shift. Different datasets have different brightness, contrast, resolution, and image style. So a model that works well on one dataset may fail on another.

---

## Slide 3 — Experimental Setting / 实验设置

### 记忆关键词
两个数据集，双向迁移，严格测试。

### 中文讲稿
论文用了两个公开数据集，分别是 STARE 和 HRF。

实验分成两个方向：第一是在 STARE 上训练、在 HRF 上测试；第二是在 HRF 上训练、在 STARE 上测试。

这里没有做目标域微调，也没有做伪标签和测试时自适应，所以这个设置很严格，也更能说明模型真正的泛化能力。

### English Script
The paper uses two public datasets, STARE and HRF.

The evaluation has two directions: train on STARE and test on HRF, and then train on HRF and test on STARE.

There is no target-domain finetuning, no pseudo-labeling, and no test-time adaptation. So the setup is strict, and the results are easy to interpret.

---

## Slide 4 — Method Variants / 方法设置

### 记忆关键词
模型不变，只改 recipe。

### 中文讲稿
这篇论文很有意思的一点是，模型骨干一直固定为标准 U-Net。

作者不想同时改太多变量，所以重点不是比较更复杂的网络，而是比较不同轻量策略的效果。

一共有四种设置，分别是 Baseline、Green plus Equalize、Balanced Only，还有 Full Improved。

其中最关键的改动是两个预处理操作，也就是绿色通道提取和直方图均衡化。

### English Script
One important design choice is that the backbone always stays as a standard U-Net.

The authors intentionally keep the model fixed, so the focus is not on a more complicated architecture. The focus is on lightweight design choices.

There are four settings: Baseline, Green plus Equalize, Balanced Only, and Full Improved.

The most important change is the preprocessing pair: green-channel extraction and histogram equalization.

---

## Slide 5 — Main Results / 主要结果

### 记忆关键词
预处理最重要。

### 中文讲稿
这一页是整篇论文最关键的一页。

在 STARE 到 HRF 这个方向上，baseline 的 Dice 只有 0.2109，但 Green plus Equalize 直接提升到 0.5789，提升非常大。

在 HRF 到 STARE 这个方向上，baseline 是 0.4016，Balanced Only 到 0.5101，Green plus Equalize 到 0.5398，Full Improved 最后是 0.5430。

所以最重要的结论不是 Full Improved 全面领先，而是预处理本身已经解释了大部分提升。

### English Script
This is the most important slide in the paper.

On STARE to HRF, the baseline Dice is only 0.2109, but Green plus Equalize raises it to 0.5789. That is a very large improvement.

On HRF to STARE, the baseline is 0.4016, Balanced Only reaches 0.5101, Green plus Equalize reaches 0.5398, and Full Improved reaches 0.5430.

So the key message is not that Full Improved wins by a huge margin. The real message is that preprocessing already explains most of the improvement.

---

## Slide 6 — Training Dynamics / 训练过程分析

### 记忆关键词
提升更快，不一定更干净。

### 中文讲稿
从训练曲线可以看到，更强的版本通常收敛更快。

但是更快并不一定代表最终结果更好。

比如在 STARE 到 HRF 这个方向上，Full Improved 前期上升更快，但后面 Green plus Equalize 反超了它。

这说明更复杂的训练 recipe 虽然能更快提高分数，但也可能带来更多噪声。

### English Script
The training curves show that the stronger variants usually improve faster.

But faster improvement does not always mean better final predictions.

For example, on STARE to HRF, Full Improved rises earlier, but Green plus Equalize later overtakes it.

This means a more aggressive recipe can raise the score faster, but it can also introduce more noise.

---

## Slide 7 — Per-Image Robustness / 单图像鲁棒性

### 记忆关键词
不能只看平均分。

### 中文讲稿
论文还特别分析了每一张测试图像的 Dice，而不是只看平均值。

这样做很重要，因为有些方法只是让简单样本更高分，但对困难样本帮助不大。

比如在 HRF 到 STARE 这个方向上，Balanced Only 在部分简单样本上很好，但最差样本掉得非常厉害。

所以如果我们真的关心泛化能力，就不能只看平均 Dice，还要看单样本表现。

### English Script
The paper also analyzes the Dice score for each individual test image instead of only reporting the mean.

This is important because some methods mainly improve easy cases, while they do not really help hard cases.

For example, on HRF to STARE, Balanced Only looks good on some easy samples, but it drops badly on the worst cases.

So if we care about generalization, we should not only look at the average Dice. We should also look at per-image behavior.

---

## Slide 8 — Qualitative Examples / 定性案例

### 记忆关键词
有些图被救回来了，但有些图会过分割。

### 中文讲稿
这页展示的是两个定性案例。

第一个案例说明，预处理对于困难样本很有帮助，它能明显恢复更多血管结构。

第二个案例说明，更激进的 Full Improved 虽然更容易提高召回率，但也会在一些样本上产生额外误检。

所以这篇论文很严谨，它不是简单地说哪一个模型永远最好，而是说明不同方法有不同的误差模式。

### English Script
This slide shows two qualitative examples.

The first case shows that preprocessing is very helpful for difficult samples because it recovers much more vessel structure.

The second case shows that the more aggressive Full Improved setting may increase recall, but it can also create extra false positives.

So the paper is careful. It does not simply claim that one model is always best. It shows that different methods have different error patterns.

---

## Slide 9 — Final Takeaways / 最终结论

### 记忆关键词
四个结论。

### 中文讲稿
最后我总结四点。

第一，最有效的改进来源是视网膜特定预处理，也就是绿色通道和直方图均衡化。

第二，平衡损失是有帮助的，但它不是主要驱动因素。

第三，更复杂的训练 recipe 往往是通过提高 sensitivity 来提高分数，但可能会牺牲 specificity。

第四，如果我们真正关心外部泛化，就不能只看一个平均分，还要看单样本分析和定性结果。

用一句话总结，这篇论文最有价值的地方在于它说明了：在小规模医学图像项目里，先把输入处理对，往往比盲目换更大模型更重要。

### English Script
To conclude, I want to leave four main points.

First, the most effective improvement comes from retinal-specific preprocessing, especially green-channel extraction and histogram equalization.

Second, balanced loss is helpful, but it is not the main driver.

Third, more complex training recipes often improve the score by increasing sensitivity, but they may reduce specificity.

Fourth, if we truly care about external generalization, we should not only report one average score. We should also inspect per-image analysis and qualitative results.

In one sentence, the main value of this paper is that in a small medical imaging project, getting the input processing right can matter more than switching to a bigger model.

---

## Optional Opening Line / 可选开场白

### 中文
因为我是从初学者角度来理解这篇论文的，所以我会尽量用最直接的方式讲清楚它做了什么、发现了什么，以及为什么这个结论是可信的。

### English
Because I approached this paper from a beginner’s perspective, I will explain it in the most direct way possible: what it does, what it finds, and why the conclusion is credible.

---

## Optional Ending Line / 可选结束语

### 中文
谢谢大家，这就是我的汇报。如果老师有问题，我可以继续从实验设置、指标含义、结果解释和论文局限这几个方面回答。

### English
Thank you very much. That is the end of my presentation. If there are any questions, I can continue from the experimental setup, the meaning of the metrics, the interpretation of the results, and the limitations of the paper.
