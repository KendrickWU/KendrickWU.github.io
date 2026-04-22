# Bilingual Q&A For Presentation And Defense

## How To Use
This document is designed for direct oral use during Q&A.

For each question, you get:

1. A short Chinese answer for quick response.
2. A short English answer for quick response.
3. A follow-up note when the teacher asks for more detail.

---

## Q1. 这篇论文的核心贡献是什么？
### 中文速答
这篇论文最大的贡献不是提出了一个更复杂的新模型，而是通过严格的跨数据集实验说明：在小数据视网膜血管分割里，视网膜特定预处理比更复杂训练技巧更重要。

### English Answer
The main contribution is not a more complex new model. The main contribution is a strict cross-dataset study showing that in small-data retinal vessel segmentation, retinal-specific preprocessing matters more than heavier training tricks.

### 追问补充
如果老师继续追问，可以补一句：论文的价值在于结论很清楚，而且实验设计比较干净，因为 backbone 一直固定为 U-Net。

---

## Q2. 为什么作者要固定使用 U-Net，而不是换更强模型？
### 中文速答
因为作者想控制变量。如果同时换 backbone 和训练策略，就很难判断提升到底来自哪里。固定 U-Net 以后，就能更清楚地比较预处理、损失函数和训练 recipe 的真实作用。

### English Answer
Because the authors want to control variables. If they change both the backbone and the training strategy, it becomes hard to know where the gain really comes from. By keeping U-Net fixed, they can isolate the effect of preprocessing, loss design, and training recipe.

### 追问补充
你也可以补充说，这样的设计更适合课程项目，因为重点是分析而不是单纯追求 SOTA。

---

## Q3. 什么是 cross-dataset setting？为什么它重要？
### 中文速答
cross-dataset setting 指的是在一个数据集上训练，在另一个数据集上直接测试。它重要是因为这更接近真实应用场景，能检验模型在域偏移下是否还可靠。

### English Answer
Cross-dataset setting means training on one dataset and directly testing on another dataset. It matters because it is closer to real deployment and shows whether the model remains reliable under domain shift.

### 追问补充
如果老师追问 domain shift，你可以接着说，就是设备、分辨率、亮度、对比度和图像风格的差异。

---

## Q4. 为什么绿色通道会有帮助？
### 中文速答
因为视网膜血管在绿色通道里通常更清楚，血管和背景的对比度更高，所以模型更容易把血管结构分出来。

### English Answer
Because retinal vessels are usually more visible in the green channel. The vessel-background contrast is stronger there, so the model can separate vessel structures more easily.

### 追问补充
可以再补一句：这说明提升并不一定要靠更复杂模型，有时更符合领域知识的输入表示更重要。

---

## Q5. 为什么直方图均衡化会有帮助？
### 中文速答
它的作用是增强对比度，并且让不同数据集之间的亮度和对比度差异变小，所以模型面对的外观偏移没有那么严重。

### English Answer
Histogram equalization improves contrast and reduces appearance differences across datasets. That makes the domain shift less severe for the model.

### 追问补充
如果老师问得更细，可以说它不是在改变血管结构，而是在让结构更容易被模型看到。

---

## Q6. 为什么 balanced loss 有帮助，但不是最核心的贡献？
### 中文速答
因为血管像素远少于背景像素，所以 balanced loss 能减轻背景偏置，提高召回率。但从结果看，它有帮助，却没有预处理带来的提升那么大，所以它是辅助因素，不是主导因素。

### English Answer
Balanced loss helps because vessel pixels are much fewer than background pixels, so it reduces background bias and improves recall. But the results show that its effect is smaller than the effect of preprocessing, so it is helpful but not the main driver.

### 追问补充
可以顺手提一句：balanced loss 往往会提高 sensitivity，但有时也会增加 false positives。

---

## Q7. 为什么 Full Improved 不是在所有情况下都最好？
### 中文速答
因为更完整的 recipe 往往把模型推向更高 sensitivity，也就是更积极地找血管。这在某些样本上有帮助，但也可能带来更多误检，所以不是每张图都更干净。

### English Answer
Because the full recipe tends to push the model toward higher sensitivity, meaning it predicts vessels more aggressively. That helps in some cases, but it can also create more false positives, so it is not cleaner on every image.

### 追问补充
如果老师追问，你可以指出论文的一个重要观点：best average model is not always best image by image.

---

## Q8. 这篇论文最关键的数字结果是什么？
### 中文速答
最关键的结果是 STARE 到 HRF 方向上，baseline 的 Dice 只有 0.2109，而 Green plus Equalize 直接提升到 0.5789。这说明预处理带来的提升非常大。

### English Answer
The most important result is on STARE to HRF. The baseline Dice is only 0.2109, while Green plus Equalize raises it to 0.5789. This shows that preprocessing provides a very large gain.

### 追问补充
你还可以补充第二个结果：在 HRF 到 STARE 上，Full Improved 是 0.5430，但 Green plus Equalize 已经到 0.5398，差距很小。

---

## Q9. 为什么论文要做 per-image analysis？
### 中文速答
因为只看平均分可能会误导。平均分高，不一定代表对困难样本更好。per-image analysis 能看出模型到底是在帮助难样本，还是只是在容易样本上刷高分。

### English Answer
Because the average score can be misleading. A higher mean does not necessarily mean better performance on difficult cases. Per-image analysis shows whether a model really helps hard samples or mainly boosts easy ones.

### 追问补充
这是这篇论文很严谨的一点，因为它不只报告平均值，还报告 lower-tail behavior。

---

## Q10. 这篇论文的创新性是不是不够强？
### 中文速答
如果从“提出全新架构”这个角度看，创新性确实不是这篇论文的重点。但它的价值在于提出了一个清楚、可信、可解释的实验结论，说明在这个任务里真正有效的是什么。

### English Answer
If we define novelty only as proposing a brand-new architecture, then that is not the main focus of this paper. Its value lies in producing a clear, credible, and interpretable experimental conclusion about what really works in this task.

### 追问补充
这是很适合课程项目的贡献类型，因为它强调 methodological clarity 而不是堆叠复杂模块。

---

## Q11. 这篇论文有哪些局限性？
### 中文速答
主要有四点：只用了两个数据集，数据规模比较小；baseline 和后续版本训练轮数不完全一致；没有加入更多第三方数据集；也没有比较更高分辨率和更多 backbone。

### English Answer
There are four main limitations: only two datasets are used, the benchmark is small, the baseline and later variants do not have exactly the same training budget, and the paper does not compare more third-party datasets, higher resolutions, or more backbones.

### 追问补充
你可以接着说，所以论文的结论是谨慎的，它没有声称 universal state of the art。

---

## Q12. 如果继续做下去，你会怎么扩展这篇论文？
### 中文速答
我会先做三件事：第一，加入第三个公开数据集，例如 CHASE_DB1；第二，在统一训练预算下重新比较所有方法；第三，测试更高分辨率输入，看看细血管是否能进一步恢复。

### English Answer
I would extend the paper in three ways. First, I would add a third public dataset such as CHASE_DB1. Second, I would compare all variants under exactly the same training budget. Third, I would test higher-resolution inputs to see whether fine vessels can be recovered better.

### 追问补充
如果老师想听更深入的方向，还可以加 domain adaptation 或 stronger encoder 的 controlled comparison。

---

## Q13. 为什么 Dice 是最重要的指标？
### 中文速答
因为这是分割任务里最常用的重叠指标，它直接衡量预测区域和真实区域重叠得有多好。对于前景区域比较小的任务，Dice 也通常比单纯 accuracy 更有意义。

### English Answer
Because Dice is the most common overlap metric for segmentation. It directly measures how well the predicted region overlaps with the ground truth. For tasks with a relatively small foreground area, Dice is usually more informative than plain accuracy.

### 追问补充
如果老师继续问 IoU，你可以说 IoU 也是重叠指标，只是比 Dice 更严格一点。

---

## Q14. sensitivity 和 specificity 在这里怎么理解？
### 中文速答
sensitivity 表示真实血管里有多少被找出来，越高说明漏检越少；specificity 表示背景里有多少被正确识别为背景，越高说明误检越少。

### English Answer
Sensitivity tells us how many true vessel pixels are recovered, so a higher value means fewer misses. Specificity tells us how many background pixels are correctly rejected, so a higher value means fewer false positives.

### 追问补充
你可以顺着说，这篇论文里 Full Improved 往往提高 sensitivity，但有时会降低 specificity。

---

## Q15. 这篇论文最适合用一句话怎么总结？
### 中文速答
一句话总结就是：在小数据跨数据集的视网膜血管分割里，做对预处理，比盲目堆更复杂的训练技巧更重要。

### English Answer
In one sentence: for small-data cross-dataset retinal vessel segmentation, getting the preprocessing right matters more than stacking heavier training tricks.

### 追问补充
这个回答很适合作为最后的收束句，也很适合在答辩最后再次强调。

---

## Q16. 如果老师问“你个人最认同论文哪一点”，你可以怎么答？
### 中文速答
我最认同的是它的实验逻辑。它没有急着换更复杂的模型，而是先用严格对照实验找出真正有效的因素。我觉得这种研究思路很扎实。

### English Answer
What I appreciate most is the experimental logic. Instead of immediately switching to a more complex model, the paper first uses controlled experiments to identify what really works. I think that is a very solid research approach.

### 追问补充
这类回答比较自然，也能体现你对论文方法论层面的理解。

---

## Q17. 如果老师问“这篇论文有没有实际应用意义”，你可以怎么答？
### 中文速答
有。它说明在真实部署前，提升外部泛化比单一数据集上的高分更重要。对医学图像来说，这种结论比单纯追求更复杂模型更有现实意义。

### English Answer
Yes. The paper shows that before real deployment, improving external generalization is more important than achieving a high score on a single dataset. For medical imaging, that insight is practically meaningful.

### 追问补充
如果继续追问，可以补充说，这种结论对小样本医学项目尤其重要，因为它能帮助我们把精力放在最有效的部分。

---

## Final Rescue Sentence / 万能收束句
### 中文
如果我用最简短的话回答，我会说：这篇论文最重要的不是模型更复杂，而是它证明了在跨数据集场景下，预处理才是主要提升来源。

### English
If I answer in the shortest possible way, I would say: the most important point of this paper is not a more complex model, but the evidence that preprocessing is the main source of improvement under cross-dataset transfer.
