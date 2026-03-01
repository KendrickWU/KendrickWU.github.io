# PriceMoE：基于影子价格的 MoE 动态路由——最终数学模型 (v2.1)

> **设计原则**：本模型以 **Management Science 期刊发表**为首要目标，在数学严谨性与证明复杂度之间寻求最优平衡。相比 v2.0，本版本 (1) 采用经典流体极限替代 Halfin-Whitt 以降低证明工作量；(2) 移除通信约束简化模型；(3) 聚焦 OR 标准术语与证明框架。

---

## 0. 核心设计决策与权衡

### 0.1 模型定位：为什么选择这个设计

| 维度 | v2.0 选择 | **v2.1 选择** | 理由 |
| ---- | ---- | ---- | ---- |
| **Scaling 机制** | Halfin-Whitt QED | **经典流体极限** | QED 需额外证明扩散收敛与状态空间坍缩，工作量翻倍；经典流体极限文献成熟、证明链更短 |
| **约束重点** | 计算+通信+质量 | **计算+质量** | 通信约束实现难度高（需 kernel 级修改）、对 congestion/load skew 改善有限、增加理论复杂度 |
| **路由形式** | 软 Softmax + Top-k | **硬 Top-k（理论） / 可选 Softmax（工程）** | 顶刊理论部分通常用确定性最优；熵正则化作为"唯一性修复"在备注中保留 |
| **价格来源** | 队列 + 资源对偶 | **队列（Backpressure）为主** | 简化对偶结构，突出"队列长度即边际拥塞成本"的经济学解释 |
| **术语体系** | CS 风格（Lyapunov, Drift-plus-Penalty） | **OR 风格（Stochastic Processing Network, Fluid Limit, Dual Price）** | 符合 MS 审稿人预期 |

### 0.2 Halfin-Whitt vs 经典流体极限：证明复杂度分析

| 维度 | 经典流体极限 | Halfin-Whitt QED |
| ---- | ---- | ---- |
| **缩放方式** | $\bar{Q}^{(n)}(t) = Q^{(n)}(nt)/n$ | $\bar{Q}^{(n)}(t) = Q^{(n)}(nt)/\sqrt{n}$ |
| **极限对象** | 确定性 ODE | 扩散过程 (Ornstein-Uhlenbeck) |
| **收敛证明** | FSLLN + 紧性 + 极限识别 | FCLT + 状态空间坍缩 + 扩散近似 |
| **核心引用** | Dai (1995), Chen & Yao | Halfin & Whitt (1981), Stolyar (2004) |
| **证明工作量** | ⭐⭐ 中等 | ⭐⭐⭐⭐ 高 |
| **额外收益** | 稳定性 + 吞吐量最优 | + 精细延迟刻画 (P99) |
| **MS 必要性** | ✅ 足够 | ⚠️ 非必须（可作为扩展） |

**结论**：对于首篇 MS 投稿，经典流体极限足以支撑核心贡献；Halfin-Whitt 可在后续期刊或会议版本中扩展。

### 0.3 与 vLLM 架构的映射（实现复杂度实况）

根据 `Documentation/PriceMoE Implementation Feasibility Analysis.md`：

| 数学概念 | vLLM 对应 | 原生支持 | 实现难度 | 工作量 |
| ---- | ---- | ---- | ---- | ---- |
| Token 到达率 $\lambda$ | `Scheduler.schedule()` 输出 | ✅ | 低 | 1 周 |
| 队列长度 $q_e(t)$ | **需新增**：batch 内路由统计 | ❌ | 中-高 | 2-3 周 |
| 路由修正 | `FusedMoE.select_experts()` | ⚠️ 需扩展 | 中 | 2-3 周 |
| 跨 GPU 通信监控 | EP All-to-All 内部 | ❌ | **极高** | 4+ 周 |
| 专家激活状态 | 无原生概念 | ❌ | 高 | 3-4 周 |

**关键发现**：
- vLLM 调度是 **per-request** 而非 **per-expert**，需定义"虚拟队列"
- 通信约束需修改 kernel 级代码，工作量与收益不成比例
- **MVP 估计**：仅实现拥塞惩罚约需 **4-6 周**

---

## 1. 随机处理网络模型（Stochastic Processing Network）

### 1.1 系统原语

**定义 1.1（系统组件）**
- **专家集合**：$\mathcal{E} = \{1, \ldots, E\}$，每专家具有服务速率 $\mu_e > 0$
- **GPU 集合**：$\mathcal{K} = \{1, \ldots, K\}$，每 GPU 具有计算容量 $C_k > 0$
- **专家放置**：$L: \mathcal{E} \to \mathcal{K}$，其中 $\mathcal{E}_k := L^{-1}(k)$
- **Token 类别**：$\mathcal{C} = \{1, \ldots, C\}$（简化模型取 $C = 1$）

**定义 1.2（允许专家集与质量约束）**
对每个 token 类别 $c$，gating 网络输出分数向量 $\mathbf{s}_c = (s_{c,e})_{e \in \mathcal{E}}$。定义允许集合：
$$
\mathcal{A}_c := \text{Top}_k(\mathbf{s}_c) = \{e_1, \ldots, e_k : s_{c,e_1} \geq \cdots \geq s_{c,e_k}\}
$$

### 1.2 随机动力学

**定义 1.3（到达过程）**
类别 $c$ 的到达 $\{A_c(t)\}_{t \geq 0}$ 为更新过程，满足泛函强大数律（FSLLN）：
$$
\frac{1}{n} A_c(nt) \xrightarrow{\text{a.s.}} \lambda_c t \quad \text{as } n \to \infty
$$
其中 $\lambda_c > 0$ 为平均到达率。

**定义 1.4（路由决策）**
定义路由变量 $x_{c,e}(t) \in [0,1]$，满足：
$$
\sum_{e \in \mathcal{A}_c} x_{c,e}(t) = 1, \quad x_{c,e}(t) = 0 \text{ if } e \notin \mathcal{A}_c
$$

诱导专家 $e$ 的瞬时到达率：
$$
\lambda_e(t) := \sum_{c \in \mathcal{C}} \lambda_c \cdot x_{c,e}(t)
$$

**定义 1.5（队列过程）**
令 $Q_e(t) \in \mathbb{Z}_{\geq 0}$ 为专家 $e$ 在时刻 $t$ 的队列长度。动力学方程：
$$
Q_e(t) = Q_e(0) + A_e(t) - S_e(t)
$$
其中：
- $A_e(t) = \sum_c A_{c,e}(t)$：累计到达
- $S_e(t)$：累计服务完成

**定义 1.6（服务过程）**
采用时间变更表示（time-change representation）：
$$
S_e(t) = N_e\left( \mu_e \int_0^t \mathbf{1}\{Q_e(s) > 0\} \, ds \right)
$$
其中 $N_e(\cdot)$ 为满足 FSLLN 的更新过程。

### 1.3 容量约束

**定义 1.7（计算容量约束）**
稳态下，GPU $k$ 的总服务负载不超过容量：
$$
\sum_{e \in \mathcal{E}_k} \frac{\lambda_e}{\mu_e} \leq C_k, \quad \forall k \in \mathcal{K}
\tag{C1}
$$

**定义 1.8（容量域）**
定义系统的稳定容量域：
$$
\mathcal{C} := \left\{ \boldsymbol{\lambda} \in \mathbb{R}_+^E : \exists \text{ routing } x \text{ s.t. } \sum_{e \in \mathcal{E}_k} \frac{\lambda_e(x)}{\mu_e} \leq C_k, \forall k \right\}
$$

---

## 2. 流体缩放与极限模型

### 2.1 流体缩放

**定义 2.1（缩放过程）**
引入规模参数 $n \to \infty$，定义流体缩放变量：
$$
\bar{Q}_e^{(n)}(t) := \frac{1}{n} Q_e^{(n)}(nt), \quad \bar{\lambda}_e^{(n)}(t) := \frac{1}{n} \lambda_e^{(n)}(nt)
$$

### 2.2 流体极限定理

**定理 2.1（流体极限存在性）**
在以下假设下：
1. **(A1)** 到达过程满足 FSLLN：$\frac{1}{n} A^{(n)}(nt) \xrightarrow{\text{a.s.}} \lambda t$
2. **(A2)** 服务过程满足 FSLLN
3. **(A3)** 初始状态满足 $\bar{Q}^{(n)}(0) \to q(0)$
4. **(A4)** 路由策略 $x^{(n)}(\cdot)$ 对 $q$ 局部 Lipschitz

则当 $n \to \infty$，缩放过程 $\bar{Q}^{(n)}(\cdot)$ 几乎必然收敛到确定性极限 $q(\cdot)$，满足常微分方程：
$$
\dot{q}_e(t) = \lambda_e(t) - \mu_e \cdot \mathbf{1}\{q_e(t) > 0\}
\tag{F-ODE}
$$

**证明要点**：
1. **紧性**：利用 Arzelà-Ascoli 定理，证明 $\{\bar{Q}^{(n)}\}$ 在 $D[0,T]$ 中相对紧
2. **极限识别**：通过鞅分解和 FSLLN，证明极限点满足 (F-ODE)
3. **唯一性**：Picard-Lindelöf 定理保证 ODE 解唯一

> **参考文献**：Dai, J.G. (1995). On positive Harris recurrence of multiclass queueing networks: A unified approach via fluid limit models. *Annals of Applied Probability*.

### 2.3 反射形式

**定义 2.2（Skorokhod 问题）**
为处理 $q_e \geq 0$ 的边界约束，采用反射形式：存在非减过程 $y_e(t)$，使得：
$$
q_e(t) = q_e(0) + \int_0^t (\lambda_e(s) - \mu_e) \, ds + y_e(t)
$$
满足互补条件：
$$
q_e(t) \geq 0, \quad y_e(0) = 0, \quad \int_0^t q_e(s) \, dy_e(s) = 0
$$

**定理 2.2（反射 ODE 唯一解）**
在假设 (A4) 下，反射 ODE 在正交象限 $\mathbb{R}_+^E$ 上存在唯一解。

---

## 3. 稳态优化与对偶价格

### 3.1 网络效用最大化问题

**定义 3.1（稳态优化问题）**
在稳态（$\dot{q}_e = 0$）下，考虑网络效用最大化（Network Utility Maximization, NUM）：
$$
\begin{aligned}
\max_{x} \quad & \sum_{c \in \mathcal{C}} \sum_{e \in \mathcal{E}} \lambda_c \cdot u_{c,e} \cdot x_{c,e} \\
\text{s.t.} \quad & \sum_{e \in \mathcal{A}_c} x_{c,e} = 1, \quad \forall c \\
& x_{c,e} \geq 0, \quad x_{c,e} = 0 \text{ if } e \notin \mathcal{A}_c \\
& \sum_{e \in \mathcal{E}_k} \frac{\lambda_e(x)}{\mu_e} \leq C_k, \quad \forall k
\end{aligned}
\tag{NUM}
$$

其中 $u_{c,e} := s_{c,e}$ 为 gating score 作为效用。

### 3.2 拉格朗日对偶与影子价格

**定义 3.2（拉格朗日函数）**
对容量约束引入对偶变量 $\nu = (\nu_k)_{k \in \mathcal{K}} \geq 0$：
$$
\mathcal{L}(x, \nu) = \sum_{c,e} \lambda_c u_{c,e} x_{c,e} - \sum_k \nu_k \left( \sum_{e \in \mathcal{E}_k} \frac{\lambda_e(x)}{\mu_e} - C_k \right)
$$

**定理 3.1（最优路由结构）**
在 Slater 条件下，最优路由满足 KKT 条件。对每个类别 $c$，最优分配集中于使"调整后效用"最大的专家：
$$
e^*(c) \in \arg\max_{e \in \mathcal{A}_c} \left\{ u_{c,e} - \frac{\nu_{L(e)}^*}{\mu_e} \right\}
$$

**经济学解释**：
- $\nu_k^*$：GPU $k$ 的**资源稀缺价格**（边际增加一单位容量对系统效用的提升）
- $\nu_{L(e)}^*/\mu_e$：选择专家 $e$ 的**机会成本**

### 3.3 动态系统与队列价格

**定理 3.2（队列长度作为边际拥塞成本）**
考虑拥塞感知的目标函数：
$$
\max_{x(t)} \left\{ \sum_{c,e} \lambda_c u_{c,e} x_{c,e}(t) - \sum_e \phi_e(q_e(t)) \right\}
$$

其中 $\phi_e(q) = \frac{1}{2} w_e q^2$ 为二次拥塞代价。

则边际拥塞成本为：
$$
p_e^{\text{queue}}(t) := \phi_e'(q_e(t)) = w_e \cdot q_e(t)
$$

**解释**：队列长度 $q_e(t)$ 乘以权重 $w_e$ 直接作为拥塞的**影子价格**。

### 3.4 统一影子价格结构

**命题 3.1（影子价格分解）**
最优路由可表示为：
$$
e^*(c, t) \in \arg\max_{e \in \mathcal{A}_c} \left\{ u_{c,e} - \underbrace{w_e \cdot q_e(t)}_{\text{拥塞价格}} - \underbrace{\frac{\nu_{L(e)}(t)}{\mu_e}}_{\text{容量价格}} \right\}
$$

在本模型中，我们采用简化形式（忽略容量对偶）：
$$
\boxed{
e^*(c, t) \in \arg\max_{e \in \mathcal{A}_c} \left\{ s_{c,e} - \alpha \cdot q_e(t) \right\}
}
\tag{PriceMoE}
$$

其中 $\alpha := w_e$（假设同质专家权重）。

### 3.5 唯一性（备注）

**备注 3.1（熵正则化与唯一性）**
若 (NUM) 存在多最优解导致路由抖动，可引入熵正则化：
$$
\max_x \left\{ \sum_{c,e} \lambda_c u_{c,e} x_{c,e} - \frac{1}{\eta} \sum_c \lambda_c \sum_e x_{c,e} \log x_{c,e} \right\}
$$

此时最优路由具有 softmax 形式，且唯一。**此扩展可在工程实现或后续会议版本中采用。**

---

## 4. PriceMoE 路由策略

### 4.1 策略定义

**算法 4.1（PriceMoE 路由）**

对每个到达 token（类别 $c$），执行：

**输入**：
- $\mathbf{s}_c = (s_{c,e})$：gating 分数向量
- $\mathbf{q}(t) = (q_e(t))$：当前专家队列长度估计
- $\alpha > 0$：拥塞惩罚系数
- $k$：Top-k 参数

**步骤**：
1. 计算修正分数：$\tilde{s}_{c,e}(t) = s_{c,e} - \alpha \cdot q_e(t)$
2. 选择 Top-k 专家：$\mathcal{E}^*(c,t) = \text{Top}_k(\tilde{\mathbf{s}}_c(t))$
3. 更新队列估计：$q_e(t^+) \leftarrow q_e(t) + \mathbf{1}\{e \in \mathcal{E}^*(c,t)\}$

**输出**：选中专家集合 $\mathcal{E}^*(c,t)$

### 4.2 虚拟队列定义

**定义 4.1（Batch 内虚拟队列）**
由于实际系统无原生 per-expert 队列，定义**虚拟队列**：
$$
q_e(t) := \sum_{j \in \text{Batch}(t)} \mathbf{1}\{e \in \text{Routed}(j)\}
$$

即当前 batch 内被路由到专家 $e$ 的 token 数量。

**定义 4.2（指数移动平均）**
跨 batch 平滑：
$$
\hat{q}_e^{(t+1)} = (1 - \rho) \cdot \hat{q}_e^{(t)} + \rho \cdot q_e(t)
$$
其中 $\rho \in (0,1)$ 为平滑系数。

### 4.3 在线价格更新

**算法 4.2（对偶价格更新）**
若需引入容量约束的对偶价格，采用投影次梯度：
$$
\nu_k(t+1) = \left[ \nu_k(t) + \beta \left( \sum_{e \in \mathcal{E}_k} \frac{\hat{\lambda}_e(t)}{\mu_e} - C_k \right) \right]_+
$$

其中 $\hat{\lambda}_e(t)$ 为专家 $e$ 到达率的滑动估计，$\beta > 0$ 为步长。

---

## 5. 理论保证

### 5.1 稳定性

**定理 5.1（流体模型稳定性）**
若到达率向量 $\boldsymbol{\lambda}$ 位于容量域内部（$\boldsymbol{\lambda} \in \text{int}(\mathcal{C})$），且拥塞惩罚系数 $\alpha > 0$，则 PriceMoE 策略下流体模型全局稳定：
$$
\lim_{t \to \infty} q_e(t) = 0, \quad \forall e \in \mathcal{E}
$$

**证明要点**：
1. 定义二次函数 $V(\mathbf{q}) = \frac{1}{2} \sum_e w_e q_e^2$
2. 计算导数：$\dot{V} = \sum_e w_e q_e (\lambda_e - \mu_e \mathbf{1}\{q_e > 0\})$
3. 代入 PriceMoE 路由，证明当 $\|\mathbf{q}\|$ 大时，$\dot{V} < 0$
4. 应用 LaSalle 不变性原理

**定理 5.2（随机系统稳定性）**
若流体模型在所有初值下满足 $q(t) \to 0$，则原始随机处理网络正常返（positive Harris recurrent），因而存在平稳分布。

> **参考**：Dai (1995), Theorem 4.2

### 5.2 吞吐量最优性

**定理 5.3（最大稳定区域）**
PriceMoE 策略使系统在容量域内稳定，即：
$$
\forall \boldsymbol{\lambda} \in \text{int}(\mathcal{C}), \quad \text{系统稳定}
$$

这是所有非预测性（non-anticipative）策略可达到的最大稳定区域。

**证明要点**：PriceMoE 具有 MaxWeight 结构，满足 Tassiulas-Ephremides (1992) 吞吐量最优性条件。

### 5.3 均衡存在性与唯一性

**定理 5.4（流体均衡存在性）**
若 $\boldsymbol{\lambda} \in \text{int}(\mathcal{C})$，则流体模型存在平衡点 $\mathbf{q}^* = \mathbf{0}$。

**定理 5.5（均衡唯一性）**
在以下任一条件下，流体均衡唯一：
1. 价格映射 $p(q) = \alpha q$ 严格单调
2. 路由采用熵正则化（softmax）

**证明**：
1. 条件 1：$V(\mathbf{q})$ 严格凸，有唯一最小化器
2. 条件 2：正则化后目标严格凹，KKT 条件确定唯一解

---

## 6. 完整推导链条

### 6.1 逻辑流程

```
┌─────────────────────────────────────────────────────────────────┐
│           随机处理网络 (Stochastic Processing Network)          │
│  • 状态: Q_e(t) ∈ ℤ₊ (专家队列长度)                             │
│  • 动力学: Q_e(t) = Q_e(0) + A_e(t) - S_e(t)                    │
│  • 约束: 计算容量 Σ λ_e/μ_e ≤ C_k                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼ 流体缩放: q_e(t) = Q_e(nt)/n, n → ∞
┌─────────────────────────────────────────────────────────────────┐
│                    流体模型 (Fluid Model)                        │
│  • ODE: dq_e/dt = λ_e(t) - μ_e · 𝟙{q_e > 0}                     │
│  • 反射形式: Skorokhod 问题                                      │
│  • 定理 2.1: FSLLN → 流体极限存在且唯一                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼ 稳态 + 拉格朗日对偶
┌─────────────────────────────────────────────────────────────────┐
│                  网络效用最大化 (NUM)                            │
│  • max Σ u_{ce} x_{ce}  s.t. 容量约束                           │
│  • 拉格朗日函数 → 对偶变量 ν_k (资源稀缺价格)                    │
│  • 拥塞代价 φ(q) = ½wq² → 边际价格 p = wq                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼ KKT 条件
┌─────────────────────────────────────────────────────────────────┐
│                   PriceMoE 路由策略                              │
│  • 修正分数: s̃_{ce}(t) = s_{ce} - α·q_e(t)                      │
│  • 路由规则: e*(c,t) = argmax_{e∈𝒜_c} s̃_{ce}(t)                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼ 理论保证
┌─────────────────────────────────────────────────────────────────┐
│  • 定理 5.1: 流体模型稳定性 (LaSalle)                           │
│  • 定理 5.2: 随机系统正常返 (Dai 1995)                          │
│  • 定理 5.3: 吞吐量最优 (MaxWeight 结构)                         │
│  • 定理 5.4-5.5: 均衡存在唯一性                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 核心参数定义

| 参数 | 符号 | 定义 | 估计方法 |
| ---- | ---- | ---- | ---- |
| 到达率 | $\lambda_c$ | 类别 $c$ 的 token 到达速率 | Scheduler 统计 |
| 服务速率 | $\mu_e$ | 专家 $e$ 的处理速率 | 离线 Profile |
| 队列长度 | $q_e(t)$ | 专家 $e$ 的虚拟队列 | Batch 内路由计数 |
| Gating 分数 | $s_{c,e}$ | Router 输出 logits | 模型前向 |
| 拥塞权重 | $\alpha$ | 拥塞惩罚系数 | 可调超参 |
| GPU 容量 | $C_k$ | GPU $k$ 的计算容量 | 硬件规格 |
| 平滑系数 | $\rho$ | EMA 衰减率 | 经验值 0.1-0.3 |

### 6.3 关键定理依赖链

| 步骤 | 内容 | 关键定理 | 所需假设 |
| ---- | ---- | ---- | ---- |
| 1 | 流体极限存在 | 定理 2.1 | A1-A4 (FSLLN, Lipschitz) |
| 2 | 反射 ODE 唯一解 | 定理 2.2 | Skorokhod 映射性质 |
| 3 | 对偶价格结构 | 定理 3.1 | Slater 条件 |
| 4 | 队列作为价格 | 定理 3.2 | 二次代价函数 |
| 5 | 流体稳定性 | 定理 5.1 | $\lambda \in \text{int}(\mathcal{C})$ |
| 6 | 随机稳定性 | 定理 5.2 | 流体稳定 (Dai 1995) |
| 7 | 吞吐量最优 | 定理 5.3 | MaxWeight 结构 |

---

## 7. MS/OR 顶刊发表可行性

### 7.1 理论贡献定位

1. **随机处理网络在 AI 系统的新应用**：首次将流体极限与影子价格理论应用于 MoE 推理负载均衡
2. **可证明的路由策略**：从数学模型到可实施策略的完整推导链
3. **吞吐量最优性**：证明 PriceMoE 达到最大稳定区域
4. **工程可落地**：策略形式简洁，可嵌入现有框架

### 7.2 与现有工作差异化

| 维度 | 现有工作 (EPLB, Capacity Factor) | PriceMoE |
| ---- | ---- | ---- |
| 理论基础 | 启发式 | 随机网络 / 流体极限 / 对偶理论 |
| 可证明性 | 无 | 稳定性 + 吞吐量最优 |
| 拥塞感知 | 静态 / 事后 | 动态 / 实时反馈 |
| 参数调优 | 需经验调参 | 有理论指导 |

### 7.3 潜在审稿人关切与回应

| 关切 | 回应 |
| ---- | ---- |
| "FSLLN 假设是否现实？" | 引用 vLLM 实际观测数据验证 |
| "为何不用 Halfin-Whitt？" | 扩展讨论；经典流体已足够支撑核心贡献 |
| "实验验证？" | E1.1 路由统计 + 仿真 + vLLM 集成 |
| "通信约束缺失？" | 备注讨论，作为 Future Work |

### 7.4 建议论文结构

1. **Introduction**：MoE 推理挑战 + 定价机制动机
2. **Related Work**：排队网络 / 流体极限 / MoE 调度
3. **Model**：随机处理网络 + 流体缩放 + 容量约束
4. **Analysis**：NUM + 对偶 + 影子价格 + 稳定性
5. **Algorithm**：PriceMoE 策略 + 在线实现
6. **Experiments**：仿真验证 + vLLM 集成（如时间允许）
7. **Conclusion**：贡献总结 + Future Work (Halfin-Whitt, 通信约束)

---

## 附录 A：证明草稿

### A.1 定理 2.1（流体极限）

**证明**：
1. **紧性**：由 FSLLN 条件，$\|\bar{Q}^{(n)}(t) - \bar{Q}^{(n)}(s)\| \leq \frac{1}{n}|A^{(n)}(nt) - A^{(n)}(ns)| + \frac{1}{n}|S^{(n)}(nt) - S^{(n)}(ns)|$。应用 Arzelà-Ascoli 定理。
2. **极限识别**：鞅分解 $M^{(n)}(t) = \bar{Q}^{(n)}(t) - \int_0^t F(\bar{Q}^{(n)}(s)) ds$。由 FSLLN，$\sup_{s \leq t} |M^{(n)}(s)| \to 0$ a.s.。
3. **唯一性**：$F(q) = \lambda - \mu \cdot \mathbf{1}\{q > 0\}$ 在 $q > 0$ 时 Lipschitz（常数）；边界用 Skorokhod 反射处理。$\square$

### A.2 定理 5.1（稳定性）

**证明**：
1. 定义 $V(\mathbf{q}) = \frac{1}{2} \sum_e w_e q_e^2$。
2. 沿流体轨迹：$\dot{V} = \sum_e w_e q_e (\lambda_e(t) - \mu_e)$。
3. PriceMoE 选择 $e^* = \arg\max\{s_{ce} - \alpha q_e\}$，故 $\lambda_e$ 偏向低队列专家。
4. 当 $\|q\| \to \infty$，高队列专家被惩罚，$\sum_e w_e q_e \lambda_e$ 减少，$\dot{V} < 0$。
5. 由 LaSalle 不变性原理，$q(t) \to 0$。$\square$

---

## 附录 B：实现复杂度说明

根据 `Documentation/PriceMoE Implementation Feasibility Analysis.md`：

| 组件 | 工作量 | 说明 |
| ---- | ---- | ---- |
| Router 修改 | 2-3 周 | 新增 `PriceMoERouter` 类 |
| 统计采集 | 1-2 周 | 扩展 `moe_stats.py` |
| 虚拟队列 | 2-3 周 | Batch 内计数 + EMA |
| Scheduler 集成 | 3-4 周 | 传递 expert_loads |
| **MVP 总计** | **4-6 周** | 仅拥塞惩罚 |
| 完整版本 | 8-12 周 | +迟滞机制 +容量对偶 |

**核心修改不止一行**：需要 Router 扩展 + 统计模块 + 状态传递。

---

*模型版本: v2.1 | 创建日期: 2026-01-21 | 目标: MS 期刊发表*
