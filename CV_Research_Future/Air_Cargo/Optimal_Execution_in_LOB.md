# Optimal Execution in Limit Order Books (LOB): A DP + Threshold-Policy Reframing

> This note rebuilds the existing **Airline Cargo Transport Recovery** model into an **Optimal Execution in Limit Order Books** story.
>
> - “Flights” → **order flow / liquidity events**
> - “Cargo backlog” → **inventory** to execute (shares to sell/buy)
> - “Direct vs transfer” → **single-venue execution vs cross-venue routing (smart order routing)**
>
> The goal is interview-ready: emphasize **dynamic programming (DP)**, **Bellman equations**, and **threshold / monotone policies**.

---

## 1. Executive summary (what problem are we solving?)

You hold an initial inventory $B$ (e.g., shares to sell). Liquidity arrives randomly over time (marketable flow, quote updates, hidden liquidity, etc.). Each time a liquidity event occurs, you may consume some available liquidity to reduce inventory.

There are **two routing channels**:

1. **Local execution (Direct)**: execute on the primary exchange (low latency / no extra routing delay).
2. **Cross-venue routing (Transfer)**: route to an alternate venue (dark pool / other exchange). This may provide additional liquidity but incurs **extra delay / uncertainty** $X$ (fill latency / queue position / adverse selection).

Objective: minimize **inventory risk + waiting cost + routing penalty**, subject to executing (largely) all inventory by horizon $T$.

---

## 2. Mapping table: airline recovery → LOB optimal execution

| Airline recovery concept | Symbol in original | LOB execution concept | Suggested symbol |
| --- | ---: | --- | ---: |
| Cargo backlog / queue | $B,\ b_t$ | Inventory to liquidate / acquire | $Q,\ q_t$ |
| Direct flight arrivals | $\mu_0$ Poisson | Primary venue liquidity events | $\mu_P$ |
| Transfer flight arrivals | $\mu_1$ Poisson | Alternate venue liquidity events | $\mu_A$ |
| Random residual capacity | $RC_0, RC_1$ with $C\sim U(0,1]$ | Random executable size at event (depth/hidden liquidity) | $L_P, L_A$ |
| Transfer delay | $X$ | Cross-venue latency / execution delay / adverse selection | $X$ |
| Waiting cost | time-weighted backlog | Inventory risk / urgency cost | holding cost $h(t)\,q_t$ |
| Static thresholds $T_0,T_1$ | time cutoffs | “spend time window on venue P vs route to A” | time threshold / switching time |
| DP threshold policy | capacity threshold $\theta$ | execute if liquidity exceeds threshold | liquidity threshold $\theta(q,t)$ |

We’ll use:

- Inventory $q_t \in \mathbb{Z}_+$ (shares remaining)
- Time $t \in [0,T]$
- Primary events: Poisson rate $\mu_P$
- Alternate events: Poisson rate $\mu_A$
- Liquidity at event: $L_P(t)=R\,C_P(t)$ and $L_A(t)=R\,C_A(t)$

---

## 3. Single-asset, two-venue execution model (baseline)

### 3.1 State, events, and information

- **State**: $(q,t)$
  - $q$: remaining inventory (shares to sell)
  - $t$: elapsed time

- **Event process** (event-driven control / SMDP):
  - Primary event occurs with intensity $\mu_P$
  - Alternate event occurs with intensity $\mu_A$

At an event, you observe realized liquidity:

- Primary event reveals $L_P \sim R\,C_P$
- Alternate event reveals $L_A \sim R\,C_A$ and routing delay $X$

Decision must be made under **revealed liquidity** (classic online decision structure).

### 3.2 Actions (what do we control?)

At a primary-venue liquidity event:

- choose executed size $x_P \in [0,\min(q, L_P)]$

At an alternate-venue event:

- choose executed size $x_A \in [0,\min(q, L_A)]$

Interpretation:

- $x$ is marketable / aggressive consumption of available liquidity.
- You could also extend to include passive limit orders, but for interview DP clarity we keep a single “consume liquidity” control.

### 3.3 Cost function (the key for Bellman)

A typical interview-friendly objective is:

- holding / urgency cost increases in time and inventory
- optional routing penalty for cross-venue

One consistent cost structure (matching your original “time-weighted backlog” flavor) is:

- instantaneous holding cost rate: $h(t)\,q_t$ with $h(t)=t$ (urgency increases with time)
- when you execute $x_P$ at time $t$: you “stop paying holding cost” for these shares; immediate cost proxy $c_P(t)\,x_P$ with $c_P(t)=t$
- when you execute $x_A$ at time $t$: you pay $c_A(t)\,x_A$ with $c_A(t)=t+X$ (extra delay / slippage)
- terminal penalty: $p\,q_T$ (enforce liquidation)

So the expected objective can be written as:

$$
J^{\pi}(q,t)=\mathbb{E}^{\pi}\Big[\int_t^T h(s)\,q_s\,ds + \sum_{k\in\mathcal{E}_P} c_P(t_k)\,x_P(t_k) + \sum_{j\in\mathcal{E}_A} c_A(t_j)\,x_A(t_j) + p\,q_T\Big].
$$

This is structurally similar to the cargo recovery cost, with $X$ capturing cross-venue “transfer delay”.

---

## 4. Static (planning) approximation: time-threshold split

Before full DP, we can build intuition via a static approximation that produces **closed-form switching rules**.

Suppose you commit to:

- use primary venue up to time $T_P$
- use alternate venue up to time $T_A$

and assume you fully consume expected liquidity in those windows.

If $\mathbb{E}[C]=1/2$ (uniform on $(0,1]$), then expected executed quantity is:

$$
\mathbb{E}[\text{volume}] \approx \frac{R\mu_P}{2}T_P + \frac{R\mu_A}{2}T_A.
$$

Expected cost (with $h(t)=t$) has the same quadratic form as your original:

$$
\mathbb{E}[\text{cost}] \approx \frac{R\mu_P}{4}T_P^2 + \frac{R\mu_A}{4}T_A^2 + \frac{R\mu_A}{2}\,\mathbb{E}[X]\,T_A.
$$

Constraint: execute initial inventory $Q$:

$$
\frac{R}{2}(\mu_P T_P + \mu_A T_A)=Q.
$$

**Takeaway**: cross-venue routing is used only if inventory is large enough to justify the extra delay $\mathbb{E}[X]$.

This motivates DP threshold policies: route when “inventory pressure” is high.

---

## 5. Dynamic Programming (stochastic control): Bellman equation

### 5.1 Value function

Let $V(q,t)$ be the minimal expected cost-to-go from state $(q,t)$.

Boundary condition:

- $V(0,t)=0$
- $V(q,T)=p\,q$ (or $T\cdot q$ in your original style)

### 5.2 Small-$\Delta t$ Bellman recursion (event-driven)

In a small interval $[t,t+\Delta t]$:

- with prob $1-(\mu_P+\mu_A)\Delta t$, no liquidity event; you pay holding cost
- with prob $\mu_P\Delta t$, primary event occurs and you choose $x_P$
- with prob $\mu_A\Delta t$, alternate event occurs and you choose $x_A$

A canonical Bellman form (suppressing higher-order terms) is:

$$
\begin{aligned}
V(q,t)=\min_{\pi}\ \mathbb{E}\Big[& (1-(\mu_P+\mu_A)\Delta t)\big(V(q,t+\Delta t)+h(t)q\,\Delta t\big)\\
&+\mu_P\Delta t\,\mathbb{E}_{L_P}\big[\min_{0\le x\le \min(q,L_P)} (V(q-x,t+\Delta t)+c_P(t)\,x)\big]\\
&+\mu_A\Delta t\,\mathbb{E}_{L_A,X}\big[\min_{0\le x\le \min(q,L_A)} (V(q-x,t+\Delta t)+c_A(t,X)\,x)\big]\Big].
\end{aligned}
$$

This is exactly the “cargo DP” story, just relabeled.

---

## 6. Why threshold policies appear (interview core)

At a liquidity event with realized liquidity $L$, you solve a 1-step minimization:

$$
\min_{0\le x\le \min(q,L)}\ \big(V(q-x,t)+\underbrace{c(t)}_{\text{immediate cost}}\,x\big).
$$

### 6.1 Convexity → bang-bang / threshold structure

If $V(\cdot,t)$ is **convex** in $q$ (standard in inventory/queueing control under holding costs), then the function

$$
F(x)=V(q-x,t)+c(t)\,x
$$

is convex in $x$. Convex minimization on a compact interval pushes solutions to the boundary:

- either $x^*=0$ (don’t execute), or
- $x^*=\min(q,L)$ (execute as much as possible)

The “switch” is characterized by a threshold comparing marginal value of inventory vs immediate execution cost.

### 6.2 Deriving a liquidity threshold (capacity threshold)

Using a second-order expansion around $q$:

$$
V(q-x,t)\approx V(q,t) - V_q(q,t)x + \tfrac{1}{2}V_{qq}(q,t)x^2.
$$

So

$$
F(x)\approx V(q,t) + (c(t)-V_q)\,x + \tfrac{1}{2}V_{qq}\,x^2.
$$

If $V_{qq}>0$, the minimizer suggests executing only if realized liquidity exceeds

$$
\theta(q,t)=\frac{2\,(V_q(q,t)-c(t))}{V_{qq}(q,t)}.
$$

- Primary venue: $c(t)=t$ hence $\theta_P(q,t)=\tfrac{2(V_q-t)}{V_{qq}}$
- Alternate venue: $c(t)=t+X$ (or $t+\mathbb{E}[X]$) hence lower propensity to route when delay is large

This is the same threshold expression as in your cargo model.

**Interview soundbite**: “Convex value function + linear execution cost → execute-all-or-nothing at events; the execution decision becomes a threshold on liquidity/price impact.”

---

## 7. A practical explicit approximation (to get closed-form thresholds)

To make the threshold computable, use a quadratic approximation for the value function:

$$
V(q,t)\approx t\,q + \alpha\,q^2.
$$

Then

- $V_q\approx t+2\alpha q$
- $V_{qq}\approx 2\alpha$

So

$$
\theta_P(q,t)\approx 2q,
$$

and for cross-venue routing (using $c_A=t+\mathbb{E}[X]$):

$$
\theta_A(q,t)\approx 2q - \frac{\mathbb{E}[X]}{\alpha}.
$$

Interpretation:

- when inventory $q$ is large, both venues are used aggressively
- when inventory is small, cross-venue is not worth the latency/adverse-selection penalty

---

## 8. Two-venue routing rule (primary vs alternate): “use alternate only when inventory pressure is high”

A clean threshold-style routing rule:

- Always execute on primary if liquidity is large enough: $L_P\ge \theta_P(q,t)$
- Execute on alternate only if $L_A\ge \theta_A(q,t)$

Because $\theta_A$ is shifted downward by the delay term, alternate routing is used more selectively.

This mirrors the airline result: “transfer is used only when backlog exceeds a critical threshold.”

---

## 9. Algorithm sketch (simulation-friendly, interview-ready)

Below is a minimal event-driven simulator + control loop.

### 9.1 Policy (threshold version)

1. Initialize $(q,t)=(Q,0)$
2. Simulate next event time $\Delta \tau\sim \mathrm{Exp}(\mu_P+\mu_A)$
3. Update time $t\leftarrow t+\Delta \tau$ and accumulate holding cost
4. Decide event type: primary with prob $\mu_P/(\mu_P+\mu_A)$ else alternate
5. Observe liquidity $L$ (and delay $X$ if alternate)
6. If $L\ge \theta(q,t)$ then execute $x=\min(q,L)$ else $x=0$
7. Update $q\leftarrow q-x$ and continue until $t\ge T$ or $q=0$

### 9.2 What interviewers will ask you next

- Why is $V$ convex? (inventory holding costs + linear dynamics)
- Why does “bang-bang” appear? (convex minimization + linear immediate cost)
- What breaks it? (price impact convex in $x$, permanent impact, state includes midprice/spread)
- How to extend? (Almgren–Chriss continuous control, Hawkes order flow, multi-asset)

---

## 10. Extensions: making it look like a real LOB execution paper

If you want to go beyond the mapping into a publication-grade model:

1. **Add midprice dynamics** (e.g., diffusion) and penalize execution shortfall / variance.
2. **Add temporary price impact**: cost $\eta x^2$; policy becomes interior (not pure bang-bang).
3. **Queue position / limit orders**: control includes posting depth; state includes queue length.
4. **Smart order routing**: alternate venue fill probability depends on routing size and venue conditions.
5. **Order flow is self-exciting**: Hawkes processes for event intensities.

---

## 10.1 How the threshold structure changes with impact or fill probability (important nuance)

The clean “bang-bang / threshold-on-liquidity” story in Sections 6–8 relies on two simplifying assumptions:

1. **Linear immediate cost in executed size**: $c(t)\,x$.
2. **If you submit $x$, it fills deterministically** (up to available liquidity $L$).

Real execution breaks (1) and (2). The good news is: **you still often get monotone/threshold-type structure**, but the threshold is no longer the same and the action may no longer be “all-or-nothing”.

### A) Add temporary price impact ⇒ interior solutions (not pure bang-bang)

Suppose the immediate cost includes a convex temporary impact term (Almgren–Chriss style, but event-driven):

$$
  ext{immediate cost} = c(t)\,x + \tfrac{\eta}{2}x^2,\qquad \eta>0.
$$

At a liquidity event, the single-step minimization becomes

$$
\min_{0\le x\le \min(q,L)}\Big(V(q-x,t)+c(t)\,x+\tfrac{\eta}{2}x^2\Big).
$$

Using the same quadratic expansion of $V(q-x,t)$ around $q$:

$$
V(q-x,t)\approx V(q,t)-V_q\,x+\tfrac{1}{2}V_{qq}x^2,
$$

the objective becomes approximately quadratic in $x$:

$$
  ext{obj}(x)\approx \text{const} + (c(t)-V_q)\,x + \tfrac{1}{2}(V_{qq}+\eta)x^2.
$$

So the unconstrained minimizer is

$$
x^\star \approx \frac{V_q(q,t)-c(t)}{V_{qq}(q,t)+\eta},
$$

and the executed size is the clipped version

$$
x^\star_{\text{clipped}}=\Big[\,x^\star\,\Big]_{[0,\min(q,L)]}.
$$

**What changes vs the original threshold policy?**

- You still execute **more** when inventory pressure (marginal value $V_q$) is higher.
- But you don’t immediately jump to “execute all available liquidity”; convex impact makes you spread execution.
- A *new* “activation threshold” still exists: execute only if $V_q(q,t) > c(t)$.

In interview terms: *impact turns bang-bang control into proportional control with clipping.*

### B) Add fill probability / adverse selection ⇒ threshold on expected marginal benefit

Now suppose execution is uncertain. If you submit $x$ to venue $v\in\{P,A\}$ at time $t$, it fills with probability

$$
p_v(t,\text{state},x)\in[0,1].
$$

For a first-order interview-friendly model, assume: fill probability depends on venue conditions but not strongly on $x$ (or use a small-$x$ approximation):

$$
p_v(t)\approx p_v\in(0,1).
$$

Then the one-step expected cost-to-go can be written as

$$
\mathbb{E}[V(q',t) + \text{execCost}]\approx p_v\,\big(V(q-x,t)+c_v(t)\,x\big) + (1-p_v)\,V(q,t).
$$

Ignoring second-order terms, you execute if the expected marginal improvement is positive:

$$
p_v\,(V_q(q,t)-c_v(t)) > 0\quad\Longleftrightarrow\quad V_q(q,t) > c_v(t).
$$

So the *activation condition* looks unchanged, but the **effective aggressiveness is reduced** because fills are probabilistic. If you include risk of adverse selection (e.g., penalty $\kappa_v x$ when filled), the activation becomes:

$$
V_q(q,t) > c_v(t) + \kappa_v.
$$

**Cross-venue routing interpretation**:

- Alternate venue often has **higher uncertainty** (lower $p_A$) and/or higher adverse-selection penalty $\kappa_A$.
- That raises the effective threshold for routing, making “use alternate only when inventory pressure is high” even more pronounced.

### C) Summary cheat-sheet (what to say in interviews)

- Linear cost + deterministic fill ⇒ **threshold + bang-bang**: execute-all-or-nothing when liquidity exceeds a threshold.
- Add convex impact ⇒ still monotone, but **interior solution**: $x^\star\propto \frac{V_q-c}{V_{qq}+\eta}$.
- Add fill probability / adverse selection ⇒ threshold shifts to **expected marginal benefit**; routing becomes more selective.

---

## 11. One-paragraph “quant interview” explanation

We model optimal execution as an inventory control problem driven by random liquidity events. Inventory $q$ is the state, and each time liquidity arrives we decide how much to execute. With convex value function in inventory and linear per-share execution cost, the one-step DP minimization becomes a convex problem whose optimum lies at the boundary—either execute nothing or execute as much as possible—leading to a threshold policy. Cross-venue routing adds an extra delay/adverse-selection term $X$, shifting the threshold so that routing is used only when inventory pressure is sufficiently high.
