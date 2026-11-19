# HPO-CRMAB demo — college board experiment

**One-line result:** delegation never monitors and contains only **5%** of fires; centralisation scans once and contains **81%**. A learned monitoring schedule gets **81%** at lower cost, close to the **91%** full-information oracle.

## What the demo shows

The GC gives each region a scalar budget `b`. The LC decides how to spend it: monitoring or suppression. In **FOpt** the LC decides (and never monitors, because the benefit is future and global); in **COpt** the central planner decides (and scans once). The cost gap is the **value of information coordination**.

## 1. Baselines (show this first)

Two regions, each a 10×10 grid. One fire starts at an unknown cell. Six epochs. Budget `b = 4` per region per epoch; a monitor costs 0.25 budget units, a suppression action costs 1.0. Lower cost is better.

| Policy | Total cost | Containment | Monitoring actions | What it is |
|---|---:|---:|---:|---|
| Random | 392.8 ± 188.8 | 7% | 24.8 | random monitor + random drop |
| Greedy | 423.8 ± 187.6 | 7% | 0 | drop on highest-belief cell, no monitor |
| FOpt | 404.1 ± 181.6 | 5% | 0 | delegated LC: all budget on suppression |
| Myopic | 167.7 ± 156.4 | 33% | 24.0 | fixed 2 monitors + 3 drops |
| **COpt** | **56.0 ± 85.6** | **81%** | 8.0 | central planner: scan once, then suppress |
| **Q-learning** | **41.6 ± 50.0** | **81%** | 27.5 | learned monitoring schedule |
| Oracle | 30.4 ± 33.1 | 91% | 0 | full information (upper bound) |

- **FOpt − COpt = 348.1**, an **86% cost reduction**.
- **FOpt − Q-learning = 362.5**, an **89.7% reduction**.
- COpt is **1.84×** the oracle cost; Q-learning is **1.37×**.
- Containment is the clearest board metric: **5% → 81% → 91%**.

The baseline ladder is exactly what a reviewer expects: random/greedy → delegated (FOpt) → heuristic (myopic) → centralised (COpt) → learned (Q-learning) → oracle.

## 2. Simulation (what the GIF shows)

`out/simulation.gif` — 42 frames, one COpt episode:

- **epoch 0:** four monitoring footprints cover the grid. The scan finds the ignition (white cell in the belief panel). Three drops land on it. The fire is out.
- **epochs 1–5:** the belief diffuses because the LC stops monitoring; the drops are placed on high-belief cells. This shows *why* information matters: without a scan the belief is a blur.

`out/grid_snapshots.png` is a static 6×3 figure (true fire / belief / uncertainty) from a COpt episode.

## 3. Q-learning curve (the RL-paper part)

`out/rl_training.png`

- **State:** discretised belief summary — `(epoch, max P(burning), mean entropy, burned fraction)`.
- **Action:** number of monitors `m = 0..4`; the suppression target uses the same heuristic as all policies.
- **Reward:** negative epoch cost.
- **Training:** 400 episodes, ε-greedy 1.0 → 0.05, α = 0.1, γ = 0.95.
- **Learning curve:** from ~200 to ~35 total cost.
- **Learned schedule:** `[4, 4, 1, 0, 3, 3]` monitors per epoch (one episode). It scans early, then monitors again late to track the fire.
- **Evaluation:** 100 episodes with ε = 0 → cost 41.6, containment 81%.

The RL agent is deliberately small: it learns **when to monitor**, not where to drop. That is enough to show the paper's point (the monitoring/suppression split is the decision that matters), and it is honest about what is learned.

## 4. Price of monitoring (the trade-off)

`out/sweep.png` — vary the budget price of one monitor, keeping at least two drops.

| `w_m` | FOpt | COpt | Gap |
|---:|---:|---:|---:|
| 0.25 | 400.6 | 62.1 | 338.5 |
| 0.50 | 400.6 | 79.2 | 321.4 |
| 1.00 | 400.6 | 232.8 | 167.8 |
| 2.00 | 400.6 | 332.0 | 68.6 |
| 4.00 | 400.6 | 400.6 | 0.0 |

When information is cheap, centralisation wins big. When a scan costs as much as a drop, the advantage disappears. This is the **value of information coordination**.

## 5. Caveats (say these out loud)

- **Toy instance**, not the calibrated wildfire model. Homogeneous spread, 10×10 grid, mean-field belief.
- **COpt is a heuristic** ("scan once, then suppress"), not the optimal POMDP policy.
- **Q-learning learns only the monitoring count**; targeting is a shared heuristic. It is not end-to-end RL.
- The oracle is an upper bound with the same action set, not deployable.
- The result is about **direction and mechanism**, not real suppression numbers.

## 6. Run

```bash
py demo_grid.py
```

Runtime ~60 s. Outputs in `out/`:

```text
policy_bars.png     baseline comparison + containment
rl_training.png     Q-learning curve + learned schedule
simulation.gif      one COpt episode, 36 frames
grid_snapshots.png  static 6-epoch view
sweep.png           cost and gap vs monitoring price
results.json        all numbers
```

## 7. Suggested board order

1. Problem: hidden ignition, two clocks, budget.
2. Simulation GIF: scan finds the fire, drops contain it.
3. Baseline bars: FOpt 5% vs COpt 81% vs Oracle 91%.
4. RL curve: a learned schedule reaches the same containment at lower cost.
5. Sweep: information is valuable only when it is affordable.
6. Takeaway: **delegation under-invests in information; centralisation internalises it.**
