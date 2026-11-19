"""
HPO-CRMAB demo on a small 10x10 grid.

Two regions (each 10x10). The GC gives each region a scalar budget b.
The LC decides how to split b between monitoring and suppression.

Policies compared:
  fopt    : LC is myopic -> spends everything on suppression, never monitors
  myopic  : fixed 1/3 monitoring, 2/3 suppression (greedy)
  copt    : central planner picks the monitoring share by a 2-epoch rollout
  oracle  : full observation (upper bound)

Outputs (in ./out):
  grid_snapshots.png
  policy_bars.png
  monitoring_share.png
  results.json
"""

import copy
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from scipy.signal import convolve2d
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- utilities
def neighbor_sum(a):
    p = np.pad(a.astype(float), 1)
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]


def entropy3(pb, pu):
    pun = np.clip(1.0 - pb - pu, 1e-9, 1.0)
    pb = np.clip(pb, 1e-9, 1.0)
    pu = np.clip(pu, 1e-9, 1.0)
    return -(pun * np.log(pun) + pb * np.log(pb) + pu * np.log(pu))


def obs_cells(center, radius, size):
    r0, c0 = center
    cells = []
    for r in range(max(0, r0 - radius), min(size, r0 + radius + 1)):
        for c in range(max(0, c0 - radius), min(size, c0 + radius + 1)):
            cells.append((r, c))
    return cells


# ---------------------------------------------------------------- environment
class Region:
    def __init__(self, size=10, p_spread=0.35, burn_time=2,
                 q_supp=0.9, wet_time=3, mon_radius=2, seed=0):
        self.size = size
        self.p_spread = p_spread
        self.burn_time = burn_time
        self.q_supp = q_supp
        self.wet_time = wet_time
        self.mon_radius = mon_radius
        self.rng = np.random.default_rng(seed)

        self.state = np.zeros((size, size), dtype=int)   # 0 unburnt, 1 burning, 2 burnt
        self.age = np.zeros((size, size), dtype=int)
        self.wet = np.zeros((size, size), dtype=int)
        self.p_burn = np.zeros((size, size))
        self.p_burnt = np.zeros((size, size))
        self.reset()

    def reset(self):
        self.state[:] = 0
        self.age[:] = 0
        self.wet[:] = 0
        self.p_burn[:] = 0.0
        self.p_burnt[:] = 0.0

        # true ignition: one cell, uniform over the whole grid
        r0 = int(self.rng.integers(0, self.size))
        c0 = int(self.rng.integers(0, self.size))
        self.state[r0, c0] = 1
        self.true_ignition = (r0, c0)

        # prior: uniform over the whole grid
        self.p_burn[:, :] = 1.0 / (self.size * self.size)

    # --- dynamics ---------------------------------------------------------
    def spread(self):
        burning = (self.state == 1)
        self.age[burning] += 1
        out = burning & (self.age >= self.burn_time)
        self.state[out] = 2

        nb = neighbor_sum(burning.astype(float))
        p_ign = 1.0 - (1.0 - self.p_spread) ** nb
        can = (self.state == 0) & (self.wet == 0)
        ign = can & (self.rng.random(self.state.shape) < p_ign)
        self.state[ign] = 1
        self.age[ign] = 0
        self.wet = np.maximum(0, self.wet - 1)

    def predict_belief(self):
        pb, pu = self.p_burn.copy(), self.p_burnt.copy()
        pun = np.clip(1.0 - pb - pu, 0.0, 1.0)

        nb = neighbor_sum(pb)
        p_ign = 1.0 - (1.0 - self.p_spread) ** nb
        p_ign[self.wet > 0] = 0.0

        nb_new = pb * (1.0 - 1.0 / self.burn_time) + pun * p_ign
        nu_new = pu + pb * (1.0 / self.burn_time)

        nb_new = np.clip(nb_new, 0.0, 1.0)
        nu_new = np.clip(nu_new, 0.0, 1.0)
        tot = nb_new + nu_new
        mask = tot > 1.0
        nb_new[mask] /= tot[mask]
        nu_new[mask] /= tot[mask]
        self.p_burn, self.p_burnt = nb_new, nu_new

    # --- observations and actions ----------------------------------------
    def observe(self, center, radius=None):
        if radius is None:
            radius = self.mon_radius
        for (r, c) in obs_cells(center, radius, self.size):
            s = self.state[r, c]
            if s == 1:
                self.p_burn[r, c], self.p_burnt[r, c] = 1.0, 0.0
            elif s == 2:
                self.p_burn[r, c], self.p_burnt[r, c] = 0.0, 1.0
            else:
                self.p_burn[r, c], self.p_burnt[r, c] = 0.0, 0.0

    def drop(self, cells):
        for (r, c) in cells:
            if self.state[r, c] == 1:
                if self.rng.random() < self.q_supp:
                    self.state[r, c] = 2
                    self.age[r, c] = 0
            elif self.state[r, c] == 0:
                self.wet[r, c] = self.wet_time

    def burned(self):
        return int((self.state == 2).sum())

    def sample_state(self):
        pb = np.clip(self.p_burn, 0.0, 1.0)
        pu = np.clip(self.p_burnt, 0.0, 1.0)
        pun = np.clip(1.0 - pb - pu, 0.0, 1.0)
        tot = pun + pb + pu
        pun, pb, pu = pun / tot, pb / tot, pu / tot

        u = self.rng.random(self.state.shape)
        s = np.zeros_like(self.state)
        s[u >= pun] = 1
        s[u >= pun + pb] = 2
        return s


# ---------------------------------------------------------------- policies
def monitor_score(reg):
    """Expected entropy covered by the monitoring footprint centred on each cell."""
    h = entropy3(reg.p_burn, reg.p_burnt)
    k = 2 * reg.mon_radius + 1
    cover = convolve2d(h, np.ones((k, k)), mode="same", boundary="fill", fillvalue=0.0)
    cover[reg.p_burn + reg.p_burnt > 0.99] = 0.0
    return cover


def choose_monitor(reg, m):
    if m <= 0:
        return []
    h = entropy3(reg.p_burn, reg.p_burnt).copy()
    h[reg.p_burn + reg.p_burnt > 0.99] = 0.0
    k = 2 * reg.mon_radius + 1
    centers = []
    for _ in range(m):
        score = convolve2d(h, np.ones((k, k)), mode="same",
                           boundary="fill", fillvalue=0.0)
        idx = int(np.argmax(score))
        r, c = np.unravel_index(idx, score.shape)
        centers.append((int(r), int(c)))
        r0, r1 = max(0, r - reg.mon_radius), min(reg.size, r + reg.mon_radius + 1)
        c0, c1 = max(0, c - reg.mon_radius), min(reg.size, c + reg.mon_radius + 1)
        h[r0:r1, c0:c1] = 0.0
    return centers


def suppression_score(reg):
    nb = neighbor_sum(reg.p_burn)
    p_ign = 1.0 - (1.0 - reg.p_spread) ** nb
    pun = np.clip(1.0 - reg.p_burn - reg.p_burnt, 0.0, 1.0)
    return np.maximum(reg.p_burn, 0.7 * p_ign * pun)


def choose_suppress(reg, s):
    if s <= 0:
        return []
    score = suppression_score(reg)
    idx = np.argsort(score.ravel())[::-1][:s]
    return [np.unravel_index(i, score.shape) for i in idx]


def choose_suppress_greedy(reg, s):
    if s <= 0:
        return []
    score = reg.p_burn.copy()
    idx = np.argsort(score.ravel())[::-1][:s]
    return [np.unravel_index(i, score.shape) for i in idx]


def choose_suppress_oracle(reg, s):
    if s <= 0:
        return []
    burning = (reg.state == 1).astype(float)
    score = 10.0 * burning + neighbor_sum(burning) * (reg.state == 0)
    idx = np.argsort(score.ravel())[::-1][:s]
    return [np.unravel_index(i, score.shape) for i in idx]


# ---------------------------------------------------------------- rollout
def simulate_epoch(reg, m, P):
    mon = choose_monitor(reg, m)
    for c in mon:
        reg.observe(c)

    s = int(np.floor((P["budget"] - P["w_m"] * m) / P["w_s"]))
    s = max(0, s)
    sup = choose_suppress(reg, s)

    reg.drop(sup)
    for _ in range(P["T"]):
        reg.spread()
    for _ in range(P["T"]):
        reg.predict_belief()

    return reg.burned() * P["value"] + P["c_m"] * m + P["c_s"] * s


def rollout_m(reg, m, P, H=2, R=3):
    costs = []
    for _ in range(R):
        sim = copy.deepcopy(reg)
        sim.state = sim.sample_state()
        sim.age[:] = 0
        total = 0.0
        for h in range(H):
            total += (P["beta"] ** h) * simulate_epoch(sim, m, P)
        costs.append(total)
    return float(np.mean(costs))


def best_m(reg, P):
    m_max = min(int(np.floor(P["budget"] / P["w_m"])), P.get("m_cap", 4))
    opts = list(range(0, m_max + 1))
    costs = [rollout_m(reg, m, P) for m in opts]
    return opts[int(np.argmin(costs))], costs


# ---------------------------------------------------------------- episode
def run_episode(policy, seed, P, keep_trace=False):
    rng = np.random.default_rng(seed * 1000 + 7)
    regs = [Region(size=P["size"], p_spread=P["p_spread"], burn_time=P["burn_time"],
                   q_supp=P["q_supp"], wet_time=P["wet_time"],
                   mon_radius=P["mon_radius"], seed=seed * 10 + i)
            for i in range(2)]

    total_cost = 0.0
    total_burned = 0
    mon_count = 0
    sup_count = 0
    trace = []

    for t in range(P["E"]):
        for ri, reg in enumerate(regs):
            if policy == "fopt" or policy == "oracle":
                m = 0
            elif policy == "greedy":
                m = 0
            elif policy == "random":
                m = int(rng.integers(0, 5))
            elif policy == "myopic":
                m = 2
            elif policy == "copt":
                # central planner: scan at epoch 0, keeping at least 2 drops
                if t == 0:
                    m = min(4, int(np.floor((P["budget"] - 2) / P["w_m"])))
                    m = max(0, m)
                else:
                    m = 0
            else:
                raise ValueError(policy)

            s = int(np.floor((P["budget"] - P["w_m"] * m) / P["w_s"]))
            s = max(0, s)

            if policy == "random":
                mon = [(int(rng.integers(0, P["size"])),
                        int(rng.integers(0, P["size"]))) for _ in range(m)]
            else:
                mon = choose_monitor(reg, m)

            for c in mon:
                reg.observe(c)

            if policy == "random":
                cells = rng.choice(P["size"] * P["size"], size=s, replace=False)
                sup = [np.unravel_index(i, (P["size"], P["size"])) for i in cells]
            elif policy == "greedy":
                sup = choose_suppress_greedy(reg, s)
            elif policy == "oracle":
                sup = choose_suppress_oracle(reg, s)
            else:
                sup = choose_suppress(reg, s)

            if keep_trace and ri == 0:
                trace.append(dict(
                    t=t,
                    state=reg.state.copy(),
                    p_burn=reg.p_burn.copy(),
                    p_burnt=reg.p_burnt.copy(),
                    mon=list(mon),
                    sup=list(sup),
                    mon_radius=reg.mon_radius,
                ))



            for c in mon:
                reg.observe(c)
            reg.drop(sup)
            for _ in range(P["T"]):
                reg.spread()
            for _ in range(P["T"]):
                reg.predict_belief()

            total_cost += reg.burned() * P["value"] + P["c_m"] * m + P["c_s"] * s
            total_burned += reg.burned()
            mon_count += m
            sup_count += s

    final_burned = sum(r.burned() for r in regs)
    return total_cost, total_burned, mon_count, sup_count, final_burned, trace


# ---------------------------------------------------------------- RL baseline
def q_index(reg, t, P):
    maxp = float(reg.p_burn.max())
    ent = float(entropy3(reg.p_burn, reg.p_burnt).mean())
    burned = float(reg.p_burnt.mean())
    b1 = min(4, int(maxp * 5))
    b2 = min(4, int(ent / 1.1 * 5))
    b3 = min(4, int(burned * 5))
    return (t, b1, b2, b3)


def q_step(reg, m, P):
    s = int(np.floor((P["budget"] - P["w_m"] * m) / P["w_s"]))
    s = max(0, s)
    mon = choose_monitor(reg, m)
    for c in mon:
        reg.observe(c)
    sup = choose_suppress(reg, s)
    reg.drop(sup)
    for _ in range(P["T"]):
        reg.spread()
    for _ in range(P["T"]):
        reg.predict_belief()
    cost = reg.burned() * P["value"] + P["c_m"] * m + P["c_s"] * s
    return cost, s


def train_q(P, episodes=400, alpha=0.10):
    Q = np.zeros((P["E"], 5, 5, 5, 5))
    curve = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        rng = np.random.default_rng(10000 + ep)
        regs = [Region(size=P["size"], p_spread=P["p_spread"], burn_time=P["burn_time"],
                       q_supp=P["q_supp"], wet_time=P["wet_time"],
                       mon_radius=P["mon_radius"], seed=ep * 10 + i)
                for i in range(2)]
        total = 0.0
        for t in range(P["E"]):
            for reg in regs:
                idx = q_index(reg, t, P)
                if rng.random() < eps:
                    a = int(rng.integers(0, 5))
                else:
                    a = int(np.argmax(Q[idx]))

                cost, _ = q_step(reg, a, P)
                total += cost
                r = -cost

                if t + 1 < P["E"]:
                    idx2 = q_index(reg, t + 1, P)
                    target = r + P["beta"] * float(np.max(Q[idx2]))
                else:
                    target = r
                Q[idx][a] += alpha * (target - Q[idx][a])
        curve.append(total)
    return Q, curve


def eval_q(Q, P, M=100):
    rows = []
    for seed in range(M):
        regs = [Region(size=P["size"], p_spread=P["p_spread"], burn_time=P["burn_time"],
                       q_supp=P["q_supp"], wet_time=P["wet_time"],
                       mon_radius=P["mon_radius"], seed=seed * 10 + i)
                for i in range(2)]
        total = 0.0
        mon = 0
        burned = 0
        for t in range(P["E"]):
            for reg in regs:
                idx = q_index(reg, t, P)
                a = int(np.argmax(Q[idx]))
                cost, _ = q_step(reg, a, P)
                total += cost
                mon += a
                burned += reg.burned()
        final = sum(r.burned() for r in regs)
        rows.append((total, burned, mon, final))
    arr = np.asarray(rows, dtype=float)
    return dict(cost_mean=float(arr[:, 0].mean()),
                cost_std=float(arr[:, 0].std()),
                burned_mean=float(arr[:, 1].mean()),
                mon_mean=float(arr[:, 2].mean()),
                final_mean=float(arr[:, 3].mean()),
                contained_rate=float((arr[:, 3] <= 5).mean()))


def q_trace(Q, P, seed=0):
    reg = Region(size=P["size"], p_spread=P["p_spread"], burn_time=P["burn_time"],
                 q_supp=P["q_supp"], wet_time=P["wet_time"],
                 mon_radius=P["mon_radius"], seed=seed)
    acts = []
    for t in range(P["E"]):
        a = int(np.argmax(Q[q_index(reg, t, P)]))
        acts.append(a)
        q_step(reg, a, P)
    return acts


def plot_rl(curve, acts_rl, acts_copt, P, path):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax[0].plot(curve, color="#718096", alpha=0.35, label="episode")
    k = 20
    if len(curve) >= k:
        ma = np.convolve(curve, np.ones(k) / k, mode="valid")
        ax[0].plot(range(k - 1, len(curve)), ma, color="#2b6cb0", lw=2, label="moving average")
    ax[0].set_xlabel("training episode")
    ax[0].set_ylabel("total cost")
    ax[0].set_title("Q-learning: when to monitor")
    ax[0].legend()

    x = np.arange(P["E"])
    w = 0.27
    ax[1].bar(x - w, [0] * P["E"], width=w, color="#c53030", label="FOpt")
    ax[1].bar(x, acts_copt, width=w, color="#2b6cb0", label="COpt (scan once)")
    ax[1].bar(x + w, acts_rl, width=w, color="#805ad5", label="Q-learning")
    ax[1].set_xticks(x)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("monitoring actions")
    ax[1].set_title("learned monitoring schedule")
    ax[1].legend(fontsize=8)
    ax[1].text(0.02, 0.06, "FOpt = 0 at every epoch", transform=ax[1].transAxes,
               fontsize=8, color="#c53030")

    fig.suptitle("RL baseline: learned allocation between monitoring and suppression")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- plots
def plot_snapshots(trace, path):
    n = len(trace)
    fig, axes = plt.subplots(n, 3, figsize=(9.2, 2.7 * n))
    cmap_fire = ListedColormap(["#e9efdf", "#e34a33", "#2f2f2f"])

    for i, snap in enumerate(trace):
        box = 2 * snap["mon_radius"] + 1
        half = (box - 1) / 2.0

        # true fire
        ax = axes[i, 0]
        ax.imshow(snap["state"], cmap=cmap_fire, vmin=0, vmax=2)
        for (r, c) in snap["mon"]:
            ax.add_patch(Rectangle((c - half, r - half), box, box, fill=False,
                                   edgecolor="#2b6cb0", lw=2.0))
        for (r, c) in snap["sup"]:
            ax.plot(c, r, marker="x", color="#00d5ff", ms=9, mew=2)
        ax.set_title(f"true fire  (epoch {snap['t']})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        # belief
        ax = axes[i, 1]
        im = ax.imshow(snap["p_burn"], cmap="inferno", vmin=0, vmax=1)
        for (r, c) in snap["mon"]:
            ax.add_patch(Rectangle((c - half, r - half), box, box, fill=False,
                                   edgecolor="#2b6cb0", lw=2.0))
        ax.set_title("belief: P(burning) [0–1]", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        # uncertainty
        ax = axes[i, 2]
        h = entropy3(snap["p_burn"], snap["p_burnt"])
        ax.imshow(h, cmap="viridis", vmin=0, vmax=1.1)
        ax.set_title("uncertainty [0–1.1]", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("HPO-CRMAB on a 10x10 grid: COpt policy\n"
                 "blue box = monitoring footprint, cyan x = suppression", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bars(results, path):
    policies = ["random", "greedy", "fopt", "myopic", "copt", "rl", "oracle"]
    labels = ["Random", "Greedy", "FOpt", "Myopic", "COpt", "Q-learn", "Oracle"]
    colors = ["#718096", "#a0522d", "#c53030", "#dd6b20", "#2b6cb0", "#805ad5", "#2f855a"]

    means = [results[p]["cost_mean"] for p in policies]
    stds = [results[p]["cost_std"] for p in policies]
    contained = [100.0 * results[p]["contained_rate"] for p in policies]
    mon = [results[p]["mon_mean"] for p in policies]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].bar(labels, means, yerr=[np.zeros(len(stds)), stds],
                color=colors, capsize=5)
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("total cost (lower is better)")
    axes[0].tick_params(axis="x", labelsize=8, rotation=15)

    axes[1].bar(labels, contained, color=colors)
    axes[1].set_ylim(0, 105)
    axes[1].set_title("containment rate (%)")
    axes[1].tick_params(axis="x", labelsize=8, rotation=15)

    axes[2].bar(labels, mon, color=colors)
    axes[2].set_title("monitoring actions")
    axes[2].tick_params(axis="x", labelsize=8, rotation=15)

    fig.suptitle("Two regions, 10x10 grid each, 6 epochs\n"
                 "FOpt = delegated (no monitoring), COpt = centralised scan, Q-learn = learned schedule",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sweep(rows, path):
    wm = [r[0] for r in rows]
    f = [r[1] for r in rows]
    c = [r[3] for r in rows]
    gap = [r[1] - r[3] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(wm, f, "o-", color="#c53030", label="FOpt (no monitor)")
    ax[0].plot(wm, c, "o-", color="#2b6cb0", label="COpt (scan once)")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("monitoring price (budget units per monitor)")
    ax[0].set_ylabel("total cost")
    ax[0].legend()
    ax[1].bar([f"{w:g}" for w in wm], gap, color="#2b6cb0")
    ax[1].set_xlabel("monitoring price")
    ax[1].set_ylabel("FOpt - COpt gap")
    fig.suptitle("Value of information vs price of monitoring")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_sweep(P, M=50):
    rows = []
    for wm in [0.25, 0.5, 1.0, 2.0, 4.0]:
        Q = dict(P)
        Q["w_m"] = wm
        f = [run_episode("fopt", s, Q)[0] for s in range(M)]
        c = [run_episode("copt", s, Q)[0] for s in range(M)]
        rows.append((wm, float(np.mean(f)), float(np.std(f)),
                     float(np.mean(c)), float(np.std(c))))
    return rows


def make_gif(P, path, seed=0):
    reg = Region(size=P["size"], p_spread=P["p_spread"], burn_time=P["burn_time"],
                 q_supp=P["q_supp"], wet_time=P["wet_time"],
                 mon_radius=P["mon_radius"], seed=seed)
    cmap_fire = ListedColormap(["#e9efdf", "#e34a33", "#2f2f2f"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    frames = []
    mon, sup = [], []

    def snap(title):
        for ax in axes:
            ax.clear()
        axes[0].imshow(reg.state, cmap=cmap_fire, vmin=0, vmax=2)
        axes[0].set_title("true fire", fontsize=9)
        axes[1].imshow(reg.p_burn, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title("belief: P(burning)", fontsize=9)
        box = 2 * reg.mon_radius + 1
        half = (box - 1) / 2.0
        for ax in axes:
            for (r, c) in mon:
                ax.add_patch(Rectangle((c - half, r - half), box, box,
                                       fill=False, edgecolor="#2b6cb0", lw=1.5))
            for (r, c) in sup:
                ax.plot(c, r, "x", color="#00d5ff", ms=7, mew=1.5)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title, fontsize=10)
        fig.text(0.5, 0.015,
                 "green = unburnt, red = burning, dark = burnt   |   blue box = monitor, x = drop",
                 ha="center", fontsize=7, color="#444444")
        fig.tight_layout(rect=[0, 0.04, 1, 0.92])
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        frames.append(img.copy())

    for t in range(P["E"]):
        m = 4 if t == 0 else 0
        s = int(np.floor((P["budget"] - P["w_m"] * m) / P["w_s"]))
        s = max(0, s)
        mon = choose_monitor(reg, m)
        for c in mon:
            reg.observe(c)

        sup = []
        snap(f"epoch {t}: scan ({m} monitors) finds the fire")
        sup = choose_suppress(reg, s)
        reg.drop(sup)
        snap(f"epoch {t}: drop ({s} suppression actions)")
        for step in range(P["T"]):
            reg.spread()
            reg.predict_belief()
            snap(f"epoch {t}, step {step + 1}")

    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=250, loop=0, optimize=True)
    plt.close(fig)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    P = dict(E=6, T=5, budget=4, size=10, mon_radius=2, m_cap=4,
             w_m=0.25, w_s=1.0,
             p_spread=0.35, burn_time=2, q_supp=0.9, wet_time=3,
             c_m=0.10, c_s=0.20, value=1.0, beta=0.95)

    M = 100
    policies = ["random", "greedy", "fopt", "myopic", "copt", "oracle"]
    results = {}

    t0 = time.time()
    for pol in policies:
        rows = []
        for seed in range(M):
            cost, burned, mon, sup, final, _ = run_episode(pol, seed, P)
            rows.append((cost, burned, mon, sup, final))
        arr = np.asarray(rows, dtype=float)
        results[pol] = dict(
            cost_mean=float(arr[:, 0].mean()),
            cost_std=float(arr[:, 0].std()),
            burned_mean=float(arr[:, 1].mean()),
            mon_mean=float(arr[:, 2].mean()),
            sup_mean=float(arr[:, 3].mean()),
            final_mean=float(arr[:, 4].mean()),
            contained_rate=float((arr[:, 4] <= 5).mean()),
        )
        print(f"{pol:7s}  cost={arr[:,0].mean():8.2f}  burned={arr[:,1].mean():7.1f}  "
              f"mon={arr[:,2].mean():5.1f}  sup={arr[:,3].mean():6.1f}  "
              f"contained={100*(arr[:,4] <= 5).mean():5.1f}%")

    # one traced COpt episode for the figure
    _, _, _, _, _, trace = run_episode("copt", 0, P, keep_trace=True)

    print("training Q-learning baseline ...")
    Q, curve = train_q(P, episodes=400)
    results["rl"] = eval_q(Q, P, M=100)
    print(f"rl      cost={results['rl']['cost_mean']:8.2f}  "
          f"burned={results['rl']['burned_mean']:7.1f}  "
          f"mon={results['rl']['mon_mean']:5.1f}")
    acts_rl = q_trace(Q, P, seed=0)
    m_first = min(4, int(np.floor((P["budget"] - 2) / P["w_m"])))
    acts_copt = [max(0, m_first)] + [0] * (P["E"] - 1)
    plot_rl(curve, acts_rl, acts_copt, P, os.path.join(OUT, "rl_training.png"))

    plot_snapshots(trace, os.path.join(OUT, "grid_snapshots.png"))
    plot_bars(results, os.path.join(OUT, "policy_bars.png"))
    make_gif(P, os.path.join(OUT, "simulation.gif"), seed=0)

    sweep = run_sweep(P, M=50)
    plot_sweep(sweep, os.path.join(OUT, "sweep.png"))

    gap = results["fopt"]["cost_mean"] - results["copt"]["cost_mean"]
    summary = dict(
        params=P,
        episodes=M,
        results=results,
        fopt_minus_copt=float(gap),
        sweep=[dict(w_m=r[0], fopt=r[1], fopt_std=r[2],
                    copt=r[3], copt_std=r[4]) for r in sweep],
        runtime_sec=float(time.time() - t0),
    )
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 60)
    print(f"FOpt - COpt cost gap : {gap:8.2f}")
    print(f"monitoring actions   : FOpt={results['fopt']['mon_mean']:.1f}  "
          f"COpt={results['copt']['mon_mean']:.1f}")
    print(f"burned cells         : FOpt={results['fopt']['burned_mean']:.1f}  "
          f"COpt={results['copt']['burned_mean']:.1f}  "
          f"Oracle={results['oracle']['burned_mean']:.1f}")
    print(f"runtime              : {time.time() - t0:.1f} s")
    print(f"figures              : {OUT}")
