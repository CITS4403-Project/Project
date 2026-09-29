"""Yard-sale wealth-exchange model: mean-field and network variants.

Prototype written from the published model equations for the CITS4403
investigation "idea 1: modification of the yard-sale model".  No code was
copied from any existing implementation.

Model
-----
N agents hold wealth w_i >= 0.  At each elementary trade a pair of agents
(i, j) is selected - a uniformly random pair in the mean-field variant, a
uniformly random edge in the network variant.  An amount

    dw = f * min(w_i, w_j)

is transferred from the loser to the winner, where f in [0, 1) is the fraction
of the poorer agent's wealth at stake.  The richer agent wins with probability
p (p = 0.5 is the unbiased yard-sale model, p > 0.5 is a wealth-attained
advantage); exact ties are resolved by a fair coin.  Total wealth is conserved
by construction.

Extensions implemented
----------------------
* topology : "complete" (mean field), "ER", "BA" (scale-free), "WS"
  (small-world), "lattice" (2-D square grid, periodic-free).
* f        : transaction size (fraction of the poorer agent's wealth).
* p        : wealth-attained advantage / bias (p = 0.5 unbiased).
* tax      : proportional tax with equal redistribution.
    - tax_mode="wealth": each sweep every agent pays tau * w_i, the pool is
      split equally.  (Boghosian-style redistribution.)
    - tax_mode="gain"  : each transaction skims tau * dw from the winner's
      winnings and adds it to a pool that is split equally at sweep end.
* rewiring : per-sweep random edge rewiring (dynamic network).  A random edge
  is detached from one endpoint and re-attached to a uniformly random node it
  is not already connected to.  Edge count is conserved.

Update convention
-----------------
One sweep performs N//2 trades (mean-field pairing of a random permutation,
or N//2 edges sampled without replacement on a network).  This is the
standard discrete-time "parallel" update used in the kinetic-exchange
literature and scales cleanly with N.  On networks an agent can take part in
several trades per sweep; trades are evaluated in chunks of `chunk` from the
wealth snapshot at the chunk start, and a loss cap guarantees that no wealth
goes negative.  The cap never binds in the mean-field variant and its
activation count is tracked (and negligible in the ranges studied here).

References (model definitions only, code written from scratch):
  A. Chakraborti, Int. J. Mod. Phys. C 13, 1315 (2002).
  C. F. Moukarzel et al., Eur. Phys. J. Spec. Top. 143, 75 (2007).
  B. M. Boghosian et al., Physica A 476, 15 (2017), arXiv:1511.00770.
  G. L. Kohlrausch & S. Goncalves, Physica A 632, 129339 (2024),
  arXiv:2302.03677 (dynamic-network extension).
"""

from __future__ import annotations

import time

import numpy as np

try:
    import networkx as nx
except Exception:  # pragma: no cover - networkx is part of the project venv
    nx = None

TOPOLOGIES = ("complete", "ER", "BA", "WS", "lattice")


# --------------------------------------------------------------------------
# network construction
# --------------------------------------------------------------------------
def build_network(topology, n, k=4, seed=None, ws_p=0.1):
    """Build a networkx graph on integer nodes 0..n-1 for `topology`."""
    if nx is None:
        raise ImportError("networkx is required for network topologies")
    if topology == "complete":
        return nx.complete_graph(n)
    if topology == "ER":
        m = int(round(n * k / 2))
        return nx.gnm_random_graph(n, m, seed=seed)
    if topology == "BA":
        m = max(1, int(round(k / 2)))
        return nx.barabasi_albert_graph(n, m, seed=seed)
    if topology == "WS":
        kk = k + (k % 2)
        return nx.watts_strogatz_graph(n, kk, ws_p, seed=seed)
    if topology == "lattice":
        side = int(round(np.sqrt(n)))
        if side * side != n:
            raise ValueError("lattice topology requires n to be a perfect square")
        g = nx.grid_2d_graph(side, side)
        return nx.convert_node_labels_to_integers(g)
    raise ValueError(f"unknown topology {topology!r}")


# --------------------------------------------------------------------------
# inequality statistics
# --------------------------------------------------------------------------
def gini(w):
    """Gini coefficient of a non-negative wealth vector (0 = equal, 1 = max)."""
    w = np.asarray(w, dtype=float)
    n = w.size
    tot = w.sum()
    if tot <= 0:
        return 0.0
    ws = np.sort(w)
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.dot(idx, ws)) / (n * tot) - (n + 1.0) / n)


def top_share(w, frac):
    """Share of total wealth held by the richest `frac` of agents."""
    w = np.asarray(w, dtype=float)
    tot = w.sum()
    if tot <= 0:
        return 0.0
    k = max(1, int(round(frac * w.size)))
    return float(np.sort(w)[-k:].sum() / tot)


def max_share(w):
    """Wealth share of the single richest agent (condensation order parameter)."""
    w = np.asarray(w, dtype=float)
    tot = w.sum()
    return 0.0 if tot <= 0 else float(w.max() / tot)


def hill_alpha(w, frac=0.1):
    """Hill tail exponent estimate for the richest `frac` of agents.

    Returns alpha where P(W > x) ~ x^-alpha (Pareto).  alpha -> large for a
    light tail; alpha near 1 signals a heavy (Zipf-like) tail.
    """
    w = np.sort(np.asarray(w, dtype=float))[::-1]
    k = max(2, int(round(frac * w.size)))
    tail = w[:k]
    xmin = tail[-1]
    if xmin <= 0 or np.any(tail <= 0):
        return float("nan")
    return float(1.0 + k / np.sum(np.log(tail / xmin)))


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
class YardSaleModel:
    """Yard-sale wealth exchange with optional networks, bias, tax, rewiring."""

    def __init__(
        self,
        n=400,
        f=0.2,
        p=0.5,
        tax=0.0,
        tax_mode="wealth",
        rewiring=0.0,
        topology="complete",
        k=4,
        seed=None,
        w0=None,
        ws_p=0.1,
        loss_cap=0.99,
        chunk=16,
    ):
        if not (0.0 <= f < 1.0):
            raise ValueError("f must be in [0, 1)")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        if tax_mode not in ("wealth", "gain"):
            raise ValueError("tax_mode must be 'wealth' or 'gain'")

        self.n = int(n)
        self.f = float(f)
        self.p = float(p)
        self.tax = float(tax)
        self.tax_mode = tax_mode
        self.rewiring = float(rewiring)
        self.topology = topology
        self.k = int(k)
        self.ws_p = float(ws_p)
        self.loss_cap = float(loss_cap)
        self.chunk = int(chunk)
        self.rng = np.random.default_rng(seed)

        if w0 is None:
            self.w = np.ones(self.n, dtype=float)
        else:
            self.w = np.asarray(w0, dtype=float).copy()
            if self.w.size != self.n:
                raise ValueError("w0 size does not match n")

        self.t = 0
        self.n_capped = 0
        self.wealth_pool = 0.0  # gain-tax proceeds awaiting redistribution
        self.pool_collected = 0.0  # cumulative tax revenue (all modes)

        # network setup -----------------------------------------------------
        self.graph = None
        self.edges = None
        self.adj = None
        if topology == "complete":
            self.mode = "meanfield"
        else:
            if topology not in TOPOLOGIES:
                raise ValueError(f"unknown topology {topology!r}")
            gseed = int(self.rng.integers(0, 2**31 - 1))
            self.graph = build_network(topology, self.n, k=self.k, seed=gseed, ws_p=self.ws_p)
            if not nx.is_connected(self.graph):
                # for the prototype, just report it; trading still works
                self.graph_connected = False
            else:
                self.graph_connected = True
            self.edges = np.array(sorted(tuple(sorted(e)) for e in self.graph.edges()), dtype=np.int64)
            self.adj = [set() for _ in range(self.n)]
            for u, v in self.edges:
                self.adj[u].add(int(v))
                self.adj[v].add(int(u))
            self.mode = "network"

    # ------------------------------------------------------------------
    @property
    def total_wealth(self):
        return float(self.w.sum() + self.wealth_pool)

    # ------------------------------------------------------------------
    def _trade_batch(self, u, v):
        """Vectorised batch of trades between partner arrays u, v.

        Trades use the wealth snapshot at batch entry ("parallel" update).
        A per-agent loss cap (fraction of snapshot wealth) keeps wealth
        non-negative; exact conservation holds because the same effective
        amount is removed from the loser and added to the winner.
        """
        w = self.w
        wu = w[u]
        wv = w[v]
        dw = self.f * np.minimum(wu, wv)
        r = self.rng.random(u.shape[0])
        # does u (the richer) win?  ties -> fair coin
        u_wins = np.where(
            wu > wv,
            r < self.p,
            np.where(wv > wu, r >= self.p, r < 0.5),
        )
        loser = np.where(u_wins, v, u)
        amt = dw  # full amount debited from the loser

        # loss cap ---------------------------------------------------------
        loss_tot = np.bincount(loser, weights=amt, minlength=self.n)
        cap = self.loss_cap * w[loser]
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(
                loss_tot[loser] > 0.0,
                np.minimum(1.0, cap / np.maximum(loss_tot[loser], 1e-300)),
                1.0,
            )
        self.n_capped += int(np.count_nonzero(scale < 1.0))
        amt = amt * scale

        winner = np.where(u_wins, u, v)
        # optional tax on the winner's winnings: full `amt` leaves the loser,
        # `amt - skim` reaches the winner, `skim` goes to the pool.
        if self.tax_mode == "gain" and self.tax > 0.0:
            skim = self.tax * amt * u_wins
            self.wealth_pool += float(skim.sum())
            self.pool_collected += float(skim.sum())
            gain = amt - skim
        else:
            gain = amt
        self.w = self.w - np.bincount(loser, weights=amt, minlength=self.n)
        self.w = self.w + np.bincount(winner, weights=gain, minlength=self.n)

    # ------------------------------------------------------------------
    def _rewire(self, n_attempts):
        for _ in range(n_attempts):
            e = int(self.rng.integers(self.edges.shape[0]))
            u, v = int(self.edges[e, 0]), int(self.edges[e, 1])
            if self.rng.random() < 0.5:
                u, v = v, u
            new = -1
            for _try in range(10):
                cand = int(self.rng.integers(self.n))
                if cand != u and cand not in self.adj[u]:
                    new = cand
                    break
            if new < 0:
                continue
            self.adj[u].discard(v)
            self.adj[v].discard(u)
            self.adj[u].add(new)
            self.adj[new].add(u)
            self.edges[e, 0] = u
            self.edges[e, 1] = new

    # ------------------------------------------------------------------
    def sweep(self):
        """Advance the model by one sweep (N // 2 elementary trades)."""
        n_trades = self.n // 2
        if self.mode == "meanfield":
            perm = self.rng.permutation(self.n)
            if self.n % 2:
                perm = perm[:-1]
            self._trade_batch(perm[0::2], perm[1::2])
        else:
            idx = self.rng.choice(self.edges.shape[0], size=n_trades, replace=False)
            e = self.edges[idx]
            for s in range(0, n_trades, self.chunk):
                self._trade_batch(e[s : s + self.chunk, 0], e[s : s + self.chunk, 1])

        if self.tax > 0.0:
            n = self.n
            if self.tax_mode == "wealth":
                pool = self.tax * self.w.sum()
                self.pool_collected += pool
                self.w *= (1.0 - self.tax)
                self.w += pool / n
            else:  # gain tax: distribute the pool accumulated this sweep
                if self.wealth_pool > 0.0:
                    self.w += self.wealth_pool / n
                    self.wealth_pool = 0.0

        if self.rewiring > 0.0 and self.edges is not None:
            n_attempts = max(1, int(round(self.rewiring * self.edges.shape[0])))
            self._rewire(n_attempts)

        self.t += 1


# --------------------------------------------------------------------------
# simulation driver
# --------------------------------------------------------------------------
def simulate(
    n=400,
    f=0.2,
    p=0.5,
    tax=0.0,
    tax_mode="wealth",
    rewiring=0.0,
    topology="complete",
    k=4,
    sweeps=2000,
    seed=0,
    sample_every=10,
    w0=None,
    ws_p=0.1,
    cond_threshold=0.5,
    stop_on_cond=False,
):
    """Run one simulation and return a dict of time series and summary stats."""
    model = YardSaleModel(
        n=n,
        f=f,
        p=p,
        tax=tax,
        tax_mode=tax_mode,
        rewiring=rewiring,
        topology=topology,
        k=k,
        seed=seed,
        w0=w0,
        ws_p=ws_p,
    )
    w0_total = model.total_wealth

    t0 = time.perf_counter()
    times, ginis, top1, top10, maxsh = [0], [gini(model.w)], [top_share(model.w, 0.01)], [
        top_share(model.w, 0.10)
    ], [max_share(model.w)]
    drift = 0.0
    t_cond = None

    for t in range(1, sweeps + 1):
        model.sweep()
        ms = max_share(model.w)
        if t_cond is None and ms > cond_threshold:
            t_cond = t
        if t % sample_every == 0 or t == sweeps:
            times.append(t)
            ginis.append(gini(model.w))
            top1.append(top_share(model.w, 0.01))
            top10.append(top_share(model.w, 0.10))
            maxsh.append(ms)
            drift = max(drift, abs(model.total_wealth - w0_total) / max(abs(w0_total), 1e-300))
        if stop_on_cond and t_cond is not None:
            break
    wall = time.perf_counter() - t0

    return {
        "params": {
            "n": n,
            "f": f,
            "p": p,
            "tax": tax,
            "tax_mode": tax_mode,
            "rewiring": rewiring,
            "topology": topology,
            "k": k,
            "seed": seed,
            "sweeps": sweeps,
        },
        "times": np.array(times),
        "gini": np.array(ginis),
        "top1": np.array(top1),
        "top10": np.array(top10),
        "max_share": np.array(maxsh),
        "t_cond": t_cond,
        "final_gini": float(ginis[-1]),
        "final_top10": float(top10[-1]),
        "final_max_share": float(maxsh[-1]),
        "max_rel_drift": drift,
        "n_capped": model.n_capped,
        "wall_seconds": wall,
        "hill_alpha": hill_alpha(model.w, 0.10),
    }


# --------------------------------------------------------------------------
def selftest(verbose=True):
    """Analytic and invariant checks.  Returns a list of (name, ok, detail)."""
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # 1. wealth conservation across every topology and tax mode -----------
    for topo in TOPOLOGIES:
        for tax_mode in ("wealth", "gain"):
            r = simulate(
                n=100,
                f=0.3,
                p=0.6,
                tax=0.02,
                tax_mode=tax_mode,
                topology=topo,
                k=4,
                sweeps=200,
                seed=7,
                sample_every=1,
            )
            add(
                f"conservation {topo}/{tax_mode}",
                r["max_rel_drift"] < 1e-10,
                f"max rel drift {r['max_rel_drift']:.2e}, capped {r['n_capped']}",
            )

    # 2. two-agent exact check: p=1, w=(2,1), f=0.5 -> (2.5, 0.5) ---------
    m = YardSaleModel(n=2, f=0.5, p=1.0, topology="complete", seed=1, w0=np.array([2.0, 1.0]))
    m._trade_batch(np.array([0]), np.array([1]))
    ok = np.allclose(np.sort(m.w), [0.5, 2.5]) and abs(m.w.sum() - 3.0) < 1e-12
    add("two-agent exact p=1 f=0.5", ok, f"w={np.sort(m.w)}")

    # 3. fair coin is a martingale: ensemble mean wealth is conserved -------
    #    (individual trajectories condense - non-ergodicity - so the time
    #    average of one trajectory is the wrong statistic; the martingale
    #    property is about the ensemble average E[w_i(t)] = w_i(0).)
    rng = np.random.default_rng(3)
    m0 = YardSaleModel(n=2, f=0.3, p=0.5, topology="complete", seed=3, w0=np.array([0.7, 0.3]))
    ens = np.empty(2000)
    for k in range(2000):
        mm = YardSaleModel(n=2, f=0.3, p=0.5, topology="complete", seed=1000 + k, w0=np.array([0.7, 0.3]))
        mm._trade_batch(np.array([0]), np.array([1]))
        ens[k] = mm.w[0]
    ok = abs(ens.mean() - 0.7) < 0.02
    add("fair-coin one-trade ensemble mean", ok, f"E[w0]={ens.mean():.4f} (exact 0.7)")

    # 3b. one-step martingale from a strongly skewed state, exact in
    #     expectation for every agent: E[dw_i] = 0 under a fair coin.
    w_skew = np.array([2.0, 1.0, 0.5, 0.01, 0.2])
    diffs = np.empty((120000, 5))
    for k in range(120000):
        m = YardSaleModel(n=5, f=0.3, p=0.5, topology="complete", seed=20000 + k, w0=w_skew)
        i, j = (0, 3) if k % 2 == 0 else (2, 4)
        m._trade_batch(np.array([i]), np.array([j]))
        diffs[k] = m.w - w_skew
    se = diffs.std(axis=0) / np.sqrt(diffs.shape[0])
    max_z = float(np.max(np.abs(diffs.mean(axis=0)) / np.maximum(se, 1e-300)))
    add("one-step martingale (skewed state)", max_z < 4.0, f"max |z| over 5 agents = {max_z:.2f}")

    # 3c. ensemble martingale over a short horizon with small f: the sample
    #     mean should converge; note that for larger t the estimator is
    #     dominated by rare condensed trajectories (non-ergodicity), which is
    #     itself the physical phenomenon - not tested here.
    w_init = rng.lognormal(mean=0.0, sigma=0.5, size=20)
    finals = np.empty((4000, 20))
    for k in range(4000):
        m = YardSaleModel(n=20, f=0.05, p=0.5, topology="complete", seed=5000 + k, w0=w_init)
        for _ in range(5):
            m.sweep()
        finals[k] = m.w
    worst = float(np.max(np.abs(finals.mean(axis=0) - w_init) / w_init))
    ok = worst < 0.05
    add("short-horizon ensemble martingale", ok, f"max rel. deviation of E[w_i] = {worst:.3f}")

    # 4. f -> 0 freezes the dynamics: Gini at fixed horizon shrinks -------
    g_small = [simulate(n=200, f=f, p=0.5, sweeps=400, seed=2, sample_every=400)["final_gini"] for f in (0.002, 0.05, 0.2)]
    ok = g_small[0] < g_small[1] < g_small[2]
    add("f->0 suppression at fixed time", ok, f"Gini(f=0.002,0.05,0.2)={['%.4f' % g for g in g_small]}")

    # 5. p=1: global richest never loses, max wealth non-decreasing --------
    m = YardSaleModel(n=50, f=0.2, p=1.0, topology="ER", k=4, seed=5)
    mx = [m.w.max()]
    for _ in range(300):
        m.sweep()
        mx.append(m.w.max())
    add("p=1 rich-never-lose", np.all(np.diff(mx) >= -1e-12), f"max wealth {mx[0]:.3f} -> {mx[-1]:.3f}")

    # 6. wealth tax tau=1 equalises perfectly after one sweep --------------
    m = YardSaleModel(n=50, f=0.2, p=0.6, tax=1.0, tax_mode="wealth", seed=6)
    m.sweep()
    add("wealth tax tau=1 -> equality", gini(m.w) < 1e-12, f"Gini={gini(m.w):.2e}")

    # 7. unbiased YSM Gini is a (weakly) increasing Lyapunov functional ----
    r = simulate(n=300, f=0.2, p=0.5, sweeps=2000, seed=9, sample_every=20)
    incr = np.mean(np.diff(r["gini"]) >= -0.02)
    add("fair YSM Gini non-decreasing (noised)", incr > 0.9, f"fraction non-decreasing = {incr:.3f}")

    # 8. complete-graph edge trading matches mean-field statistics --------
    a = simulate(n=200, f=0.25, p=0.6, topology="complete", sweeps=1500, seed=11, sample_every=1500)
    b = simulate(n=200, f=0.25, p=0.6, topology="ER", k=199, sweeps=1500, seed=11, sample_every=1500)
    ok = abs(a["final_gini"] - b["final_gini"]) < 0.05
    add("complete graph vs dense ER agreement", ok, f"Gini {a['final_gini']:.4f} vs {b['final_gini']:.4f}")

    n_fail = sum(not ok for _, ok, _ in checks)
    if verbose:
        print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed")
    return checks


if __name__ == "__main__":
    selftest()