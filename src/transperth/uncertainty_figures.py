"""P2.5 figures; intervals show trigger Monte Carlo uncertainty only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transperth.plotting import apply_style, save_figure
from transperth.tail_fit import discrete_pmf


def render_uncertainty_figures(
    failure: pd.DataFrame,
    collapse: pd.DataFrame,
    convergence: pd.DataFrame,
    paired: pd.DataFrame,
    avalanche: pd.DataFrame,
    fits: Sequence[Mapping[str, Any]],
    grid: pd.DataFrame,
    directory: str | Path,
) -> list[Path]:
    apply_style()
    paths = []
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(failure.fraction, failure["mean"], color="C0")
    ax.fill_between(
        failure.fraction,
        failure.low,
        failure.high,
        color="C0",
        alpha=0.25,
        label="95% percentile bootstrap CI",
    )
    ax.set(
        xlabel="Removed station fraction",
        ylabel="GCC / original 86 stations",
        title="Random station failures · fixed rail graph",
        ylim=(-0.02, 1.02),
    )
    ax.legend()
    fig.tight_layout()
    paths.append(save_figure(fig, "random_failure_ci.png", figures_dir=directory))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, rule in zip(axes, ("equal", "capacity")):
        selected = collapse[
            collapse.rule.eq(rule) & collapse.metric.eq("failed_fraction")
        ]
        ax.plot(selected.alpha, selected["mean"])
        ax.fill_between(
            selected.alpha,
            selected.low,
            selected.high,
            alpha=0.25,
            label="95% bootstrap CI",
        )
        ax.set(
            xlabel="Tolerance α",
            ylabel="Mean failed fraction",
            title=f"{rule} redistribution · static",
            ylim=(-0.02, 1.02),
        )
        ax.legend()
    fig.tight_layout()
    paths.append(save_figure(fig, "cascade_ci.png", figures_dir=directory))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for family, group in convergence.groupby("family"):
        worst = group.groupby("n")[["absolute_change", "half_width"]].max()
        for ax, metric in zip(axes, ("absolute_change", "half_width")):
            ax.plot(worst.index, worst[metric], marker="o", label=family)
    for ax, threshold, title in zip(
        axes,
        (0.02, 0.03),
        ("Worst mean change from full sample", "Worst 95% CI half width"),
    ):
        ax.axhline(threshold, color=".4", ls="--", label=f"Target {threshold:g}")
        ax.set(
            xlabel="Nested sample size",
            ylabel="Fraction units",
            title=title,
            xscale="log",
        )
        ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(save_figure(fig, "seed_convergence.png", figures_dir=directory))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, (comparison, selected_dynamic) in zip(
        axes.flat,
        [
            (comparison, dynamic)
            for dynamic in (False, True)
            for comparison in ("capacity_minus_equal", "load_minus_betweenness")
        ],
    ):
        for (dynamic, mode), group in paired[paired.comparison.eq(comparison)].groupby(
            ["dynamic", "load_mode"]
        ):
            if dynamic != selected_dynamic:
                continue
            ax.plot(group.alpha, group["mean"], label=mode)
            ax.fill_between(group.alpha, group.low, group.high, alpha=0.15)
        ax.axhline(0, color=".4", ls="--")
        ax.set(
            xlabel="Tolerance α",
            ylabel="Paired failed-fraction difference",
            title=comparison.replace("_", " ")
            + f" · {'dynamic' if selected_dynamic else 'static'}",
        )
        ax.legend(fontsize=7)
    fig.suptitle(
        "Paired model sensitivity · 95% CIs · demand also changes reference capacity"
    )
    fig.tight_layout()
    paths.append(save_figure(fig, "paired_sensitivity.png", figures_dir=directory))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, fit in zip(axes.flat, [item for item in fits if not item["dynamic"]]):
        values = avalanche[
            ~avalanche.dynamic & avalanche.alpha.eq(fit["alpha"])
        ].post_trigger_size.to_numpy()
        support = np.arange(1, fit["upper"] + 1)
        empirical = np.array([np.mean(values >= value) for value in support])
        ax.step(support, empirical, where="post", label="Empirical CCDF")
        if "exponent" in fit:
            support, probability = discrete_pmf(
                fit["exponent"], fit["xmin"], fit["upper"]
            )
            ax.plot(
                support,
                probability[::-1].cumsum()[::-1] * fit["n_tail"] / fit["n"],
                ls="--",
                label="Bounded power-law fit",
            )
        text = fit["status"].replace("_", " ")
        if fit["p_value"] is not None:
            text += f"\np={fit['p_value']:.4f}; KS={fit['ks']:.3f}"
        ax.text(0.03, 0.07, text, transform=ax.transAxes, fontsize=8)
        ax.set(
            title=f"α={fit['alpha']:g} · P(S=0)={fit['p_zero']:.3f}",
            xlabel="Secondary failures S",
            ylabel="P(S ≥ s)",
            xscale="log",
            yscale="log",
            xlim=(1, 86),
            ylim=(0.001, 1.05),
        )
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Separate fixed-condition tail tests · dynamic samples contain only zeros"
    )
    fig.tight_layout()
    paths.append(save_figure(fig, "tail_diagnostics.png", figures_dir=directory))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for rule, group in grid.groupby("rule"):
        axes[0].plot(group.step, group.alpha_star, marker="o", label=rule)
        axes[1].plot(group.step, group.max_interpolation_error, marker="o", label=rule)
    for ax, title in zip(
        axes,
        (
            "First sampled targeted containment α",
            "Worst random-mean interpolation error",
        ),
    ):
        ax.set(xlabel="Alpha grid step", title=title)
        ax.legend()
    fig.tight_layout()
    paths.append(save_figure(fig, "alpha_grid_sensitivity.png", figures_dir=directory))
    return paths
