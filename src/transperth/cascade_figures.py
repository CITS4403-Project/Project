"""Publication figures drawn from the saved P2.2 experiment tables."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from transperth.plotting import apply_style, save_figure


def render_cascade_figures(
    graph, full, sensitivity, avalanche, line_runs, maps, outcomes, directory: Path
) -> list[Path]:
    from transperth.cascade_experiments import empirical_ccdf

    apply_style()
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, dynamic in zip(axes, (False, True)):
        table = sensitivity if dynamic else full
        table = table[
            table.rule.eq("capacity")
            & table.dynamic.eq(dynamic)
            & table.load_mode.eq("betweenness")
        ]
        for trigger, label in [("load", "Max load"), ("degree", "Max degree")]:
            rows = table[table.trigger.eq(trigger)].sort_values("alpha")
            ax.plot(rows.alpha, rows.failed_fraction, label=label)
        random = (
            table[table.trigger.eq("random")]
            .groupby("alpha")
            .failed_fraction.agg(["mean", "std"])
        )
        ax.plot(random.index, random["mean"], label="Random mean", color="C2")
        ax.fill_between(
            random.index,
            np.clip(random["mean"] - random["std"], 0, 1),
            np.clip(random["mean"] + random["std"], 0, 1),
            color="C2",
            alpha=0.15,
            label="Random ±1 SD",
        )
        ax.set(
            title=f"{'Dynamic routing' if dynamic else 'Static redistribution'}",
            xlabel="Tolerance α",
            ylim=(-0.02, 1.02),
        )
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Failed stations / 86 (including trigger)")
    fig.suptitle("Frozen 86-station tree · capacity redistribution")
    fig.tight_layout()
    paths.append(save_figure(fig, "collapse.png", figures_dir=directory))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, dynamic in zip(axes, (False, True)):
        for alpha, group in avalanche[avalanche.dynamic.eq(dynamic)].groupby("alpha"):
            curve = empirical_ccdf(group.post_trigger_size.to_numpy())
            if len(curve):
                ax.step(curve["size"], curve.ccdf, where="post", label=f"α={alpha:g}")
        ax.set(
            xlabel="Secondary failures S",
            ylabel="P(S ≥ s)",
            title="Dynamic" if dynamic else "Static",
            xlim=(1, 86),
            ylim=(0.001, 1.05),
            xscale="log",
            yscale="log",
        )
        if ax.lines:
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No secondary failures\nP(S = 0) = 1",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
    fig.suptitle("Independent random station triggers · zeros included in denominator")
    fig.tight_layout()
    paths.append(save_figure(fig, "avalanche_ccdf.png", figures_dir=directory))

    fig, axes = plt.subplots(1, 2, figsize=(11, 8))
    for ax, row in zip(axes, maps):
        outcome = outcomes[row["outcome_id"]]
        initial = outcome["n_initial_failed"]
        rounds = {node: 0 for node in outcome["failed"][:initial]}
        cursor = initial
        for index, size in enumerate(outcome["avalanche_sizes"], 1):
            rounds.update(
                {node: index for node in outcome["failed"][cursor : cursor + size]}
            )
            cursor += size
        for a, b in graph.edges:
            ax.plot(
                [graph.nodes[a]["lon"], graph.nodes[b]["lon"]],
                [graph.nodes[a]["lat"], graph.nodes[b]["lat"]],
                color=".8",
                lw=0.6,
                zorder=1,
            )
        survivors = [node for node in graph if node not in rounds]
        ax.scatter(
            [graph.nodes[node]["lon"] for node in survivors],
            [graph.nodes[node]["lat"] for node in survivors],
            s=14,
            color=".6",
            label="Surviving",
            zorder=2,
        )
        nodes = [node for node in rounds if rounds[node] > 0]
        if nodes:
            scatter = ax.scatter(
                [graph.nodes[node]["lon"] for node in nodes],
                [graph.nodes[node]["lat"] for node in nodes],
                c=[rounds[node] for node in nodes],
                cmap="plasma",
                vmin=1,
                vmax=max(rounds.values()),
                s=22,
                zorder=3,
            )
            fig.colorbar(scatter, ax=ax, shrink=0.65, label="Secondary failure round")
        nodes = [node for node in rounds if rounds[node] == 0]
        ax.scatter(
            [graph.nodes[node]["lon"] for node in nodes],
            [graph.nodes[node]["lat"] for node in nodes],
            color="C0",
            s=65,
            marker="*",
            label="Initial trigger",
            zorder=4,
        )
        ax.set_aspect(1 / math.cos(math.radians(-32)))
        ax.set(
            title=f"{'Dynamic' if row['dynamic'] else 'Static'}: {row['n_failed']}/86 failed\n{row['rounds']} secondary rounds",
            xlabel="Longitude",
            ylabel="Latitude",
        )
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Perth station trigger · α=0.2 · capacity rule")
    fig.tight_layout()
    paths.append(save_figure(fig, "cascade_map.png", figures_dir=directory))

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey=True)
    for ax, (rule, dynamic) in zip(
        axes.flat,
        [("equal", False), ("capacity", False), ("equal", True), ("capacity", True)],
    ):
        table = sensitivity[
            sensitivity.rule.eq(rule)
            & sensitivity.dynamic.eq(dynamic)
            & sensitivity.trigger.eq("random")
        ]
        grid = table.pivot_table(
            index="load_mode", columns="alpha", values="failed_fraction", aggfunc="mean"
        )
        picture = ax.imshow(
            grid.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis"
        )
        ax.set(
            xticks=range(len(grid.columns)),
            xticklabels=[f"{value:g}" for value in grid.columns],
            yticks=range(len(grid.index)),
            yticklabels=grid.index,
            title=f"{rule} · {'dynamic' if dynamic else 'static'}",
            xlabel="Tolerance α",
        )
    fig.colorbar(
        picture,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        label="Mean failed fraction (random triggers)",
    )
    fig.subplots_adjust(left=0.23, right=0.8, bottom=0.1, top=0.9, hspace=0.4)
    paths.append(save_figure(fig, "load_rule_sensitivity.png", figures_dir=directory))

    selected = line_runs[
        line_runs.alpha.eq(0.2) & line_runs.rule.eq("capacity") & ~line_runs.dynamic
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        selected.line,
        selected.n_initial_failed,
        label="Initially closed (exclusive stations)",
    )
    ax.bar(
        selected.line,
        selected.post_trigger_size,
        bottom=selected.n_initial_failed,
        label="Secondary failures",
    )
    ax.set(ylabel="Stations", title="Line closures · static capacity rule · α=0.2")
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend()
    fig.tight_layout()
    paths.append(save_figure(fig, "line_closures.png", figures_dir=directory))
    return paths
