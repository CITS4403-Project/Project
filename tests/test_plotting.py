"""Tests for the shared plotting helpers in :mod:`transperth.plotting`."""

from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt

from transperth.plotting import (
    apply_style,
    plot_bus_service_mechanism,
    plot_network,
    plot_robustness_cascade,
    plot_series,
    plot_uncertainty_avalanche,
    save_figure,
)


def test_save_figure_writes_png(tmp_path):
    apply_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = save_figure(fig, "demo.png", figures_dir=tmp_path)
    assert path == tmp_path / "demo.png"
    assert path.is_file()
    assert path.stat().st_size > 0


def test_save_figure_creates_missing_directory(tmp_path):
    fig, _ = plt.subplots()
    path = save_figure(fig, "nested/demo.png", figures_dir=tmp_path)
    assert path.is_file()


def test_plot_series_uses_given_axes():
    table = pd.DataFrame({"x": [0, 1, 2], "y": [1, 4, 9]})
    fig, ax = plt.subplots()
    returned = plot_series(table, "x", "y", ax=ax, label="y = x^2")
    assert returned is ax
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["y = x^2"]
    plt.close(fig)


def test_plot_network_draws_edges_and_nodes():
    stations = pd.DataFrame(
        {
            "station_id": ["1", "2", "3"],
            "lat": [-31.90, -31.95, -32.00],
            "lon": [115.80, 115.85, 115.90],
            "trips_served": [100, 500, 50],
        }
    )
    edges = pd.DataFrame({"station_a": ["1", "2"], "station_b": ["2", "3"]})
    fig, ax = plt.subplots()
    returned = plot_network(stations, edges, ax=ax)
    assert returned is ax
    assert len(ax.lines) == 2
    assert len(ax.collections) == 1
    plt.close(fig)


def test_plot_robustness_cascade_draws_both_panels():
    removal = {
        name: pd.DataFrame({"fraction": [0.0, 0.1], "gcc_fraction": [1.0, 0.4]})
        for name in ["random_degree", "targeted_degree", "targeted_betweenness"]
    }
    runs = pd.DataFrame(
        {
            "rule": ["capacity", "capacity", "equal", "equal"],
            "trigger": ["load"] * 4,
            "alpha": [0.0, 0.1, 0.0, 0.1],
            "failed_fraction": [1.0, 0.5, 1.0, 0.8],
        }
    )
    ci = pd.DataFrame(
        {
            "rule": ["capacity", "capacity"],
            "metric": ["failed_fraction", "failed_fraction"],
            "alpha": [0.0, 0.1],
            "mean": [1.0, 0.5],
            "low": [0.9, 0.4],
            "high": [1.0, 0.6],
        }
    )
    fig = plot_robustness_cascade(removal, runs, ci)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylabel() == "GCC / 86"
    assert fig.axes[1].get_title() == "(b) Static secondary failure"
    plt.close(fig)


def test_plot_bus_service_mechanism_draws_all_three_scenarios():
    scenarios = ["rail_only", "manual_bus", "candidate_bus"]
    bay = pd.DataFrame(
        {"unmet_fraction": [0.3, 0.1, 0.0], "n_failed": [6, 4, 2]},
        index=scenarios,
    )
    ratios = pd.DataFrame({"load_ratio": [0.68, 0.9, 1.3]}, index=scenarios)
    fig = plot_bus_service_mechanism(bay, ratios)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylabel() == "Served terminal-pair fraction"
    assert fig.axes[1].get_title() == "(b) Fremantle, first load check"
    plt.close(fig)


def test_plot_uncertainty_avalanche_keeps_discrete_masses():
    recommendations = {
        "seed_candidates": [
            {
                "n": 100,
                "all_condition_prefixes_available": True,
                "worst_ci_half_width": 0.09,
            },
            {
                "n": 1000,
                "all_condition_prefixes_available": True,
                "worst_ci_half_width": 0.02,
            },
        ]
    }
    avalanche = pd.DataFrame(
        {
            "dynamic": [False, False, False, False],
            "alpha": [0.1, 0.1, 0.2, 0.2],
            "post_trigger_size": [0, 85, 0, 84],
        }
    )
    fig = plot_uncertainty_avalanche(recommendations, avalanche)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_xlabel() == "Random-trigger trials"
    assert fig.axes[1].get_title() == "(b) Finite avalanche outcomes"
    assert len(fig.axes[1].collections) == 2
    plt.close(fig)
