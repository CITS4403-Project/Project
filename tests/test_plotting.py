"""Tests for the shared plotting helpers in :mod:`transperth.plotting`."""

from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt

from transperth.plotting import apply_style, plot_network, plot_series, save_figure


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