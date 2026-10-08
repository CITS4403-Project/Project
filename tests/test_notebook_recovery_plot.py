"""The recovery plot must agree with served service in the frozen scan."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.utils.capture import capture_output

from transperth import config


def test_notebook04_service_curves_match_reachable_fractions(monkeypatch, tmp_path):
    path = (
        Path(__file__).resolve().parents[1] / "notebooks/04-recovery-strategies.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    plot = next(cell for cell in code if "bus_cascade_deltas.png" in cell)
    monkeypatch.chdir(path.parents[1])
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    namespace = {}
    with capture_output():
        exec(compile(code[0], str(path) + ":setup", "exec"), namespace)  # noqa: S102 - trusted local notebook setup
    figures = []
    namespace["show_figure"] = lambda figure, name: figures.append(figure)
    try:
        exec(compile(plot, str(path) + ":bus-deltas", "exec"), namespace)  # noqa: S102 - exercise the actual tracked plot
        axis = figures[0].axes[1]
        assert axis.get_ylabel() == "served-fraction change (pp, bus - rail)"
        scan = namespace["scan"]
        for trigger, (_, name) in namespace["TRIGGER_STYLE"].items():
            rail = scan[
                (scan.trigger == trigger) & (scan.scenario == "rail_only")
            ].set_index("alpha")
            for scenario in ("manual_bus", "candidate_bus"):
                bus = scan[
                    (scan.trigger == trigger) & (scan.scenario == scenario)
                ].set_index("alpha")
                label = f"{name}, {scenario.replace('_', ' ')}"
                line = next(line for line in axis.lines if line.get_label() == label)
                alpha = line.get_xdata()
                # Independent scan rows report unmet service, rather than the paired delta table.
                rail_served = 1.0 - rail.loc[alpha, "unmet_fraction"].to_numpy()
                bus_served = 1.0 - bus.loc[alpha, "unmet_fraction"].to_numpy()
                np.testing.assert_allclose(
                    line.get_ydata(), 100.0 * (bus_served - rail_served), atol=1e-10
                )
                if trigger == 23 and scenario == "candidate_bus":
                    focus = np.isclose(alpha, 0.2)
                    np.testing.assert_allclose(
                        line.get_ydata()[focus], [32.99589603283174]
                    )
    finally:
        for figure in figures:
            plt.close(figure)
