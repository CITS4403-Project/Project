"""Deterministic selection of budgeted, evidenced terminal backup links."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from itertools import combinations
from numbers import Integral

import networkx as nx
import pandas as pd

from transperth.config import DEFAULT_SEED
from transperth.multilayer import BACKUP_KINDS

Pair = tuple[str, str]


def _id(value: object) -> str:
    if value is None or pd.isna(value):
        raise ValueError("station ID cannot be missing")
    text = str(value).strip()
    if text.startswith(("T:", "R:")):
        text = text[2:]
    if not text:
        raise ValueError("station ID cannot be empty")
    return text


def _pair(a: object, b: object) -> Pair:
    pair = tuple(sorted((_id(a), _id(b))))
    if pair[0] == pair[1]:
        raise ValueError("self-pairs are not allowed")
    return pair


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _context(
    graph: nx.Graph,
    *,
    budget: int,
    failed: Collection[str],
    demand: Mapping[Pair, float] | None,
    seed: int,
) -> tuple[nx.Graph, dict[Pair, float]]:
    _integer(budget, "budget")
    _integer(seed, "seed")
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("strategies require an undirected simple layered graph")
    working = graph.copy()
    terminals = sorted(
        node[2:] for node in graph if isinstance(node, str) and node.startswith("T:")
    )
    for station in failed:
        station = _id(station)
        if "T:" + station not in graph:
            raise ValueError(f"unknown failed station {station!r}")
        if "R:" + station in working:
            working.remove_node("R:" + station)
    for a, b, data in working.edges(data=True):
        minutes = float(data.get("minutes", math.nan))
        if not math.isfinite(minutes) or minutes <= 0:
            raise ValueError(f"edge {a!r}-{b!r} needs finite positive minutes")
    if demand is None:
        weights = {pair: 1.0 for pair in combinations(terminals, 2)}
    else:
        weights = {}
        for endpoints, raw in demand.items():
            pair = _pair(*endpoints)
            if any("T:" + station not in working for station in pair):
                raise ValueError(f"unknown demand terminal in {pair!r}")
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError("demand weights must be finite and non-negative")
            if pair in weights:
                raise ValueError("duplicate undirected demand pair")
            weights[pair] = value
    return working, weights


def _components(graph: nx.Graph) -> dict[str, int]:
    return {
        node: index
        for index, component in enumerate(nx.connected_components(graph))
        for node in component
    }


def _gain(
    pair: Pair, components: dict[str, int], weights: Mapping[Pair, float]
) -> float:
    left, right = (components["T:" + station] for station in pair)
    if left == right:
        return 0.0
    return math.fsum(
        weight
        for (a, b), weight in weights.items()
        if {components["T:" + a], components["T:" + b]} == {left, right}
    )


class NoBackup:
    """Return no extra links; callers measure the already failed baseline."""

    name = "no_backup"

    def deploy(
        self,
        graph: nx.Graph,
        *,
        budget: int,
        failed: Collection[str] = (),
        demand: Mapping[Pair, float] | None = None,
        seed: int = DEFAULT_SEED,
    ) -> list[Pair]:
        """Validate shared inputs and leave deployment empty."""
        _context(graph, budget=budget, failed=failed, demand=demand, seed=seed)
        return []

    def selected_table(self, pairs: Collection[Pair]) -> pd.DataFrame:
        """Return an empty table accepted by P1.5's backup application."""
        if pairs:
            raise ValueError("NoBackup cannot select links")
        return pd.DataFrame(columns=["station_a", "station_b", "minutes", "kind"])


class _Candidates:
    """Own a canonical copy of feasible links; retain all evidence columns."""

    name: str

    def __init__(self, candidates: pd.DataFrame, *, rail: nx.Graph | None = None):
        required = {"station_a", "station_b", "minutes", "kind"}
        if not required <= set(candidates.columns):
            raise ValueError(
                "candidates require station_a, station_b, minutes and kind"
            )
        self.rail = rail.copy() if rail is not None else None
        rows = []
        for source in candidates.to_dict("records"):
            pair = _pair(source["station_a"], source["station_b"])
            minutes = float(source["minutes"])
            if not math.isfinite(minutes) or minutes <= 0:
                raise ValueError(f"candidate {pair!r} needs finite positive minutes")
            if source["kind"] not in BACKUP_KINDS:
                raise ValueError(f"unknown candidate kind {source['kind']!r}")
            source.update(station_a=pair[0], station_b=pair[1], minutes=minutes)
            # Equal-time rows are resolved by complete evidence, not input order.
            evidence_key = json.dumps(
                {key: str(value) for key, value in source.items()}, sort_keys=True
            )
            rows.append((pair, minutes, evidence_key, source))
        self._rows: dict[Pair, dict] = {}
        for pair, _, _, row in sorted(rows, key=lambda item: item[:3]):
            self._rows.setdefault(pair, row)

    def selected_table(self, pairs: Collection[Pair]) -> pd.DataFrame:
        """Return original fastest candidate rows for deployment, with evidence."""
        requested = [_pair(*pair) for pair in pairs]
        if len(set(requested)) != len(requested):
            raise ValueError("duplicate selected pairs")
        if any(pair not in self._rows for pair in requested):
            raise ValueError("selection contains a pair outside the candidate set")
        if not requested:
            return pd.DataFrame(columns=["station_a", "station_b", "minutes", "kind"])
        return pd.DataFrame([dict(self._rows[pair]) for pair in requested])

    def _available(self, graph: nx.Graph) -> dict[Pair, dict]:
        available = {}
        for pair, row in self._rows.items():
            a, b = ("T:" + station for station in pair)
            if a not in graph or b not in graph:
                raise ValueError(f"unknown candidate terminal in {pair!r}")
            if graph.has_edge(a, b) and graph[a][b]["minutes"] <= row["minutes"]:
                continue
            available[pair] = row
        return available

    def deploy(
        self,
        graph: nx.Graph,
        *,
        budget: int,
        failed: Collection[str] = (),
        demand: Mapping[Pair, float] | None = None,
        seed: int = DEFAULT_SEED,
    ) -> list[Pair]:
        """Choose links without mutating inputs; seed is validated, ties lexical."""
        failed = tuple(failed)
        working, weights = _context(
            graph, budget=budget, failed=failed, demand=demand, seed=seed
        )
        available = self._available(working)
        if self.name == "existing_bus":
            available = {
                pair: row
                for pair, row in available.items()
                if row["kind"] in ("existing_bus", "gtfs_candidate")
            }
            return sorted(
                available, key=lambda pair: (available[pair]["minutes"], pair)
            )[:budget]
        if self.name == "shuttle_bridging":
            # Unit OD weights rank the gain in terminal-pair connectivity.
            terminals = sorted(
                node[2:]
                for node in working
                if isinstance(node, str) and node.startswith("T:")
            )
            weights = {pair: 1.0 for pair in combinations(terminals, 2)}
        corridor_scores = (
            self._corridor_scores(failed)
            if self.name == "corridor_reinforcement"
            else {}
        )
        selected = []
        while available and len(selected) < budget:
            components = _components(working)
            gains = {pair: _gain(pair, components, weights) for pair in available}
            if self.name == "corridor_reinforcement":
                viable = [
                    pair for pair in available if corridor_scores.get(pair, 0) > 0
                ]
                key = lambda pair: (
                    -corridor_scores[pair],
                    -gains[pair],
                    available[pair]["minutes"],
                    pair,
                )
            else:
                viable = [pair for pair in available if gains[pair] > 0]
                key = lambda pair: (-gains[pair], available[pair]["minutes"], pair)
            if not viable:
                break
            chosen = min(viable, key=key)
            row = available.pop(chosen)
            selected.append(chosen)
            working.add_edge("T:" + chosen[0], "T:" + chosen[1], minutes=row["minutes"])
        return selected

    def _corridor_scores(self, failed: Collection[str]) -> dict[Pair, float]:
        if self.rail is None:
            raise ValueError("CorridorReinforcement requires the intact rail graph")

        def lines(station: str) -> set[str]:
            if station not in self.rail:
                raise ValueError(f"unknown intact rail station {station!r}")
            return {
                value.strip()
                for value in str(self.rail.nodes[station].get("lines", "")).split(";")
                if value.strip()
            }

        disrupted = set().union(*(lines(_id(station)) for station in failed))
        loads: dict[str, float] = {line: 0 for line in disrupted}
        for station in sorted(self.rail):
            flow = float(self.rail.nodes[station].get("trips_served", 0))
            if not math.isfinite(flow) or flow < 0:
                raise ValueError(
                    "corridor trips_served must be finite and non-negative"
                )
            for line in lines(station) & disrupted:
                loads[line] += flow
        return {
            pair: math.fsum(
                loads[line] for line in (lines(pair[0]) | lines(pair[1])) & disrupted
            )
            for pair in self._rows
        }


class ExistingBus(_Candidates):
    """Select the shortest inactive existing/GTFS links; baseline buses stay active."""

    name = "existing_bus"


class ShuttleBridging(_Candidates):
    """Greedily bridge components by the number of newly reachable terminal pairs."""

    name = "shuttle_bridging"


class CorridorReinforcement(_Candidates):
    """Prioritise failed line corridors by summed scheduled trips_served proxy."""

    name = "corridor_reinforcement"


class DemandAdaptive(_Candidates):
    """Greedily restore unmet weighted OD demand; absent demand uses unit pairs."""

    name = "demand_adaptive"
