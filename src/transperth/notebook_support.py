"""Offline maps and frozen-input checks for the live demonstration notebooks."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from html import escape
from pathlib import Path

import networkx as nx
import pandas as pd

from transperth.experiments import file_sha256, load_meta


def verify_processed(root: Path) -> dict:
    """Check every published processed file before displaying the baseline."""
    directory = root / "data" / "processed"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = directory / entry["name"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Frozen input changed: {path.name}; see docs/data.md")
    return manifest


def read_result(root: Path, relative: str, **kwargs) -> pd.DataFrame:
    """Check a saved table against its family manifest and graph provenance.

    Only the processed inputs are required locally: raw GTFS downloads are
    deliberately unnecessary for the demonstration. Engine hashes remain in
    the sidecar for audit; displaying a historical result does not rerun it.
    """
    path = root / relative
    meta = load_meta(path)
    for name, digest in meta["inputs"].items():
        if name.startswith("data/processed/") and file_sha256(root / name) != digest:
            raise ValueError(f"Result uses different frozen input: {name}")
    family = path.parent / "experiment_manifest.json"
    if family.exists():
        manifest = json.loads(family.read_text(encoding="utf-8"))
        expected = manifest["artifact_sha256"].get(path.name)
        if expected is None or file_sha256(path) != expected:
            raise ValueError(f"Result disagrees with family manifest: {relative}")
    return pd.read_csv(path, **kwargs)


def network_map_html(
    graph: nx.Graph,
    *,
    backups: pd.DataFrame | None = None,
    failed_rounds: Mapping[str, int] | None = None,
    title: str = "Frozen station network",
) -> str:
    """Return a sandboxed, self-contained SVG map with pan, zoom and layers.

    Coordinates are a local equirectangular projection, not a street map.
    ``failed_rounds`` uses 0 for the external trigger and positive update rounds.
    Bus lines connect verified endpoints; they do not depict bus-road geometry.
    No tiles, CDN, external scripts or additional plotting dependency is used.
    """
    if not graph:
        raise ValueError("map requires at least one station")
    failure = dict(failed_rounds or {})
    if not set(failure) <= set(graph):
        raise ValueError("failed station is absent from graph")
    if any(not isinstance(r, int) or r < 0 for r in failure.values()):
        raise ValueError("failure rounds must be non-negative integers")
    coordinates = {n: (float(d["lon"]), float(d["lat"])) for n, d in graph.nodes(data=True)}
    if any(not math.isfinite(v) for xy in coordinates.values() for v in xy):
        raise ValueError("map coordinates must be finite")
    mean_lat = sum(xy[1] for xy in coordinates.values()) / len(coordinates)
    points = {n: (lon * math.cos(math.radians(mean_lat)), -lat) for n, (lon, lat) in coordinates.items()}
    xmin, xmax = min(x for x, _ in points.values()), max(x for x, _ in points.values())
    ymin, ymax = min(y for _, y in points.values()), max(y for _, y in points.values())
    scale = min(650 / max(xmax - xmin, 0.001), 470 / max(ymax - ymin, 0.001))
    points = {n: (375 + (x - (xmin + xmax) / 2) * scale, 275 + (y - (ymin + ymax) / 2) * scale) for n, (x, y) in points.items()}

    def line(a, b, **attributes):
        ax, ay = points[a]
        bx, by = points[b]
        attrs = " ".join(f'{k}="{escape(str(v), quote=True)}"' for k, v in attributes.items())
        return f'<line x1="{ax:.2f}" y1="{ay:.2f}" x2="{bx:.2f}" y2="{by:.2f}" {attrs}/>'

    rails = "".join(line(a, b, stroke="#64748b", **{"stroke-width": 2}) for a, b in sorted(graph.edges))
    buses = ""
    if backups is not None:
        for row in backups.itertuples(index=False):
            a, b = str(row.station_a), str(row.station_b)
            if a not in graph or b not in graph:
                raise ValueError("backup endpoint is absent from graph")
            buses += line(a, b, stroke="#08916b", **{"stroke-width": 3, "stroke-dasharray": "6 4"})
    stations = []
    maximum = max(failure.values(), default=1)
    for n in sorted(graph):
        x, y = points[n]
        d = graph.nodes[n]
        r = failure.get(n)
        colour = "#dc2626" if r == 0 else (f"hsl({25 + 250 * r / max(maximum, 1):.0f} 65% 45%)" if r is not None else "#2563eb")
        status = "external trigger" if r == 0 else (f"failed in round {r}" if r is not None else "surviving facility" if failure else "baseline")
        name = f'{n}: {d.get("name", n)} | {status} | degree {graph.degree(n)} | AM stops {d.get("am_peak_stops", "?")}'
        stations.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{6 if r == 0 else 4}" fill="{colour}" stroke="white" stroke-width="0.7" tabindex="0" data-detail="{escape(name, quote=True)}"><title>{escape(name)}</title></circle>')
    content = '''<!doctype html><html lang="en"><meta charset="utf-8">
<style>body{font:14px system-ui;margin:0;color:#172338;background:#f8fafc}h3{margin:12px}label,button{margin:6px}button{cursor:pointer}svg{width:100%;height:470px;touch-action:none;cursor:grab}#detail{padding:10px;min-height:35px}circle:focus,circle:hover{stroke:#111;stroke-width:2}small{display:block;margin:8px;color:#475569}</style>
<h3>__TITLE__</h3><label><input type="checkbox" data-layer="rail" checked>Rail</label><label><input type="checkbox" data-layer="bus" checked>Verified buses</label><label><input type="checkbox" data-layer="stations" checked>Stations</label><button id="reset">Reset view</button>
<small>Drag to pan; wheel to zoom; hover or focus a station for details. North is up. Dashed links show bus endpoint connections, not road routes. Trigger: red; secondary failures: round colours; survivors: blue.</small>
<svg id="map" viewBox="0 0 750 550" role="img" aria-label="Station coordinate map"><g id="rail">__RAIL__</g><g id="bus">__BUS__</g><g id="stations">__STATIONS__</g></svg><div id="detail" aria-live="polite">Select a station.</div>
<script>
const svg=document.getElementById('map');let view=[0,0,750,550],drag=null;
const render=()=>svg.setAttribute('viewBox',view.join(' '));
document.querySelectorAll('[data-layer]').forEach(c=>c.onchange=()=>document.getElementById(c.dataset.layer).style.display=c.checked?'':'none');
document.getElementById('reset').onclick=()=>{view=[0,0,750,550];render()};
svg.addEventListener('wheel',e=>{e.preventDefault();const rect=svg.getBoundingClientRect(),factor=e.deltaY>0?1.15:1/1.15;if(view[2]*factor<15||view[2]*factor>3000)return;const rx=(e.clientX-rect.left)/rect.width,ry=(e.clientY-rect.top)/rect.height;view=[view[0]+rx*view[2]*(1-factor),view[1]+ry*view[3]*(1-factor),view[2]*factor,view[3]*factor];render()},{passive:false});
svg.onpointerdown=e=>{drag=[e.clientX,e.clientY,...view];svg.setPointerCapture(e.pointerId)};
svg.onpointermove=e=>{if(!drag)return;const rect=svg.getBoundingClientRect();view=[drag[2]-(e.clientX-drag[0])*drag[4]/rect.width,drag[3]-(e.clientY-drag[1])*drag[5]/rect.height,drag[4],drag[5]];render()};
svg.onpointerup=svg.onpointercancel=()=>{drag=null};
document.querySelectorAll('[data-detail]').forEach(n=>{const show=()=>document.getElementById('detail').textContent=n.dataset.detail;n.onmouseover=show;n.onfocus=show});
</script></html>'''
    content = content.replace("__TITLE__", escape(title)).replace("__RAIL__", rails).replace("__BUS__", buses).replace("__STATIONS__", "".join(stations))
    return f'<iframe title="{escape(title, quote=True)}" sandbox="allow-scripts" style="width:100%;height:660px;border:1px solid #cbd5e1;border-radius:8px" srcdoc="{escape(content, quote=True)}"></iframe>'
