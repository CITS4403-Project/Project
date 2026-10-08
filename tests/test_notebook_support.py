"""The demo must reject changed evidence and render labels as text."""
import json
from html import unescape

import networkx as nx
import pandas as pd
import pytest

from transperth.experiments import RunMeta, file_sha256, save_table
from transperth.notebook_support import network_map_html, read_result, verify_processed


def test_map_escapes_station_labels_and_has_no_network_dependencies():
    graph = nx.Graph()
    graph.add_node('a', lat=-31.9, lon=115.8, name='<script>alert(1)</script>')
    graph.add_node('b', lat=-32.0, lon=115.9, name='Other')
    graph.add_edge('a', 'b')
    html = network_map_html(graph, failed_rounds={'a': 0, 'b': 1})
    document = unescape(html.split('srcdoc="')[1].rsplit('"></iframe>', 1)[0])
    assert '<script>alert(1)</script>' not in document
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in document
    assert 'failed in round 1' in document
    assert 'https://' not in document and 'src=' not in document
    assert 'sandbox="allow-scripts"' in html


def test_result_refuses_changed_graph_and_changed_table(tmp_path):
    data = tmp_path / 'data/processed'
    data.mkdir(parents=True)
    (data / 'stations.csv').write_text('id\na\n')
    result = tmp_path / 'results/cascade/runs.csv'
    meta = RunMeta.create('cascade', seed=0, inputs=[data / 'stations.csv'])
    # Publication sidecars use portable repository-relative paths.
    meta.inputs.clear()
    meta.inputs['data/processed/stations.csv'] = file_sha256(data / 'stations.csv')
    save_table(pd.DataFrame({'failed': [1]}), result, meta)
    (result.parent / 'experiment_manifest.json').write_text(json.dumps({'artifact_sha256': {'runs.csv': file_sha256(result)}}))
    assert read_result(tmp_path, 'results/cascade/runs.csv')['failed'].tolist() == [1]
    result.write_text('failed\n99\n')
    with pytest.raises(ValueError, match='family manifest'):
        read_result(tmp_path, 'results/cascade/runs.csv')
    (data / 'stations.csv').write_text('id\nb\n')
    with pytest.raises(ValueError, match='different frozen input'):
        read_result(tmp_path, 'results/cascade/runs.csv')


def test_processed_manifest_refuses_changed_data(tmp_path):
    directory = tmp_path / 'data/processed'
    directory.mkdir(parents=True)
    table = directory / 'stations.csv'
    table.write_text('id\na\n')
    (directory / 'manifest.json').write_text(json.dumps({'files': [{'name': table.name, 'sha256': file_sha256(table)}]}))
    verify_processed(tmp_path)
    table.write_text('id\nb\n')
    with pytest.raises(ValueError, match='Frozen input changed'):
        verify_processed(tmp_path)
