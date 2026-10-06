"""Smoke tests for the transperth package scaffold."""

from __future__ import annotations

import importlib

import transperth


def test_package_exposes_version():
    assert transperth.__version__ == "0.1.0"


def test_skeleton_modules_import():
    for name in ("transperth.config", "transperth.experiments", "transperth.plotting"):
        assert importlib.import_module(name) is not None