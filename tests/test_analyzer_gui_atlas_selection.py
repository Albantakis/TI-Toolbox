"""Regression coverage for changing ROIs in a running Analyzer tab."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
from PyQt5 import QtWidgets

from tit.gui.analyzer_tab import AnalyzerTab
from tit.paths import PathManager


@pytest.fixture
def tab(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(
        "tit.gui.analyzer_tab.get_path_manager", lambda: PathManager(str(tmp_path))
    )
    widget = AnalyzerTab()
    monkeypatch.setattr(widget, "get_selected_subjects", lambda: ["002"])
    monkeypatch.setattr(
        widget, "get_available_atlas_files",
        lambda sid: [
            (name, f"/{sid}/{name}.nii.gz")
            for name in ("posterior", "anterior", "bilateral")
        ],
    )
    widget.space_voxel.setChecked(True)
    widget.type_cortical.setChecked(True)
    widget.update_atlas_combo()
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_switching_roi_clears_previous_region_without_restart(tab):
    for index, region in enumerate(("posterior", "anterior", "bilateral")):
        tab.atlas_combo.setCurrentIndex(index)
        assert tab._get_regions() == []
        tab.region_input.setText(f"thalamus_{region}_bilateral")
        assert len(tab._get_regions()) == 1


def test_refresh_preserves_atlas_and_regions(tab):
    tab.atlas_combo.setCurrentIndex(1)
    tab.region_input.setText("thalamus_anterior_bilateral")
    for _ in range(3):
        tab.update_atlas_combo()
        tab.update_atlas_visibility()
    assert tab.atlas_combo.currentData() == "/002/anterior.nii.gz"
    assert tab._get_regions() == ["thalamus_anterior_bilateral"]


def test_subject_change_clears_regions_even_with_same_atlas_name(tab, monkeypatch):
    tab.region_input.setText("thalamus_posterior_bilateral")
    monkeypatch.setattr(tab, "get_selected_subjects", lambda: ["009"])
    tab.update_atlas_combo()
    assert tab.atlas_combo.currentData() == "/009/posterior.nii.gz"
    assert tab._get_regions() == []


def test_mesh_space_and_atlas_changes_clear_regions(tab):
    tab.region_input.setText("thalamus_posterior_bilateral")
    tab.space_mesh.setChecked(True)
    assert tab._get_regions() == []
    tab.region_input.setText("precentral")
    tab.atlas_name_combo.setCurrentIndex(
        (tab.atlas_name_combo.currentIndex() + 1) % tab.atlas_name_combo.count()
    )
    assert tab._get_regions() == []


def test_group_refresh_preserves_regions_but_atlas_change_clears_them(tab, monkeypatch):
    tab.pairs_table.setRowCount(2)
    monkeypatch.setattr(tab, "get_selected_subjects", lambda: ["002", "009"])
    tab.update_group_atlas_options()
    tab.atlas_combo.setCurrentIndex(1)
    tab.region_input.setText("thalamus_anterior_bilateral")
    tab.update_group_atlas_options()
    assert tab._get_regions() == ["thalamus_anterior_bilateral"]
    assert tab.group_atlas_config["009"]["path"] == "/009/anterior.nii.gz"
    tab.atlas_combo.setCurrentIndex(2)
    assert tab._get_regions() == []
    assert tab.group_atlas_config["009"]["path"] == "/009/bilateral.nii.gz"
