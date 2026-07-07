import json
from pathlib import Path

import numpy as np


class _FakeHeader:
    def copy(self):
        return _FakeHeader()

    def set_data_dtype(self, _dtype):
        return None


class _FakeImage:
    header = _FakeHeader()
    affine = np.eye(4)


def _make_project(tmp_path: Path, subject_id: str = "001") -> Path:
    m2m = (
        tmp_path
        / "derivatives"
        / "SimNIBS"
        / f"sub-{subject_id}"
        / f"m2m_{subject_id}"
    )
    m2m.mkdir(parents=True)
    (m2m / "T1.nii.gz").touch()
    return tmp_path


def test_create_subject_space_rois_warps_all_project_roi_masks(tmp_path, monkeypatch):
    from tit.tools import subject_rois

    project = _make_project(tmp_path)
    template_root = project / "derivatives" / "ti-toolbox" / "rois"
    nested = template_root / "thalamus_functional_mni"
    nested.mkdir(parents=True)
    (template_root / "motor_MNI.nii.gz").touch()
    (template_root / "motor_MNI.json").write_text(
        json.dumps({"space": "MNI", "labels": [{"id": 1, "name": "motor"}]})
    )
    (nested / "anterior_left_MNI.nii.gz").touch()

    saved_masks = {}

    def fake_warp(_source, _m2m_dir, output, _reference):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).touch()

    def fake_load(path):
        path = Path(path)
        if path in saved_masks:
            return saved_masks[path], _FakeImage()
        mask = np.zeros((2, 2, 2), dtype=bool)
        mask[0, 0, 0] = True
        return mask, _FakeImage()

    def fake_save(mask, _reference_img, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        saved_masks[output_path] = mask.copy()

    monkeypatch.setattr(subject_rois, "_warp_volume", fake_warp)
    monkeypatch.setattr(subject_rois, "_load_binary_mask", fake_load)
    monkeypatch.setattr(subject_rois, "_save_binary_mask", fake_save)

    results = subject_rois.create_subject_space_rois(project, "001")

    paths = {result.path.relative_to(project) for result in results}
    assert paths == {
        Path(
            "derivatives/SimNIBS/sub-001/m2m_001/ROIs/motor_sub-001.nii.gz"
        ),
        Path(
            "derivatives/SimNIBS/sub-001/m2m_001/ROIs/"
            "thalamus_functional/anterior_left_sub-001.nii.gz"
        ),
    }
    sidecar = json.loads(
        (
            project
            / "derivatives/SimNIBS/sub-001/m2m_001/ROIs/motor_sub-001.json"
        ).read_text()
    )
    assert sidecar["name"] == "motor"
    assert sidecar["source_space"] == "MNI"
    assert sidecar["labels"] == [{"id": 1, "name": "motor"}]
