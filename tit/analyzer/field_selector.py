"""Field selection utilities for automatic field file determination.

Resolves the correct field file path and SimNIBS field name for a given
subject, simulation, and analysis space (mesh or voxel).

Public API
----------
select_field_file
    Resolve the field path and SimNIBS field name for a subject/simulation.

See Also
--------
tit.analyzer.analyzer : Analyzer class that consumes the resolved paths.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from tit.paths import get_path_manager
from tit import constants as const

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldTarget:
    """Analyzable field file for a TI/mTI measure."""

    measure: str
    field_path: Path
    field_name: str


def select_field_file(
    subject_id: str,
    simulation: str,
    space: str,
    tissue_type: str = "GM",
    measure: str | None = None,
) -> tuple[Path, str]:
    """Return the field file path and SimNIBS field name.

    Detects whether the simulation is TI (2-pair) or mTI (4-pair) by checking
    for the existence of the mTI mesh directory.

    Parameters
    ----------
    subject_id : str
        Subject identifier (without ``sub-`` prefix).
    simulation : str
        Simulation (montage) folder name.
    space : str
        ``"mesh"`` or ``"voxel"``.
    tissue_type : str, optional
        ``"GM"``, ``"WM"``, or ``"both"`` (voxel only). Default ``"GM"``.

    Returns
    -------
    field_path : pathlib.Path
        Resolved absolute path to the field file.
    field_name : str
        SimNIBS field name (e.g. ``"TI_max"``, ``"mTI_max"``).

    Raises
    ------
    FileNotFoundError
        If the expected field file does not exist.
    ValueError
        If *space* is not ``"mesh"`` or ``"voxel"``.

    See Also
    --------
    Analyzer : Consumes the resolved path to load and analyze fields.
    """
    targets = list_field_targets(subject_id, simulation, space, tissue_type)
    if not targets:
        sim_dir = Path(get_path_manager().simulation(subject_id, simulation))
        raise FileNotFoundError(f"No analyzable field targets found in {sim_dir}")

    if measure is not None:
        for target in targets:
            if target.measure == measure:
                return target.field_path, target.field_name
        raise FileNotFoundError(
            f"No field target found for measure {measure!r} in simulation {simulation!r}"
        )

    first = targets[0]
    return first.field_path, first.field_name


def list_field_targets(
    subject_id: str,
    simulation: str,
    space: str,
    tissue_type: str = "GM",
) -> list[FieldTarget]:
    """Return all analyzable TI/mTI field targets for a simulation."""
    pm = get_path_manager()
    sim_dir = Path(pm.simulation(subject_id, simulation))

    if space not in {"mesh", "voxel"}:
        raise ValueError(f"Unsupported space: {space!r} (expected 'mesh' or 'voxel')")

    is_mti = (sim_dir / "mTI" / "mesh").is_dir()
    if is_mti:
        if space == "mesh":
            return _list_mti_mesh_targets(sim_dir, simulation)
        return _list_mti_voxel_targets(sim_dir, simulation, tissue_type)

    if space == "mesh":
        path, field_name = _select_mesh(sim_dir, simulation, is_mti=False)
    else:
        path, field_name = _select_voxel(sim_dir, is_mti=False, tissue_type=tissue_type)
    return [FieldTarget("TI", path, field_name)]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _select_mesh(sim_dir: Path, simulation: str, is_mti: bool) -> tuple[Path, str]:
    """Resolve a mesh (.msh) field file."""
    if is_mti:
        mesh_path = sim_dir / "mTI" / "mesh" / f"{simulation}_mTI.msh"
        field_name = const.FIELD_MTI_MAX
    else:
        mesh_path = sim_dir / "TI" / "mesh" / f"{simulation}_TI.msh"
        field_name = const.FIELD_TI_MAX

    if not mesh_path.exists():
        high_freq = sim_dir / "high_Frequency"
        hint = ""
        if high_freq.is_dir():
            hint = (
                " The high_Frequency folder exists, but TI/mTI post-processing "
                "outputs are missing. Check the simulation log for post-processing "
                "or NIfTI/mesh conversion errors."
            )
        raise FileNotFoundError(
            f"Mesh field file not found: {mesh_path}.{hint} Expected TI output at "
            f"{sim_dir / 'TI' / 'mesh'} or mTI output at {sim_dir / 'mTI' / 'mesh'}."
        )

    logger.debug("Selected mesh field file: %s (field=%s)", mesh_path, field_name)
    return mesh_path, field_name


def _list_mti_mesh_targets(sim_dir: Path, simulation: str) -> list[FieldTarget]:
    """Resolve available mTI metric meshes in the current shared-folder layout."""
    mesh_dir = sim_dir / "mTI" / "mesh"
    targets: list[FieldTarget] = []

    legacy = mesh_dir / f"{simulation}_mTI.msh"
    if legacy.exists():
        targets.append(FieldTarget("recursive_ti", legacy, const.FIELD_MTI_MAX))

    prefix = f"{simulation}_mTI_"
    for mesh_path in sorted(mesh_dir.glob(f"{prefix}*.msh")):
        name = mesh_path.name
        if name.startswith(("grey_", "white_")) or name.endswith("_normal.msh"):
            continue
        measure = mesh_path.stem[len(prefix) :]
        if measure and not any(target.measure == measure for target in targets):
            targets.append(FieldTarget(measure, mesh_path, const.FIELD_MTI_MAX))

    if not targets:
        path, field_name = _select_mesh(sim_dir, simulation, is_mti=True)
        targets.append(FieldTarget("recursive_ti", path, field_name))
    return targets


def _list_mti_voxel_targets(
    sim_dir: Path,
    simulation: str,
    tissue_type: str,
) -> list[FieldTarget]:
    """Resolve available mTI metric NIfTIs in the current shared-folder layout."""
    nifti_dir = sim_dir / "mTI" / "niftis"
    measures = _mti_measures_from_meshes(sim_dir, simulation)

    if not measures:
        measures = _mti_measures_from_niftis(nifti_dir, simulation)

    targets: list[FieldTarget] = []
    for measure in measures:
        try:
            path = _select_voxel_from_dir(
                nifti_dir,
                const.FIELD_MTI_MAX,
                tissue_type,
                simulation=simulation,
                measure=measure,
            )
        except FileNotFoundError:
            continue
        targets.append(FieldTarget(measure, path, const.FIELD_MTI_MAX))

    if not targets:
        path, field_name = _select_voxel(sim_dir, is_mti=True, tissue_type=tissue_type)
        targets.append(FieldTarget("recursive_ti", path, field_name))
    return targets


def _mti_measures_from_meshes(sim_dir: Path, simulation: str) -> list[str]:
    mesh_dir = sim_dir / "mTI" / "mesh"
    measures: list[str] = []

    legacy = mesh_dir / f"{simulation}_mTI.msh"
    if legacy.exists():
        measures.append("recursive_ti")

    prefix = f"{simulation}_mTI_"
    for mesh_path in sorted(mesh_dir.glob(f"{prefix}*.msh")):
        name = mesh_path.name
        if name.startswith(("grey_", "white_")) or name.endswith("_normal.msh"):
            continue
        measure = mesh_path.stem[len(prefix) :]
        if measure and measure not in measures:
            measures.append(measure)
    return measures


def _mti_measures_from_niftis(nifti_dir: Path, simulation: str) -> list[str]:
    if not nifti_dir.is_dir():
        return []

    measures: list[str] = []
    for nii in sorted(nifti_dir.iterdir()):
        if not (nii.name.endswith(".nii.gz") or nii.name.endswith(".nii")):
            continue
        measure = _measure_from_nifti_name(nii.name, simulation)
        if measure and measure not in measures:
            measures.append(measure)
    return measures


def _measure_from_nifti_name(name: str, simulation: str) -> str | None:
    stem = name.removesuffix(".nii.gz").removesuffix(".nii")
    for prefix in ("grey_", "white_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break

    for marker in ("_subject_", "_MNI_"):
        if marker in stem:
            stem = stem.split(marker, 1)[0]
            break

    if stem == f"{simulation}_mTI":
        return "recursive_ti"
    prefix = f"{simulation}_mTI_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return None


def _select_voxel(sim_dir: Path, is_mti: bool, tissue_type: str) -> tuple[Path, str]:
    """Resolve a voxel (.nii.gz) field file."""
    subdir = "mTI" if is_mti else "TI"
    nifti_dir = sim_dir / subdir / "niftis"
    field_name = const.FIELD_MTI_MAX if is_mti else const.FIELD_TI_MAX

    if not nifti_dir.is_dir():
        high_freq = sim_dir / "high_Frequency"
        hint = ""
        if high_freq.is_dir():
            hint = (
                " The high_Frequency folder exists, but TI/mTI NIfTI outputs are "
                "missing. Check the simulation log for post-processing or mesh-to-NIfTI "
                "conversion errors."
            )
        raise FileNotFoundError(f"NIfTI directory not found: {nifti_dir}.{hint}")

    return _select_voxel_from_dir(nifti_dir, field_name, tissue_type), field_name


def _select_voxel_from_dir(
    nifti_dir: Path,
    field_name: str,
    tissue_type: str,
    *,
    simulation: str | None = None,
    measure: str | None = None,
) -> Path:
    """Resolve a voxel (.nii.gz) field file from an explicit NIfTI directory."""
    if not nifti_dir.is_dir():
        high_freq = nifti_dir.parent.parent / "high_Frequency"
        hint = ""
        if high_freq.is_dir():
            hint = (
                " The high_Frequency folder exists, but TI/mTI NIfTI outputs are "
                "missing. Check the simulation log for post-processing or mesh-to-NIfTI "
                "conversion errors."
            )
        raise FileNotFoundError(f"NIfTI directory not found: {nifti_dir}.{hint}")

    niftis = sorted(
        p
        for p in nifti_dir.iterdir()
        if p.name.endswith(".nii.gz") or p.name.endswith(".nii")
    )

    if not niftis:
        raise FileNotFoundError(f"No NIfTI files found in {nifti_dir}")

    tissue = str(tissue_type or "GM").strip().lower()
    prefix_map = {"gm": "grey_", "wm": "white_", "both": None}
    if tissue not in prefix_map:
        raise ValueError(
            f"Unsupported tissue_type: {tissue_type!r} (expected 'GM', 'WM', or 'both')"
        )

    def matches_measure(path: Path) -> bool:
        if not simulation or not measure:
            return True
        return _measure_from_nifti_name(path.name, simulation) == measure

    preferred_prefix = prefix_map[tissue]
    if preferred_prefix is None:
        # Prefer subject-space, full-field files (no tissue prefix, no MNI tag).
        for nii in niftis:
            name = nii.name
            if (
                not name.startswith(("grey_", "white_"))
                and "_MNI" not in name
                and field_name in name
                and matches_measure(nii)
            ):
                logger.debug(
                    "Selected voxel field file: %s (field=%s, tissue=%s)",
                    nii,
                    field_name,
                    tissue,
                )
                return nii
    else:
        for nii in niftis:
            name = nii.name
            if (
                name.startswith(preferred_prefix)
                and "_MNI" not in name
                and field_name in name
                and matches_measure(nii)
            ):
                logger.debug(
                    "Selected voxel field file: %s (field=%s, tissue=%s)",
                    nii,
                    field_name,
                    tissue,
                )
                return nii

    if preferred_prefix is None:
        for nii in niftis:
            name = nii.name
            if (
                not name.startswith(("grey_", "white_"))
                and "_MNI" not in name
                and matches_measure(nii)
            ):
                logger.debug(
                    "Selected voxel field file (non-field fallback): %s (field=%s, tissue=%s)",
                    nii,
                    field_name,
                    tissue,
                )
                return nii
    else:
        for nii in niftis:
            name = nii.name
            if (
                name.startswith(preferred_prefix)
                and "_MNI" not in name
                and matches_measure(nii)
            ):
                logger.debug(
                    "Selected voxel field file (non-field fallback): %s (field=%s, tissue=%s)",
                    nii,
                    field_name,
                    tissue,
                )
                return nii

    raise FileNotFoundError(f"No {tissue_type} NIfTI file found in {nifti_dir}")
