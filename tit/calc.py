#!/usr/bin/env simnibs_python
"""Temporal interference field calculation utilities.

Vectorised NumPy implementations of the TI modulation-amplitude algorithm
from Grossman et al. (2017), extended to multi-channel (mTI) configurations.
Used by the simulation engine and optimisation tools.

Public API
----------
get_TI_vectors
    Compute TI modulation-amplitude vectors for a single electrode pair.
get_nTI_vectors
    Generalised N-channel TI via recursive binary-tree pairing.
get_mTI_vectors
    4-channel mTI (convenience wrapper around :func:`get_TI_vectors`).
compute_mti_metric_field
    Compute scalar mTI metric fields for multipolar search workflows.
"""

import numpy as np


MTI_METRIC_RECURSIVE_TI = "recursive_ti"
MTI_METRIC_BOTZANOWSKI_MAGNITUDE_AM = "botzanowski_magnitude_am"
MTI_METRIC_BOTZANOWSKI_DIRECTIONAL_AM = "botzanowski_directional_am"
MTI_METRIC_BOTZANOWSKI_DIRECTIONAL_AM_AVG = "botzanowski_directional_am_ti_avg"
MTI_METRIC_GROSSMAN_EXT_DIRECTIONAL_AM = "grossman_ext_directional_am"
MTI_METRIC_GROSSMAN_EXT_DIRECTIONAL_AM_AVG = "grossman_ext_directional_am_ti_avg"

MTI_METRICS = {
    MTI_METRIC_RECURSIVE_TI,
    MTI_METRIC_BOTZANOWSKI_MAGNITUDE_AM,
    MTI_METRIC_BOTZANOWSKI_DIRECTIONAL_AM,
    MTI_METRIC_BOTZANOWSKI_DIRECTIONAL_AM_AVG,
    MTI_METRIC_GROSSMAN_EXT_DIRECTIONAL_AM,
    MTI_METRIC_GROSSMAN_EXT_DIRECTIONAL_AM_AVG,
}


def get_TI_vectors(E1_org, E2_org):
    """
    Calculate the temporal interference (TI) modulation amplitude vectors.

    This function implements the Grossman et al. 2017 algorithm for computing
    TI vectors that represent both the direction and magnitude of maximum
    modulation amplitude when two sinusoidal electric fields interfere.

    PHYSICAL INTERPRETATION:
    When two electric fields E1(t) = E1*cos(2πf1*t) and E2(t) = E2*cos(2πf2*t)
    with slightly different frequencies are applied simultaneously, they create
    a beating pattern. The TI vector indicates:
    - DIRECTION: Spatial direction of maximum envelope modulation
    - MAGNITUDE: Maximum envelope amplitude = 2 * effective_amplitude

    ALGORITHM (Grossman et al. 2017):
    1. Preprocessing: Ensure |E1| ≥ |E2| and acute angle α < π/2
    2. Regime selection based on geometric relationship:
       - Regime 1 (parallel): |E2| ≤ |E1|cos(α) → TI = 2*E2
       - Regime 2 (oblique): |E2| > |E1|cos(α) → TI = 2*E2_perpendicular_to_h
       where h = E1 - E2

    Parameters
    ----------
    E1_org : np.ndarray, shape (N, 3)
        Electric field vectors from electrode pair 1 [V/m]
    E2_org : np.ndarray, shape (N, 3)
        Electric field vectors from electrode pair 2 [V/m]

    Returns
    -------
    TI_vectors : np.ndarray, shape (N, 3)
        TI modulation amplitude vectors [V/m]
        Direction: Maximum modulation direction
        Magnitude: Maximum envelope amplitude

    References
    ----------
    Grossman, N. et al. (2017). Noninvasive Deep Brain Stimulation via
    Temporally Interfering Electric Fields. Cell, 169(6), 1029-1041.

    See Also
    --------
    get_mTI_vectors : 4-channel mTI (two pairs of pairs).
    get_nTI_vectors : Generalised N-channel recursive TI.
    """
    # Input validation
    assert E1_org.shape == E2_org.shape, "E1 and E2 must have same shape"
    assert E1_org.shape[1] == 3, "Vectors must be 3D"

    # Work with copies to avoid modifying input arrays
    E1 = E1_org.copy()
    E2 = E2_org.copy()

    # =================================================================
    # PREPROCESSING STEP 1: Magnitude ordering |E1| ≥ |E2|
    # =================================================================
    # Ensures consistency by always treating E1 as the "stronger" field
    # This simplifies the subsequent regime analysis
    idx_swap = np.linalg.norm(E2, axis=1) > np.linalg.norm(E1, axis=1)
    E1[idx_swap], E2[idx_swap] = E2[idx_swap], E1_org[idx_swap]

    # =================================================================
    # PREPROCESSING STEP 2: Acute angle constraint α < π/2
    # =================================================================
    # Ensures constructive interference by flipping E2 if dot product < 0
    # This avoids destructive interference scenarios
    idx_flip = np.sum(E1 * E2, axis=1) < 0
    E2[idx_flip] = -E2[idx_flip]

    # =================================================================
    # GEOMETRIC PARAMETERS CALCULATION
    # =================================================================
    # Calculate field magnitudes and angle between vectors
    normE1 = np.linalg.norm(E1, axis=1)
    normE2 = np.linalg.norm(E2, axis=1)

    # Safe cosine calculation to avoid division by zero and numerical errors
    denom = normE1 * normE2
    denom[denom == 0] = 1.0  # Prevent division by zero
    cosalpha = np.clip(np.sum(E1 * E2, axis=1) / denom, -1.0, 1.0)

    # =================================================================
    # REGIME SELECTION CRITERION
    # =================================================================
    # Critical condition from Grossman 2017: |E2| ≤ |E1| * cos(α)
    # This determines whether E2 is "small" relative to E1's projection
    regime1_mask = normE2 <= normE1 * cosalpha

    # Initialize output array
    TI_vectors = np.zeros_like(E1)

    # =================================================================
    # REGIME 1: PARALLEL ALIGNMENT (|E2| ≤ |E1| cos(α))
    # =================================================================
    # Physical interpretation: E2 is effectively "contained" within E1's projection
    # The TI amplitude is determined entirely by E2's magnitude and direction
    # Formula: TI = 2 * E2
    TI_vectors[regime1_mask] = 2.0 * E2[regime1_mask]

    # =================================================================
    # REGIME 2: OBLIQUE CONFIGURATION (|E2| > |E1| cos(α))
    # =================================================================
    # Physical interpretation: E2 has significant perpendicular component to E1
    # The TI is determined by the component of E2 perpendicular to h = E1 - E2
    # Formula: TI = 2 * E2_perpendicular_to_h
    regime2_mask = ~regime1_mask
    if np.any(regime2_mask):
        # Calculate difference vector h = E1 - E2
        h = E1[regime2_mask] - E2[regime2_mask]
        h_norm = np.linalg.norm(h, axis=1)

        # Handle degenerate case (h = 0) by setting unit norm
        h_norm[h_norm == 0] = 1.0
        e_h = h / h_norm[:, None]  # Unit vector along h

        # Project E2 onto h, then subtract to get perpendicular component
        # E2_perp = E2 - proj_h(E2) = E2 - (E2·ĥ)ĥ
        E2_parallel_component = np.sum(E2[regime2_mask] * e_h, axis=1)[:, None] * e_h
        E2_perp = E2[regime2_mask] - E2_parallel_component

        # The TI vector in regime 2 is twice the perpendicular component
        TI_vectors[regime2_mask] = 2.0 * E2_perp

    return TI_vectors


def get_nTI_vectors(fields):
    """Compute TI vectors for *N* E-fields using recursive binary-tree pairing.

    *N* must be even.  Fields are paired sequentially — ``(E1, E2)``,
    ``(E3, E4)``, etc. — then intermediate TI results are paired recursively
    until a single result remains.

    For 2 fields: ``TI(E1, E2)``
    For 4 fields: ``TI(TI(E1,E2), TI(E3,E4))``
    For 6 fields: ``TI(TI(TI(E1,E2), TI(E3,E4)), TI(E5,E6))``
    For 8 fields: ``TI(TI(TI(E1,E2), TI(E3,E4)), TI(TI(E5,E6), TI(E7,E8)))``

    Parameters
    ----------
    fields : list of np.ndarray, each shape (N, 3)
        Electric field vectors, one per electrode pair.

    Returns
    -------
    result : np.ndarray, shape (N, 3)
        Combined TI modulation amplitude vectors.

    Raises
    ------
    ValueError
        If number of fields is not even or less than 2.

    See Also
    --------
    get_TI_vectors : Core 2-field TI calculation.
    get_mTI_vectors : 4-channel convenience wrapper.
    """
    n = len(fields)
    if n < 2 or n % 2 != 0:
        raise ValueError(
            f"get_nTI_vectors requires an even number of fields >= 2, got {n}"
        )

    # First round: pair adjacent fields
    current = []
    for i in range(0, n, 2):
        current.append(get_TI_vectors(fields[i], fields[i + 1]))

    # Recursive rounds: pair results until one remains
    while len(current) > 1:
        next_round = []
        i = 0
        while i + 1 < len(current):
            next_round.append(get_TI_vectors(current[i], current[i + 1]))
            i += 2
        # Odd element carries forward
        if i < len(current):
            next_round.append(current[i])
        current = next_round

    return current[0]


def compute_mti_vectors(fields, metric=MTI_METRIC_RECURSIVE_TI):
    """Compute vector-valued mTI output for metrics with a best direction."""
    metric = _normalize_mti_metric(metric)
    if metric == MTI_METRIC_RECURSIVE_TI:
        return get_nTI_vectors(fields)
    if metric == MTI_METRIC_BOTZANOWSKI_DIRECTIONAL_AM:
        return compute_botzanowski_directional_am_stats(fields)["vectors"]
    if metric == MTI_METRIC_GROSSMAN_EXT_DIRECTIONAL_AM:
        return compute_grossman_ext_directional_am_stats(fields)["vectors"]
    raise ValueError(f"Metric does not produce mTI vectors: {metric!r}")


def compute_mti_metric_field(fields, metric=MTI_METRIC_RECURSIVE_TI):
    """Compute a scalar mTI field for one selected multipolar metric.

    Parameters
    ----------
    fields : list of np.ndarray
        Electric field vectors, one array per bipolar electrode pair.
    metric : str
        One of :data:`MTI_METRICS`.

    Returns
    -------
    np.ndarray
        Scalar metric value for each mesh element/voxel.
    """
    metric = _normalize_mti_metric(metric)
    if metric == MTI_METRIC_RECURSIVE_TI:
        return np.linalg.norm(get_nTI_vectors(fields), axis=1)
    if metric == MTI_METRIC_BOTZANOWSKI_MAGNITUDE_AM:
        return compute_botzanowski_magnitude_am_vectors(fields)
    if metric == MTI_METRIC_BOTZANOWSKI_DIRECTIONAL_AM:
        return np.linalg.norm(
            compute_botzanowski_directional_am_stats(fields)["vectors"], axis=1
        )
    if metric == MTI_METRIC_BOTZANOWSKI_DIRECTIONAL_AM_AVG:
        return compute_botzanowski_directional_am_stats(fields)["avg"]
    if metric == MTI_METRIC_GROSSMAN_EXT_DIRECTIONAL_AM:
        return np.linalg.norm(
            compute_grossman_ext_directional_am_stats(fields)["vectors"], axis=1
        )
    if metric == MTI_METRIC_GROSSMAN_EXT_DIRECTIONAL_AM_AVG:
        return compute_grossman_ext_directional_am_stats(fields)["avg"]
    raise ValueError(f"Unsupported mTI metric: {metric!r}")


def compute_botzanowski_magnitude_am_vectors(fields):
    """Compute direct-field AM from the envelope of ``||E(t)||``."""
    mti_amp, _env_max = _botzanowski_magnitude_am_components(fields)
    return mti_amp


def compute_botzanowski_directional_am_stats(fields):
    """Return best-direction and orientation-averaged Botzanowski AM fields."""
    vectors, peak_env, avg = _botzanowski_directional_am_components(fields)
    return {"vectors": vectors, "avg": avg, "peak_env": peak_env}


def compute_grossman_ext_directional_am_stats(fields):
    """Return best-direction and orientation-averaged full-field AM fields."""
    vectors, peak_env, avg = _grossman_ext_directional_am_components(fields)
    return {"vectors": vectors, "avg": avg, "peak_env": peak_env}


def _normalize_mti_metric(metric):
    metric = metric.value if hasattr(metric, "value") else str(metric)
    if metric not in MTI_METRICS:
        raise ValueError(f"Unsupported mTI metric: {metric!r}")
    return metric


def _botzanowski_magnitude_am_components(fields):
    arrs = _validate_field_list(fields)
    weights = _pair_weights(len(arrs))

    s0 = np.zeros(arrs[0].shape[0], dtype=np.float64)
    for field in arrs:
        s0 += np.sum(field * field, axis=1)
    s0 *= 0.5

    b = np.zeros_like(s0)
    for pair_idx, weight in enumerate(weights):
        f1 = arrs[2 * pair_idx]
        f2 = arrs[2 * pair_idx + 1]
        b += weight * np.sum(f1 * f2, axis=1)

    abs_b = np.abs(b)
    smin = np.maximum(s0 - abs_b, 0.0)
    smax = np.maximum(s0 + abs_b, 0.0)
    env_min = np.sqrt(2 * smin)
    env_max = np.sqrt(2 * smax)
    return env_max - env_min, env_max


def _botzanowski_directional_am_components(fields):
    arrs = _validate_field_list(fields)
    weights = _pair_weights(len(arrs))
    directions = _fibonacci_sphere(192)
    chunk_size = 16384

    n_vox = arrs[0].shape[0]
    best_vectors = np.zeros((n_vox, 3), dtype=np.float64)
    best_peak = np.zeros(n_vox, dtype=np.float64)
    avg_amp = np.zeros(n_vox, dtype=np.float64)

    for start in range(0, n_vox, chunk_size):
        stop = min(start + chunk_size, n_vox)
        proj_fields = [field[start:stop] @ directions.T for field in arrs]

        s0 = np.zeros_like(proj_fields[0], dtype=np.float64)
        for proj in proj_fields:
            s0 += proj * proj
        s0 *= 0.5

        b = np.zeros_like(s0, dtype=np.float64)
        for pair_idx, weight in enumerate(weights):
            p1 = proj_fields[2 * pair_idx]
            p2 = proj_fields[2 * pair_idx + 1]
            b += weight * (p1 * p2)

        abs_b = np.abs(b)
        smin = np.maximum(s0 - abs_b, 0.0)
        smax = np.maximum(s0 + abs_b, 0.0)
        env_min = np.sqrt(2 * smin)
        env_max = np.sqrt(2 * smax)
        amp = env_max - env_min

        avg_amp[start:stop] = np.mean(amp, axis=1)
        best_idx = np.argmax(amp, axis=1)
        rows = np.arange(stop - start)
        best_amp = amp[rows, best_idx]
        best_dirs = directions[best_idx]
        best_vectors[start:stop] = best_dirs * best_amp[:, None]
        best_peak[start:stop] = env_max[rows, best_idx]

    return best_vectors, best_peak, avg_amp


def _grossman_ext_directional_am_components(fields):
    arrs = _validate_field_list(fields)
    directions = _fibonacci_sphere(192)
    chunk_size = 16384

    n_vox = arrs[0].shape[0]
    best_vectors = np.zeros((n_vox, 3), dtype=np.float64)
    best_peak_env = np.zeros(n_vox, dtype=np.float64)
    avg_amp = np.zeros(n_vox, dtype=np.float64)

    for start in range(0, n_vox, chunk_size):
        stop = min(start + chunk_size, n_vox)
        proj_fields = [field[start:stop] @ directions.T for field in arrs]

        env_psi0 = np.zeros((stop - start, directions.shape[0]), dtype=np.float64)
        env_psi_pi = np.zeros_like(env_psi0)

        for pair_idx in range(len(arrs) // 2):
            a = proj_fields[2 * pair_idx]
            b = proj_fields[2 * pair_idx + 1]
            env_psi0 += np.abs(a + b)
            env_psi_pi += np.abs(a - b)

        env_hi = np.maximum(env_psi0, env_psi_pi)
        env_lo = np.minimum(env_psi0, env_psi_pi)
        amp = env_hi - env_lo

        avg_amp[start:stop] = np.mean(amp, axis=1)
        best_idx = np.argmax(amp, axis=1)
        rows = np.arange(stop - start)
        best_amp = amp[rows, best_idx]
        best_peak_env[start:stop] = env_hi[rows, best_idx]
        best_vectors[start:stop] = directions[best_idx] * best_amp[:, None]

    return best_vectors, best_peak_env, avg_amp


def _validate_field_list(fields):
    arrs = [np.asarray(field, dtype=np.float64) for field in fields]
    n = len(arrs)
    if n < 2 or n % 2 != 0:
        raise ValueError(f"mTI requires an even number of fields >= 2, got {n}")
    ref_shape = arrs[0].shape
    if len(ref_shape) != 2 or ref_shape[1] != 3:
        raise ValueError(f"Fields must have shape (N, 3), got {ref_shape}")
    for i, arr in enumerate(arrs[1:], start=2):
        if arr.shape != ref_shape:
            raise ValueError(
                "All fields must have identical shape; "
                f"field 1 has {ref_shape}, field {i} has {arr.shape}"
            )
    return arrs


def _pair_weights(num_fields: int):
    return [1.0] * (num_fields // 2)


def _fibonacci_sphere(num_dirs: int) -> np.ndarray:
    """Return approximately uniform unit vectors on the sphere."""
    if num_dirs < 2:
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    i = np.arange(num_dirs, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * i / (num_dirs - 1)
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.stack((x, y, z), axis=1)


def get_mTI_vectors(E1_org, E2_org, E3_org, E4_org):
    """Calculate multi-temporal interference (mTI) vectors from four E-fields.

    Computes TI between channels 1 and 2 to get ``TI_A``, TI between
    channels 3 and 4 to get ``TI_B``, and finally TI between ``TI_A`` and
    ``TI_B`` to produce the mTI vector field.

    Parameters
    ----------
    E1_org : np.ndarray, shape (N, 3)
        Electric field vectors for channel 1.
    E2_org : np.ndarray, shape (N, 3)
        Electric field vectors for channel 2.
    E3_org : np.ndarray, shape (N, 3)
        Electric field vectors for channel 3.
    E4_org : np.ndarray, shape (N, 3)
        Electric field vectors for channel 4.

    Returns
    -------
    mTI_vectors : np.ndarray, shape (N, 3)
        Multi-TI modulation amplitude vectors [V/m].

    Raises
    ------
    ValueError
        If any input array does not have shape ``(N, 3)`` or if the shapes
        are not identical.

    See Also
    --------
    get_TI_vectors : Core 2-field TI calculation.
    get_nTI_vectors : Generalised N-channel recursive TI.
    """
    # Validate shapes
    for i, arr in enumerate([E1_org, E2_org, E3_org, E4_org], start=1):
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"E{i}_org must have shape (N, 3), got {arr.shape}")

    if not (E1_org.shape == E2_org.shape == E3_org.shape == E4_org.shape):
        raise ValueError(
            "All input arrays must have identical shapes. "
            f"Got: {[E1_org.shape, E2_org.shape, E3_org.shape, E4_org.shape]}"
        )

    # Step 1: TI between (E1, E2)
    TI_A = get_TI_vectors(E1_org, E2_org)

    # Step 2: TI between (E3, E4)
    TI_B = get_TI_vectors(E3_org, E4_org)

    # Step 3: TI between (TI_A, TI_B) → mTI
    mTI_vectors = get_TI_vectors(TI_A, TI_B)

    return mTI_vectors
