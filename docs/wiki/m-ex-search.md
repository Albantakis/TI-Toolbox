---
layout: wiki
title: Multipolar Ex-Search
permalink: /wiki/m-ex-search/
---

Multipolar Ex-Search (`m-ex-search`) is an experimental branch workflow for
exhaustively ranking four-pair mTI montages from leadfield data. It is intended
for refinement searches around an existing montage: define electrode buckets,
choose one mTI metric, and evaluate all valid eight-electrode candidates.

## Scope

`m-ex-search` currently supports:

- four bipolar pairs: `E1+`, `E1-`, `E2+`, `E2-`, `E3+`, `E3-`, `E4+`, `E4-`
- one fixed current value applied to every pair
- bucket loading from JSON files
- optional left/right symmetry constraints
- duplicate-electrode filtering, so overlapping buckets are allowed but a
  montage cannot use the same electrode twice
- one selected mTI metric per search run

It does not perform a current-ratio search. To compare different current
levels, run separate searches with different current values.

## GUI Workflow

1. Open the GUI and go to the **Optimizer** tab.
2. Set **Select Optimization Method** to **m-ex-search**.
3. Select the subject, leadfield, EEG net, and ROI as in regular Ex-Search.
4. Load an m-ex-search bucket JSON file.
5. Set **Current per Pair**. The default is `5` mA.
6. Select one **mTI Metric**.
7. If needed, enable symmetry and choose the symmetry pairing:
   - `within_pairs`: each plus bucket mirrors its paired minus bucket.
   - `cross_pairs`: `E1` mirrors `E3`, and `E2` mirrors `E4`.
8. Run the search.

For left/right montage refinement where `E3` should mirror `E1` and `E4`
should mirror `E2`, use `cross_pairs`.

## Bucket JSON

Bucket files contain one list per electrode position. Electrode labels should
match the selected EEG net.

```json
{
  "e1_plus": ["39", "34", "27"],
  "e1_minus": ["70", "71", "72"],
  "e2_plus": ["220", "212", "204"],
  "e2_minus": ["177", "169", "159"],
  "e3_plus": ["218", "223", "230"],
  "e3_minus": ["188", "187", "186"],
  "e4_plus": ["38", "46", "54"],
  "e4_minus": ["83", "91", "101"]
}
```

The loader also accepts full Ex-Search-style config JSON files when the bucket
lists are nested under an `electrodes` key.

## Metrics

The available mTI metric values are:

| Value | Meaning |
|-------|---------|
| `recursive_ti` | Recursive pairwise TI combination |
| `botzanowski_magnitude_am` | Botzanowski magnitude AM |
| `botzanowski_directional_am` | Botzanowski directional AM peak |
| `botzanowski_directional_am_ti_avg` | Botzanowski directional AM average |
| `grossman_ext_directional_am` | Grossman-extended directional AM peak |
| `grossman_ext_directional_am_ti_avg` | Grossman-extended directional AM average |

Each run ranks candidates using only the selected metric.

## Outputs

Results are written under the subject's `m-ex-search` output folder. The main
files follow the regular Ex-Search pattern:

- `final_output.csv`: ranked montage table with ROI, grey-matter, focality, and
  current columns
- `best_composite.csv`: highest composite-index montage summary
- `config.json`: saved run configuration
- distribution plots for the generated result table

The electrode scalp-map output is not currently generated for `m-ex-search`
because the older plot parser assumes the regular four-electrode Ex-Search
format.

## CLI Entrypoint

The same workflow can be launched from a config JSON:

```bash
simnibs_python -m tit.opt.mex path/to/m-ex-search-config.json
```

Example config shape:

```json
{
  "project_dir": "/path/to/TI-Toolbox/project",
  "subject_id": "001",
  "leadfield_hdf": "/path/to/leadfield.hdf5",
  "roi_name": "my_roi.csv",
  "current_mA": 5.0,
  "mti_metric": "recursive_ti",
  "symmetric_bucket": true,
  "symmetry_eeg_csv": "/path/to/GSN-256.csv",
  "symmetry_pairing": "cross_pairs",
  "electrodes": {
    "e1_plus": ["39", "34", "27"],
    "e1_minus": ["70", "71", "72"],
    "e2_plus": ["220", "212", "204"],
    "e2_minus": ["177", "169", "159"],
    "e3_plus": ["218", "223", "230"],
    "e3_minus": ["188", "187", "186"],
    "e4_plus": ["38", "46", "54"],
    "e4_minus": ["83", "91", "101"]
  }
}
```

## Performance Notes

Runtime scales with the number of valid bucket combinations and with the cost
of the selected mTI metric. Symmetry constraints can dramatically reduce the
search space. The GUI output reports progress and an ETA during the run,
including a clearer progress estimate every 500 candidates.

If a search is too large, reduce one or more buckets, enable `cross_pairs`
symmetry when appropriate, or start with a reduced bucket generated from the
best candidates of a prior run.
