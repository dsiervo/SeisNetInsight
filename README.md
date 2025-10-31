# SeisNetInsight

SeisNetInsight provides tooling for the seismic community to evaluate station coverage, compute monitoring priority grids, and produce interactive maps. The new Python package bundles the grid computations, session persistence, and a Streamlit web application for exploratory analysis.

## Package overview

The package exposes utilities for:

- Loading and validating seismic event, station, and SWD well catalogues.
- Generating subject proximity (primary/secondary), ΔGap90, and SWD influence grids on a configurable mesh.
- Computing a composite monitoring index with user-defined weights and half-life decay.
- Exporting map artefacts (PNG, PDF, KML, Shapefile) from the computed grids.
- Launching an interactive Streamlit UI to guide users through data loading, grid computation, and map visualisation.

## Installation

```bash
pip install .
```

This installs the `seisnetinsight` package along with the `seisnetinsight-app` console script for the Streamlit interface.

## Running the Streamlit app

After installation, launch the interface with:

```bash
seisnetinsight-app
```

The UI is divided into three sections:

1. **Data loading** – Restore a previous session or configure a new one by uploading events, stations, optional SWD well catalogues, and BNA polygons. The interface shows the first 10 rows of each dataset, warns about missing expected columns, and optionally applies BallTree catalogue reduction.
2. **Grids computation** – Compute subject, ΔGap90, SWD, and composite grids with progress indicators, stop controls, and automatic session caching.
3. **Maps** – Explore interactive pydeck-based maps for each feature and the composite priority map. Download PNG, KML, Shapefile, or a consolidated PDF for enabled maps.

### GUI example:
<img width="600" height="600" alt="Screenshot 2025-10-31 182146" src="https://github.com/user-attachments/assets/4ad14cc1-7041-4b1e-ab58-414b401d17c5" />

## Programmatic usage

```python
from seisnetinsight import (
    default_parameters,
    generate_grid,
    compute_subject_grids,
    compute_gap_grid,
    compute_swd_grid,
    compute_composite_index,
)

params = default_parameters()
# events_df, stations_df, swd_df are pandas DataFrames with standard columns

grid = generate_grid(params)
subjects = compute_subject_grids(events_df, stations_df, grid, params)
gap = compute_gap_grid(events_df, stations_df, grid, params)
swd = compute_swd_grid(swd_df, grid, params)
combined = compute_composite_index(subjects.merge(gap, on=["latitude", "longitude"]).merge(swd, on=["latitude", "longitude"]), params)
```

Refer to the Streamlit app for a guided workflow and session persistence utilities.

# Output Examples
(a–c) are station-impact maps: brighter colors indicate grid cells where placing a station would affect a larger number of recency-weighted events. (a) Recency-weighted count of events that would be brought within 4 km by a station at each grid cell. (b) Same as (a) for a 10 km radius. (c) Recency-weighted count of events whose maximum azimuthal gap would be reduced to ≤90° by a station at each cell. (d) Cumulative salt-water-disposal volume within 25 km (barrels). Black outline denotes the Midland Basin:
<img width="520" height="500" alt="image" src="https://github.com/user-attachments/assets/3f67335f-5525-4fbe-b931-56f7d7fa2c0d" />

(a) Composite impact index (0–1) combining the four metrics—S₄, S₁₀, ΔGap ≤ 90°, and SWD—after recency weighting, min–max scaling, and user-set weights; brighter cells indicate grid locations where a new station would yield the largest combined benefit. (b) Priority regions derived by k-means clustering of the scaled metrics and ranked by mean composite index (Very High, High, Medium, Low). Blue triangles mark existing stations; the black outline denotes the Midland Basin.
<img width="900" height="521" alt="image" src="https://github.com/user-attachments/assets/53a6fec4-0de5-4c7e-8e01-c358656033da" />



