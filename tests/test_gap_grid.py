import pandas as pd
from pandas.testing import assert_frame_equal

from seisnetinsight.config import GridParameters
from seisnetinsight.grids import compute_gap_grid, generate_grid


def test_gap_grid_clips_inputs_to_aoi_bounds():
    params = GridParameters(lats=(0.0, 1.0), lons=(0.0, 1.0), grid_step=0.5, gap_search_km=50.0)
    grid = generate_grid(params)

    events = pd.DataFrame(
        {
            "latitude": [0.5, 1.01],  # second event sits just outside AOI
            "longitude": [0.5, 0.5],
            "origin_time": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        }
    )
    stations = pd.DataFrame(
        {
            "latitude": [0.5, 0.4, 1.05],  # third station falls outside AOI
            "longitude": [0.4, 0.5, 0.5],
        }
    )

    filtered_events = events.iloc[[0]].copy()
    filtered_stations = stations.iloc[:2].copy()

    expected = compute_gap_grid(filtered_events, filtered_stations, grid, params)
    result = compute_gap_grid(events, stations, grid, params)

    assert_frame_equal(result, expected)
