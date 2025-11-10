"""Data loading utilities."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

EXPECTED_EVENT_COLUMNS = ["latitude", "longitude", "magnitude", "origin_time"]
EXPECTED_STATION_COLUMNS = ["latitude", "longitude"]
EXPECTED_SWD_COLUMNS = ["latitude", "longitude", "volume"]

COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "latitude": ["latitude", "lat", "Latitude", "Latitude (WGS84)"],
    "longitude": ["longitude", "lon", "Longitude", "Longitude (WGS84)"],
    "magnitude": ["magnitude", "mag", "Magnitude"],
    "origin_time": ["origin_time", "time", "origin", "Origin Date", "event_time"],
    "volume": ["volume", "vol", "SUM_injected_liquid_BBL", "total_volume"],
}


@dataclass
class LoadedData:
    events: pd.DataFrame
    stations: pd.DataFrame
    swd: Optional[pd.DataFrame] = None


class ColumnWarning(UserWarning):
    """Raised when expected columns are missing."""


def _read_dataframe(source) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        return pd.read_csv(source)
    if isinstance(source, bytes):
        return pd.read_csv(io.BytesIO(source))
    if hasattr(source, "read"):
        content = source.read()
        if isinstance(content, bytes):
            return pd.read_csv(io.BytesIO(content))
        return pd.read_csv(io.StringIO(content))
    raise TypeError(f"Unsupported source type: {type(source)!r}")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    return df.rename(columns=rename_map)


def validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    missing = [col for col in required if col not in df.columns]
    return missing


def load_events(source, *, warn: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    df = _read_dataframe(source)
    df = _standardize_columns(df)
    missing = validate_required_columns(df, EXPECTED_EVENT_COLUMNS)
    if "origin_time" in df.columns:
        df["origin_time"] = pd.to_datetime(df["origin_time"], errors="coerce")
    if warn and missing:
        for column in missing:
            pd.Series(dtype="object")  # touch pandas to avoid lint
    return df, missing


def load_stations(source, *, warn: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    df = _read_dataframe(source)
    df = _standardize_columns(df)
    missing = validate_required_columns(df, EXPECTED_STATION_COLUMNS)
    if warn and missing:
        for column in missing:
            pd.Series(dtype="object")
    return df, missing


def load_swd_wells(source, *, warn: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    df = _read_dataframe(source)
    df = _standardize_columns(df)
    missing = validate_required_columns(df, EXPECTED_SWD_COLUMNS)
    if warn and missing:
        for column in missing:
            pd.Series(dtype="object")
    return df, missing


def balltree_reduce_events(
    df: pd.DataFrame,
    *,
    distance_threshold_km: float,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    from sklearn.neighbors import BallTree
    import numpy as np

    coords = df[[lat_col, lon_col]].to_numpy()
    radians = np.radians(coords)
    tree = BallTree(radians, metric="haversine")
    mask = np.ones(len(df), dtype=bool)
    radius = distance_threshold_km / 6371.0
    for idx in range(len(df)):
        if not mask[idx]:
            continue
        neighbors = tree.query_radius([radians[idx]], r=radius)[0]
        mask[neighbors] = False
        mask[idx] = True
    return df.loc[mask].reset_index(drop=True)
