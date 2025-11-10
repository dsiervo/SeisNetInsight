"""Configuration helpers for SeisNetInsight."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import streamlit as st


@dataclass
class GridParameters:
    """User-configurable parameters for grid generation and weighting."""

    lons: Tuple[float, float] = (-102.9, -100.4)
    lats: Tuple[float, float] = (30.5, 34.0)
    grid_step: float = 0.01
    dist_threshold_sub4: float = 4.0
    min_sta_sub4: int = 1
    weight_sub4: float = 0.4
    dist_threshold_sub10: float = 10.0
    min_sta_sub10: int = 1
    weight_sub10: float = 0.35
    gap_search_km: float = 30.0
    weight_gap: float = 0.05
    swd_radius_km: float = 25.0
    weight_swd: float = 0.2
    half_time_years: float = 5.0
    overwrite: bool = False

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "longitude": self.lons,
            "latitude": self.lats,
        }

    def normalized_weights(self) -> Dict[str, float]:
        weights = {
            "subject4": max(self.weight_sub4, 0.0),
            "subject10": max(self.weight_sub10, 0.0),
            "gap": max(self.weight_gap, 0.0),
            "swd": max(self.weight_swd, 0.0),
        }
        total = sum(weights.values())
        if total == 0:
            return {k: 0.0 for k in weights}
        return {k: v / total for k, v in weights.items()}


DEFAULT_PARAMETER_NAMES = [
    "LONS",
    "LATS",
    "GRID_STEP",
    "DIST_THRESHOLD_SUB4",
    "MIN_STA_SUB4",
    "WEIGHT_SUB4",
    "DIST_THRESHOLD_SUB10",
    "MIN_STA_SUB10",
    "WEIGHT_SUB10",
    "GAP_SEARCH_KM",
    "WEIGHT_GAP",
    "SWD_RADIUS_KM",
    "WEIGHT_SWD",
    "HALF_TIME_YEARS",
    "OVERWRITE",
]


def default_parameters() -> GridParameters:
    return GridParameters()


def parse_bounds(value: str, fallback: Tuple[float, float]) -> Tuple[float, float]:
    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except Exception:  # pragma: no cover - defensive
        return fallback
    if len(parts) != 2:
        return fallback
    lo, hi = min(parts), max(parts)
    if lo == hi:
        hi = lo + 1e-6
    return (lo, hi)


def warn_missing_parameter(name: str, default_value: object, logger: Optional[Callable[[str], None]] = None) -> None:
    message = f"Parameter '{name}' missing; using default value {default_value!r}."
    if logger is not None:
        logger(message)
    else:
        st.warning(message)


def parameter_from_inputs(inputs: Dict[str, object], logger: Optional[Callable[[str], None]] = None) -> GridParameters:
    params = default_parameters()

    def get(key: str, default: object) -> object:
        if key not in inputs or inputs[key] in (None, ""):
            warn_missing_parameter(key, default, logger)
            return default
        return inputs[key]

    params.lons = parse_bounds(str(get("LONS", ",".join(map(str, params.lons)))), params.lons)
    params.lats = parse_bounds(str(get("LATS", ",".join(map(str, params.lats)))), params.lats)
    params.grid_step = float(get("GRID_STEP", params.grid_step))
    params.dist_threshold_sub4 = float(get("DIST_THRESHOLD_SUB4", params.dist_threshold_sub4))
    params.min_sta_sub4 = int(get("MIN_STA_SUB4", params.min_sta_sub4))
    params.weight_sub4 = float(get("WEIGHT_SUB4", params.weight_sub4))
    params.dist_threshold_sub10 = float(get("DIST_THRESHOLD_SUB10", params.dist_threshold_sub10))
    params.min_sta_sub10 = int(get("MIN_STA_SUB10", params.min_sta_sub10))
    params.weight_sub10 = float(get("WEIGHT_SUB10", params.weight_sub10))
    params.gap_search_km = float(get("GAP_SEARCH_KM", params.gap_search_km))
    params.weight_gap = float(get("WEIGHT_GAP", params.weight_gap))
    params.swd_radius_km = float(get("SWD_RADIUS_KM", params.swd_radius_km))
    params.weight_swd = float(get("WEIGHT_SWD", params.weight_swd))
    params.half_time_years = float(get("HALF_TIME_YEARS", params.half_time_years))
    params.overwrite = bool(get("OVERWRITE", params.overwrite))
    return params


def parameter_dict(params: GridParameters) -> Dict[str, object]:
    return {
        "LONS": f"{params.lons[0]},{params.lons[1]}",
        "LATS": f"{params.lats[0]},{params.lats[1]}",
        "GRID_STEP": params.grid_step,
        "DIST_THRESHOLD_SUB4": params.dist_threshold_sub4,
        "MIN_STA_SUB4": params.min_sta_sub4,
        "WEIGHT_SUB4": params.weight_sub4,
        "DIST_THRESHOLD_SUB10": params.dist_threshold_sub10,
        "MIN_STA_SUB10": params.min_sta_sub10,
        "WEIGHT_SUB10": params.weight_sub10,
        "GAP_SEARCH_KM": params.gap_search_km,
        "WEIGHT_GAP": params.weight_gap,
        "SWD_RADIUS_KM": params.swd_radius_km,
        "WEIGHT_SWD": params.weight_swd,
        "HALF_TIME_YEARS": params.half_time_years,
        "OVERWRITE": params.overwrite,
    }
