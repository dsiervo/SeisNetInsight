"""Session persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import GridParameters, parameter_dict

SESSION_ROOT = Path.home() / ".seisnetinsight" / "sessions"
SESSION_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class SessionFiles:
    events: Optional[str] = None
    stations: Optional[str] = None
    swd: Optional[str] = None
    bna_files: List[str] = field(default_factory=list)


@dataclass
class SessionState:
    name: str
    parameters: GridParameters
    files: SessionFiles
    grids: Dict[str, str] = field(default_factory=dict)

    def directory(self) -> Path:
        return SESSION_ROOT / self.name

    def save_metadata(self) -> None:
        directory = self.directory()
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "parameters": parameter_dict(self.parameters),
            "files": asdict(self.files),
            "grids": self.grids,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, name: str) -> "SessionState":
        directory = SESSION_ROOT / name
        if not directory.exists():
            raise FileNotFoundError(f"Session '{name}' does not exist")
        metadata = json.loads((directory / "metadata.json").read_text())
        params = GridParameters()
        params = params.__class__(**{
            "lons": tuple(map(float, metadata["parameters"]["LONS"].split(","))),
            "lats": tuple(map(float, metadata["parameters"]["LATS"].split(","))),
            "grid_step": float(metadata["parameters"]["GRID_STEP"]),
            "dist_threshold_sub4": float(metadata["parameters"]["DIST_THRESHOLD_SUB4"]),
            "min_sta_sub4": int(metadata["parameters"]["MIN_STA_SUB4"]),
            "weight_sub4": float(metadata["parameters"]["WEIGHT_SUB4"]),
            "dist_threshold_sub10": float(metadata["parameters"]["DIST_THRESHOLD_SUB10"]),
            "min_sta_sub10": int(metadata["parameters"]["MIN_STA_SUB10"]),
            "weight_sub10": float(metadata["parameters"]["WEIGHT_SUB10"]),
            "gap_search_km": float(metadata["parameters"]["GAP_SEARCH_KM"]),
            "weight_gap": float(metadata["parameters"]["WEIGHT_GAP"]),
            "swd_radius_km": float(metadata["parameters"]["SWD_RADIUS_KM"]),
            "weight_swd": float(metadata["parameters"]["WEIGHT_SWD"]),
            "half_time_years": float(metadata["parameters"]["HALF_TIME_YEARS"]),
            "overwrite": bool(metadata["parameters"]["OVERWRITE"]),
        })
        files = SessionFiles(**metadata["files"])
        return cls(name=name, parameters=params, files=files, grids=metadata.get("grids", {}))

    def save_dataframe(self, df: pd.DataFrame, key: str) -> None:
        directory = self.directory()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key}.parquet"
        df.to_parquet(path, index=False)
        self.grids[key] = path.name
        self.save_metadata()

    def load_dataframe(self, key: str) -> pd.DataFrame:
        directory = self.directory()
        if key not in self.grids:
            raise KeyError(f"Grid '{key}' not available in session '{self.name}'")
        return pd.read_parquet(directory / self.grids[key])

    def save_source(self, df: pd.DataFrame, key: str) -> None:
        directory = self.directory()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key}_source.parquet"
        df.to_parquet(path, index=False)
        setattr(self.files, key, path.name)
        self.save_metadata()

    def save_bna(self, filename: str, data: bytes) -> None:
        directory = self.directory()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_bytes(data)
        if filename not in self.files.bna_files:
            self.files.bna_files.append(filename)
        self.save_metadata()

    def load_source(self, key: str) -> Optional[pd.DataFrame]:
        filename = getattr(self.files, key)
        if not filename:
            return None
        path = self.directory() / filename
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def list_bna(self) -> List[Path]:
        directory = self.directory()
        return [directory / name for name in self.files.bna_files if (directory / name).exists()]


def list_sessions() -> List[str]:
    return sorted([p.name for p in SESSION_ROOT.glob("*") if p.is_dir() and (p / "metadata.json").exists()])
