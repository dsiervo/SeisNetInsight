"""Map creation and export utilities."""

from __future__ import annotations

import io
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
from matplotlib.backends.backend_pdf import PdfPages

from .config import GridParameters


FEATURE_COLUMNS = [
    "subject4_within_4km_weighted",
    "subject10_within_10km_weighted",
    "delta_gap90_weighted",
    "swd_volume_25km_bbl",
    "composite_index",
]


@dataclass
class MapArtifacts:
    feature_name: str
    deck: pdk.Deck
    png_bytes: bytes
    kml_bytes: Optional[bytes]
    shp_bytes: Optional[bytes]


def _grid_to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geometry = gpd.points_from_xy(df["longitude"], df["latitude"], crs="EPSG:4326")
    return gpd.GeoDataFrame(df.copy(), geometry=geometry)


def _priority_labels(values: pd.Series) -> pd.Series:
    quantiles = values.quantile([0.25, 0.5, 0.75]).to_list()
    bins = [-math.inf] + quantiles + [math.inf]
    labels = ["Low", "Medium", "High", "Very High"]
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True)


def create_feature_map(df: pd.DataFrame, feature: str, *, tooltip: bool = True) -> pdk.Deck:
    data = df[["longitude", "latitude", feature]].rename(columns={feature: "value"})
    layer = pdk.Layer(
        "HeatmapLayer",
        data=data,
        get_position="[longitude, latitude]",
        aggregation="SUM",
        get_weight="value",
        radiusPixels=40,
    )
    tooltip_conf = {"html": f"<b>{feature}</b>: {{value}}", "style": {"backgroundColor": "#2E2E2E", "color": "white"}} if tooltip else None
    view_state = pdk.ViewState(
        longitude=float(data["longitude"].mean()),
        latitude=float(data["latitude"].mean()),
        zoom=6,
        pitch=0,
    )
    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip_conf)


def create_priority_map(df: pd.DataFrame) -> pdk.Deck:
    data = df[["longitude", "latitude", "composite_index"]].copy()
    data["priority"] = _priority_labels(data["composite_index"]).astype(str)
    color_map = {
        "Very High": [179, 0, 0],
        "High": [255, 128, 0],
        "Medium": [255, 215, 0],
        "Low": [34, 139, 34],
    }
    data["color"] = data["priority"].map(color_map)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[longitude, latitude]",
        get_color="color",
        get_radius=1500,
        pickable=True,
        auto_highlight=True,
    )
    tooltip_conf = {
        "html": "<b>Priority:</b> {priority}<br/><b>Composite:</b> {composite_index:.2f}",
        "style": {"backgroundColor": "#2E2E2E", "color": "white"},
    }
    view_state = pdk.ViewState(
        longitude=float(data["longitude"].mean()),
        latitude=float(data["latitude"].mean()),
        zoom=6,
    )
    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip_conf)


def _export_static_map(df: pd.DataFrame, feature: str) -> bytes:
    pivot = df.pivot_table(index="latitude", columns="longitude", values=feature)
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(pivot.columns, pivot.index, pivot.values, shading="auto")
    fig.colorbar(mesh, ax=ax, label=feature)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{feature} map")
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def _export_kml(gdf: gpd.GeoDataFrame, feature: str) -> Optional[bytes]:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            kml_path = Path(tmpdir) / f"{feature}.kml"
            gdf.to_file(kml_path, driver="KML")
            return kml_path.read_bytes()
    except Exception:
        return None


def _export_shapefile(gdf: gpd.GeoDataFrame, feature: str) -> Optional[bytes]:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_dir = Path(tmpdir) / feature
            shp_dir.mkdir(parents=True, exist_ok=True)
            shp_path = shp_dir / f"{feature}.shp"
            gdf.to_file(shp_path)
            archive_path = Path(tmpdir) / f"{feature}.zip"
            import zipfile

            with zipfile.ZipFile(archive_path, "w") as zf:
                for file in shp_dir.iterdir():
                    zf.write(file, file.name)
            return archive_path.read_bytes()
    except Exception:
        return None


def export_feature_map(df: pd.DataFrame, feature: str) -> MapArtifacts:
    deck = create_feature_map(df, feature)
    png_bytes = _export_static_map(df, feature)
    gdf = _grid_to_geodataframe(df[["latitude", "longitude", feature]])
    kml_bytes = _export_kml(gdf, feature)
    shp_bytes = _export_shapefile(gdf, feature)
    return MapArtifacts(feature, deck, png_bytes, kml_bytes, shp_bytes)


def export_priority_map(df: pd.DataFrame) -> MapArtifacts:
    deck = create_priority_map(df)
    png_bytes = _export_static_map(df, "composite_index")
    gdf = _grid_to_geodataframe(df[["latitude", "longitude", "composite_index"]])
    kml_bytes = _export_kml(gdf, "priority")
    shp_bytes = _export_shapefile(gdf, "priority")
    return MapArtifacts("priority", deck, png_bytes, kml_bytes, shp_bytes)


def export_pdf(artifacts: Iterable[MapArtifacts]) -> bytes:
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for artifact in artifacts:
            image = plt.imread(io.BytesIO(artifact.png_bytes))
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(artifact.feature_name)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()
