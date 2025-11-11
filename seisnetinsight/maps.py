"""Map creation and export utilities."""

from __future__ import annotations

import io
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import matplotlib as mpl
from .config import GridParameters


FEATURE_COLUMNS = [
    "subject_primary_weighted",
    "subject_secondary_weighted",
    "delta_gap90_weighted",
    "swd_volume_25km_bbl",
    "composite_index",
]
DEFAULT_PRIORITY_FEATURES = [
    "subject_primary_weighted",
    "subject_secondary_weighted",
    "delta_gap90_weighted",
    "swd_volume_25km_bbl",
]
PRIORITY_LEVELS = ["Very High", "High", "Medium", "Low"]


def _priority_color_lookup(levels: Iterable[str]) -> Dict[str, list[int]]:
    palette = mpl.colormaps["hot_r"](np.linspace(0.75, 0.35, len(PRIORITY_LEVELS)))
    colors: Dict[str, list[int]] = {}
    for label, rgba in zip(PRIORITY_LEVELS, palette):
        rgb = [int(channel * 255) for channel in rgba[:3]]
        colors[label] = rgb
    # Map any additional labels (if provided) to nearest color sequence order.
    for label in levels:
        if label not in colors:
            colors[label] = list(colors[PRIORITY_LEVELS[min(len(PRIORITY_LEVELS) - 1, len(colors))]])
    return colors


PRIORITY_COLORS = _priority_color_lookup(PRIORITY_LEVELS)


@dataclass
class MapArtifacts:
    feature_name: str
    deck: pdk.Deck
    png_bytes: bytes
    kml_bytes: Optional[bytes]
    kmz_bytes: Optional[bytes]
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
    data["color"] = data["priority"].map(PRIORITY_COLORS)
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


def _export_kmz(gdf: gpd.GeoDataFrame, feature: str) -> Optional[bytes]:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            kml_path = tmp_path / f"{feature}.kml"
            gdf.to_file(kml_path, driver="KML")
            kmz_path = tmp_path / f"{feature}.kmz"
            import zipfile

            with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(kml_path, arcname=kml_path.name)
            return kmz_path.read_bytes()
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
    kmz_bytes = _export_kmz(gdf, feature)
    shp_bytes = _export_shapefile(gdf, feature)
    return MapArtifacts(feature, deck, png_bytes, kml_bytes, kmz_bytes, shp_bytes)


def export_priority_map(df: pd.DataFrame) -> MapArtifacts:
    deck = create_priority_map(df)
    png_bytes = _export_static_map(df, "composite_index")
    gdf = _grid_to_geodataframe(df[["latitude", "longitude", "composite_index"]])
    kml_bytes = _export_kml(gdf, "priority")
    kmz_bytes = _export_kmz(gdf, "priority")
    shp_bytes = _export_shapefile(gdf, "priority")
    return MapArtifacts("priority", deck, png_bytes, kml_bytes, kmz_bytes, shp_bytes)


def classify_priority_clusters(
    df: pd.DataFrame,
    *,
    features: Optional[Iterable[str]] = None,
    n_clusters: int = 4,
    random_state: int = 42,
    n_init: int = 20,
) -> pd.DataFrame:
    if features is None:
        features = DEFAULT_PRIORITY_FEATURES
    missing = [name for name in features if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Cannot classify priorities on an empty DataFrame.")
    feature_frame = df[list(features)].fillna(0.0)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feature_frame.to_numpy(dtype=float))
    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
    )
    clusters = model.fit_predict(scaled)
    result = df.copy()
    result["priority_cluster"] = clusters

    def _label_mapping(series: pd.Series) -> Dict[int, str]:
        order = series.sort_values(ascending=False).index.tolist()
        labels = PRIORITY_LEVELS.copy()
        if len(order) > len(labels):
            labels.extend([f"Priority {idx + 1}" for idx in range(len(labels), len(order))])
        mapping: Dict[int, str] = {}
        for idx, cluster in enumerate(order):
            mapping[cluster] = labels[idx] if idx < len(labels) else labels[-1]
        return mapping

    if "composite_index" in result.columns:
        mean_scores = result.groupby("priority_cluster")["composite_index"].mean()
        mapping = _label_mapping(mean_scores)
    else:
        sums = result.groupby("priority_cluster")[list(features)].mean().sum(axis=1)
        mapping = _label_mapping(sums)
    result["priority_label"] = result["priority_cluster"].map(mapping)
    return result


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
