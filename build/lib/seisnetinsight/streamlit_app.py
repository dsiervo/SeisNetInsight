"""Streamlit application for SeisNetInsight."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from .config import GridParameters, default_parameters, parameter_from_inputs
from .data import (
    EXPECTED_EVENT_COLUMNS,
    EXPECTED_STATION_COLUMNS,
    EXPECTED_SWD_COLUMNS,
    balltree_reduce_events,
    load_events,
    load_stations,
    load_swd_wells,
)
from .grids import (
    GridDefinition,
    compute_composite_index,
    compute_gap_grid,
    compute_subject_grids,
    compute_swd_grid,
    generate_grid,
    merge_grids,
)
from .maps import FEATURE_COLUMNS, export_feature_map, export_pdf, export_priority_map
from .sessions import SessionFiles, SessionState, list_sessions


@dataclass
class WorkingSession:
    name: str
    parameters: GridParameters = field(default_factory=default_parameters)
    events: Optional[pd.DataFrame] = None
    stations: Optional[pd.DataFrame] = None
    swd: Optional[pd.DataFrame] = None
    grid: Optional[GridDefinition] = None
    grids: Dict[str, pd.DataFrame] = field(default_factory=dict)
    column_warnings: Dict[str, List[str]] = field(default_factory=dict)
    balltree_enabled: bool = False
    balltree_distance: float = 1.0
    storage: Optional[SessionState] = None
    bna_files: Dict[str, bytes] = field(default_factory=dict)

    @property
    def data_loaded(self) -> bool:
        return self.events is not None and self.stations is not None

    def ensure_grid(self) -> GridDefinition:
        if self.grid is None:
            self.grid = generate_grid(self.parameters)
        return self.grid

    def merged(self) -> Optional[pd.DataFrame]:
        frames = []
        if "subject" in self.grids:
            frames.append(self.grids["subject"])
        if "gap" in self.grids:
            frames.append(self.grids["gap"])
        if "swd" in self.grids:
            frames.append(self.grids["swd"])
        if not frames:
            return None
        merged = merge_grids(*frames)
        if "composite" in self.grids:
            merged = merged.merge(
                self.grids["composite"][
                    [
                        "latitude",
                        "longitude",
                        "composite_index",
                        "contrib_subject4",
                        "contrib_subject10",
                        "contrib_gap",
                        "contrib_swd",
                    ]
                ],
                on=["latitude", "longitude"],
                how="left",
            )
        return merged


SESSION_KEY = "seisnetinsight_session"
STOP_PREFIX = "stop_"
RUN_PREFIX = "run_"


def _default_session_name() -> str:
    return dt.datetime.utcnow().strftime("session-%Y%m%d-%H%M%S")


def get_working_session() -> WorkingSession:
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = WorkingSession(name=_default_session_name())
    return st.session_state[SESSION_KEY]


def set_working_session(session: WorkingSession) -> None:
    st.session_state[SESSION_KEY] = session


def _stop_key(name: str) -> str:
    return f"{STOP_PREFIX}{name}"


def _run_key(name: str) -> str:
    return f"{RUN_PREFIX}{name}"


def _stop_requested(name: str) -> bool:
    return bool(st.session_state.get(_stop_key(name), False))


def _reset_stop(name: str) -> None:
    st.session_state[_stop_key(name)] = False


def _show_column_warnings(warnings: Dict[str, List[str]]) -> None:
    for dataset, missing in warnings.items():
        if missing:
            st.warning(
                f"{dataset} is missing expected columns: {', '.join(missing)}."
            )


def _load_session_from_disk(name: str) -> WorkingSession:
    state = SessionState.load(name)
    ws = WorkingSession(name=name, parameters=state.parameters, storage=state)
    ws.events = state.load_source("events")
    ws.stations = state.load_source("stations")
    ws.swd = state.load_source("swd")
    ws.grid = generate_grid(ws.parameters)
    warnings: Dict[str, List[str]] = {}
    warnings["Events"] = [] if ws.events is not None else EXPECTED_EVENT_COLUMNS
    warnings["Stations"] = [] if ws.stations is not None else EXPECTED_STATION_COLUMNS
    warnings["SWD wells"] = [] if ws.swd is not None else []
    ws.column_warnings = warnings
    for key in ["subject", "gap", "swd", "composite"]:
        if key in state.grids:
            ws.grids[key] = state.load_dataframe(key)
    return ws


def _save_sources(session: WorkingSession, events: pd.DataFrame, stations: pd.DataFrame, swd: Optional[pd.DataFrame]) -> None:
    if session.storage is None:
        session.storage = SessionState(session.name, session.parameters, SessionFiles())
    session.storage.save_source(events, "events")
    session.storage.save_source(stations, "stations")
    if swd is not None:
        session.storage.save_source(swd, "swd")
    session.storage.save_metadata()


def _save_bna_files(session: WorkingSession) -> None:
    if session.storage is None:
        return
    for name, data in session.bna_files.items():
        session.storage.save_bna(name, data)


def _run_subject_grid(session: WorkingSession) -> None:
    status = st.empty()
    progress = st.progress(0.0, text="Starting subject grids…")

    def update(fraction: float, message: str) -> None:
        progress.progress(min(fraction, 1.0), text=message)

    try:
        grid = session.ensure_grid()
        result = compute_subject_grids(
            session.events,
            session.stations,
            grid,
            session.parameters,
            progress=update,
            should_stop=lambda: _stop_requested("subject"),
        )
        session.grids["subject"] = result
        if session.storage:
            session.storage.save_dataframe(result, "subject")
        status.success("Subject grids computed.")
    except InterruptedError:
        status.warning("Subject grids computation stopped by user.")
    except Exception as exc:
        status.error(f"Failed to compute subject grids: {exc}")
    finally:
        progress.empty()
        _reset_stop("subject")
        st.session_state[_run_key("subject")] = False


def _run_gap_grid(session: WorkingSession) -> None:
    status = st.empty()
    progress = st.progress(0.0, text="Starting ΔGap90 grid…")

    def update(fraction: float, message: str) -> None:
        progress.progress(min(fraction, 1.0), text=message)

    try:
        grid = session.ensure_grid()
        result = compute_gap_grid(
            session.events,
            session.stations,
            grid,
            session.parameters,
            progress=update,
            should_stop=lambda: _stop_requested("gap"),
        )
        session.grids["gap"] = result
        if session.storage:
            session.storage.save_dataframe(result, "gap")
        status.success("ΔGap90 grid computed.")
    except InterruptedError:
        status.warning("ΔGap90 computation stopped by user.")
    except Exception as exc:
        status.error(f"Failed to compute ΔGap90 grid: {exc}")
    finally:
        progress.empty()
        _reset_stop("gap")
        st.session_state[_run_key("gap")] = False


def _run_swd_grid(session: WorkingSession) -> None:
    status = st.empty()
    progress = st.progress(0.0, text="Starting SWD grid…")

    def update(fraction: float, message: str) -> None:
        progress.progress(min(fraction, 1.0), text=message)

    try:
        grid = session.ensure_grid()
        result = compute_swd_grid(
            session.swd,
            grid,
            session.parameters,
            progress=update,
            should_stop=lambda: _stop_requested("swd"),
        )
        session.grids["swd"] = result
        if session.storage:
            session.storage.save_dataframe(result, "swd")
        status.success("SWD grid computed.")
    except InterruptedError:
        status.warning("SWD computation stopped by user.")
    except Exception as exc:
        status.error(f"Failed to compute SWD grid: {exc}")
    finally:
        progress.empty()
        _reset_stop("swd")
        st.session_state[_run_key("swd")] = False


def _run_composite_grid(session: WorkingSession) -> None:
    merged = session.merged()
    if merged is None:
        st.warning("Compute subject, gap, and SWD grids before the composite index.")
        return
    try:
        composite = compute_composite_index(merged, session.parameters)
        session.grids["composite"] = composite
        if session.storage:
            session.storage.save_dataframe(composite, "composite")
        st.success("Composite index grid computed.")
    except Exception as exc:
        st.error(f"Failed to compute composite index: {exc}")


def _render_data_loading(session: WorkingSession) -> None:
    st.header("1. Data Loading")
    existing_sessions = list_sessions()
    if existing_sessions:
        st.markdown("**Restore previous session**")
        restore_choice = st.checkbox("Restore a session", key="restore_toggle")
        selected_name = st.selectbox("Available sessions", existing_sessions, disabled=not restore_choice)
        if restore_choice and st.button("Load session"):
            try:
                new_session = _load_session_from_disk(selected_name)
                set_working_session(new_session)
                st.success(f"Session '{selected_name}' loaded.")
            except Exception as exc:
                st.error(f"Failed to load session: {exc}")
    st.markdown("---")
    st.subheader("New session setup")
    session_name = st.text_input("Session name", value=session.name)
    lons_input = st.text_input("Longitude bounds (min,max)", value=",".join(map(str, session.parameters.lons)))
    lats_input = st.text_input("Latitude bounds (min,max)", value=",".join(map(str, session.parameters.lats)))
    grid_step = st.number_input("Grid step (degrees)", value=float(session.parameters.grid_step), min_value=0.001, step=0.001)
    dist_sub4 = st.number_input("Subject-4 distance threshold (km)", value=float(session.parameters.dist_threshold_sub4))
    min_sub4 = st.number_input("Minimum stations for subject-4", value=int(session.parameters.min_sta_sub4), min_value=0)
    weight_sub4 = st.number_input("Weight subject-4", value=float(session.parameters.weight_sub4))
    dist_sub10 = st.number_input("Subject-10 distance threshold (km)", value=float(session.parameters.dist_threshold_sub10))
    min_sub10 = st.number_input("Minimum stations for subject-10", value=int(session.parameters.min_sta_sub10), min_value=0)
    weight_sub10 = st.number_input("Weight subject-10", value=float(session.parameters.weight_sub10))
    gap_search = st.number_input("Gap search radius (km)", value=float(session.parameters.gap_search_km))
    weight_gap = st.number_input("Weight ΔGap90", value=float(session.parameters.weight_gap))
    swd_radius = st.number_input("SWD radius (km)", value=float(session.parameters.swd_radius_km))
    weight_swd = st.number_input("Weight SWD", value=float(session.parameters.weight_swd))
    half_time = st.number_input("Half-time (years)", value=float(session.parameters.half_time_years))
    overwrite = st.checkbox("Overwrite cached grids", value=session.parameters.overwrite)

    balltree_enabled = st.checkbox("Apply BallTree data reduction", value=session.balltree_enabled)
    balltree_distance = st.number_input("BallTree distance threshold (km)", value=float(session.balltree_distance), min_value=0.1, step=0.1)

    events_file = st.file_uploader("Events file", type=["csv"])
    stations_file = st.file_uploader("Stations file", type=["csv"])
    swd_file = st.file_uploader("SWD wells file (optional)", type=["csv"], key="swd")
    bna_files = st.file_uploader("BNA files (optional)", type=["bna"], accept_multiple_files=True)

    inputs = {
        "LONS": lons_input,
        "LATS": lats_input,
        "GRID_STEP": grid_step,
        "DIST_THRESHOLD_SUB4": dist_sub4,
        "MIN_STA_SUB4": min_sub4,
        "WEIGHT_SUB4": weight_sub4,
        "DIST_THRESHOLD_SUB10": dist_sub10,
        "MIN_STA_SUB10": min_sub10,
        "WEIGHT_SUB10": weight_sub10,
        "GAP_SEARCH_KM": gap_search,
        "WEIGHT_GAP": weight_gap,
        "SWD_RADIUS_KM": swd_radius,
        "WEIGHT_SWD": weight_swd,
        "HALF_TIME_YEARS": half_time,
        "OVERWRITE": overwrite,
    }
    parameters = parameter_from_inputs(inputs, logger=st.warning)

    def process_inputs(run_all: bool) -> None:
        if events_file is None or stations_file is None:
            st.error("Events and stations files are required.")
            return
        events_df, events_missing = load_events(events_file)
        stations_df, stations_missing = load_stations(stations_file)
        swd_df = None
        swd_missing: List[str] = []
        if swd_file is not None:
            swd_df, swd_missing = load_swd_wells(swd_file)
        warnings = {
            "Events": events_missing,
            "Stations": stations_missing,
            "SWD wells": swd_missing,
        }
        if balltree_enabled:
            events_df = balltree_reduce_events(events_df, distance_threshold_km=balltree_distance)
        new_session = WorkingSession(
            name=session_name,
            parameters=parameters,
            events=events_df,
            stations=stations_df,
            swd=swd_df,
            balltree_enabled=balltree_enabled,
            balltree_distance=balltree_distance,
            column_warnings=warnings,
        )
        new_session.grid = generate_grid(parameters)
        if bna_files:
            for file in bna_files:
                new_session.bna_files[file.name] = file.getvalue()
        set_working_session(new_session)
        _show_column_warnings(warnings)
        st.dataframe(events_df.head(10), use_container_width=True)
        st.dataframe(stations_df.head(10), use_container_width=True)
        if swd_df is not None:
            st.dataframe(swd_df.head(10), use_container_width=True)
        _save_sources(new_session, events_df, stations_df, swd_df)
        _save_bna_files(new_session)
        if run_all:
            st.session_state[_run_key("subject")] = True
            st.session_state[_run_key("gap")] = True
            st.session_state[_run_key("swd")] = True
            _run_subject_grid(new_session)
            _run_gap_grid(new_session)
            _run_swd_grid(new_session)
            _run_composite_grid(new_session)

    col1, col2 = st.columns(2)
    if col1.button("Load data"):
        process_inputs(False)
    if col2.button("Load data and run all"):
        process_inputs(True)



def _render_grid_section(session: WorkingSession) -> None:
    st.header("2. Grids Computation")
    if not session.data_loaded:
        st.info("Load data to enable grid computations.")
        return

    def run_with_stop(name: str, label: str, runner) -> None:
        trigger_key = _run_key(name)
        if st.button(f"Re-compute {label}"):
            st.session_state[trigger_key] = True
        if st.session_state.get(trigger_key):
            stop_placeholder = st.empty()
            stop_placeholder.button(
                "Stop", key=f"stop_btn_{name}", on_click=lambda: st.session_state.update({_stop_key(name): True})
            )
            runner(session)
            stop_placeholder.empty()

    run_with_stop("subject", "subject grids", _run_subject_grid)
    run_with_stop("gap", "ΔGap90 grid", _run_gap_grid)
    run_with_stop("swd", "SWD grid", _run_swd_grid)

    if st.button("Re-compute composite index"):
        _run_composite_grid(session)

    merged = session.merged()
    if merged is not None:
        st.success("Merged grid summary")
        st.dataframe(merged.describe(), use_container_width=True)



def _render_maps_section(session: WorkingSession) -> None:
    st.header("3. Maps")
    merged = session.merged()
    if merged is None:
        st.info("Compute grids to enable map previews.")
        return

    st.subheader("Feature maps")
    enabled_features = {}
    artifacts_cache: Dict[str, object] = {}
    for feature in FEATURE_COLUMNS:
        enabled_features[feature] = st.checkbox(f"Show {feature}", value=True, key=f"feature_{feature}")
        if enabled_features[feature]:
            artifacts = export_feature_map(merged, feature)
            artifacts_cache[feature] = artifacts
            st.pydeck_chart(artifacts.deck)
            st.download_button(
                label=f"Download {feature} PNG",
                data=artifacts.png_bytes,
                file_name=f"{feature}.png",
            )
            if artifacts.kml_bytes:
                st.download_button(
                    label=f"Download {feature} KML",
                    data=artifacts.kml_bytes,
                    file_name=f"{feature}.kml",
                )
            if artifacts.shp_bytes:
                st.download_button(
                    label=f"Download {feature} Shapefile",
                    data=artifacts.shp_bytes,
                    file_name=f"{feature}.zip",
                )
    st.subheader("Priority regions map")
    priority_enabled = st.checkbox("Show priority regions", value=True)
    artifacts_to_export = []
    if priority_enabled:
        priority_artifacts = export_priority_map(merged)
        st.pydeck_chart(priority_artifacts.deck)
        st.download_button("Download priority PNG", priority_artifacts.png_bytes, file_name="priority.png")
        if priority_artifacts.kml_bytes:
            st.download_button("Download priority KML", priority_artifacts.kml_bytes, file_name="priority.kml")
        if priority_artifacts.shp_bytes:
            st.download_button("Download priority Shapefile", priority_artifacts.shp_bytes, file_name="priority.zip")
        artifacts_to_export.append(priority_artifacts)
    for feature, enabled in enabled_features.items():
        if enabled:
            artifacts_to_export.append(artifacts_cache.get(feature) or export_feature_map(merged, feature))
    if artifacts_to_export:
        pdf_bytes = export_pdf(artifacts_to_export)
        st.download_button("Download PDF with selected maps", pdf_bytes, file_name="seisnetinsight_maps.pdf")


def main() -> None:
    st.set_page_config(page_title="SeisNetInsight", layout="wide")
    st.title("SeisNetInsight")
    st.caption("Interactive seismic monitoring prioritization tools")
    session = get_working_session()
    _render_data_loading(session)
    session = get_working_session()
    _render_grid_section(session)
    session = get_working_session()
    _render_maps_section(session)


if __name__ == "__main__":
    main()
