"""Streamlit application for SeisNetInsight."""

from __future__ import annotations

import datetime as dt
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from seisnetinsight.config import (
    GridParameters,
    default_parameters,
    parameter_from_inputs,
)
from seisnetinsight.data import (
    EXPECTED_EVENT_COLUMNS,
    EXPECTED_STATION_COLUMNS,
    EXPECTED_SWD_COLUMNS,
    COLUMN_ALIASES,
    balltree_reduce_events,
    load_events,
    load_stations,
    load_swd_wells,
)
from seisnetinsight.grids import (
    GridDefinition,
    compute_composite_index,
    compute_gap_grid,
    compute_subject_grids,
    compute_swd_grid,
    generate_grid,
    merge_grids,
)
from seisnetinsight.legacy_maps import (
    LegacyMapConfig,
    figure_png_bytes,
    render_legacy_contour,
    render_priority_clusters,
)
from seisnetinsight.maps import PRIORITY_LEVELS, classify_priority_clusters
from seisnetinsight.sessions import SessionFiles, SessionState, list_sessions


def _default_event_columns() -> Dict[str, str]:
    return {
        "latitude": "latitude",
        "longitude": "longitude",
        "magnitude": "magnitude",
        "origin_time": "origin_time",
    }


def _default_station_columns() -> Dict[str, str]:
    return {
        "latitude": "latitude",
        "longitude": "longitude",
    }


def _default_swd_columns() -> Dict[str, str]:
    return {
        "latitude": "latitude",
        "longitude": "longitude",
        "volume": "volume",
    }


def _normalize_column_mapping(raw: Dict[str, str], defaults: Dict[str, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, fallback in defaults.items():
        value = raw.get(key, "")
        normalized[key] = value.strip() or fallback
    return normalized


def _extract_columns(uploaded_file) -> List[str]:
    if uploaded_file is None:
        return []
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    try:
        preview = pd.read_csv(uploaded_file, nrows=0)
    except Exception:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        return []
    columns = [str(col) for col in preview.columns]
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    return columns


def _match_column(canonical: str, available: List[str], current: str) -> str:
    if current:
        for candidate in available:
            if candidate == current:
                return candidate
        for candidate in available:
            if candidate.lower() == current.lower():
                return candidate
    for alias in COLUMN_ALIASES.get(canonical, []):
        for candidate in available:
            if candidate.lower() == alias.lower():
                return candidate
    for candidate in available:
        if candidate.lower() == canonical.lower():
            return candidate
    return ""


def _render_column_selectors(
    fields: List[tuple[str, str]],
    available_columns: List[str],
    current_mapping: Dict[str, str],
    defaults: Dict[str, str],
    key_prefix: str,
) -> Dict[str, str]:
    if not available_columns:
        return current_mapping
    placeholder = "Select column…"
    columns_ui = st.columns(len(fields))
    selections: Dict[str, str] = {}
    for idx, (canonical, label) in enumerate(fields):
        default_choice = _match_column(canonical, available_columns, current_mapping.get(canonical, ""))
        index = 0
        if default_choice:
            try:
                index = available_columns.index(default_choice) + 1
            except ValueError:
                index = 0
        options = [placeholder] + available_columns
        selection = columns_ui[idx].selectbox(
            label,
            options,
            index=index,
            key=f"{key_prefix}_{canonical}",
        )
        if selection == placeholder:
            selection = default_choice or current_mapping.get(canonical, "")
        selections[canonical] = selection
    return _normalize_column_mapping(selections, defaults)


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
    events_columns: Dict[str, str] = field(default_factory=_default_event_columns)
    stations_columns: Dict[str, str] = field(default_factory=_default_station_columns)
    swd_columns: Dict[str, str] = field(default_factory=_default_swd_columns)

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
                        "contrib_subject_primary",
                        "contrib_subject_secondary",
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

LEGACY_SUBJECT_COLUMN_MAP = {
    "subject4_within_4km": "subject_primary_count",
    "subject10_within_10km": "subject_secondary_count",
    "subject4_within_4km_weighted": "subject_primary_weighted",
    "subject10_within_10km_weighted": "subject_secondary_weighted",
}

LEGACY_COMPOSITE_COLUMN_MAP = {
    "contrib_subject4": "contrib_subject_primary",
    "contrib_subject10": "contrib_subject_secondary",
}

LEGACY_FEATURE_ALIAS = {
    "subject4_within_4km_weighted": "subject_primary_weighted",
    "subject10_within_10km_weighted": "subject_secondary_weighted",
    "delta_gap90_weighted": "delta_gap90_weighted",
    "swd_volume_25km_bbl": "swd_volume_25km_bbl",
    "composite_index": "composite_index",
}

LEGACY_FEATURE_ORDER = [
    ("subject4_within_4km_weighted", "S4 (recency-weighted)"),
    ("subject10_within_10km_weighted", "S10 (recency-weighted)"),
    ("delta_gap90_weighted", "ΔGap90 (recency-weighted)"),
    ("swd_volume_25km_bbl", "SWD volume within 25 km"),
    ("composite_index", "Composite index"),
]


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


def _bna_bytes(session: WorkingSession) -> Optional[bytes]:
    if session.bna_files:
        name = sorted(session.bna_files)[0]
        return session.bna_files[name]
    if session.storage:
        for path in session.storage.list_bna():
            try:
                return path.read_bytes()
            except OSError:
                continue
    return None


def _show_column_warnings(warnings: Dict[str, List[str]]) -> None:
    for dataset, missing in warnings.items():
        if missing:
            st.warning(
                f"{dataset} is missing expected columns: {', '.join(missing)}."
            )


def _load_session_from_disk(name: str) -> WorkingSession:
    state = SessionState.load(name)
    events_cols = _normalize_column_mapping(
        state.column_mapping.get("events", {}),
        _default_event_columns(),
    )
    stations_cols = _normalize_column_mapping(
        state.column_mapping.get("stations", {}),
        _default_station_columns(),
    )
    swd_cols = _normalize_column_mapping(
        state.column_mapping.get("swd", {}),
        _default_swd_columns(),
    )
    ws = WorkingSession(
        name=name,
        parameters=state.parameters,
        storage=state,
        events_columns=events_cols,
        stations_columns=stations_cols,
        swd_columns=swd_cols,
    )
    for path in state.list_bna():
        try:
            ws.bna_files[path.name] = path.read_bytes()
        except OSError:
            continue
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
            frame = state.load_dataframe(key)
            if key == "subject":
                frame = frame.rename(columns=LEGACY_SUBJECT_COLUMN_MAP)
            if key == "composite":
                frame = frame.rename(columns=LEGACY_COMPOSITE_COLUMN_MAP)
            ws.grids[key] = frame
    return ws


def _save_sources(session: WorkingSession, events: pd.DataFrame, stations: pd.DataFrame, swd: Optional[pd.DataFrame]) -> None:
    if session.storage is None:
        session.storage = SessionState(session.name, session.parameters, SessionFiles(), column_mapping={})
    session.storage.column_mapping["events"] = dict(session.events_columns)
    session.storage.column_mapping["stations"] = dict(session.stations_columns)
    session.storage.column_mapping["swd"] = dict(session.swd_columns)
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
    primary_radius = st.number_input(
        "Subject primary radius (km)", value=float(session.parameters.subject_primary_radius_km)
    )
    primary_min = st.number_input(
        "Minimum stations within primary radius",
        value=int(session.parameters.subject_primary_min_stations),
        min_value=0,
    )
    primary_weight = st.number_input(
        "Weight subject primary", value=float(session.parameters.subject_primary_weight)
    )
    secondary_radius = st.number_input(
        "Subject secondary radius (km)", value=float(session.parameters.subject_secondary_radius_km)
    )
    secondary_min = st.number_input(
        "Minimum stations within secondary radius",
        value=int(session.parameters.subject_secondary_min_stations),
        min_value=0,
    )
    secondary_weight = st.number_input(
        "Weight subject secondary", value=float(session.parameters.subject_secondary_weight)
    )
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

    events_bytes = events_file.getvalue() if events_file is not None else None
    stations_bytes = stations_file.getvalue() if stations_file is not None else None
    swd_bytes = swd_file.getvalue() if swd_file is not None else None

    if events_bytes is not None:
        event_columns = _extract_columns(io.BytesIO(events_bytes))
        if not event_columns:
            st.warning("Could not read column names from events file.")
        else:
            st.caption("Events column mapping")
            session.events_columns = _render_column_selectors(
                [
                    ("latitude", "Latitude"),
                    ("longitude", "Longitude"),
                    ("magnitude", "Magnitude"),
                    ("origin_time", "Origin time"),
                ],
                event_columns,
                session.events_columns,
                _default_event_columns(),
                "events_column",
            )

    if stations_bytes is not None:
        station_columns = _extract_columns(io.BytesIO(stations_bytes))
        if not station_columns:
            st.warning("Could not read column names from stations file.")
        else:
            st.caption("Stations column mapping")
            session.stations_columns = _render_column_selectors(
                [
                    ("latitude", "Latitude"),
                    ("longitude", "Longitude"),
                ],
                station_columns,
                session.stations_columns,
                _default_station_columns(),
                "stations_column",
            )

    if swd_bytes is not None:
        swd_columns = _extract_columns(io.BytesIO(swd_bytes))
        if not swd_columns:
            st.warning("Could not read column names from SWD file.")
        else:
            st.caption("SWD column mapping")
            session.swd_columns = _render_column_selectors(
                [
                    ("latitude", "Latitude"),
                    ("longitude", "Longitude"),
                    ("volume", "Volume"),
                ],
                swd_columns,
                session.swd_columns,
                _default_swd_columns(),
                "swd_column",
            )

    inputs = {
        "LONS": lons_input,
        "LATS": lats_input,
        "GRID_STEP": grid_step,
        "SUBJECT_PRIMARY_RADIUS_KM": primary_radius,
        "SUBJECT_PRIMARY_MIN_STATIONS": primary_min,
        "SUBJECT_PRIMARY_WEIGHT": primary_weight,
        "SUBJECT_SECONDARY_RADIUS_KM": secondary_radius,
        "SUBJECT_SECONDARY_MIN_STATIONS": secondary_min,
        "SUBJECT_SECONDARY_WEIGHT": secondary_weight,
        "GAP_SEARCH_KM": gap_search,
        "WEIGHT_GAP": weight_gap,
        "SWD_RADIUS_KM": swd_radius,
        "WEIGHT_SWD": weight_swd,
        "HALF_TIME_YEARS": half_time,
        "OVERWRITE": overwrite,
    }
    parameters = parameter_from_inputs(inputs, logger=st.warning)

    def process_inputs(run_all: bool) -> None:
        if events_bytes is None or stations_bytes is None:
            st.error("Events and stations files are required.")
            return
        events_df, events_missing = load_events(io.BytesIO(events_bytes), column_map=session.events_columns)
        stations_df, stations_missing = load_stations(io.BytesIO(stations_bytes), column_map=session.stations_columns)
        swd_df = None
        swd_missing: List[str] = []
        if swd_bytes is not None:
            swd_df, swd_missing = load_swd_wells(io.BytesIO(swd_bytes), column_map=session.swd_columns)
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
            events_columns=session.events_columns.copy(),
            stations_columns=session.stations_columns.copy(),
            swd_columns=session.swd_columns.copy(),
        )
        new_session.grid = generate_grid(parameters)
        if bna_files:
            for file in bna_files:
                new_session.bna_files[file.name] = file.getvalue()
        set_working_session(new_session)
        _show_column_warnings(warnings)
        st.markdown("**Events preview**")
        st.dataframe(events_df.head(10), width="stretch")
        st.markdown("**Stations preview**")
        st.dataframe(stations_df.head(10), width="stretch")
        if swd_df is not None:
            st.markdown("**SWD wells preview**")
            st.dataframe(swd_df.head(10), width="stretch")
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
        st.dataframe(merged.describe(), width="stretch")



def _render_maps_section(session: WorkingSession) -> None:
    st.header("3. Maps")
    merged = session.merged()
    if merged is None:
        st.info("Compute grids to enable map previews.")
        return

    legacy_df = merged.rename(
        columns={alias: legacy for legacy, alias in LEGACY_FEATURE_ALIAS.items()},
        errors="ignore",
    )

    bna_bytes = _bna_bytes(session)
    config = LegacyMapConfig()

    st.subheader("Legacy contour maps")
    for feature, label in LEGACY_FEATURE_ORDER:
        if feature not in legacy_df.columns:
            continue
        if st.checkbox(f"Show {label}", value=True, key=f"legacy_feature_{feature}"):
            fig = render_legacy_contour(
                legacy_df,
                feature,
                session.parameters,
                config=config,
                bna_bytes=bna_bytes,
                title=label,
            )
            png_bytes = figure_png_bytes(fig)
            st.image(png_bytes, caption=label)
            st.download_button(
                label=f"Download {label} PNG",
                data=png_bytes,
                file_name=f"{feature}.png",
                mime="image/png",
            )

    st.subheader("K-means priority map")
    if "composite_index" not in merged.columns:
        st.info("Compute the composite index to enable the priority map.")
        return

    k_col, init_col = st.columns(2)
    max_clusters = max(2, min(len(PRIORITY_LEVELS), 6))
    default_k = 4 if 4 <= max_clusters else max_clusters
    n_clusters = k_col.slider(
        "Number of clusters (k)",
        min_value=2,
        max_value=max_clusters,
        value=default_k,
        step=1,
        key="priority_kmeans_cluster_count",
    )
    n_init = init_col.slider(
        "K-means initializations",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        key="priority_kmeans_n_init",
    )

    try:
        prioritized = classify_priority_clusters(
            merged,
            n_clusters=n_clusters,
            n_init=int(n_init),
        )
    except Exception as exc:
        st.error(f"Priority clustering failed: {exc}")
        return

    with st.expander("Decoration options", expanded=False):
        scale_col1, scale_col2, scale_col3 = st.columns(3)
        arrow_col1, arrow_col2, arrow_col3 = st.columns(3)

        scale_bar_length = scale_col1.slider(
            "Scale length (km)",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="priority_scale_length",
        )
        scale_bar_x = scale_col2.slider(
            "Scale X",
            min_value=0.0,
            max_value=1.0,
            value=0.22,
            step=0.01,
            key="priority_scale_x",
        )
        scale_bar_y = scale_col3.slider(
            "Scale Y",
            min_value=0.0,
            max_value=1.0,
            value=0.06,
            step=0.01,
            key="priority_scale_y",
        )

        north_arrow_x = arrow_col1.slider(
            "Arrow X",
            min_value=0.0,
            max_value=1.0,
            value=0.92,
            step=0.01,
            key="priority_arrow_x",
        )
        north_arrow_y = arrow_col2.slider(
            "Arrow Y",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.01,
            key="priority_arrow_y",
        )
        north_arrow_length = arrow_col3.slider(
            "Arrow length",
            min_value=0.02,
            max_value=0.15,
            value=0.03,
            step=0.01,
            key="priority_arrow_length",
        )

    fig = render_priority_clusters(
        prioritized,
        session.parameters,
        stations=session.stations,
        bna_bytes=bna_bytes,
        title=f"Priority areas (K-means, k={n_clusters})",
        scale_bar_length_km=float(scale_bar_length),
        scale_bar_location=(float(scale_bar_x), float(scale_bar_y)),
        north_arrow_location=(float(north_arrow_x), float(north_arrow_y)),
        north_arrow_length=float(north_arrow_length),
    )
    st.pyplot(fig)
    png_bytes = figure_png_bytes(fig)
    st.download_button(
        label="Download priority map PNG",
        data=png_bytes,
        file_name=f"priority_map_k{n_clusters}.png",
        mime="image/png",
    )

    summary = (
        prioritized.groupby("priority_label")
        .size()
        .rename("grid_cells")
        .reset_index()
        .sort_values("priority_label")
    )
    st.dataframe(summary, use_container_width=True)

    export_columns = ["latitude", "longitude", "priority_cluster", "priority_label"]
    if "composite_index" in prioritized.columns:
        export_columns.append("composite_index")
    csv_bytes = prioritized[export_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download priority labels CSV",
        data=csv_bytes,
        file_name=f"priority_labels_k{n_clusters}.csv",
        mime="text/csv",
    )


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
