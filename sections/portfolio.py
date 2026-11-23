"""Top row: 2D portfolio map + clickable buildings table, and the building selector.

This runs BEFORE the per-building analysis because it decides which building is analysed.
"""

import pandas as pd
import streamlit as st

from plots.portfolio_map import build_portfolio_map


def render_portfolio(
    filtered_df: pd.DataFrame,
    intra_df: pd.DataFrame,
    building_datasets: dict[int, dict],
    show_map: bool = True,
    show_table: bool = True,
) -> int:
    """Draw the enabled portfolio widgets and return the selected building id."""
    available_ids = filtered_df["building"].tolist()
    if "selected_bld" not in st.session_state or st.session_state["selected_bld"] not in available_ids:
        st.session_state["selected_bld"] = available_ids[0]

    if show_map and show_table:
        map_col, table_col = st.columns([0.52, 0.48], gap="medium")
    else:
        map_col = st.container() if show_map else None
        table_col = st.container() if show_table else None

    # ---- table (can set the selection by row click)
    if table_col is not None:
        with table_col:
            st.subheader(f"Buildings ({len(filtered_df)} / {len(intra_df)})")
            display_cols = [
                c for c in [
                    "building", "construction_year", "n_pts", "h_min_m", "h_max_m",
                    "h_range_m",
                ]
                if c in filtered_df.columns
            ]
            table_event = st.dataframe(
                filtered_df[display_cols].round(3),
                use_container_width=True,
                height=440,
                hide_index=True,
                column_config={
                    "building": st.column_config.NumberColumn("Building #", format="%d"),
                    "construction_year": st.column_config.NumberColumn("Built", format="%d"),
                    "n_pts": st.column_config.NumberColumn("PS Pts", format="%d"),
                    "h_min_m": st.column_config.NumberColumn("Min H Ground (m)", format="%.1f"),
                    "h_max_m": st.column_config.NumberColumn("Max H Ground (m)", format="%.1f"),
                    "h_range_m": st.column_config.NumberColumn("H Span (m)", format="%.1f"),
                },
                on_select="rerun",
                selection_mode="single-row",
            )
            if table_event.selection.rows:
                clicked_row = table_event.selection.rows[0]
                clicked_id = int(filtered_df.iloc[clicked_row]["building"])
                if st.session_state.get("last_clicked_row") != clicked_row:
                    st.session_state["last_clicked_row"] = clicked_row
                    st.session_state["selected_bld"] = clicked_id

    # ---- selector (always shown: it is how a building gets chosen)
    # `key="selected_bld"` makes session_state the single source of truth, so a table click
    # (which sets it above) and a dropdown pick never fight each other.
    selected_id = int(
        st.selectbox(
            "Selected Building for Detailed Inspection:",
            options=available_ids,
            key="selected_bld",
            format_func=lambda b: f"Building #{b}",
        )
    )

    # ---- map
    if map_col is not None:
        with map_col:
            st.subheader("Map of Buildings with PS Scatterers")
            mc1, mc2, mc3 = st.columns([0.28, 0.28, 0.28])
            with mc1:
                basemap_choice = st.selectbox(
                    "Basemap",
                    options=[
                        "Light Theme",
                        "OpenStreetMap",
                    ],
                    index=0,
                    key="map_basemap_sel",
                )
            with mc2:
                map_color_by = st.selectbox(
                    "Marker Color",
                    options=[
                        "Median LOS Velocity (mm/yr)",
                        "PS Point Count",
                        "Max Height wrt Ground (m)",
                    ],
                    index=0,
                    key="map_color_sel",
                )
            with mc3:
                map_focus = st.selectbox(
                    "Map View",
                    options=["Fit All Filtered Buildings", "Zoom to Selected Building"],
                    index=0,
                    key="map_focus_sel",
                )
            show_footprints_and_ps = st.checkbox(
                "Show building footprint polygons & individual PS points on map",
                value=True,
                key="map_footprints_chk",
            )
            st.plotly_chart(
                build_portfolio_map(
                    df=filtered_df,
                    selected_id=selected_id,
                    building_datasets=building_datasets,
                    basemap_choice=basemap_choice,
                    color_by=map_color_by,
                    camera_focus=map_focus,
                    show_footprints_and_ps=show_footprints_and_ps,
                ),
                use_container_width=True,
            )

    return selected_id
