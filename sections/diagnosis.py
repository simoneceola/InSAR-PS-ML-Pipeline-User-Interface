"""Section: automated diagnosis banner, 'why' explanation and ML guide."""

import streamlit as st

from analysis.diagnosis import generate_building_diagnosis
from app_context import AppContext

# One table row per ML module; only rows for enabled modules are shown.
GUIDE_ROWS = {
    "dtw": r"| **DTW Clustering** | Aligns seasonal waves shifted by 1–3 epochs (different thermal inertia across concrete, brick, steel). |",
    "hdbscan": r"| **HDBSCAN** | Finds variable-density clusters across dense roof scatterers & sparse walls without fixed `eps`. |",
    "thermal_inertia": r"| **Lagged Ridge & GBDT** | Separates delayed thermal expansion ($T_t \dots T_{t-3}$) from true permanent settlement rate. |",
    "autoencoder": r"| **1D-CNN Autoencoder** | Learns raw trajectory shapes (coherence-weighted); automatically flags errors $> \text{Med} + 3.5\,\text{MAD}$. |",
    "stgnn": r"| **3D ST-GNN** | Smooths phase noise over a 3D $(E, N, H)$ k-NN graph to isolate contiguous structural zones. |",
}


def render(ctx: AppContext) -> None:
    res = ctx.res
    diagnosis = generate_building_diagnosis(res, temp_is_measured=ctx.temp_is_measured)

    yr = ctx.building_data.get("construction_year")
    yr_tag = f" (Built {yr})" if yr else ""
    st.subheader(f"Automated Diagnosis & Full ML Breakdown: Building #{ctx.selected_id}{yr_tag}")

    if diagnosis["severity"] == "warning":
        st.warning(f"**{diagnosis['headline']}**", icon="⚠️")
    elif diagnosis["severity"] == "info":
        st.info(f"**{diagnosis['headline']}**", icon="ℹ️")
    else:
        st.success(f"**{diagnosis['headline']}**", icon="✅")

    guide_rows = [row for key, row in GUIDE_ROWS.items() if key in res["enabled_ml"]]
    n_models = len(res["enabled_ml"])

    if guide_rows:
        col_diag, col_guide = st.columns([0.58, 0.42], gap="medium")
    else:
        col_diag, col_guide = st.container(), None

    with col_diag:
        with st.expander(f"Why is this happening? (Single-Orbit LOS + {n_models}-Model ML Breakdown)", expanded=True):
            for bullet in diagnosis["bullets"]:
                st.markdown(f"- {bullet}")

    if col_guide is not None:
        with col_guide:
            with st.expander("How to Read the ML Layers (Non-Expert Guide)", expanded=True):
                header = "| ML Algorithm | What It Reveals About the Building |\n| :--- | :--- |\n"
                st.markdown(header + "\n".join(guide_rows))
