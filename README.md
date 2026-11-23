# Single-Orbit InSAR ML Inspector
To run: (from Git Bash)

    bash run_app.sh

The launch script stores Python bytecode in `/tmp/insar-ps-pycache`, outside the
repository, so new runs no longer recreate `.pyc`/`__pycache__` data in the repo.

**To choose what appears on the page, edit `page_layout.py`** (3 switches: top row, page sections, ML models).

| Folder / file | Purpose |
|---|---|
| `page_layout.py` | ON/OFF switches + section order |
| `app_config.py` | paths, colours, defaults, `AnalysisConfig` |
| `app.py` | thin entry point that wires everything together |
| `data_io/` | CSV + GeoPackage loader, synthetic demo data, temperature series |
| `analysis/` | rate fitting, strata, vertical gradient, pipeline, diagnosis text |
| `ml/` | one file per model (RF, DTW, HDBSCAN, thermal, autoencoder, IF, ST-GNN) |
| `plots/` | pure Plotly figure builders |
| `sections/` | one file per page section; `__init__.py` is the registry |
| `ui/` | sidebar controls, scatterer selector |

Add a section: create `sections/my_section.py` with `render(ctx)`, register it in `sections/__init__.py`, add its key to `page_layout.SECTIONS`.
