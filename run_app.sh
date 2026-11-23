#!/usr/bin/env bash

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/insar-ps-pycache}"

py -m streamlit run app.py --server.port 8507