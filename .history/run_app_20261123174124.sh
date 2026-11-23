#!/usr/bin/env bash

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/insar-ps-pycache}"

py streamlit run app.py


$ py -m streamlit run app.py --server.port 8507
