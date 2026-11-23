#!/usr/bin/env bash

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/insar-ps-pycache}"

py streamlit run app.py


