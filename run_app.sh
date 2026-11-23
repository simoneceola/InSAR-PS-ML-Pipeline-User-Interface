#!/usr/bin/env bash

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/insar-ps-pycache}"

streamlit run app.py
