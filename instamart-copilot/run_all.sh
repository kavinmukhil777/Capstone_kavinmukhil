#!/usr/bin/env bash
# Runs the whole Copilot end to end (same order as the README Reproduce section, after the venv is active).
set -euo pipefail
python generate_data.py
python src/part_a_clean.py
python src/part_a_validate.py
python src/part_a_sql_pandas.py
python src/part_b_segments.py
python src/part_b_lapse.py
python src/part_c_triage.py
python src/part_d_rag.py
python src/part_d_agent.py
