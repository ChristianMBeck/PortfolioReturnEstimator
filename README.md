# Quantitative Portfolio Optimization Dashboard

A Python dashboard for historical financial data pipelines, modern portfolio optimization (MPT), and predictive returns estimation.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 app.py
```

## Project Structure

- `app/` — Streamlit entry point
- `src/data/` — Data fetching, pipeline, and SQLite storage
- `src/optimization/` — MPT engine (SciPy)
- `src/prediction/` — Returns estimation (Scikit-learn)
- `src/visualization/` — Plotly chart builders
- `tests/` — Unit tests for computational engines

See [PRD.md](PRD.md) for full scope and requirements.
