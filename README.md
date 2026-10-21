# Bank Churn Dashboard

Files
- `pipeline.py` - data prep, feature engineering, 6 models, statistical tests
- `app.py` - Streamlit dashboard (6 pages)
- `data/Churn_Modelling.csv` - your dataset
- `data/model_results.csv` - model metrics copied from your notebook
- `data/stat_tests.json` - test results (created by `python pipeline.py`)

Run locally (PowerShell, inside this folder)
    pip install -r requirements.txt
    python pipeline.py          # optional: retrains all models, refreshes metrics
    streamlit run app.py

Host free (Streamlit Community Cloud)
1. Create a GitHub repo and upload this whole folder (keep `data/`).
2. Go to share.streamlit.io, sign in with GitHub, click "Create app".
3. Pick the repo, branch `main`, main file `app.py`, then Deploy.
4. You get a public link to share.
