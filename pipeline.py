"""
pipeline.py  -  Churn prediction pipeline (file 1 of 2)

Steps (same as your notebook):
  1. Load data  ->  2. Clean  ->  3. Feature engineering  ->  4. Encode
  5. Train/test split  ->  6. Scale + train 6 models  ->  7. Save metrics

Run once to (re)create results:   python pipeline.py
app.py imports the helper functions from this file.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "Churn_Modelling.csv"
RESULTS_PATH = BASE / "data" / "model_results.csv"
STATS_PATH = BASE / "data" / "stat_tests.json"

FEATURES = ["CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary", "balance_to_salary",
            "tenure_by_age", "Geography_Germany", "Geography_Spain"]


# ---------- 1-2. load & clean ----------
def load_raw(path=DATA_PATH):
    return pd.read_csv(path)


def clean(df):
    return df.drop(["CustomerId", "Surname", "RowNumber"], axis=1, errors="ignore")


# ---------- 3-4. features & encoding ----------
def make_features(df):
    """Raw columns -> the 13 model features (works for 1 customer or many)."""
    d = df.copy()
    d["balance_to_salary"] = d["Balance"] / d["EstimatedSalary"]
    d["tenure_by_age"] = d["Tenure"] / d["Age"]
    d["Gender"] = d["Gender"].map({"Female": 0, "Male": 1})          # = LabelEncoder
    d["Geography_Germany"] = (d["Geography"] == "Germany").astype(int)  # = get_dummies(drop_first)
    d["Geography_Spain"] = (d["Geography"] == "Spain").astype(int)
    return d[FEATURES]


def get_split(df):
    d = clean(df)
    X, y = make_features(d), d["Exited"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------- 5-6. models ----------
def get_models():
    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN Classifier": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        # ANN: same 12-6-1 layout as your Keras model, using scikit-learn (no TensorFlow needed)
        "ANN Model": MLPClassifier(hidden_layer_sizes=(12, 6), max_iter=500, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["XG Boost Classifier"] = XGBClassifier(random_state=42)
    except ImportError:
        print("xgboost not installed - skipping XGBoost (pip install xgboost)")
    return models


def make_model_pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def train_all(df):
    X_train, X_test, y_train, y_test = get_split(df)
    rows, fitted = [], {}
    for name, model in get_models().items():
        pipe = make_model_pipeline(model).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred),
                     "Recall": recall_score(y_test, pred),
                     "F1 Score": f1_score(y_test, pred)})
        fitted[name] = pipe
    return pd.DataFrame(rows), fitted


def train_rf_for_app(df):
    """Random Forest pipeline + feature importances, used by the Predict page."""
    X_train, _, y_train, _ = get_split(df)
    pipe = make_model_pipeline(RandomForestClassifier(random_state=42)).fit(X_train, y_train)
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=FEATURES)
    return pipe, imp.sort_values(ascending=False)


def predict_customer(pipe, **c):
    row = make_features(pd.DataFrame([c]))
    return float(pipe.predict_proba(row)[0, 1])


# ---------- statistical tests (same tests as statistical_testing.ipynb) ----------
def run_stat_tests(df):
    out = {}
    ch, rt = df[df.Exited == 1]["Balance"], df[df.Exited == 0]["Balance"]

    # Test 1: Welch t-test (Balance vs churn)
    lev = stats.levene(ch, rt)
    t, p = stats.ttest_ind(ch, rt, equal_var=False)
    pooled = np.sqrt(((len(ch) - 1) * ch.std() ** 2 + (len(rt) - 1) * rt.std() ** 2) / (len(ch) + len(rt) - 2))
    out["ttest"] = dict(churn_mean=ch.mean(), retain_mean=rt.mean(), levene_p=lev.pvalue,
                        t=t, p=p, cohens_d=(ch.mean() - rt.mean()) / pooled)

    # Test 2: Chi-square (IsActiveMember vs churn)
    ct = pd.crosstab(df["Exited"], df["IsActiveMember"])
    chi2, p2, dof, exp = stats.chi2_contingency(ct)
    out["chi2"] = dict(table=ct.values.tolist(), chi2=chi2, p=p2, dof=int(dof),
                       min_expected=float(exp.min()),
                       cramers_v=float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))))

    # Test 3: ANOVA (Tenure across Geography)
    groups = [g["Tenure"].values for _, g in df.groupby("Geography")]
    lev3 = stats.levene(*groups)
    f, p3 = stats.f_oneway(*groups)
    allv = df["Tenure"].values
    ssb = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    sst = ((allv - allv.mean()) ** 2).sum()
    out["anova"] = dict(means=df.groupby("Geography")["Tenure"].mean().to_dict(),
                        levene_p=lev3.pvalue, f=f, p=p3, eta_sq=ssb / sst)
    return json.loads(json.dumps(out, default=float))


if __name__ == "__main__":
    data = load_raw()
    results, _ = train_all(data)
    results.to_csv(RESULTS_PATH, index=False)
    STATS_PATH.write_text(json.dumps(run_stat_tests(data), indent=2))
    print(results.round(4).to_string(index=False))
    print(f"\nSaved: {RESULTS_PATH.name}, {STATS_PATH.name}")
