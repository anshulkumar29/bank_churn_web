"""
app.py  -  Streamlit dashboard (file 2 of 2)
Run:  streamlit run app.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import pipeline as pl

st.set_page_config(page_title="Bank Churn Dashboard", page_icon="🏦", layout="wide")
sns.set_theme(style="whitegrid")
COLORS = {0: "#2a9d8f", 1: "#e76f51"}  # retained / churned


# ---------- cached loaders ----------
@st.cache_data
def load_data():
    return pl.load_raw()


@st.cache_data
def load_results():
    return pd.read_csv(pl.RESULTS_PATH)


@st.cache_data
def load_stats(df):
    if pl.STATS_PATH.exists():
        return json.loads(pl.STATS_PATH.read_text())
    return pl.run_stat_tests(df)


@st.cache_resource
def load_predictor(df):
    return pl.train_rf_for_app(df)


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


df = load_data()
results = load_results()
stats_ = load_stats(df)

st.sidebar.title("🏦 Churn Dashboard")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Statistical Tests",
                                  "Model Comparison", "Predict a Customer", "Recommendations"])
st.sidebar.caption("Data: Churn_Modelling.csv (10,000 bank customers)")


# =====================================================================
if page == "Overview":
    st.title("Bank Customer Churn Prediction")
    st.write("**Goal:** predict whether a customer will **leave (churn)** or **stay (retain)** "
             "using features such as geography, age, balance, credit score and activity.")
    churn_rate = df["Exited"].mean()
    c = st.columns(4)
    c[0].metric("Customers", f"{len(df):,}")
    c[1].metric("Churned", f"{int(df['Exited'].sum()):,}")
    c[2].metric("Churn rate", f"{churn_rate:.1%}")
    c[3].metric("Features used", len(pl.FEATURES))

    st.subheader("How this project works")
    st.markdown("""
1. **Data** – 10,000 customers, 14 columns (ID columns dropped).
2. **Feature engineering** – `balance_to_salary` and `tenure_by_age` added (13 features in total after encoding).
3. **Encoding** – Gender label-encoded, Geography one-hot encoded.
4. **Models** – Logistic Regression, KNN, Naive Bayes, Random Forest, XGBoost, ANN (80/20 split, scaled).
5. **Statistical tests** – Welch t-test, Chi-square and ANOVA with effect sizes.
6. **Dashboard** – use the sidebar to explore each part.
""")
    st.subheader("Data sample")
    st.dataframe(df.drop(columns=["RowNumber", "CustomerId", "Surname"]).head(10), use_container_width=True)
    st.info("Class imbalance: only ~20% of customers churn, so **accuracy alone is misleading** — "
            "a model that says 'everyone stays' already scores ~80%. Recall and F1 matter more.")


# =====================================================================
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    d = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Churn by category", "Churn vs numbers", "Correlation"])

    with tab1:
        col = st.selectbox("Pick a numeric feature", ["Age", "Balance", "CreditScore", "EstimatedSalary", "Tenure"])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.histplot(d[col], bins=30, kde=True, ax=ax, color="#264653")
        ax.set_title(f"Distribution of {col}")
        show(fig)
        st.caption("Balance has a big spike at 0 (about 36% of customers hold no balance).")

    with tab2:
        col = st.selectbox("Pick a category", ["Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember"])
        rate = d.groupby(col)["Exited"].mean().mul(100).reset_index()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.barplot(data=rate, x=col, y="Exited", ax=ax, color="#e76f51")
        ax.set_ylabel("Churn rate (%)")
        ax.set_title(f"Churn rate by {col}")
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)
        show(fig)
        st.caption("Germany, female customers, inactive members and customers with 3-4 products churn most.")

    with tab3:
        col = st.selectbox("Pick a numeric feature ", ["Age", "Balance", "CreditScore", "EstimatedSalary", "Tenure"])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.boxplot(data=d, x="Exited", y=col, ax=ax, palette=[COLORS[0], COLORS[1]], hue="Exited", legend=False)
        ax.set_xticks([0, 1], ["Retained", "Churned"])
        ax.set_xlabel("")
        show(fig)
        st.caption("Age shows the clearest gap: churners are noticeably older.")

    with tab4:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(d.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"size": 8})
        show(fig)
        corr = d.corr(numeric_only=True)["Exited"].drop("Exited").sort_values(key=abs, ascending=False)
        st.write("Strongest correlations with churn:", ", ".join(f"**{k}** ({v:+.2f})" for k, v in corr.head(3).items()))


# =====================================================================
elif page == "Statistical Tests":
    st.title("Statistical Tests")
    st.write("Each test asks a business question. **p-value < 0.05** → the difference is statistically real. "
             "**Effect size** tells us whether it is big enough to matter.")
    t, c, a = stats_["ttest"], stats_["chi2"], stats_["anova"]

    tab1, tab2, tab3, tab4 = st.tabs(["T-test", "Chi-square", "ANOVA", "Summary"])

    with tab1:
        st.subheader("Welch t-test: do churners have a different balance?")
        st.caption("H0: mean balance is equal for churned and retained customers.")
        k = st.columns(4)
        k[0].metric("Churned mean", f"{t['churn_mean']:,.0f}")
        k[1].metric("Retained mean", f"{t['retain_mean']:,.0f}")
        k[2].metric("p-value", f"{t['p']:.1e}")
        k[3].metric("Cohen's d", f"{t['cohens_d']:.2f}")
        st.caption(f"Levene p = {t['levene_p']:.1e} (< 0.05 → unequal variances, so Welch's version is the right one).")
        st.success("**Result: significant (p < 0.05).** Churned customers hold higher balances on average. "
                   "Cohen's d ≈ 0.30 is a **small** effect, so balance helps but is not a strong driver on its own.")
        st.warning("Note: the interpretation text in your statistical_testing notebook says 'p-value is more than 0.05 / "
                   "no difference' and 'd = 0.03'. Your own output shows p = 6e-35 and d = 0.30, so the correct "
                   "reading is the one above. You may want to fix that markdown cell.")

    with tab2:
        st.subheader("Chi-square: is activity linked to churn?")
        st.caption("H0: churn and IsActiveMember are independent.")
        ct = pd.DataFrame(c["table"], index=["Retained", "Churned"], columns=["Inactive", "Active"])
        st.dataframe(ct)
        k = st.columns(3)
        k[0].metric("Chi²", f"{c['chi2']:.1f}")
        k[1].metric("p-value", f"{c['p']:.1e}")
        k[2].metric("Cramér's V", f"{c['cramers_v']:.2f}")
        st.caption(f"All expected counts ≥ 5 (smallest = {c['min_expected']:.0f}), so the assumption holds.")
        st.success("**Result: significant.** Inactive members churn far more (about 27% vs 14%). "
                   "Cramér's V ≈ 0.16 is a **weak-to-moderate** association, so activity matters but is not the only driver.")

    with tab3:
        st.subheader("ANOVA: does tenure differ by country?")
        st.caption("H0: average tenure is the same in France, Germany and Spain.")
        st.dataframe(pd.DataFrame(a["means"], index=["Mean tenure (years)"]).round(2))
        k = st.columns(3)
        k[0].metric("F", f"{a['f']:.3f}")
        k[1].metric("p-value", f"{a['p']:.2f}")
        k[2].metric("Eta²", f"{a['eta_sq']:.5f}")
        st.caption(f"Levene p = {a['levene_p']:.2f} (> 0.05 → variances are equal, so standard ANOVA is valid).")
        st.success("**Result: not significant.** Tenure is the same across countries and the effect is essentially zero. "
                   "Country differences in churn therefore do not come from tenure.")
        st.warning("Note: your notebook says 'p-value is more than 0.05, H0 is rejected, variances are not equal' for "
                   "Levene. With p = 0.12 you actually **fail to reject** equal variances.")

    with tab4:
        summary = pd.DataFrame([
            ["Welch t-test", "Balance vs churn", f"{t['p']:.1e}", "Significant", f"d = {t['cohens_d']:.2f} (small)"],
            ["Chi-square", "IsActiveMember vs churn", f"{c['p']:.1e}", "Significant", f"V = {c['cramers_v']:.2f} (weak-moderate)"],
            ["ANOVA", "Tenure across Geography", f"{a['p']:.2f}", "Not significant", f"η² = {a['eta_sq']:.5f} (negligible)"],
        ], columns=["Test", "Question", "p-value", "Verdict", "Effect size"])
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.info("**What this means for modelling:** activity status and balance carry churn signal, so the models should "
                "use them. Tenure does not vary by country, so it adds little for separating geographies. "
                "No single feature is strong alone, which is why combining many features in a model helps.")
        st.caption("Your notebooks contain no A/B test, so none is shown here.")


# =====================================================================
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.caption("Metrics on the 20% hold-out test set (2,000 customers; 393 churners).")
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    res = results.sort_values("F1 Score", ascending=False).reset_index(drop=True)

    st.dataframe(res.style.format({m: "{:.3f}" for m in metrics})
                 .highlight_max(subset=metrics, color="#b7e4c7"), use_container_width=True, hide_index=True)

    sel = st.multiselect("Metrics to plot", metrics, default=metrics)
    long = res.melt(id_vars="Model", value_vars=sel, var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=long, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    show(fig)

    st.subheader("Look at one model")
    name = st.selectbox("Model", res["Model"])
    r = res[res["Model"] == name].iloc[0]
    # rebuild confusion matrix from precision / recall (393 churners in the test set of 2000)
    pos, n = 393, 2000
    tp = round(r["Recall"] * pos)
    fp = round(tp / r["Precision"] - tp) if r["Precision"] > 0 else 0
    fn, tn = pos - tp, n - pos - fp
    cm = np.array([[tn, fp], [fn, tp]])
    k = st.columns([1, 1])
    with k[0]:
        fig, ax = plt.subplots(figsize=(3.6, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Retain", "Churn"], yticklabels=["Retain", "Churn"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        show(fig)
    with k[1]:
        st.markdown(f"""
- Caught **{tp}** of {pos} real churners (recall {r['Recall']:.0%}).
- Missed **{fn}** churners (they leave unnoticed).
- Raised **{fp}** false alarms (precision {r['Precision']:.0%}).
""")
    st.caption("Metric meanings — **Precision**: of customers flagged as churn, how many really churn. "
               "**Recall**: of customers who really churn, how many we catch. **F1**: balance of the two.")


# =====================================================================
elif page == "Predict a Customer":
    st.title("Predict: Churn or Retain?")
    st.caption("Uses a Random Forest trained on the full pipeline.")
    pipe, importance = load_predictor(df)

    with st.form("f"):
        a, b, c3 = st.columns(3)
        geo = a.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = b.selectbox("Gender", ["Female", "Male"])
        age = c3.slider("Age", 18, 92, 40)
        score = a.slider("Credit score", 350, 850, 650)
        balance = b.number_input("Balance", 0.0, 260000.0, 76000.0, step=1000.0)
        salary = c3.number_input("Estimated salary", 10.0, 200000.0, 100000.0, step=1000.0)
        tenure = a.slider("Tenure (years)", 0, 10, 5)
        prods = b.selectbox("Number of products", [1, 2, 3, 4], index=0)
        card = c3.selectbox("Has credit card?", ["Yes", "No"])
        active = a.selectbox("Active member?", ["Yes", "No"])
        go = st.form_submit_button("Predict")

    if go:
        prob = pl.predict_customer(pipe, CreditScore=score, Geography=geo, Gender=gender, Age=age,
                                   Tenure=tenure, Balance=balance, NumOfProducts=prods,
                                   HasCrCard=int(card == "Yes"), IsActiveMember=int(active == "Yes"),
                                   EstimatedSalary=salary)
        left, right = st.columns([1, 2])
        left.metric("Churn probability", f"{prob:.0%}")
        if prob >= 0.5:
            right.error("🔴 **Likely to CHURN** – consider a retention offer or a call.")
        elif prob >= 0.3:
            right.warning("🟠 **At risk** – not predicted to churn, but worth monitoring.")
        else:
            right.success("🟢 **Likely to RETAIN.**")
        st.progress(min(prob, 1.0))

    st.subheader("What the model relies on most")
    top = importance.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(top.index, top.values, color="#264653")
    ax.set_xlabel("Importance")
    show(fig)


# =====================================================================
else:
    st.title("Recommendations")
    best_f1 = results.loc[results["F1 Score"].idxmax()]
    best_rec = results.loc[results["Recall"].idxmax()]
    best_acc = results.loc[results["Accuracy"].idxmax()]
    best_pre = results.loc[results["Precision"].idxmax()]
    worst = results.loc[results["F1 Score"].idxmin()]

    st.subheader("Best model")
    st.success(f"**{best_f1['Model']}** has the best balance (F1 = {best_f1['F1 Score']:.3f}, "
               f"recall = {best_f1['Recall']:.0%}, precision = {best_f1['Precision']:.0%}). "
               "Recommended as the main churn model.")
    c = st.columns(4)
    c[0].metric("Best F1", best_f1["Model"], f"{best_f1['F1 Score']:.3f}", delta_color="off")
    c[1].metric("Best recall", best_rec["Model"], f"{best_rec['Recall']:.3f}", delta_color="off")
    c[2].metric("Best precision", best_pre["Model"], f"{best_pre['Precision']:.3f}", delta_color="off")
    c[3].metric("Best accuracy", best_acc["Model"], f"{best_acc['Accuracy']:.3f}", delta_color="off")

    st.subheader("Decisions")
    st.markdown(f"""
- **Use {best_f1['Model']} for churn scoring.** If the aim is to catch as many leavers as possible, pick
  **{best_rec['Model']}** (highest recall). If retention offers are expensive and you want few false alarms,
  pick **{best_pre['Model']}** (highest precision).
- **Do not judge models by accuracy alone.** About 80% of customers stay, so even weak models look "good" on accuracy.
  {worst['Model']} has {worst['Accuracy']:.0%} accuracy but only {worst['Recall']:.0%} recall — it misses most churners.
- **Tree models and the neural network beat the simple models** because churn depends on combinations of features
  (for example age together with product count), which Logistic Regression and Naive Bayes cannot capture.
- **Tests back this up.** Activity (chi-square, significant) and balance (t-test, significant but small effect) carry real signal,
  but no single feature is strong, so a multi-feature model is justified.
""")
    st.subheader("Business actions")
    st.markdown("""
- Target **inactive members** with re-engagement campaigns (churn about 27% vs 14% for active).
- Review **German customers** and customers with **3-4 products**; both churn far above average.
- Watch **older customers (45+)**, who leave more often.
- Use the model's churn probability to rank customers and contact the top group first.
""")
    st.subheader("Next steps to improve")
    st.markdown("""
- Handle class imbalance (class weights or SMOTE) to raise recall.
- Lower the decision threshold from 0.5 (say 0.35) to catch more churners.
- Tune Random Forest / XGBoost with cross-validation instead of default settings.
""")
