"""
app.py — Vi Telecom Customer Churn Dashboard
Streamlit dashboard with 4 tabs, AI-powered insights, interactive filters.
"""

import os, warnings, json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vi Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Background */
.main { background: #F8FAFC; }
.stApp { background: #F8FAFC; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1A1F3C 0%, #0F3460 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #E8EAF0 !important; }
[data-testid="stSidebar"] .stMultiSelect > div { background: rgba(255,255,255,0.08) !important; border-color: rgba(255,255,255,0.15) !important; }
[data-testid="stSidebar"] .stSlider > div { color: #E8EAF0 !important; }

/* KPI Cards */
.kpi-card {
    background: white;
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-left: 4px solid #4361EE;
    margin-bottom: 8px;
}
.kpi-label { font-size: 12px; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 0.8px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #111827; font-family: 'Space Grotesk', sans-serif; }
.kpi-delta { font-size: 13px; margin-top: 2px; }
.kpi-delta.up { color: #EF4444; }
.kpi-delta.down { color: #10B981; }

/* Section header */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px; font-weight: 700;
    color: #1A1F3C;
    margin: 24px 0 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid #4361EE22;
}

/* AI insight box */
.ai-insight {
    background: linear-gradient(135deg, #EFF6FF 0%, #F0FDF4 100%);
    border-left: 4px solid #4361EE;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0 20px;
    font-size: 14px;
    color: #374151;
    line-height: 1.6;
}
.ai-badge {
    display: inline-block;
    background: #4361EE;
    color: white;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 6px;
    letter-spacing: 0.8px;
}

/* Metric table */
.metric-good { color: #10B981; font-weight: 600; }
.metric-bad { color: #EF4444; font-weight: 600; }

/* Tab styling */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 14px;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #4361EE !important;
    border-bottom-color: #4361EE !important;
}

/* Logo / title */
.app-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px; font-weight: 700;
    background: linear-gradient(135deg, #4361EE, #F77F00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-subtitle { font-size: 13px; color: #6B7280; margin-top: -4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────────
C = {
    "churned": "#EF4444",
    "retained": "#10B981",
    "prepaid": "#F77F00",
    "postpaid": "#4361EE",
    "blue": "#4361EE",
    "orange": "#F77F00",
    "green": "#10B981",
    "red": "#EF4444",
    "purple": "#7C3AED",
    "gray": "#6B7280",
}
MODEL_COLORS = {"Decision Tree": "#F77F00", "Random Forest": "#4361EE", "XGBoost": "#10B981"}


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    from utils import load_and_clean, handle_missing
    path = "data/VI_Customer_Churn.csv"
    df = load_and_clean(path)
    df = handle_missing(df)
    return df


@st.cache_data(show_spinner=False)
def load_encoded() -> pd.DataFrame:
    from utils import load_and_clean, handle_missing, encode_features, engineer_features
    path = "data/VI_Customer_Churn.csv"
    df = load_and_clean(path)
    df = handle_missing(df)
    df_enc = encode_features(df, save_path="data/encoding_mappings.csv")
    df_enc = engineer_features(df_enc)
    return df_enc


@st.cache_resource(show_spinner=False)
def load_models_and_data():
    from utils import load_models, split_data, evaluate_models, get_roc_data, get_pr_data
    from utils import get_confusion_matrices, get_feature_importance, get_permutation_importance
    import joblib

    df_enc = load_encoded()

    # Check if models pre-trained
    if not (os.path.exists("models/decision_tree.joblib") and
            os.path.exists("models/random_forest.joblib") and
            os.path.exists("models/xgboost.joblib")):
        # Train on the fly
        from utils import train_all_models, save_models
        X_train, X_test, y_train, y_test = split_data(df_enc)
        trained = train_all_models(X_train, y_train)
        save_models(trained, "models")
    else:
        trained = load_models("models")
        X_train, X_test, y_train, y_test = split_data(df_enc)

    feature_names = list(X_train.columns)
    metrics_df = evaluate_models(trained, X_train, X_test, y_train, y_test)
    roc_data = get_roc_data(trained, X_test, y_test)
    pr_data = get_pr_data(trained, X_test, y_test)
    cms = get_confusion_matrices(trained, X_test, y_test)
    importances = get_feature_importance(trained, feature_names)
    perm_imp = get_permutation_importance(trained, X_test, y_test, feature_names)

    return {
        "trained": trained,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "metrics": metrics_df,
        "roc": roc_data,
        "pr": pr_data,
        "cms": cms,
        "importances": importances,
        "perm_imp": perm_imp,
        "feature_names": feature_names,
    }


def ai_insight(prompt: str, key: str) -> str:
    """Call Anthropic API for a 2-line chart insight."""
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"⚠️ AI insight unavailable ({e})"


def insight_box(text: str):
    st.markdown(f"""
    <div class="ai-insight">
      <span class="ai-badge">✦ AI INSIGHT</span><br>
      {text}
    </div>""", unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

def build_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.markdown('<div class="app-title">📡 Vi Churn AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Customer Intelligence Platform</div>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### 🔍 Filters")

        # Gender
        genders = sorted(df["gender"].dropna().unique().tolist())
        sel_gender = st.multiselect("Gender", genders, default=genders)

        # Payment method
        payments = sorted(df["payment_method"].dropna().unique().tolist())
        sel_payment = st.multiselect("Payment Method", payments, default=payments)

        # Plan type
        plans = sorted(df["plan_type"].dropna().unique().tolist())
        sel_plan = st.multiselect("Plan Type", plans, default=plans)

        # Top 5 states
        top5_states = df["state"].value_counts().head(5).index.tolist()
        all_states = ["All"] + top5_states
        sel_states = st.multiselect("State (Top 5)", top5_states, default=top5_states)

        st.markdown("---")

        # Sliders
        min_charge, max_charge = int(df["monthly_charges_inr"].min()), int(df["monthly_charges_inr"].max())
        charge_range = st.slider("Monthly Charges (₹)", min_charge, max_charge, (min_charge, max_charge))

        min_ten, max_ten = int(df["contract_tenure_months"].min()), int(df["contract_tenure_months"].max())
        tenure_range = st.slider("Tenure (months)", min_ten, max_ten, (min_ten, max_ten))

        st.markdown("---")
        st.markdown("**Vi Telecom Churn Intelligence**")
        st.markdown("*Powered by XGBoost + Claude AI*")

    # Apply filters
    mask = (
        df["gender"].isin(sel_gender) &
        df["payment_method"].isin(sel_payment) &
        df["plan_type"].isin(sel_plan) &
        df["state"].isin(sel_states) &
        df["monthly_charges_inr"].between(*charge_range) &
        df["contract_tenure_months"].between(*tenure_range)
    )
    return df[mask].copy()


# ─────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────

def tab_overview(df: pd.DataFrame):
    section("📊 Key Performance Indicators")

    total = len(df)
    churned = df["churned"].sum()
    churn_rate = churned / total * 100 if total > 0 else 0

    retained_df = df[df["churned"] == 0]
    avg_charges = retained_df["monthly_charges_inr"].mean() if len(retained_df) else 0
    avg_tenure = retained_df["contract_tenure_months"].mean() if len(retained_df) else 0

    # High-risk segment
    high_risk = df[
        (df["complaints_last_6months"] >= 3) &
        (df["late_payment_count"] >= 2) &
        (df["customer_satisfaction_score"] <= 2)
    ]
    risk_count = len(high_risk)
    risk_churn = high_risk["churned"].mean() * 100 if risk_count > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_txt = "High churn risk" if churn_rate > 30 else "Within target"
        delta_cls = "up" if churn_rate > 30 else "down"
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #EF4444;">
          <div class="kpi-label">📉 Churn Rate</div>
          <div class="kpi-value">{churn_rate:.1f}%</div>
          <div class="kpi-delta {delta_cls}">{churned} of {total} customers</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #10B981;">
          <div class="kpi-label">💰 Avg Monthly Charges (Retained)</div>
          <div class="kpi-value">₹{avg_charges:.0f}</div>
          <div class="kpi-delta down">Retained customers</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #4361EE;">
          <div class="kpi-label">📅 Avg Tenure (Retained)</div>
          <div class="kpi-value">{avg_tenure:.0f} mo</div>
          <div class="kpi-delta down">Retained customers</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #F77F00;">
          <div class="kpi-label">⚠️ High-Risk Segment</div>
          <div class="kpi-value">{risk_count} cust.</div>
          <div class="kpi-delta up">{risk_churn:.1f}% churn in segment</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top 5 states by churn %
    section("🗺️ Top 5 States by Churn Rate")
    state_churn = (
        df.groupby("state")["churned"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "customers"})
        .reset_index()
    )
    state_churn["churn_pct"] = (state_churn["churn_rate"] * 100).round(1)
    top5 = state_churn.nlargest(5, "churn_pct")

    fig_state = px.bar(
        top5, x="churn_pct", y="state", orientation="h",
        text="churn_pct",
        color="churn_pct",
        color_continuous_scale=["#10B981", "#F77F00", "#EF4444"],
        labels={"churn_pct": "Churn Rate (%)", "state": "State"},
    )
    fig_state.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
    fig_state.update_layout(
        height=320, margin=dict(l=10, r=30, t=10, b=10),
        coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(categoryorder="total ascending"),
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_state, use_container_width=True)

    # AI insight
    insight_prompt = f"""
    You are a telecom analyst. Based on the following data, write exactly 2 concise insight sentences (no bullet points):
    - Total customers: {total}, Churn rate: {churn_rate:.1f}%
    - Top churning state: {top5.iloc[0]['state'] if len(top5)>0 else 'N/A'} at {top5.iloc[0]['churn_pct'] if len(top5)>0 else 0:.1f}%
    - High-risk segment: {risk_count} customers ({risk_churn:.1f}% churn rate)
    Focus on business impact for Vi Telecom India.
    """
    key = f"overview_{churn_rate:.1f}_{total}"
    if f"insight_{key}" not in st.session_state:
        st.session_state[f"insight_{key}"] = ai_insight(insight_prompt, key)
    insight_box(st.session_state[f"insight_{key}"])


# ─────────────────────────────────────────────────────────────────
# TAB 2: CUSTOMER EDA
# ─────────────────────────────────────────────────────────────────

def tab_eda(df: pd.DataFrame):
    section("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    # ── Chart 1: Churn % by Plan Type & Gender ──
    with col1:
        st.markdown("**Churn Rate by Plan Type & Gender**")
        grp = df.groupby(["plan_type", "gender"])["churned"].mean().reset_index()
        grp["churn_pct"] = grp["churned"] * 100
        color_map = {"Prepaid": C["prepaid"], "Postpaid": C["postpaid"]}
        fig1 = px.bar(
            grp, x="gender", y="churn_pct", color="plan_type",
            barmode="group",
            color_discrete_map=color_map,
            labels={"churn_pct": "Churn (%)", "gender": "Gender"},
            text_auto=".1f",
        )
        fig1.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
                           legend_title="Plan Type", margin=dict(t=10, b=10),
                           font=dict(family="DM Sans"))
        st.plotly_chart(fig1, use_container_width=True)

        grp_summary = grp.to_dict(orient="records")
        insight_prompt1 = f"""
        Telecom analyst. 2 sentences, no bullets:
        Churn rate by plan type and gender: {grp_summary}
        Insight for Vi Telecom retention strategy.
        """
        k1 = f"eda1_{df['churned'].sum()}_{len(df)}"
        if f"insight_{k1}" not in st.session_state:
            st.session_state[f"insight_{k1}"] = ai_insight(insight_prompt1, k1)
        insight_box(st.session_state[f"insight_{k1}"])

    # ── Chart 2: Violin — Monthly Charges ──
    with col2:
        st.markdown("**Monthly Charges: Churned vs Retained**")
        df_plot = df.copy()
        df_plot["Status"] = df_plot["churned"].map({1: "Churned", 0: "Retained"})
        fig2 = px.violin(
            df_plot, x="Status", y="monthly_charges_inr",
            color="Status",
            color_discrete_map={"Churned": C["churned"], "Retained": C["retained"]},
            box=True, points="outliers",
            labels={"monthly_charges_inr": "Monthly Charges (₹)"},
        )
        fig2.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
                           showlegend=False, margin=dict(t=10, b=10),
                           font=dict(family="DM Sans"))
        st.plotly_chart(fig2, use_container_width=True)

        churned_med = df[df["churned"]==1]["monthly_charges_inr"].median()
        retained_med = df[df["churned"]==0]["monthly_charges_inr"].median()
        insight_prompt2 = f"""
        Telecom analyst. 2 sentences, no bullets:
        Median monthly charges — Churned: ₹{churned_med:.0f}, Retained: ₹{retained_med:.0f}
        Sample: {len(df)} customers. Insight for Vi Telecom.
        """
        k2 = f"eda2_{churned_med:.0f}_{retained_med:.0f}"
        if f"insight_{k2}" not in st.session_state:
            st.session_state[f"insight_{k2}"] = ai_insight(insight_prompt2, k2)
        insight_box(st.session_state[f"insight_{k2}"])

    col3, col4 = st.columns(2)

    # ── Chart 3: Box — Contract Tenure ──
    with col3:
        st.markdown("**Contract Tenure by Churn Status**")
        df_plot2 = df.copy()
        df_plot2["Status"] = df_plot2["churned"].map({1: "Churned", 0: "Retained"})
        fig3 = px.box(
            df_plot2, x="Status", y="contract_tenure_months",
            color="Status",
            color_discrete_map={"Churned": C["churned"], "Retained": C["retained"]},
            points="outliers",
            labels={"contract_tenure_months": "Contract Tenure (months)"},
        )
        fig3.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
                           showlegend=False, margin=dict(t=10, b=10),
                           font=dict(family="DM Sans"))
        st.plotly_chart(fig3, use_container_width=True)

        churned_ten = df[df["churned"]==1]["contract_tenure_months"].median()
        retained_ten = df[df["churned"]==0]["contract_tenure_months"].median()
        insight_prompt3 = f"""
        Telecom analyst. 2 sentences, no bullets:
        Median tenure — Churned: {churned_ten:.0f} months, Retained: {retained_ten:.0f} months.
        Insight on tenure's role in Vi Telecom churn.
        """
        k3 = f"eda3_{churned_ten:.0f}_{retained_ten:.0f}"
        if f"insight_{k3}" not in st.session_state:
            st.session_state[f"insight_{k3}"] = ai_insight(insight_prompt3, k3)
        insight_box(st.session_state[f"insight_{k3}"])

    # ── Chart 4: Correlation Heatmap ──
    with col4:
        st.markdown("**Correlation Heatmap — Top 12 Numeric Features**")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        num_cols = [c for c in num_cols if not c.endswith("_was_imputed") and c != "churned"][:12]
        corr = df[num_cols + ["churned"]].corr()
        fig4 = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig4.update_layout(height=320, margin=dict(t=10, b=10),
                           font=dict(family="DM Sans", size=10))
        st.plotly_chart(fig4, use_container_width=True)

        top_corr = corr["churned"].drop("churned").abs().sort_values(ascending=False).head(3)
        insight_prompt4 = f"""
        Telecom analyst. 2 sentences, no bullets:
        Top 3 features correlated with churn: {top_corr.index.tolist()} with correlations {top_corr.values.tolist()}.
        Insight for Vi Telecom feature selection.
        """
        k4 = f"eda4_{'_'.join(top_corr.index.tolist()[:2])}"
        if f"insight_{k4}" not in st.session_state:
            st.session_state[f"insight_{k4}"] = ai_insight(insight_prompt4, k4)
        insight_box(st.session_state[f"insight_{k4}"])


# ─────────────────────────────────────────────────────────────────
# TAB 3: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────

def tab_models(model_data: dict):
    section("🤖 Model Performance Comparison")
    metrics_df = model_data["metrics"]

    # Styled metrics table
    def color_metric(val, col):
        if col in ["Test Acc", "F1", "ROC AUC", "Recall"]:
            if val >= 0.75: return "metric-good"
            if val < 0.60: return "metric-bad"
        return ""

    st.markdown("**📋 Metrics Comparison Table**")
    cols_show = ["Model", "Train Acc", "Test Acc", "Precision", "Recall", "F1", "ROC AUC", "PR AUC"]
    html_rows = ""
    for _, row in metrics_df[cols_show].iterrows():
        html_rows += "<tr>"
        for i, (col, val) in enumerate(row.items()):
            if i == 0:
                html_rows += f"<td style='font-weight:600;color:#1A1F3C'>{val}</td>"
            else:
                cls = color_metric(float(val), col)
                html_rows += f"<td class='{cls}'>{val:.4f}</td>"
        html_rows += "</tr>"

    st.markdown(f"""
    <div style='overflow-x:auto;'>
    <table style='width:100%;border-collapse:collapse;font-size:14px;'>
    <thead><tr style='background:#1A1F3C;color:white;'>
    {''.join(f"<th style='padding:10px 16px;text-align:left'>{c}</th>" for c in cols_show)}
    </tr></thead>
    <tbody>
    {html_rows}
    </tbody></table></div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion Matrices ──
    section("🔢 Confusion Matrices")
    cms = model_data["cms"]
    cm_cols = st.columns(3)
    for idx, (name, cm) in enumerate(cms.items()):
        with cm_cols[idx]:
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            labels = [["TN", "FP"], ["FN", "TP"]]
            values = [[tn, fp], [fn, tp]]
            pcts = [[f"{tn/total*100:.1f}%", f"{fp/total*100:.1f}%"],
                    [f"{fn/total*100:.1f}%", f"{tp/total*100:.1f}%"]]
            text = [[f"{v}<br><small>{p}</small>" for v, p in zip(row_v, row_p)]
                    for row_v, row_p in zip(values, pcts)]

            fig_cm = go.Figure(go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                text=text,
                texttemplate="%{text}",
                colorscale=[[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, "#1D4ED8"]],
                showscale=False,
            ))
            fig_cm.update_layout(
                title=dict(text=name, font=dict(size=13, family="Space Grotesk")),
                height=240, margin=dict(t=40, b=10, l=10, r=10),
                font=dict(family="DM Sans", size=11),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC Curves ──
    section("📈 ROC Curves — All Models")
    roc = model_data["roc"]
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="#9CA3AF", width=1))
    for name, rd in roc.items():
        fig_roc.add_trace(go.Scatter(
            x=rd["fpr"], y=rd["tpr"],
            name=f"{name} (AUC={rd['auc']:.3f})",
            line=dict(color=MODEL_COLORS.get(name, "#666"), width=2.5),
        ))
    fig_roc.update_layout(
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.1, bgcolor="rgba(255,255,255,0.9)"),
        margin=dict(t=10, b=30), font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    best_model = metrics_df.loc[metrics_df["ROC AUC"].idxmax(), "Model"]
    best_auc = metrics_df["ROC AUC"].max()
    insight_prompt_m = f"""
    Telecom ML analyst. 2 sentences, no bullets:
    Model AUC scores: {', '.join([f"{n}: {d['auc']:.3f}" for n, d in roc.items()])}.
    Best model: {best_model} (AUC={best_auc:.3f}). Insight on model choice for Vi Telecom churn prediction.
    """
    km = f"model_{best_model}_{best_auc:.3f}"
    if f"insight_{km}" not in st.session_state:
        st.session_state[f"insight_{km}"] = ai_insight(insight_prompt_m, km)
    insight_box(st.session_state[f"insight_{km}"])


# ─────────────────────────────────────────────────────────────────
# TAB 4: RETENTION INSIGHTS
# ─────────────────────────────────────────────────────────────────

def tab_retention(model_data: dict):
    section("🔍 Feature Importance — Model-Based")
    importances = model_data["importances"]

    imp_cols = st.columns(len(importances))
    for idx, (name, imp) in enumerate(importances.items()):
        with imp_cols[idx]:
            top10 = imp.head(10).reset_index()
            top10.columns = ["feature", "importance"]
            fig_imp = px.bar(
                top10, x="importance", y="feature", orientation="h",
                color="importance",
                color_continuous_scale=["#BFDBFE", MODEL_COLORS.get(name, "#4361EE")],
                labels={"importance": "Importance", "feature": ""},
                title=name,
            )
            fig_imp.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                                   coloraxis_showscale=False, margin=dict(t=40, b=10),
                                   yaxis=dict(categoryorder="total ascending"),
                                   font=dict(family="DM Sans", size=11))
            st.plotly_chart(fig_imp, use_container_width=True)

    # ── Permutation Importance ──
    section("🎲 Permutation Importance (XGBoost, with Error Bars)")
    perm = model_data["perm_imp"].head(10)
    fig_perm = go.Figure(go.Bar(
        x=perm["importance"],
        y=perm["feature"],
        orientation="h",
        error_x=dict(type="data", array=perm["std"].tolist()),
        marker_color=C["blue"],
        marker_line_width=0,
    ))
    fig_perm.update_layout(
        height=360, plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Mean AUC Drop", yaxis_title="",
        yaxis=dict(categoryorder="total ascending"),
        margin=dict(t=10, b=30), font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_perm, use_container_width=True)

    # AI insight on feature importance
    top5_features = list(model_data["importances"].get("XGBoost", list(model_data["importances"].values())[0]).head(5).index)
    insight_prompt_f = f"""
    Vi Telecom analyst. 2 sentences, no bullets:
    Top 5 churn drivers: {top5_features}.
    Provide concise business interpretation of what these features mean for customer retention.
    """
    kf = f"feat_{'_'.join(top5_features[:2])}"
    if f"insight_{kf}" not in st.session_state:
        st.session_state[f"insight_{kf}"] = ai_insight(insight_prompt_f, kf)
    insight_box(st.session_state[f"insight_{kf}"])

    # ── Business Action List ──
    section("💡 Top 5 Churn Drivers & Vi Retention Strategies")

    actions = [
        ("📊 Customer Satisfaction Score",
         "Low satisfaction is the most direct signal of imminent churn.",
         "Deploy proactive NPS surveys at months 3, 6, and 12. Route low-scorers (<3) to dedicated retention specialists within 48 hours."),
        ("📋 Complaints Last 6 Months",
         "Repeat complainers have disproportionately high churn rates.",
         "Implement a complaint escalation SLA: >2 complaints in 6 months triggers automatic account review and a personalised goodwill offer."),
        ("💳 Late Payment Count",
         "Payment difficulties correlate with both financial stress and low loyalty.",
         "Offer flexible EMI plans or bill-shock alerts via SMS/app. Auto-enrol high-risk accounts into Vi Pay Later."),
        ("📅 Contract Tenure",
         "Short-tenure customers churn at 2–3x the rate of long-tenure ones.",
         "Introduce a 'Loyalty Rewards' programme with increasing data/call benefits at 6, 12, and 24-month milestones."),
        ("📡 Usage Ratio (Data Used / Data Limit)",
         "Customers consistently hitting data caps churn due to frustration.",
         "Trigger a personalised upgrade prompt when usage_ratio > 0.85 for 2 consecutive months; offer a ₹50 data booster pack free for first month."),
    ]

    for icon_title, driver, strategy in actions:
        with st.expander(icon_title, expanded=False):
            st.markdown(f"**Driver:** {driver}")
            st.markdown(f"**Strategy:** {strategy}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    # Load raw data for EDA/sidebar (with original categorical columns)
    df_raw = load_raw()

    # Apply sidebar filters
    df_filtered = build_sidebar(df_raw)

    if len(df_filtered) == 0:
        st.warning("No data matches the current filters. Please widen your selection.")
        return

    # Load models (uses encoded data internally)
    with st.spinner("Loading models & computing metrics…"):
        model_data = load_models_and_data()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Overview",
        "📊 Customer EDA",
        "🤖 Churn Models",
        "💡 Retention Insights",
    ])

    with tab1:
        tab_overview(df_filtered)

    with tab2:
        tab_eda(df_filtered)

    with tab3:
        tab_models(model_data)

    with tab4:
        tab_retention(model_data)


if __name__ == "__main__":
    main()
