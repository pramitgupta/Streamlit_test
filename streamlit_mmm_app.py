
# Streamlit MMM Playground (Colab-friendly)
# ----------------------------------------
# Features:
# - Load sample or your own weekly dataset with channel spends
# - Adstock (geometric) + saturation (log/sqrt/none) feature engineering
# - Ridge regression fit with holdout
# - Channel contributions, response curves, what-if, and a simple greedy budget allocator
# - Designed for class demos: fast, visual, robust
#
# Sample data columns expected:
#   date, sales, price, promo, competitor_index, spend_tv, spend_social, spend_search, spend_display, spend_email
#
# Notes for Colab:
# - In Colab, install and run:
#     !pip -q install streamlit altair scikit-learn pandas numpy pyngrok
#     from pyngrok import ngrok; import os
#     os.environ["STREAMLIT_SERVER_PORT"] = "8501"
#     public_url = ngrok.connect(8501)
#     public_url
#     !streamlit run /content/streamlit_mmm_app.py --server.headless true --server.port 8501
#
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

st.set_page_config(page_title="MMM Playground", layout="wide")

# ----------------------------
# Utilities
# ----------------------------
def adstock_geometric(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, val in enumerate(x):
        out[i] = val + decay * carry
        carry = out[i]
    return out

def saturate(x: np.ndarray, kind: str, scale: float = 10000.0) -> np.ndarray:
    if kind == "log1p":
        return np.log1p(x / scale)
    elif kind == "sqrt":
        return np.sqrt(x / scale)
    elif kind == "none":
        return x / scale
    else:
        raise ValueError("Unknown saturation kind")

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

@dataclass
class MMMConfig:
    channels: List[str]
    adstock_decays: Dict[str, float]
    saturation: str
    control_vars: List[str]
    ridge_alpha: float
    holdout_weeks: int
    scale: float = 10000.0

# ----------------------------
# Load data
# ----------------------------
st.sidebar.title("MMM Playground")
st.sidebar.markdown("Upload your CSV **or** use the sample toy dataset.")

uploaded = st.sidebar.file_uploader("Upload CSV (weekly data)", type=["csv"], help="Must include 'date', 'sales', and 'spend_*' columns.")
use_sample = st.sidebar.toggle("Use sample dataset", value=(uploaded is None))

if use_sample:
    # Path is relative to where you run streamlit; adjust if needed in Colab
    default_path = "mmm_toy_data.csv"
    try:
        df = pd.read_csv(default_path, parse_dates=["date"])
    except Exception:
        st.error("Sample dataset not found next to the app. Upload the CSV or place mmm_toy_data.csv in the working directory.")
        st.stop()
else:
    if uploaded is None:
        st.info("Upload a dataset or toggle 'Use sample dataset'.")
        st.stop()
    df = pd.read_csv(uploaded, parse_dates=["date"])

df = df.sort_values("date").reset_index(drop=True)
channel_cols = [c for c in df.columns if c.startswith("spend_")]
non_media_controls_guess = [c for c in ["price", "promo", "competitor_index"] if c in df.columns]

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.subheader("Model Settings")
decay_defaults = {ch: 0.5 for ch in channel_cols}
adstock_decays = {}
for ch in channel_cols:
    adstock_decays[ch] = st.sidebar.slider(f"Adstock decay for {ch}", 0.0, 0.9, decay_defaults[ch], 0.05)

saturation = st.sidebar.selectbox("Saturation transform", ["log1p", "sqrt", "none"], index=0)
ridge_alpha = st.sidebar.slider("Ridge alpha (L2 strength)", 0.0, 50.0, 5.0, 0.5)
holdout_weeks = st.sidebar.slider("Holdout weeks (from end)", 0, min(26, max(0, len(df)//5)), 12, 1)

controls = st.sidebar.multiselect("Control variables", non_media_controls_guess, default=non_media_controls_guess)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Try adjusting decays and comparing validation metrics.")

# ----------------------------
# Feature engineering
# ----------------------------
def make_features(data: pd.DataFrame, cfg: MMMConfig) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(index=data.index)
    for ch in cfg.channels:
        decay = cfg.adstock_decays[ch]
        ads = adstock_geometric(data[ch].values, decay)
        X[f"{ch}_adstock"] = ads
        X[f"{ch}_feat"] = saturate(ads, cfg.saturation, cfg.scale)
    for c in cfg.control_vars:
        X[c] = data[c].values
    # Add simple time features (trend & seasonality)
    t = np.arange(len(data))
    X["trend"] = (t - t.mean()) / (t.std() + 1e-9)
    X["dow_sin"] = np.sin(2*np.pi*t/52)
    X["dow_cos"] = np.cos(2*np.pi*t/52)
    y = data["sales"].values
    return X, pd.Series(y, index=data.index)

cfg = MMMConfig(
    channels=channel_cols,
    adstock_decays=adstock_decays,
    saturation=saturation,
    control_vars=controls,
    ridge_alpha=ridge_alpha,
    holdout_weeks=holdout_weeks,
)

X, y = make_features(df, cfg)

# ----------------------------
# Train / Validation split
# ----------------------------
n = len(df)
if cfg.holdout_weeks > 0:
    train_idx = np.arange(0, n - cfg.holdout_weeks)
    valid_idx = np.arange(n - cfg.holdout_weeks, n)
else:
    train_idx = np.arange(n)
    valid_idx = np.array([], dtype=int)

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx] if len(valid_idx) > 0 else (pd.DataFrame(), np.array([]))

# ----------------------------
# Fit Ridge regression (on engineered features)
# ----------------------------
model = Ridge(alpha=cfg.ridge_alpha, fit_intercept=True, random_state=42)
model.fit(X_train, y_train)
yhat_train = model.predict(X_train)
r2_tr = r2_score(y_train, yhat_train)
mape_tr = mean_absolute_percentage_error(y_train, yhat_train)
rmse_tr = rmse(y_train, yhat_train)

if len(valid_idx) > 0:
    yhat_valid = model.predict(X_valid)
    r2_v = r2_score(y_valid, yhat_valid)
    mape_v = mean_absolute_percentage_error(y_valid, yhat_valid)
    rmse_v = rmse(y_valid, yhat_valid)
else:
    yhat_valid = np.array([])
    r2_v = mape_v = rmse_v = np.nan

# ----------------------------
# Layout
# ----------------------------
st.title("📊 Media Mix Model (MMM) Playground")
st.write("Quickly explore adstock, saturation, model fit, response curves, and budget allocation.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Data Overview", "🧠 Model Fit", "📈 Response Curves", "💰 Budget Optimizer", "🧪 What‑if Simulator"
])

with tab1:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(12), use_container_width=True)
        st.caption(f"Rows: {len(df):,} | Channels: {len(channel_cols)} — {', '.join(channel_cols)}")
    with right:
        st.subheader("Sales & Total Spend (weekly)")
        plot_df = df.copy()
        plot_df["total_spend"] = df[channel_cols].sum(axis=1)
        c1 = alt.Chart(plot_df).mark_line().encode(
            x="date:T", y=alt.Y("sales:Q", title="Sales")
        ).properties(height=250)
        c2 = alt.Chart(plot_df).mark_line().encode(
            x="date:T", y=alt.Y("total_spend:Q", title="Total Spend")
        ).properties(height=250)
        st.altair_chart(alt.layer(c1, c2).resolve_scale(y='independent').interactive(), use_container_width=True)

with tab2:
    st.subheader("Fit Metrics")
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Train R²", f"{r2_tr:.3f}")
    mcol2.metric("Train MAPE", f"{mape_tr*100:.1f}%")
    mcol3.metric("Train RMSE", f"{rmse_tr:.2f}")

    if len(valid_idx) > 0:
        vcol1, vcol2, vcol3 = st.columns(3)
        vcol1.metric("Valid R²", f"{r2_v:.3f}")
        vcol2.metric("Valid MAPE", f"{mape_v*100:.1f}%")
        vcol3.metric("Valid RMSE", f"{rmse_v:.2f}")
        v_plot = pd.DataFrame({
            "date": df.loc[valid_idx, "date"],
            "actual": y_valid,
            "pred": yhat_valid
        })
        st.altair_chart(
            alt.Chart(v_plot.melt("date", var_name="series", value_name="value"))
              .mark_line()
              .encode(x="date:T", y="value:Q", color="series:N")
              .properties(height=250).interactive(),
            use_container_width=True
        )

    # Coefficients for media features (on transformed features)
    st.subheader("Channel Coefficients (on engineered features)")
    coef_rows = []
    for ch in channel_cols:
        coef_rows.append({
            "channel": ch,
            "coef_on_feat": model.coef_[X.columns.get_loc(f"{ch}_feat")],
            "decay_used": cfg.adstock_decays[ch],
        })
    coef_df = pd.DataFrame(coef_rows).sort_values("coef_on_feat", ascending=False)
    st.dataframe(coef_df, use_container_width=True)

with tab3:
    st.subheader("Response Curve")
    st.caption("Approximate steady‑state assumption: adstock ≈ spend / (1 - decay). Saturation applied on adstocked spend.")

    ch = st.selectbox("Channel", channel_cols, index=0)
    # Baseline: use mean of the engineered features for other vars
    feat_means = X.mean(axis=0)

    # Build response for selected channel
    spend_max = float(df[ch].quantile(0.95) * 2.0)
    spend_grid = np.linspace(0, spend_max, 60)

    k = 1.0 / max(1e-6, (1.0 - cfg.adstock_decays[ch]))  # steady-state multiplier
    def channel_feat_from_spend(s):
        ad = s * k
        return saturate(np.array([ad]), cfg.saturation, cfg.scale)[0]

    base_vec = feat_means.copy()
    # Zero out this channel's engineered feature for incremental view
    base_vec[f"{ch}_feat"] = 0.0

    preds = []
    for s in spend_grid:
        x = base_vec.copy()
        x[f"{ch}_feat"] = channel_feat_from_spend(s)
        # Use mean for non-media engineered cols
        for other in X.columns:
            if other.endswith("_feat") or other in cfg.control_vars or other in ["trend","dow_sin","dow_cos"]:
                if other not in x.index:
                    x[other] = feat_means[other]
        yhat = float(model.intercept_ + np.dot(model.coef_, x.values))
        preds.append(yhat)

    curve_df = pd.DataFrame({"spend": spend_grid, "pred_sales": preds})
    st.altair_chart(
        alt.Chart(curve_df).mark_line().encode(x="spend:Q", y="pred_sales:Q").properties(height=280).interactive(),
        use_container_width=True
    )
    st.caption("This shows predicted sales vs spend for the selected channel, holding others around their average effect.")

with tab4:
    st.subheader("Greedy Budget Optimizer (concave curves)")
    st.caption("Allocates budget in small steps to the channel with the highest current marginal return, using the same response approximation as above.")

    total_budget = st.number_input("Total budget to allocate (same currency as dataset)", min_value=0.0, value=float(df[channel_cols].sum(axis=1).median()), step=1000.0, format="%.2f")
    step = st.slider("Allocation step", min_value=100.0, max_value=5000.0, value=1000.0, step=100.0)
    min_share = st.slider("Min % per channel", 0, 50, 0, 1) / 100.0
    max_share = st.slider("Max % per channel", 50, 100, 100, 1) / 100.0

    # Precompute channel response params via the same function as tab3
    k_map = {c: 1.0 / max(1e-6, (1.0 - cfg.adstock_decays[c])) for c in channel_cols}

    # baseline features at average
    feat_means = X.mean(axis=0)
    base_vec = feat_means.copy()
    for c in channel_cols:
        base_vec[f"{c}_feat"] = 0.0

    def pred_with_spends(sp_map: Dict[str, float]) -> float:
        x = base_vec.copy()
        for c in channel_cols:
            ad = sp_map[c] * k_map[c]
            x[f"{c}_feat"] = saturate(np.array([ad]), cfg.saturation, cfg.scale)[0]
        # fill remaining engineered columns if missing
        for other in X.columns:
            if other not in x.index:
                x[other] = feat_means[other]
        return float(model.intercept_ + np.dot(model.coef_, x.values))

    # Greedy allocation
    alloc = {c: 0.0 for c in channel_cols}
    # Enforce min shares initially
    for c in channel_cols:
        alloc[c] = total_budget * min_share / len(channel_cols)
    spent = sum(alloc.values())
    remaining = max(0.0, total_budget - spent)

    def marginal_gain(c, current_spend):
        # approximate derivative by finite difference around current spend
        s0 = current_spend
        s1 = s0 + step
        y0 = pred_with_spends({**alloc, c: s0})
        y1 = pred_with_spends({**alloc, c: s1})
        return (y1 - y0) / step

    # allocate greedily
    while remaining >= step - 1e-9:
        # obey max share
        candidates = [c for c in channel_cols if alloc[c] <= max_share * total_budget - step + 1e-9]
        if not candidates:
            break
        mg = {c: marginal_gain(c, alloc[c]) for c in candidates}
        best = max(mg, key=mg.get)
        alloc[best] += step
        remaining -= step

    alloc_df = pd.DataFrame({
        "channel": channel_cols,
        "allocated_spend": [alloc[c] for c in channel_cols],
        "share_%": [100*alloc[c]/max(1e-9, total_budget) for c in channel_cols],
    }).sort_values("allocated_spend", ascending=False)

    expected_sales = pred_with_spends(alloc)

    st.write(f"**Optimized allocation** for total budget of {total_budget:,.0f}")
    st.dataframe(alloc_df, use_container_width=True)
    st.metric("Expected sales (under allocation)", f"{expected_sales:,.2f}")

    st.altair_chart(
        alt.Chart(alloc_df).mark_bar().encode(x=alt.X("channel:N", sort="-y"), y="allocated_spend:Q", tooltip=["channel","allocated_spend","share_%"]).properties(height=260),
        use_container_width=True
    )

with tab5:
    st.subheader("What‑if Simulator")
    st.caption("Set weekly spends and preview predicted sales & channel contributions.")

    cols = st.columns(len(channel_cols))
    sim_spend = {}
    for i, ch in enumerate(channel_cols):
        with cols[i]:
            sim_spend[ch] = st.number_input(f"{ch}", min_value=0.0, value=float(df[ch].median()), step=1000.0, format="%.2f")

    # Build one-row feature vector
    feat_means = X.mean(axis=0)
    x = feat_means.copy()
    for ch in channel_cols:
        k = 1.0 / max(1e-6, (1.0 - cfg.adstock_decays[ch]))
        ad = sim_spend[ch] * k
        x[f"{ch}_feat"] = saturate(np.array([ad]), cfg.saturation, cfg.scale)[0]

    # fill missing engineered columns
    for other in X.columns:
        if other not in x.index:
            x[other] = feat_means[other]

    yhat = float(model.intercept_ + np.dot(model.coef_, x.values))

    # Contribution breakdown (approx, using linear terms on engineered features)
    contrib = []
    for ch in channel_cols:
        val = model.coef_[X.columns.get_loc(f"{ch}_feat")] * x[f"{ch}_feat"]
        contrib.append({"channel": ch, "pred_contribution": val})
    contrib_df = pd.DataFrame(contrib).sort_values("pred_contribution", ascending=False)

    left, right = st.columns([1,1])
    with left:
        st.metric("Predicted Sales", f"{yhat:,.2f}")
    with right:
        st.altair_chart(
            alt.Chart(contrib_df).mark_bar().encode(x=alt.X("channel:N", sort="-y"), y="pred_contribution:Q"),
            use_container_width=True
        )

    st.dataframe(contrib_df, use_container_width=True)

st.markdown("---")
st.caption("Educational tool. For production MMM, consider Bayesian calibration, geo‑experiments, and rigorous feature diagnostics.")
