"""
╔══════════════════════════════════════════════════════════════╗
║  BLOCKCHAIN AML MONITOR: AZTEC NETWORK                      ║
║  Real-Time Blockchain Threat Intelligence Dashboard         ║
║  Powered by XGBoost + Elliptic Bitcoin Dataset              ║
╚══════════════════════════════════════════════════════════════╝
"""

import time
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="BLOCKCHAIN AML MONITOR: AZTEC NETWORK",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# AZTEC THEME CSS
# ═══════════════════════════════════════════════════════════════
AZTEC_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;500;600;700;800;900&family=VT323&display=swap');

    :root {
        --bg-dark: #0a1612;
        --bg-panel: rgba(12, 28, 22, 0.92);
        --bg-panel-alt: rgba(18, 40, 32, 0.85);
        --border-main: #2a5e4a;
        --border-glow: #3d8b6e;
        --border-gold: #c4a245;

        --text-gold: #d4a843;
        --text-bright-gold: #f0d060;
        --text-green: #5dbb8a;
        --text-green-bright: #7ddb9a;
        --text-dim: #4a8a6e;
        --text-white: #d0e8d8;

        --accent-red: #e74c3c;
        --accent-amber: #e6a817;
        --accent-green: #27ae60;
        --accent-teal: #1abc9c;

        --font-mono: 'Share Tech Mono', 'Courier New', monospace;
        --font-display: 'Orbitron', monospace;
        --font-terminal: 'VT323', monospace;

        --patina-gradient: linear-gradient(
            145deg,
            #0a1612 0%,
            #0d1f18 20%,
            #122a20 40%,
            #0f231a 60%,
            #0b1913 80%,
            #0a1612 100%
        );
    }

    /* ── GLOBAL ────────────────────────────────────── */
    .stApp {
        background: var(--patina-gradient) !important;
        color: var(--text-green) !important;
        font-family: var(--font-mono) !important;
    }

    .stApp > header {
        background: rgba(10, 22, 18, 0.97) !important;
        border-bottom: 2px solid var(--border-main) !important;
    }

    #MainMenu, footer, .stDeployButton { display: none !important; }

    /* ── SCROLLBAR ─────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb {
        background: var(--border-main);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-gold); }

    /* ── SIDEBAR ───────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081510 0%, #0d1f18 50%, #081510 100%) !important;
        border-right: 2px solid var(--border-main) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-gold) !important;
        font-family: var(--font-display) !important;
    }

    /* ── METRIC CARDS ──────────────────────────────── */
    div[data-testid="stMetric"] {
        background: var(--bg-panel) !important;
        border: 2px solid var(--border-main) !important;
        border-radius: 4px !important;
        padding: 14px 16px !important;
        box-shadow: 0 0 15px rgba(42, 94, 74, 0.2), inset 0 0 30px rgba(10, 22, 18, 0.5) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--text-gold) !important;
        box-shadow: 0 0 20px rgba(196, 162, 69, 0.2), inset 0 0 30px rgba(10, 22, 18, 0.5) !important;
    }
    div[data-testid="stMetric"] label {
        color: var(--text-dim) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--text-bright-gold) !important;
        font-family: var(--font-display) !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        text-shadow: 0 0 10px rgba(212, 168, 67, 0.3);
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] > div {
        color: var(--text-green-bright) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.78rem !important;
    }

    /* ── HEADERS ────────────────────────────────────── */
    h1 {
        color: var(--text-bright-gold) !important;
        font-family: var(--font-display) !important;
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        letter-spacing: 3px !important;
        text-shadow: 0 0 15px rgba(212, 168, 67, 0.4);
    }
    h2 {
        color: var(--text-gold) !important;
        font-family: var(--font-display) !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
    }
    h3 {
        color: var(--text-green) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.9rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    /* ── DATAFRAMES ────────────────────────────────── */
    .stDataFrame {
        border: 2px solid var(--border-main) !important;
        border-radius: 4px !important;
        overflow: hidden;
        background: var(--bg-panel) !important;
    }
    div[data-testid="stDataFrame"] > div {
        background-color: var(--bg-dark) !important;
    }

    /* ── BUTTONS ───────────────────────────────────── */
    .stButton > button {
        font-family: var(--font-mono) !important;
        font-size: 0.8rem !important;
        letter-spacing: 2px !important;
        border-radius: 3px !important;
        padding: 8px 16px !important;
        text-transform: uppercase !important;
        border: 2px solid var(--border-main) !important;
        background: rgba(42, 94, 74, 0.15) !important;
        color: var(--text-green) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        background: rgba(42, 94, 74, 0.35) !important;
        border-color: var(--text-gold) !important;
        color: var(--text-gold) !important;
        box-shadow: 0 0 15px rgba(196, 162, 69, 0.25) !important;
    }
    .stButton > button:active {
        transform: scale(0.97) !important;
    }

    /* ── SLIDER ────────────────────────────────────── */
    .stSlider > div > div > div {
        color: var(--text-gold) !important;
    }

    /* ── DIVIDERS ──────────────────────────────────── */
    hr {
        border-color: var(--border-main) !important;
        opacity: 0.5 !important;
    }

    /* ── PLOTLY ────────────────────────────────────── */
    .stPlotlyChart {
        border: 2px solid var(--border-main) !important;
        border-radius: 4px !important;
        background: var(--bg-panel) !important;
        box-shadow: 0 0 15px rgba(42, 94, 74, 0.15) !important;
    }

    /* ── PROGRESS BAR ─────────────────────────────── */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent-teal), var(--text-gold)) !important;
        border-radius: 2px !important;
    }

    /* ── TABS ─────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: rgba(10, 22, 18, 0.9);
        border: 2px solid var(--border-main);
        border-radius: 4px;
        padding: 3px;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-dim) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.78rem !important;
        border-radius: 3px !important;
        letter-spacing: 1px !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(42, 94, 74, 0.3) !important;
        color: var(--text-gold) !important;
    }

    /* ── PANEL CONTAINERS ─────────────────────────── */
    .aztec-panel {
        background: var(--bg-panel);
        border: 2px solid var(--border-main);
        border-radius: 4px;
        padding: 18px;
        box-shadow: 0 0 15px rgba(42, 94, 74, 0.2), inset 0 0 40px rgba(10, 22, 18, 0.6);
        position: relative;
        overflow: hidden;
    }
    .aztec-panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--text-gold), transparent);
        opacity: 0.4;
    }

    .panel-title {
        font-family: var(--font-mono);
        font-size: 0.78rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text-gold);
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(42, 94, 74, 0.5);
    }

    /* ── HEADER BAR ───────────────────────────────── */
    .aztec-header {
        background: rgba(12, 28, 22, 0.95);
        border: 2px solid var(--border-main);
        border-radius: 4px;
        padding: 14px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        box-shadow: 0 0 20px rgba(42, 94, 74, 0.25);
    }
    .aztec-header-title {
        font-family: var(--font-display);
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--text-bright-gold);
        letter-spacing: 3px;
        text-shadow: 0 0 12px rgba(212, 168, 67, 0.4);
    }
    .aztec-header-subtitle {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        color: var(--text-green);
        letter-spacing: 1px;
        margin-top: 2px;
    }
    .aztec-header-right {
        text-align: right;
        font-family: var(--font-mono);
    }
    .aztec-header-time {
        font-size: 1.1rem;
        color: var(--text-bright-gold);
        font-weight: bold;
        letter-spacing: 2px;
    }
    .aztec-header-status {
        font-size: 0.72rem;
        color: var(--text-green);
        letter-spacing: 1px;
    }

    /* ── TRANSACTION LIST ─────────────────────────── */
    .tx-list {
        background: rgba(8, 18, 14, 0.85);
        border: 1px solid rgba(42, 94, 74, 0.4);
        border-radius: 3px;
        padding: 10px 14px;
        font-family: var(--font-mono);
        font-size: 0.78rem;
        line-height: 1.9;
        height: 340px;
        overflow-y: auto;
    }
    .tx-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 3px 0;
        border-bottom: 1px solid rgba(42, 94, 74, 0.15);
    }
    .tx-hash { color: var(--text-green); }
    .tx-amount { color: var(--text-white); }
    .tx-risk-high { color: #e74c3c; font-weight: bold; }
    .tx-risk-med { color: #e6a817; }
    .tx-risk-low { color: #27ae60; }

    /* ── TERMINAL FEED ────────────────────────────── */
    .terminal-feed {
        background: rgba(8, 18, 14, 0.85);
        border: 1px solid rgba(42, 94, 74, 0.4);
        border-radius: 3px;
        padding: 14px 16px;
        font-family: var(--font-mono);
        font-size: 0.78rem;
        line-height: 1.8;
        height: 260px;
        overflow-y: auto;
    }
    .terminal-feed .safe { color: var(--accent-green); }
    .terminal-feed .illicit { color: var(--accent-red); font-weight: bold; }
    .terminal-feed .header-line { color: var(--text-dim); }
    .terminal-feed .timestamp { color: var(--text-dim); opacity: 0.6; }

    /* ── ALERT BANNERS ────────────────────────────── */
    .alert-banner {
        text-align: center;
        padding: 10px 20px;
        font-family: var(--font-mono);
        font-size: 0.78rem;
        letter-spacing: 2px;
        border-radius: 3px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }
    .alert-secure {
        background: rgba(39, 174, 96, 0.1);
        border: 2px solid rgba(39, 174, 96, 0.4);
        color: var(--accent-green);
    }
    .alert-elevated {
        background: rgba(230, 168, 23, 0.1);
        border: 2px solid rgba(230, 168, 23, 0.4);
        color: var(--accent-amber);
    }
    .alert-critical {
        background: rgba(231, 76, 60, 0.1);
        border: 2px solid rgba(231, 76, 60, 0.5);
        color: var(--accent-red);
        box-shadow: 0 0 15px rgba(231, 76, 60, 0.2);
    }

    /* ── STATUS INDICATORS ────────────────────────── */
    .status-secure { color: var(--accent-green); }
    .status-warning { color: var(--accent-amber); }
    .status-danger { color: var(--accent-red); }

    /* ── SIDEBAR STATS ────────────────────────────── */
    .sidebar-stat-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid rgba(42, 94, 74, 0.2);
        font-family: var(--font-mono);
        font-size: 0.82rem;
    }
    .sidebar-stat-label { color: var(--text-dim); }
    .sidebar-stat-value { color: var(--text-bright-gold); font-weight: bold; }

    /* ── LIVE PULSE ────────────────────────────────── */
    @keyframes aztec-pulse {
        0%, 100% { opacity: 1; text-shadow: 0 0 8px currentColor; }
        50% { opacity: 0.5; text-shadow: 0 0 2px currentColor; }
    }
    .live-pulse {
        animation: aztec-pulse 1.5s ease-in-out infinite;
    }

    /* ── BAR INDICATOR ────────────────────────────── */
    .bar-container {
        background: rgba(42, 94, 74, 0.2);
        border: 1px solid var(--border-main);
        border-radius: 2px;
        height: 22px;
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.5s ease;
    }
    .bar-fill-green {
        background: linear-gradient(90deg, #1a6b40, #27ae60, #2ecc71);
        box-shadow: 0 0 8px rgba(39, 174, 96, 0.4);
    }
    .bar-fill-amber {
        background: linear-gradient(90deg, #b8860b, #e6a817, #f0c040);
        box-shadow: 0 0 8px rgba(230, 168, 23, 0.4);
    }
    .bar-fill-red {
        background: linear-gradient(90deg, #8b1a1a, #e74c3c, #ff6b6b);
        box-shadow: 0 0 8px rgba(231, 76, 60, 0.4);
    }
</style>
"""

st.markdown(AZTEC_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING & MODEL (cached)
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_all_data():
    """Load and prepare the full dataset for streaming."""
    parquet_path = DATA_DIR / "elliptic_features_optimized.parquet"
    csv_path = DATA_DIR / "elliptic_txs_features.csv"
    
    if parquet_path.exists():
        features_df = pd.read_parquet(parquet_path)
    else:
        features_df = pd.read_csv(csv_path, header=None)
        
    features_df.columns = (
        ["txId", "time_step"]
        + [f"feat_{i}" for i in range(features_df.shape[1] - 2)]
    )

    classes_df = pd.read_csv(DATA_DIR / "elliptic_txs_classes.csv")
    classes_df.columns = ["txId", "class"]

    edgelist_df = pd.read_csv(DATA_DIR / "elliptic_txs_edgelist.csv")
    edgelist_df.columns = ["txId1", "txId2"]

    # Merge
    df = features_df.merge(classes_df, on="txId", how="left")
    df["label"] = df["class"].map({"1": 1, "2": 0, 1: 1, 2: 0})

    return df, edgelist_df


@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained XGBoost model and metadata."""
    model = joblib.load(MODEL_DIR / "xgb_aml_model.pkl")
    metadata = joblib.load(MODEL_DIR / "model_metadata.pkl")
    return model, metadata


def generate_tx_hash(txid):
    """Generate a pseudo-hash for display from txId."""
    h = hashlib.md5(str(txid).encode()).hexdigest()
    return f"0x{h[:6].upper()}..."


def compute_neighbor_features_batch(batch_df, all_df, edgelist_df, n_agg_feats=20):
    """Compute neighbor features for a batch of transactions."""
    from collections import defaultdict

    feature_cols = [c for c in all_df.columns if c.startswith("feat_")]

    # Build adjacency from full edgelist
    adjacency = defaultdict(set)
    tx_set = set(all_df["txId"].values)
    batch_txids = set(batch_df["txId"].values)

    for _, row in edgelist_df.iterrows():
        t1, t2 = row["txId1"], row["txId2"]
        if t1 in tx_set and t2 in tx_set:
            if t1 in batch_txids or t2 in batch_txids:
                adjacency[t1].add(t2)
                adjacency[t2].add(t1)

    # Feature lookup from full dataset
    feat_matrix = all_df.set_index("txId")[feature_cols].values
    txid_to_idx = {txid: idx for idx, txid in enumerate(all_df["txId"].values)}

    n_agg = min(n_agg_feats, len(feature_cols))
    feat_subset = feat_matrix[:, :n_agg]

    result = batch_df.copy()
    agg_cols = []

    for prefix in ["nb_mean", "nb_sum", "nb_max"]:
        for j in range(n_agg):
            col = f"{prefix}_{j}"
            result[col] = 0.0
            agg_cols.append(col)

    result["neighbor_count"] = 0
    agg_cols.append("neighbor_count")

    for i, row in result.iterrows():
        txid = row["txId"]
        if txid in adjacency:
            nbs = [txid_to_idx[n] for n in adjacency[txid] if n in txid_to_idx]
            if len(nbs) > 0:
                nb_feats = feat_subset[nbs]
                for j in range(n_agg):
                    result.at[i, f"nb_mean_{j}"] = nb_feats[:, j].mean()
                    result.at[i, f"nb_sum_{j}"] = nb_feats[:, j].sum()
                    result.at[i, f"nb_max_{j}"] = nb_feats[:, j].max()
                result.at[i, "neighbor_count"] = len(nbs)

    return result, agg_cols


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "current_timestep": 0,
        "is_streaming": False,
        "transaction_buffer": [],
        "timestep_stats": [],
        "live_feed_log": deque(maxlen=200),
        "total_processed": 0,
        "total_illicit": 0,
        "total_licit": 0,
        "stream_speed": 3,
        "initialized": False,
        "auto_stream": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ═══════════════════════════════════════════════════════════════
# STREAMING ENGINE
# ═══════════════════════════════════════════════════════════════
def process_timestep(ts, df, edgelist_df, model, metadata):
    """Process a single timestep: filter, predict, update state."""
    batch = df[df["time_step"] == ts].copy()
    if len(batch) == 0:
        return

    feature_cols = metadata["feature_columns"]

    # Compute neighbor features for this batch
    batch_enriched, _ = compute_neighbor_features_batch(batch, df, edgelist_df)

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in batch_enriched.columns:
            batch_enriched[col] = 0.0

    # Run inference
    X = batch_enriched[feature_cols].values
    risk_scores = model.predict_proba(X)[:, 1]

    batch_enriched["risk_score"] = risk_scores
    batch_enriched["predicted_class"] = (risk_scores > 0.5).astype(int)
    batch_enriched["tx_hash"] = batch_enriched["txId"].apply(generate_tx_hash)

    # Use feat_0 as a volume proxy
    batch_enriched["volume_proxy"] = batch_enriched["feat_0"].abs().round(4)

    # Update session state
    n_illicit = (batch_enriched["predicted_class"] == 1).sum()
    n_licit = (batch_enriched["predicted_class"] == 0).sum()
    fraud_rate = n_illicit / max(len(batch_enriched), 1) * 100

    st.session_state.total_processed += len(batch_enriched)
    st.session_state.total_illicit += int(n_illicit)
    st.session_state.total_licit += int(n_licit)

    # Timestep stats
    st.session_state.timestep_stats.append({
        "timestep": ts,
        "total_txs": len(batch_enriched),
        "illicit": int(n_illicit),
        "licit": int(n_licit),
        "fraud_rate": fraud_rate,
        "avg_risk": float(risk_scores.mean()),
        "max_risk": float(risk_scores.max()),
    })

    # Live feed log
    sorted_batch = batch_enriched.nlargest(min(15, len(batch_enriched)), "risk_score")
    ts_time = time.strftime("%H:%M:%S")

    for _, row in sorted_batch.iterrows():
        status = "ILLICIT" if row["predicted_class"] == 1 else "SAFE"
        css_class = "illicit" if status == "ILLICIT" else "safe"
        marker = "⚠" if status == "ILLICIT" else "·"

        entry = (
            f'<span class="timestamp">[{ts_time}]</span> '
            f'<span class="{css_class}">{marker} TX {row["tx_hash"]} '
            f'| {row["volume_proxy"]:.4f} BTC '
            f'| Risk: {row["risk_score"]:.3f} '
            f'| {status}</span>'
        )
        st.session_state.live_feed_log.appendleft(entry)

    # Store high-risk transactions in buffer
    high_risk = batch_enriched[batch_enriched["risk_score"] > 0.3].copy()
    records = high_risk[["tx_hash", "time_step", "risk_score", "volume_proxy",
                          "predicted_class"]].to_dict("records")
    st.session_state.transaction_buffer.extend(records)

    st.session_state.current_timestep = ts


# ═══════════════════════════════════════════════════════════════
# PLOTLY CHARTS (AZTEC THEME)
# ═══════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(10, 22, 18, 0.6)",
    plot_bgcolor="rgba(0, 0, 0, 0)",
    font=dict(family="Share Tech Mono, monospace", color="#4a8a6e", size=11),
    margin=dict(l=45, r=15, t=45, b=35),
    xaxis=dict(
        gridcolor="rgba(42, 94, 74, 0.15)",
        zerolinecolor="rgba(42, 94, 74, 0.15)",
        linecolor="rgba(42, 94, 74, 0.4)",
    ),
    yaxis=dict(
        gridcolor="rgba(42, 94, 74, 0.15)",
        zerolinecolor="rgba(42, 94, 74, 0.15)",
        linecolor="rgba(42, 94, 74, 0.4)",
    ),
)


def build_threat_chart(stats):
    """Small sparkline-style threat chart for the metrics panel."""
    if not stats:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=120,
            margin=dict(l=5, r=5, t=5, b=5),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    df_stats = pd.DataFrame(stats)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_stats["timestep"], y=df_stats["fraud_rate"],
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(39, 174, 96, 0.12)",
        line=dict(color="#27ae60", width=2, shape="spline"),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=120,
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(visible=False, gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(visible=False, gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


def build_fraud_ratio_chart(stats):
    """Cumulative fraud ratio over timesteps."""
    if not stats:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="FRAUD RATIO", height=260)
        return fig

    df_stats = pd.DataFrame(stats)
    cumulative_illicit = df_stats["illicit"].cumsum()
    cumulative_total = df_stats["total_txs"].cumsum()
    cumulative_ratio = (cumulative_illicit / cumulative_total * 100).round(2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_stats["timestep"], y=cumulative_ratio,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.08)",
        line=dict(color="#e74c3c", width=2, shape="spline"),
        name="Fraud %",
    ))

    fig.add_hline(y=5, line_dash="dash", line_color="#e6a817",
                  annotation_text="THRESHOLD 5%",
                  annotation=dict(font_color="#e6a817", font_size=10,
                                  font_family="Share Tech Mono"))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="[CUMULATIVE FRAUD INDEX]",
                   font=dict(color="#d4a843", size=12, family="Share Tech Mono")),
        xaxis_title="BLOCK",
        yaxis_title="FRAUD %",
        showlegend=False,
        height=260,
    )
    return fig


def build_volume_chart(stats):
    """Transaction volume by timestep."""
    if not stats:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="TX VOLUME", height=260)
        return fig

    df_stats = pd.DataFrame(stats)

    colors = []
    for _, row in df_stats.iterrows():
        ratio = row["fraud_rate"] / 100
        if ratio > 0.08:
            colors.append("rgba(231, 76, 60, 0.7)")
        elif ratio > 0.04:
            colors.append("rgba(230, 168, 23, 0.7)")
        else:
            colors.append("rgba(39, 174, 96, 0.6)")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_stats["timestep"], y=df_stats["total_txs"],
        marker=dict(
            color=colors,
            line=dict(color="rgba(42, 94, 74, 0.4)", width=1),
        ),
        name="Transactions",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="[BLOCK VOLUME MONITOR]",
                   font=dict(color="#d4a843", size=12, family="Share Tech Mono")),
        xaxis_title="BLOCK",
        yaxis_title="TX COUNT",
        showlegend=False,
        height=260,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# DASHBOARD RENDERING
# ═══════════════════════════════════════════════════════════════
def render_dashboard():
    """Main dashboard rendering function."""

    # ── CHECK DATA & MODEL ─────────────────────────────────────
    data_ready = (DATA_DIR / "elliptic_txs_features.csv").exists() or (DATA_DIR / "elliptic_features_optimized.parquet").exists()
    model_ready = (MODEL_DIR / "xgb_aml_model.pkl").exists()

    if not data_ready or not model_ready:
        st.markdown("# 🏛️ BLOCKCHAIN AML MONITOR")
        st.markdown("---")
        
        if not model_ready:
            st.error("⚠ MODEL NOT FOUND — Run `python model_engine.py` locally to train the model first.")
            return

        if not data_ready:
            st.warning("⚠ DATASET NOT FOUND — Initiating automated download (This may take a few minutes...)")
            with st.spinner('Downloading Elliptic Dataset from HuggingFace/Kaggle...'):
                import data_fetcher
                data_fetcher.main()
                st.success("Download complete! Refreshing...")
                time.sleep(2)
                st.rerun()
        return

    # ── LOAD DATA & MODEL ──────────────────────────────────────
    df, edgelist_df = load_all_data()
    model, metadata = load_model()
    all_timesteps = sorted(df["time_step"].unique())
    max_ts = int(max(all_timesteps))

    # Session values
    current_ts = st.session_state.current_timestep
    total_proc = st.session_state.total_processed
    total_ill = st.session_state.total_illicit
    total_lic = st.session_state.total_licit

    if total_proc > 0:
        overall_fraud_pct = total_ill / total_proc * 100
    else:
        overall_fraud_pct = 0.0

    risk_score_val = int(overall_fraud_pct * 10) if total_proc > 0 else 0
    risk_score_val = min(risk_score_val, 100)

    if overall_fraud_pct > 8:
        system_status = "BREACH DETECTED"
        status_class = "status-danger"
        alert_class = "alert-critical"
    elif overall_fraud_pct > 4:
        system_status = "ELEVATED RISK"
        status_class = "status-warning"
        alert_class = "alert-elevated"
    else:
        system_status = "SECURE"
        status_class = "status-secure"
        alert_class = "alert-secure"

    current_time = time.strftime("%H:%M:%S")

    # ══════════════════════════════════════════════════════════
    # SIDEBAR — AZTEC MONITOR
    # ══════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center; padding: 12px 0 6px 0;">'
            '<div style="font-family: Orbitron, monospace; font-size: 1.15rem; '
            'font-weight: 800; color: #f0d060; letter-spacing: 2px; '
            'text-shadow: 0 0 12px rgba(212, 168, 67, 0.4);">'
            '🏛️ Aztec<br>Monitor</div>'
            '<div style="font-family: Share Tech Mono, monospace; font-size: 0.65rem; '
            'color: #4a8a6e; letter-spacing: 1px; margin-top: 4px;">'
            'BLOCKCHAIN INTELLIGENCE</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Stream Controls ────────────────────────────────────
        st.markdown(
            '<div class="panel-title">⚡ STREAM CONTROLS</div>',
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            start_btn = st.button("▶ START", key="start_btn", use_container_width=True)
        with col_b:
            stop_btn = st.button("■ STOP", key="stop_btn", use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            step_btn = st.button("⏭ STEP", key="step_btn", use_container_width=True)
        with col_d:
            reset_btn = st.button("↺ RESET", key="reset_btn", use_container_width=True)

        st.markdown("")
        speed = st.slider("SPEED (sec)", 1, 10, st.session_state.stream_speed, key="speed_slider")
        st.session_state.stream_speed = speed

        # Stream status indicator
        if st.session_state.auto_stream:
            st.markdown(
                '<div style="text-align:center; padding: 6px; margin-top: 6px; '
                'background: rgba(39, 174, 96, 0.1); border: 1px solid rgba(39, 174, 96, 0.35); '
                'border-radius: 3px;">'
                '<span class="live-pulse" style="color: #27ae60; font-family: Share Tech Mono; '
                'font-size: 0.75rem; letter-spacing: 2px;">● STREAMING</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align:center; padding: 6px; margin-top: 6px; '
                'background: rgba(42, 94, 74, 0.1); border: 1px solid rgba(42, 94, 74, 0.25); '
                'border-radius: 3px;">'
                '<span style="color: #4a8a6e; font-family: Share Tech Mono; '
                'font-size: 0.75rem; letter-spacing: 2px;">○ PAUSED</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Dataset Info ───────────────────────────────────────
        st.markdown(
            '<div class="panel-title">📊 DATASET</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sidebar-stat-row">'
            f'<span class="sidebar-stat-label">Timesteps</span>'
            f'<span class="sidebar-stat-value">{max_ts}</span></div>'
            f'<div class="sidebar-stat-row">'
            f'<span class="sidebar-stat-label">Nodes</span>'
            f'<span class="sidebar-stat-value">{len(df):,}</span></div>'
            f'<div class="sidebar-stat-row">'
            f'<span class="sidebar-stat-label">Edges</span>'
            f'<span class="sidebar-stat-value">{len(edgelist_df):,}</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Model Stats ───────────────────────────────────────
        model_metrics = metadata.get("metrics", {})
        if model_metrics:
            st.markdown(
                '<div class="panel-title">🤖 MODEL</div>',
                unsafe_allow_html=True,
            )
            f1_val = model_metrics.get("f1_illicit", 0)
            pr_val = model_metrics.get("pr_auc", 0)
            st.markdown(
                f'<div class="sidebar-stat-row">'
                f'<span class="sidebar-stat-label">F1 Illicit</span>'
                f'<span class="sidebar-stat-value" style="color:#27ae60;">{f1_val:.4f}</span></div>'
                f'<div class="sidebar-stat-row">'
                f'<span class="sidebar-stat-label">PR-AUC</span>'
                f'<span class="sidebar-stat-value" style="color:#1abc9c;">{pr_val:.4f}</span></div>',
                unsafe_allow_html=True,
            )

    # ── HANDLE BUTTON ACTIONS ──────────────────────────────────
    if reset_btn:
        for key in ["current_timestep", "transaction_buffer", "timestep_stats",
                     "live_feed_log", "total_processed", "total_illicit",
                     "total_licit", "auto_stream"]:
            if key == "transaction_buffer":
                st.session_state[key] = []
            elif key == "timestep_stats":
                st.session_state[key] = []
            elif key == "live_feed_log":
                st.session_state[key] = deque(maxlen=200)
            elif key == "auto_stream":
                st.session_state[key] = False
            else:
                st.session_state[key] = 0
        st.rerun()

    if start_btn:
        st.session_state.auto_stream = True

    if stop_btn:
        st.session_state.auto_stream = False

    if step_btn and st.session_state.current_timestep < max_ts:
        next_ts = st.session_state.current_timestep + 1
        process_timestep(next_ts, df, edgelist_df, model, metadata)

    # ══════════════════════════════════════════════════════════
    # MAIN DASHBOARD
    # ══════════════════════════════════════════════════════════

    # ── HEADER BAR ─────────────────────────────────────────────
    st.markdown(
        f'<div class="aztec-header">'
        f'<div>'
        f'<div class="aztec-header-title">BLOCKCHAIN AML MONITOR: AZTEC NETWORK</div>'
        f'<div class="aztec-header-subtitle">STATUS: KINGDOM IS {system_status}</div>'
        f'</div>'
        f'<div class="aztec-header-right">'
        f'<div class="aztec-header-time">{current_time}</div>'
        f'<div class="aztec-header-status">GLOBAL NETWORK: CONNECTED</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── PROGRESS ───────────────────────────────────────────────
    progress = current_ts / max_ts if max_ts > 0 else 0
    st.progress(progress)

    # ══════════════════════════════════════════════════════════
    # THREE-PANEL LAYOUT (like the reference image)
    # ══════════════════════════════════════════════════════════
    panel_left, panel_center, panel_right = st.columns([3, 4, 3])

    # ── LEFT PANEL: THREAT METRICS ─────────────────────────────
    with panel_left:
        st.markdown(
            '<div class="aztec-panel">'
            '<div class="panel-title">⚔ THREAT METRICS</div>',
            unsafe_allow_html=True,
        )

        # System Status
        if system_status == "SECURE":
            status_icon = "🟢"
            status_color = "#27ae60"
        elif system_status == "ELEVATED RISK":
            status_icon = "🟡"
            status_color = "#e6a817"
        else:
            status_icon = "🔴"
            status_color = "#e74c3c"

        st.markdown(
            f'<div style="margin-bottom: 16px;">'
            f'<div style="color: #4a8a6e; font-size: 0.72rem; letter-spacing: 2px; '
            f'margin-bottom: 4px;">SYSTEM STATUS:</div>'
            f'<div style="color: {status_color}; font-family: Orbitron, monospace; '
            f'font-size: 1.3rem; font-weight: 700; letter-spacing: 2px; '
            f'text-shadow: 0 0 10px {status_color}40;">'
            f'{status_icon} {system_status}</div></div>',
            unsafe_allow_html=True,
        )

        # Tx/s counter
        txs_val = total_proc if total_proc > 0 else 0
        st.markdown(
            f'<div style="margin-bottom: 14px;">'
            f'<div style="color: #4a8a6e; font-size: 0.72rem; letter-spacing: 2px; '
            f'margin-bottom: 4px;">TX/s:</div>'
            f'<div style="display: flex; align-items: center; gap: 12px;">'
            f'<span style="color: #f0d060; font-family: Orbitron, monospace; '
            f'font-size: 1.5rem; font-weight: 700;">{txs_val:,}</span>'
            f'</div>'
            f'<div class="bar-container" style="margin-top: 6px;">'
            f'<div class="bar-fill bar-fill-green" style="width: {min(progress * 100, 100):.0f}%;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Risk Score
        if risk_score_val > 60:
            risk_bar_class = "bar-fill-red"
        elif risk_score_val > 30:
            risk_bar_class = "bar-fill-amber"
        else:
            risk_bar_class = "bar-fill-green"

        st.markdown(
            f'<div style="margin-bottom: 14px;">'
            f'<div style="color: #4a8a6e; font-size: 0.72rem; letter-spacing: 2px; '
            f'margin-bottom: 4px;">RISK SCORE:</div>'
            f'<div style="color: #f0d060; font-family: Orbitron, monospace; '
            f'font-size: 1.5rem; font-weight: 700;">{risk_score_val}</div>'
            f'<div class="bar-container" style="margin-top: 6px;">'
            f'<div class="bar-fill {risk_bar_class}" style="width: {risk_score_val}%;"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Mini threat chart
        fig_mini = build_threat_chart(st.session_state.timestep_stats)
        st.plotly_chart(fig_mini, key="threat_mini_chart", use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown('</div>', unsafe_allow_html=True)

    # ── CENTER PANEL: REAL-TIME NODE NETWORK ───────────────────
    with panel_center:
        st.markdown(
            '<div class="aztec-panel">'
            '<div class="panel-title">🌐 REAL-TIME NODE NETWORK</div>',
            unsafe_allow_html=True,
        )

        recent_txs = []
        if st.session_state.transaction_buffer:
            recent_txs = st.session_state.transaction_buffer[-10:]
        tx_data_json = json.dumps(recent_txs)

        # Node network canvas with world map style
        network_html = """
        <div style="position: relative; width: 100%%; height: 340px; overflow: hidden;">
            <canvas id="nodeNetCanvas" style="width: 100%%; height: 340px; display: block;"></canvas>
        </div>
        <script>
        (function() {
            const canvas = document.getElementById('nodeNetCanvas');
            const ctx = canvas.getContext('2d');

            function resize() {
                canvas.width = canvas.clientWidth;
                canvas.height = canvas.clientHeight;
            }
            resize();
            setTimeout(resize, 100);

            const W = canvas.width;
            const H = canvas.height;
            const txData = __TX_DATA__;

            // Draw world map outline (simplified)
            function drawWorldMap() {
                ctx.strokeStyle = 'rgba(42, 94, 74, 0.25)';
                ctx.lineWidth = 1;
                ctx.setLineDash([2, 4]);

                // Grid lines (longitude/latitude)
                for (let x = 0; x < W; x += W/8) {
                    ctx.beginPath();
                    ctx.moveTo(x, 0);
                    ctx.lineTo(x, H);
                    ctx.stroke();
                }
                for (let y = 0; y < H; y += H/6) {
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(W, y);
                    ctx.stroke();
                }
                ctx.setLineDash([]);

                // Continent shapes (simplified polygons)
                ctx.fillStyle = 'rgba(42, 94, 74, 0.12)';
                ctx.strokeStyle = 'rgba(42, 94, 74, 0.35)';
                ctx.lineWidth = 1.5;

                // North America
                ctx.beginPath();
                ctx.moveTo(W*0.1, H*0.15);
                ctx.lineTo(W*0.25, H*0.12);
                ctx.lineTo(W*0.3, H*0.2);
                ctx.lineTo(W*0.28, H*0.35);
                ctx.lineTo(W*0.22, H*0.45);
                ctx.lineTo(W*0.15, H*0.5);
                ctx.lineTo(W*0.1, H*0.45);
                ctx.lineTo(W*0.08, H*0.3);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();

                // South America
                ctx.beginPath();
                ctx.moveTo(W*0.2, H*0.55);
                ctx.lineTo(W*0.25, H*0.52);
                ctx.lineTo(W*0.28, H*0.6);
                ctx.lineTo(W*0.26, H*0.75);
                ctx.lineTo(W*0.22, H*0.85);
                ctx.lineTo(W*0.18, H*0.78);
                ctx.lineTo(W*0.19, H*0.65);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();

                // Europe
                ctx.beginPath();
                ctx.moveTo(W*0.42, H*0.15);
                ctx.lineTo(W*0.52, H*0.12);
                ctx.lineTo(W*0.55, H*0.2);
                ctx.lineTo(W*0.52, H*0.32);
                ctx.lineTo(W*0.45, H*0.35);
                ctx.lineTo(W*0.4, H*0.28);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();

                // Africa
                ctx.beginPath();
                ctx.moveTo(W*0.42, H*0.38);
                ctx.lineTo(W*0.52, H*0.36);
                ctx.lineTo(W*0.55, H*0.5);
                ctx.lineTo(W*0.52, H*0.7);
                ctx.lineTo(W*0.48, H*0.75);
                ctx.lineTo(W*0.44, H*0.65);
                ctx.lineTo(W*0.4, H*0.5);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();

                // Asia
                ctx.beginPath();
                ctx.moveTo(W*0.55, H*0.1);
                ctx.lineTo(W*0.78, H*0.12);
                ctx.lineTo(W*0.85, H*0.2);
                ctx.lineTo(W*0.82, H*0.35);
                ctx.lineTo(W*0.72, H*0.45);
                ctx.lineTo(W*0.6, H*0.4);
                ctx.lineTo(W*0.55, H*0.28);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();

                // Australia
                ctx.beginPath();
                ctx.moveTo(W*0.75, H*0.6);
                ctx.lineTo(W*0.85, H*0.58);
                ctx.lineTo(W*0.88, H*0.68);
                ctx.lineTo(W*0.83, H*0.78);
                ctx.lineTo(W*0.75, H*0.75);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
            }

            // Node locations
            const nodes = [
                { name: 'NODE-NY', x: W*0.18, y: H*0.32, active: true },
                { name: 'NODE-LDN', x: W*0.45, y: H*0.22, active: true },
                { name: 'NODE-TKO', x: W*0.78, y: H*0.3, active: true },
                { name: 'NODE-SYD', x: W*0.8, y: H*0.68, active: true },
                { name: 'NODE-SGP', x: W*0.72, y: H*0.48, active: true },
            ];

            let frame = 0;

            function drawNode(node) {
                const pulse = Math.sin(frame * 0.03 + nodes.indexOf(node) * 1.5) * 0.3 + 0.7;

                // Outer glow
                ctx.shadowBlur = 15;
                ctx.shadowColor = 'rgba(39, 174, 96, 0.5)';

                // Node circle
                ctx.fillStyle = `rgba(39, 174, 96, ${0.15 * pulse})`;
                ctx.beginPath();
                ctx.arc(node.x, node.y, 22, 0, Math.PI * 2);
                ctx.fill();

                ctx.strokeStyle = `rgba(39, 174, 96, ${0.6 * pulse})`;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(node.x, node.y, 22, 0, Math.PI * 2);
                ctx.stroke();

                // Inner dot
                ctx.shadowBlur = 8;
                ctx.fillStyle = `rgba(39, 174, 96, ${0.9 * pulse})`;
                ctx.beginPath();
                ctx.arc(node.x, node.y, 6, 0, Math.PI * 2);
                ctx.fill();

                // Bitcoin/blockchain icon (₿)
                ctx.shadowBlur = 0;
                ctx.fillStyle = `rgba(212, 168, 67, ${0.8 * pulse})`;
                ctx.font = 'bold 11px Share Tech Mono';
                ctx.textAlign = 'center';
                ctx.fillText('₿', node.x, node.y + 4);

                // Label
                ctx.fillStyle = '#d4a843';
                ctx.font = 'bold 10px Share Tech Mono';
                ctx.textAlign = 'center';
                ctx.fillText(node.name, node.x, node.y + 38);
            }

            function drawConnections() {
                for (let i = 0; i < nodes.length; i++) {
                    for (let j = i + 1; j < nodes.length; j++) {
                        const progress = (Math.sin(frame * 0.02 + i + j) + 1) / 2;
                        const alpha = 0.08 + progress * 0.12;

                        ctx.strokeStyle = `rgba(42, 94, 74, ${alpha})`;
                        ctx.lineWidth = 1;
                        ctx.setLineDash([4, 6]);
                        ctx.beginPath();
                        ctx.moveTo(nodes[i].x, nodes[i].y);
                        ctx.lineTo(nodes[j].x, nodes[j].y);
                        ctx.stroke();
                        ctx.setLineDash([]);

                        // Moving data packet dot
                        const px = nodes[i].x + (nodes[j].x - nodes[i].x) * progress;
                        const py = nodes[i].y + (nodes[j].y - nodes[i].y) * progress;
                        ctx.fillStyle = `rgba(212, 168, 67, ${0.6 + progress * 0.4})`;
                        ctx.beginPath();
                        ctx.arc(px, py, 2, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
            }

            // Illicit transaction markers from real data
            const illicitMarkers = [];
            txData.forEach((tx, idx) => {
                if (tx.predicted_class === 1) {
                    illicitMarkers.push({
                        x: 30 + Math.random() * (W - 60),
                        y: 30 + Math.random() * (H - 60),
                        hash: tx.tx_hash,
                        risk: tx.risk_score,
                        phase: Math.random() * Math.PI * 2
                    });
                }
            });

            function drawIllicitMarkers() {
                illicitMarkers.forEach(m => {
                    const pulse = Math.sin(frame * 0.05 + m.phase) * 0.3 + 0.7;
                    ctx.strokeStyle = `rgba(231, 76, 60, ${0.5 * pulse})`;
                    ctx.lineWidth = 1.5;

                    // Warning triangle
                    ctx.beginPath();
                    ctx.moveTo(m.x, m.y - 8);
                    ctx.lineTo(m.x + 7, m.y + 5);
                    ctx.lineTo(m.x - 7, m.y + 5);
                    ctx.closePath();
                    ctx.stroke();

                    ctx.fillStyle = `rgba(231, 76, 60, ${0.3 * pulse})`;
                    ctx.fill();
                });
            }

            function animate() {
                ctx.clearRect(0, 0, W, H);
                frame++;

                drawWorldMap();
                drawConnections();
                nodes.forEach(drawNode);
                drawIllicitMarkers();

                requestAnimationFrame(animate);
            }

            animate();
        })();
        </script>
        """.replace("__TX_DATA__", tx_data_json)

        st.components.v1.html(network_html, height=360)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT PANEL: TRANSACTION HASHES ────────────────────────
    with panel_right:
        st.markdown(
            '<div class="aztec-panel">'
            '<div class="panel-title">📜 TRANSACTION HASHES</div>',
            unsafe_allow_html=True,
        )

        # Build transaction list HTML
        buffer = st.session_state.transaction_buffer
        if buffer:
            # Show latest transactions
            recent = list(reversed(buffer[-25:]))
            tx_rows_html = ""
            for tx in recent:
                risk = tx["risk_score"]
                vol = tx["volume_proxy"]
                tx_h = tx["tx_hash"]

                # Determine risk label and color
                if risk > 0.7:
                    risk_label = "Risk: High"
                    risk_class = "tx-risk-high"
                    risk_icon = "🔴"
                elif risk > 0.4:
                    risk_label = "Risk: Med"
                    risk_class = "tx-risk-med"
                    risk_icon = "🟡"
                else:
                    risk_label = "Risk: Low"
                    risk_class = "tx-risk-low"
                    risk_icon = "🟢"

                # Random currency for display variety
                currencies = ["BTC", "ETH", "USDT"]
                currency = currencies[hash(tx_h) % len(currencies)]

                tx_rows_html += (
                    f'<div class="tx-row">'
                    f'<span class="tx-hash">{tx_h}</span> '
                    f'<span class="tx-amount">{vol:.2f} {currency}</span> '
                    f'<span class="{risk_class}">{risk_label} {risk_icon}</span>'
                    f'</div>'
                )

            st.markdown(
                f'<div class="tx-list">{tx_rows_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="tx-list" style="display:flex; align-items:center; '
                'justify-content:center; text-align:center;">'
                '<div style="color: #4a8a6e;">'
                'AWAITING TRANSACTIONS...<br>'
                '<span style="font-size: 0.7rem; opacity: 0.6;">'
                'Press ▶ START to begin monitoring</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # BOTTOM SECTION: CHARTS + LIVE FEED
    # ══════════════════════════════════════════════════════════

    bottom_left, bottom_right = st.columns([5, 5])

    with bottom_left:
        # Fraud Ratio Chart
        fig_fraud = build_fraud_ratio_chart(st.session_state.timestep_stats)
        st.plotly_chart(fig_fraud, key="fraud_chart", use_container_width=True)

    with bottom_right:
        # Volume Chart
        fig_vol = build_volume_chart(st.session_state.timestep_stats)
        st.plotly_chart(fig_vol, key="vol_chart", use_container_width=True)

    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

    # ── LIVE FEED ──────────────────────────────────────────────
    st.markdown(
        '<div class="aztec-panel">'
        '<div class="panel-title">📡 LIVE TRANSACTION FEED</div>',
        unsafe_allow_html=True,
    )

    feed_lines = list(st.session_state.live_feed_log)
    if feed_lines:
        header = (
            '<div class="header-line">'
            '  TIME      TX HASH         VOLUME     RISK   STATUS<br>'
            '────────────────────────────────────────────────────────</div>'
        )
        feed_html = header + "<br>".join(feed_lines[:60])
    else:
        feed_html = (
            '<span class="header-line">'
            'AWAITING INCOMING TRANSACTIONS...<br>'
            'Press ▶ START or ⏭ STEP in sidebar to begin streaming.<br>'
            '&gt; _</span>'
        )
    st.markdown(
        f'<div class="terminal-feed">{feed_html}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

    # ── FLAGGED TABLE ──────────────────────────────────────────
    st.markdown(
        '<div class="aztec-panel">'
        '<div class="panel-title">🚨 FLAGGED TRANSACTIONS — HIGH RISK</div>',
        unsafe_allow_html=True,
    )

    if buffer:
        flagged_df = pd.DataFrame(buffer)
        flagged_df = flagged_df.sort_values("risk_score", ascending=False)
        flagged_df["STATUS"] = flagged_df["predicted_class"].map(
            {1: "⚠ ILLICIT", 0: "SAFE"}
        )
        display_df = flagged_df[["tx_hash", "time_step", "risk_score",
                                  "volume_proxy", "STATUS"]].copy()
        display_df.columns = ["TX HASH", "BLOCK", "RISK SCORE", "VOLUME", "STATUS"]

        st.dataframe(
            display_df.head(200).style.apply(
                lambda row: [
                    "color: #e74c3c; font-weight: bold;" if row["STATUS"] == "⚠ ILLICIT"
                    else "color: #27ae60;"
                ] * len(row),
                axis=1,
            ),
            use_container_width=True,
            height=280,
            hide_index=True,
        )

        st.markdown(
            f'<div style="text-align:center; color:#4a8a6e; font-size:0.72rem; '
            f'margin-top:6px; font-family: Share Tech Mono, monospace;">'
            f'Showing {min(200, len(display_df))} of '
            f'{len(display_df):,} flagged transactions (risk > 0.3)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:center; padding: 40px; color: #4a8a6e; '
            'font-family: Share Tech Mono, monospace;">'
            'NO FLAGGED TRANSACTIONS YET<br>'
            '<span style="font-size: 0.72rem; opacity: 0.6;">'
            'Begin streaming to populate threat intelligence</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── FOOTER ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#1a4a3a; font-size:0.72rem; '
        'font-family: Share Tech Mono, monospace; padding: 10px; letter-spacing: 2px;">'
        'BLOCKCHAIN AML MONITOR: AZTEC NETWORK v3.0 | ELLIPTIC BITCOIN DATASET | '
        'XGBOOST ENGINE<br>'
        '© 2026 — ALL SYSTEMS OPERATIONAL</div>',
        unsafe_allow_html=True,
    )

    # ── AUTO-STREAM ────────────────────────────────────────────
    if st.session_state.auto_stream and current_ts < max_ts:
        next_ts = current_ts + 1
        process_timestep(next_ts, df, edgelist_df, model, metadata)
        time.sleep(st.session_state.stream_speed)
        st.rerun()
    elif st.session_state.auto_stream and current_ts >= max_ts:
        st.session_state.auto_stream = False
        st.toast("✓ Stream complete — All timesteps processed.", icon="🏛️")


# ── MAIN ───────────────────────────────────────────────────────
render_dashboard()
