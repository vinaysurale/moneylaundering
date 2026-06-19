"""
╔══════════════════════════════════════════════════════════════╗
║  AML MODEL ENGINE — XGBoost + Structural Neighbor Features  ║
║  Elliptic Bitcoin Dataset Node Classification Pipeline      ║
╚══════════════════════════════════════════════════════════════╝

Trains an XGBoost classifier to detect illicit (class 1) Bitcoin
transactions using local features + 1-hop neighbor aggregations.
Temporal train/val split: timesteps 1-34 / 35-49.
"""

import os
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    auc,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"
EDGELIST_FILE = DATA_DIR / "elliptic_txs_edgelist.csv"
MODEL_PATH = MODEL_DIR / "xgb_aml_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.pkl"

# ── Config ─────────────────────────────────────────────────────
TRAIN_TIMESTEPS = range(1, 35)   # timesteps 1-34
VAL_TIMESTEPS = range(35, 50)    # timesteps 35-49
N_NEIGHBOR_AGGS = 3              # mean, sum, max


def log(msg: str, level: str = "INFO"):
    ts = time.strftime("%H:%M:%S")
    prefix = {"INFO": "✓", "WARN": "⚠", "ERROR": "✗", "STEP": "►"}
    print(f"  [{ts}] {prefix.get(level, '·')} {msg}")


# ═══════════════════════════════════════════════════════════════
# STEP 1: LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════
def load_data():
    """Load features, classes, and edgelist CSVs."""
    log("Loading raw CSV data...", "STEP")

    # Features: first col = txId, second col = time_step, remaining = features
    features_df = pd.read_csv(FEATURES_FILE, header=None)
    features_df.columns = (
        ["txId", "time_step"]
        + [f"feat_{i}" for i in range(features_df.shape[1] - 2)]
    )
    log(f"  Features: {features_df.shape[0]:,} rows × {features_df.shape[1]} cols")

    # Classes
    classes_df = pd.read_csv(CLASSES_FILE)
    classes_df.columns = ["txId", "class"]
    log(f"  Classes:  {classes_df.shape[0]:,} rows")

    # Edgelist
    edgelist_df = pd.read_csv(EDGELIST_FILE)
    edgelist_df.columns = ["txId1", "txId2"]
    log(f"  Edges:    {edgelist_df.shape[0]:,} rows")

    return features_df, classes_df, edgelist_df


# ═══════════════════════════════════════════════════════════════
# STEP 2: MERGE & LABEL
# ═══════════════════════════════════════════════════════════════
def merge_and_label(features_df, classes_df):
    """Merge features with class labels; encode labels."""
    log("Merging features with class labels...", "STEP")

    df = features_df.merge(classes_df, on="txId", how="left")

    # Label encoding: 1 (illicit) → 1, 2 (licit) → 0, unknown → NaN
    df["label"] = df["class"].map({"1": 1, "2": 0, 1: 1, 2: 0})
    labeled_mask = df["label"].notna()

    log(f"  Total nodes:   {len(df):,}")
    log(f"  Labeled nodes: {labeled_mask.sum():,}")
    log(f"  Unlabeled:     {(~labeled_mask).sum():,}")

    if labeled_mask.sum() > 0:
        illicit_count = (df.loc[labeled_mask, "label"] == 1).sum()
        licit_count = (df.loc[labeled_mask, "label"] == 0).sum()
        log(f"  Illicit: {illicit_count:,} | Licit: {licit_count:,}")

    return df


# ═══════════════════════════════════════════════════════════════
# STEP 3: 1-HOP NEIGHBOR FEATURE AGGREGATION
# ═══════════════════════════════════════════════════════════════
def build_neighbor_features(df, edgelist_df):
    """Compute mean/sum/max of neighbor features for each node."""
    log("Building 1-hop neighbor structural features...", "STEP")

    feature_cols = [c for c in df.columns if c.startswith("feat_")]

    # Build adjacency list (undirected for aggregation)
    adjacency = defaultdict(set)
    tx_set = set(df["txId"].values)
    for _, row in edgelist_df.iterrows():
        t1, t2 = row["txId1"], row["txId2"]
        if t1 in tx_set and t2 in tx_set:
            adjacency[t1].add(t2)
            adjacency[t2].add(t1)

    log(f"  Adjacency built: {len(adjacency):,} nodes with neighbors")

    # Index features by txId for fast lookup
    feat_matrix = df.set_index("txId")[feature_cols].values
    txid_to_idx = {txid: idx for idx, txid in enumerate(df["txId"].values)}

    # Pre-select a subset of feature columns for aggregation (first 20)
    # to keep computation tractable
    n_agg_feats = min(20, len(feature_cols))
    feat_subset = feat_matrix[:, :n_agg_feats]

    # Compute aggregations
    n_nodes = len(df)
    agg_mean = np.zeros((n_nodes, n_agg_feats), dtype=np.float32)
    agg_sum = np.zeros((n_nodes, n_agg_feats), dtype=np.float32)
    agg_max = np.full((n_nodes, n_agg_feats), -np.inf, dtype=np.float32)
    neighbor_count = np.zeros(n_nodes, dtype=np.int32)

    for txid, neighbors in adjacency.items():
        if txid not in txid_to_idx:
            continue
        idx = txid_to_idx[txid]
        neighbor_indices = [txid_to_idx[n] for n in neighbors if n in txid_to_idx]
        if len(neighbor_indices) == 0:
            continue

        nb_feats = feat_subset[neighbor_indices]
        agg_mean[idx] = nb_feats.mean(axis=0)
        agg_sum[idx] = nb_feats.sum(axis=0)
        agg_max[idx] = nb_feats.max(axis=0)
        neighbor_count[idx] = len(neighbor_indices)

    # Replace -inf with 0 for nodes with no neighbors
    agg_max[agg_max == -np.inf] = 0.0

    # Build neighbor feature columns
    agg_cols = []
    for prefix, arr in [("nb_mean", agg_mean), ("nb_sum", agg_sum), ("nb_max", agg_max)]:
        for j in range(n_agg_feats):
            col_name = f"{prefix}_{j}"
            df[col_name] = arr[:, j]
            agg_cols.append(col_name)

    df["neighbor_count"] = neighbor_count
    agg_cols.append("neighbor_count")

    log(f"  Added {len(agg_cols)} neighbor-aggregated features")
    return df, agg_cols


# ═══════════════════════════════════════════════════════════════
# STEP 4: TEMPORAL SPLIT
# ═══════════════════════════════════════════════════════════════
def temporal_split(df):
    """Split labeled data by timestep for temporal evaluation."""
    log("Performing temporal train/val split...", "STEP")

    labeled = df[df["label"].notna()].copy()
    labeled["label"] = labeled["label"].astype(int)

    train = labeled[labeled["time_step"].isin(TRAIN_TIMESTEPS)]
    val = labeled[labeled["time_step"].isin(VAL_TIMESTEPS)]

    log(f"  Train: {len(train):,} samples (timesteps 1-34)")
    log(f"  Val:   {len(val):,} samples (timesteps 35-49)")

    if len(train) > 0:
        log(f"  Train illicit: {(train['label']==1).sum():,} "
            f"({(train['label']==1).mean()*100:.1f}%)")
    if len(val) > 0:
        log(f"  Val illicit:   {(val['label']==1).sum():,} "
            f"({(val['label']==1).mean()*100:.1f}%)")

    return train, val


# ═══════════════════════════════════════════════════════════════
# STEP 5: TRAIN XGBOOST
# ═══════════════════════════════════════════════════════════════
def train_model(train_df, val_df, feature_cols):
    """Train XGBoost classifier with class imbalance handling."""
    log("Training XGBoost classifier...", "STEP")

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values

    # Compute scale_pos_weight for class imbalance
    n_licit = (y_train == 0).sum()
    n_illicit = (y_train == 1).sum()
    scale_weight = n_licit / max(n_illicit, 1)
    log(f"  Class imbalance ratio: {scale_weight:.1f}:1 (licit:illicit)")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    log("  Training complete ✓")
    return model, X_val, y_val


# ═══════════════════════════════════════════════════════════════
# STEP 6: EVALUATE
# ═══════════════════════════════════════════════════════════════
def evaluate_model(model, X_val, y_val):
    """Full evaluation with classification report + confusion matrix."""
    log("Evaluating model performance...", "STEP")

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Classification report
    report = classification_report(y_val, y_pred, target_names=["Licit", "Illicit"])
    print("\n  ┌─── CLASSIFICATION REPORT ───────────────────────┐")
    for line in report.strip().split("\n"):
        print(f"  │ {line:<50s}│")
    print("  └──────────────────────────────────────────────────┘\n")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("  ┌─── CONFUSION MATRIX ────────────────────────────┐")
    print(f"  │                Predicted Licit  Predicted Illicit│")
    print(f"  │ Actual Licit     {cm[0,0]:>6,}          {cm[0,1]:>6,}       │")
    print(f"  │ Actual Illicit   {cm[1,0]:>6,}          {cm[1,1]:>6,}       │")
    print("  └──────────────────────────────────────────────────┘\n")

    # Key metrics
    f1_illicit = f1_score(y_val, y_pred, pos_label=1)
    precision, recall, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = auc(recall, precision)

    log(f"  F1 (illicit): {f1_illicit:.4f}")
    log(f"  PR-AUC:       {pr_auc:.4f}")

    if f1_illicit >= 0.70:
        log(f"  ✓ TARGET MET: F1 ≥ 0.70 on illicit class", "INFO")
    else:
        log(f"  ⚠ Below target: F1 = {f1_illicit:.4f} (target: 0.70)", "WARN")

    return {
        "f1_illicit": f1_illicit,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# STEP 7: SAVE MODEL
# ═══════════════════════════════════════════════════════════════
def save_model(model, feature_cols, metrics):
    """Persist trained model and metadata."""
    log("Saving model artifacts...", "STEP")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    log(f"  Model saved: {MODEL_PATH}")

    metadata = {
        "feature_columns": feature_cols,
        "metrics": metrics,
        "train_timesteps": list(TRAIN_TIMESTEPS),
        "val_timesteps": list(VAL_TIMESTEPS),
    }
    joblib.dump(metadata, METADATA_PATH)
    log(f"  Metadata saved: {METADATA_PATH}")


# ═══════════════════════════════════════════════════════════════
# INFERENCE API (used by streaming dashboard)
# ═══════════════════════════════════════════════════════════════
def load_trained_model():
    """Load the saved model and metadata for inference."""
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


def predict_batch(model, feature_df, feature_cols):
    """
    Run inference on a batch of transactions.
    Returns array of illicit probabilities (0.0–1.0).
    """
    X = feature_df[feature_cols].values
    probas = model.predict_proba(X)[:, 1]
    return probas


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
def main():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║  AML MODEL ENGINE — TRAINING PIPELINE            ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    t_start = time.time()

    # 1. Load
    features_df, classes_df, edgelist_df = load_data()

    # 2. Merge & label
    df = merge_and_label(features_df, classes_df)

    # 3. Neighbor features
    df, agg_cols = build_neighbor_features(df, edgelist_df)

    # 4. Define all feature columns
    base_feature_cols = [c for c in df.columns if c.startswith("feat_")]
    all_feature_cols = base_feature_cols + agg_cols

    # 5. Temporal split
    train_df, val_df = temporal_split(df)

    # 6. Train
    model, X_val, y_val = train_model(train_df, val_df, all_feature_cols)

    # 7. Evaluate
    metrics = evaluate_model(model, X_val, y_val)

    # 8. Save
    save_model(model, all_feature_cols, metrics)

    elapsed = time.time() - t_start
    print()
    log(f"PIPELINE COMPLETE in {elapsed:.1f}s ✓", "STEP")
    print()


if __name__ == "__main__":
    main()
