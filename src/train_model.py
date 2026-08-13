"""
Stage 3 — Train the LightGBM "price will rise in 24h" classifier.

Reads the canonical offers table, builds features (features.py), does a
time-aware train/test split, trains LightGBM, reports AUC / accuracy /
precision-recall, and saves the model + metrics. Also scores the latest
snapshot per trip and writes a `predictions` table for the API to serve.

Cross-validation is distributed across time-ordered folds with Ray when it's
installed (Ray has no Windows wheel for Python 3.13 yet — use a 3.10-3.12
env for this), and the run is tracked with MLflow when it's installed.
Both are optional: without them training falls back to the plain single-split
run exactly as before.

Outputs:
  models/price_rise_lgbm.txt   trained booster
  models/metrics.json          evaluation metrics + feature importance
  data/aggregator.db:predictions   per-trip rise probability
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from sqlalchemy import text

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "models")

import db
import features as F  # noqa: E402


def _time_folds(df: pd.DataFrame, n_folds: int = 4):
    """Expanding-window time folds: each fold trains on everything before a
    cut point and tests on the slice right after it (no shuffling — captures
    stay in capture order, same principle as the main 80/20 split)."""
    df = df.sort_values("captured_at").reset_index(drop=True)
    n = len(df)
    bounds = np.linspace(int(n * 0.5), n, n_folds + 1, dtype=int)
    folds = []
    for i in range(n_folds):
        train_df, test_df = df.iloc[:bounds[i]], df.iloc[bounds[i]:bounds[i + 1]]
        if len(test_df) < 20:
            continue
        folds.append((train_df, test_df))
    return folds


def _train_one_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict) -> dict:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, accuracy_score

    Xtr, ytr = train_df[F.FEATURE_COLS], train_df["price_will_rise"]
    Xte, yte = test_df[F.FEATURE_COLS], test_df["price_will_rise"]
    dtrain = lgb.Dataset(Xtr, label=ytr)
    dtest = lgb.Dataset(Xte, label=yte, reference=dtrain)
    booster = lgb.train(
        params, dtrain, num_boost_round=400, valid_sets=[dtest],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    proba = booster.predict(Xte)
    return {
        "auc": float(roc_auc_score(yte, proba)),
        "accuracy": float(accuracy_score(yte, (proba >= 0.5).astype(int))),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }


def _ray_cross_validate(df: pd.DataFrame, params: dict, n_folds: int = 4) -> list[dict]:
    """Train each time fold as a parallel Ray task. Returns [] if Ray isn't
    installed or no folds are large enough to evaluate."""
    try:
        import ray
    except ImportError:
        print("  ray not installed — skipping distributed CV "
              "(pip install ray; needs Python <=3.12 on Windows)")
        return []

    folds = _time_folds(df, n_folds=n_folds)
    if not folds:
        return []

    ray.init(ignore_reinit_error=True, log_to_driver=False, include_dashboard=False)
    try:
        remote_fold = ray.remote(_train_one_fold)
        futures = [remote_fold.remote(tr, te, params) for tr, te in folds]
        results = ray.get(futures)
    finally:
        ray.shutdown()
    return results


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

    try:
        import mlflow
        HAS_MLFLOW = True
    except ImportError:
        print("  mlflow not installed — skipping run tracking (pip install mlflow)")
        HAS_MLFLOW = False

    engine = db.get_engine()
    df = F.build_training_frame(engine)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "is_unbalance": True,
        "verbose": -1,
    }

    print("  running time-series cross-validation...")
    cv_results = _ray_cross_validate(df, params, n_folds=4)
    cv_auc_mean = float(np.mean([r["auc"] for r in cv_results])) if cv_results else None
    if cv_results:
        print(f"  Ray CV ({len(cv_results)} folds): mean AUC = {cv_auc_mean:.3f}")

    if HAS_MLFLOW:
        mlflow.set_experiment("naijafare-price-rise")
        run_ctx = mlflow.start_run()
    else:
        from contextlib import nullcontext
        run_ctx = nullcontext()

    with run_ctx:
        if HAS_MLFLOW:
            mlflow.log_params(params)
            mlflow.log_param("n_cv_folds", len(cv_results))
            if cv_auc_mean is not None:
                mlflow.log_metric("cv_auc_mean", cv_auc_mean)

        # Time-aware split: earlier captures train, most recent captures test.
        df = df.sort_values("captured_at")
        cut = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:cut], df.iloc[cut:]

        Xtr, ytr = train_df[F.FEATURE_COLS], train_df["price_will_rise"]
        Xte, yte = test_df[F.FEATURE_COLS], test_df["price_will_rise"]

        dtrain = lgb.Dataset(Xtr, label=ytr)
        dtest = lgb.Dataset(Xte, label=yte, reference=dtrain)
        booster = lgb.train(
            params, dtrain, num_boost_round=400, valid_sets=[dtrain, dtest],
            valid_names=["train", "test"],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
        )

        proba = booster.predict(Xte)
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(yte, proba)
        acc = accuracy_score(yte, pred)
        base_rate = float(ytr.mean())

        print(f"  train rows: {len(train_df):,} | test rows: {len(test_df):,}")
        print(f"  base rate (price rises): {base_rate:.1%}")
        print(f"  test AUC:      {auc:.3f}")
        print(f"  test accuracy: {acc:.1%}")
        print("  " + classification_report(yte, pred, digits=3, zero_division=0).replace("\n", "\n  "))

        importance = dict(sorted(
            zip(F.FEATURE_COLS, booster.feature_importance(importance_type="gain")),
            key=lambda kv: -kv[1],
        ))

        booster.save_model(os.path.join(MODEL_DIR, "price_rise_lgbm.txt"))
        metrics = {
            "auc": round(float(auc), 4),
            "accuracy": round(float(acc), 4),
            "base_rate": round(base_rate, 4),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "best_iteration": booster.best_iteration,
            "feature_importance_gain": {k: float(round(v, 1)) for k, v in importance.items()},
        }
        if cv_auc_mean is not None:
            metrics["cv_auc_mean"] = round(cv_auc_mean, 4)
            metrics["cv_folds"] = len(cv_results)
        with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        if HAS_MLFLOW:
            mlflow.log_metrics({"auc": auc, "accuracy": acc, "base_rate": base_rate})
            mlflow.log_artifact(os.path.join(MODEL_DIR, "price_rise_lgbm.txt"))
            mlflow.log_artifact(os.path.join(MODEL_DIR, "metrics.json"))

        _write_predictions(engine, booster)

    print(f"  saved model + metrics to {MODEL_DIR}")
    return metrics


def _write_predictions(engine, booster):
    """Score the latest snapshot per trip and persist for the API."""
    live = F.build_live_features(engine)
    live["rise_prob"] = booster.predict(live[F.FEATURE_COLS])
    cols = ["trip_id", "route_id", "carrier", "mode", "departure_date",
            "departure_hour", "duration_min", "price_ngn", "days_to_departure",
            "rise_prob"]
    out = live[cols].copy()
    out["departure_date"] = out["departure_date"].dt.date.astype(str)
    with engine.begin() as con:
        con.execute(text("DROP TABLE IF EXISTS predictions"))
        out.to_sql("predictions", con, index=False, if_exists="append")
    print(f"  wrote {len(out)} live predictions to predictions table")


if __name__ == "__main__":
    print("Stage 3: training LightGBM price-rise model...")
    train()