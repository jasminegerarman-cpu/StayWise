"""Train on original hotel bookings; select via validation, report untouched test.

Original rows are never duplicated/augmented. No fabricated benchmark metrics.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
from src.config import DATA_PATH, MODEL_PATH, REPORT_PATH, FEATURES, TARGET
from src.pipeline import load_dataset, make_preprocessor
from download_data import download_dataset


def evaluate(model, X, y):
    pred = model.predict(X)
    classes = list(model.classes_)
    probability = model.predict_proba(X)[:, classes.index(1)]
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, probability)), 4),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, pred, digits=3, zero_division=0),
        "class_0_count": int(np.sum(np.asarray(y) == 0)),
        "class_1_count": int(np.sum(np.asarray(y) == 1)),
    }


def train():
    download_dataset()
    df = load_dataset(DATA_PATH)
    original_count = len(__import__('pandas').read_csv(DATA_PATH, usecols=[TARGET]))
    print(f"\nOriginal dataset: {original_count:,} bookings", flush=True)
    print(f"Valid original records after cleaning: {len(df):,}", flush=True)
    X, y = df[FEATURES], df[TARGET]

    # 60% TRAIN / 20% VALIDATION / 20% UNTOUCHED TEST.
    # Model selection uses validation, never the final test results.
    X_build, X_test, y_build, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_build, y_build, test_size=0.25, random_state=42, stratify=y_build)

    algorithms = {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1400, class_weight="balanced", solver="lbfgs"),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100, max_depth=17, min_samples_leaf=4,
            n_jobs=-1, random_state=42, class_weight="balanced_subsample"),
    }
    comparison = {}
    for name, new_estimator in algorithms.items():
        print(f"[TRAIN] {name} on {len(X_train):,} training rows...", flush=True)
        model = Pipeline([("preprocess", make_preprocessor()), ("model", new_estimator())])
        model.fit(X_train, y_train)
        comparison[name] = evaluate(model, X_val, y_val)
        print(f"[VALIDATION] {name}: F1={comparison[name]['f1']:.4f}; "
              f"ROC-AUC={comparison[name]['roc_auc']:.4f}", flush=True)

    selected = max(comparison, key=lambda k: (comparison[k]["f1"], comparison[k]["roc_auc"]))
    print(f"[SELECTED] {selected}, by VALIDATION F1 only", flush=True)
    best_model = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", algorithms[selected]()),
    ])
    best_model.fit(X_build, y_build)  # Train + validation after selection
    final_test = evaluate(best_model, X_test, y_test)  # one-time independent test
    print(f"[FINAL INDEPENDENT TEST] accuracy={final_test['accuracy']:.4f}, "
          f"F1={final_test['f1']:.4f}, ROC-AUC={final_test['roc_auc']:.4f}", flush=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH, compress=3)
    report = {
        "project": "StayWise | Machine Learning-Based Hotel Booking Cancellation Prediction",
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "dataset_source": "António, de Almeida & Nunes (2019), mirrored by TidyTuesday",
        "raw_records": int(original_count),
        "modeling_records": int(len(df)),
        "train_records": int(len(X_train)),
        "validation_records": int(len(X_val)),
        "refit_train_records": int(len(X_build)),
        "test_records": int(len(X_test)),
        "cancel_rate": round(float(y.mean()), 4),
        "best_model": selected,
        "selection_metric": "Validation F1 for class 1 = canceled; ROC-AUC tie breaker",
        "split_method": "Stratified 60/20/20 train/validation/test, random_state=42",
        "data_leakage_exclusions": ["reservation_status", "reservation_status_date", "assigned_room_type", "booking_changes"],
        "features": FEATURES,
        "model_comparison": comparison,
        "final_test": final_test,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[SAVED] {MODEL_PATH}\n[SAVED] {REPORT_PATH}", flush=True)
    return report


if __name__ == "__main__":
    try:
        train()
    except Exception as exc:
        print(f"\nTRAINING FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
