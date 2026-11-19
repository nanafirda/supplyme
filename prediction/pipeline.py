"""Orchestration du module prédictif : features -> évaluation -> modèle final -> table predictions."""
from __future__ import annotations

import logging

import pandas as pd

from config import DB_PATH, ModelConfig
from database.db import set_metadata, write_table
from prediction.features import (build_dataset, load_events, load_orders, load_static,
                                 make_snapshots, snapshot_features)
from prediction.models import evaluate, fit_final, predict_next

log = logging.getLogger(__name__)


def run_prediction(cfg: ModelConfig, db_path=DB_PATH) -> dict:
    events, orders = load_events(db_path), load_orders(db_path)
    customers, products = load_static(db_path)
    plan = make_snapshots(events, cfg)
    log.info("Snapshots train : %s -> %s (%d) | test : %s",
             plan.train[0].date(), plan.train[-1].date(), len(plan.train),
             [s.date().isoformat() for s in plan.test])

    train = build_dataset(plan.train, events, orders, customers, products, cfg)
    test = build_dataset(plan.test, events, orders, customers, products, cfg)
    results = evaluate(train, test, cfg)

    reg, clf = fit_final(pd.concat([train, test], ignore_index=True), cfg)
    current = snapshot_features(plan.inference, events, orders, customers, products, cfg, with_target=False)
    preds = predict_next(current, reg, clf, cfg)
    write_table(preds[["customer_id", "product_id", "as_of_date", "last_purchase_date", "predicted_date",
                       "predicted_interval", "baseline_interval", "purchase_proba", "model_name"]],
                "predictions", db_path, replace_rows=True)

    rows = [(model, metric, value, "test")
            for model, metrics in results.items() for metric, value in metrics.items()]
    write_table(pd.DataFrame(rows, columns=["model_name", "metric", "value", "split"]),
                "model_metrics", db_path, replace_rows=True)
    set_metadata({
        "as_of_date": plan.inference.date().isoformat(),
        "train_snapshots": len(plan.train), "test_snapshots": len(plan.test),
        "train_rows": len(train), "test_rows": len(test),
        "censored_share": round(float(train["censored"].mean()), 4),
    }, db_path)
    return results
