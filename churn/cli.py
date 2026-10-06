"""
Command-line interface for the churn prediction model.
Provides commands to train, evaluate, and predict on batch data.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd  # type: ignore

from churn import APP_NAME, __version__
from churn.data import load_data, validate_batch
from churn.model import compare_models, feature_importance, save, score_frame, threshold_table, train


def cmd_train(args) -> None:
    model = train()
    s = model.summary
    print(f"Trained on {s['train_rows']} customers, tested on {s['test_rows']} (churn rate {s['churn_rate']:.1%}).")
    print(f"Stopped after {s['epochs']} epochs; {s['n_features']} encoded features.\n")
    m = model.metrics_at(args.threshold)
    print(
        f"At threshold {args.threshold:.2f}:  accuracy {m['accuracy']:.1%}  precision {m['precision']:.1%}  "
        f"recall {m['recall']:.1%}  F1 {m['f1']:.1%}  ROC-AUC {s['roc_auc']:.3f}"
    )
    print(f"(Always predicting 'no churn' scores {model.majority_baseline:.1%} accuracy.)")
    print(f"Confusion matrix: TN {m['tn']}  FP {m['fp']}  FN {m['fn']}  TP {m['tp']}")
    if not args.no_save:
        print(f"\nSaved model to {save(model)}")


def cmd_evaluate(args) -> None:
    model = train()
    print("Model comparison (same split, same preprocessing, threshold 0.5)\n")
    print(compare_models(model).to_string(index=False))
    print("\nThreshold trade-off for the ANN\n")
    print(threshold_table(model.y_test, model.proba_test).round(3).to_string(index=False))
    print("\nPermutation importance (drop in ROC-AUC)\n")
    print(feature_importance(model).head(8).round(4).to_string(index=False))


def cmd_predict(args) -> None:
    df = pd.read_csv(args.csv)
    errors, warnings = validate_batch(df)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if errors:
        sys.exit("error: " + "; ".join(errors))
    scored = score_frame(train(load_data()), df, args.threshold).sort_values("churn_probability", ascending=False)
    scored.to_csv(args.out, index=False)
    print(f"Scored {len(scored)} customers -> {args.out}")
    print(scored["risk_band"].value_counts().to_string())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="churn", description=f"{APP_NAME} {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("train", help="train the ANN and print its test metrics")
    sp.add_argument("--threshold", type=float, default=0.5)
    sp.add_argument("--no-save", action="store_true", help="do not write models/churn_model.joblib")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="compare with baselines, threshold trade-off, feature importance")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("predict", help="score a CSV of customers")
    sp.add_argument("csv")
    sp.add_argument("--out", default="scored_customers.csv")
    sp.add_argument("--threshold", type=float, default=0.5)
    sp.set_defaults(func=cmd_predict)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
