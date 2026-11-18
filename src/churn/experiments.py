"""Grid rastreado: seleção por validação e teste final apenas do vencedor."""

import itertools

from churn.config import Settings, settings
from churn.model import (
    configure_tracking,
    evaluate_holdout,
    split_data,
    train_candidate,
    write_json,
)


def run_grid(
    cfg: Settings = settings, n_estimators=(100, 200, 500), depths=(None, 8, 16)
):
    configure_tracking(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(cfg)
    candidates = []
    best_pipeline = best = None
    for n, depth in itertools.product(n_estimators, depths):
        candidate_cfg = cfg.model_copy(update={"n_estimators": n, "max_depth": depth})
        pipeline, candidate, _ = train_candidate(
            candidate_cfg, X_train, y_train, X_val, y_val
        )
        candidates.append(candidate)
        if best is None or candidate["val_roc_auc"] > best["val_roc_auc"]:
            best, best_pipeline = candidate, pipeline
    if best is None:
        raise ValueError("O grid precisa conter ao menos uma combinação.")
    evaluate_holdout(best_pipeline, best, X_test, y_test)
    summary = {"candidates": candidates, "best_run_id": best["run_id"]}
    write_json(cfg.candidates_path, summary)
    return summary


if __name__ == "__main__":
    summary = run_grid()
    print(f"Melhor run: {summary['best_run_id']}")
