"""Split, fit, and evaluate the seven-feature classifier."""
import importlib.metadata
import json
import os
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, make_scorer, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from data_io import CLASSES, ROOT, digest, write_json
from morphology import FEATURES

LABELS = np.arange(len(CLASSES))
SCORE = make_scorer(f1_score, average="macro", labels=LABELS, zero_division=0)
GRID = {
    "forest__n_estimators": [160],
    "forest__max_depth": [14, None],
    "forest__min_samples_leaf": [2, 8],
    "forest__max_features": ["sqrt"],
    "forest__class_weight": [None, "balanced_subsample"],
}


def distribution(labels):
    return np.bincount(labels, minlength=len(CLASSES)).tolist()


def checked_split(a, b, y, groups):
    assert len(a) + len(b) == len(y)
    assert np.array_equal(np.sort(np.r_[a, b]), np.arange(len(y)))
    assert not np.intersect1d(groups[a], groups[b]).size
    assert np.array_equal(np.unique(y[a]), LABELS)
    assert np.array_equal(np.unique(y[b]), LABELS)


def splits(y, groups, k, seed):
    x = np.zeros(len(y))
    if len(np.unique(groups)) == len(groups):
        pairs = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(x, y))
    else:
        pairs = list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(x, y, groups))
    for a, b in pairs:
        checked_split(a, b, y, groups)
    assert np.array_equal(np.sort(np.concatenate([b for _, b in pairs])), np.arange(len(y)))
    return pairs


def split_plan(table, seed=42):
    usable = table.loc[table.usable].reset_index(drop=True)
    y, groups, row = (usable[col].to_numpy() for col in ("label", "group_id", "row_id"))
    candidates = splits(y, groups, 5, seed)
    global_fraction = np.bincount(y, minlength=len(CLASSES)) / len(y)

    def holdout_distance(pair):
        test = pair[1]
        return abs(len(test) / len(y) - .2) + np.abs(
            np.bincount(y[test], minlength=len(CLASSES)) / len(test) - global_fraction).sum()

    dev, test = min(candidates, key=holdout_distance)

    def ids(a, b):
        return {"train": row[a].tolist(), "validation": row[b].tolist()}

    outer = []
    for i, (a, b) in enumerate(splits(y[dev], groups[dev], 5, seed + 1), 1):
        train, val = dev[a], dev[b]
        inner = [ids(train[c], train[d]) for c, d in
                 splits(y[train], groups[train], 3, seed + 10 + i)]
        outer.append({**ids(train, val), "fold": i, "inner": inner})
    final = [ids(dev[a], dev[b]) for a, b in splits(y[dev], groups[dev], 5, seed + 2)]
    plan = {"seed": seed, "development": row[dev].tolist(), "test": row[test].tolist(),
            "outer": outer, "final_tuning": final}
    verify_plan(table, plan)
    return plan


def verify_plan(table, plan):
    groups = table.set_index("row_id").group_id
    usable = set(table.loc[table.usable, "row_id"])

    def check(a, b, parent):
        assert len(set(a)) == len(a) and len(set(b)) == len(b)
        assert set(a).isdisjoint(b) and set(a).union(b) == parent
        assert set(groups.loc[a]).isdisjoint(groups.loc[b])

    def folds(parts, parent, n):
        assert len(parts) == n
        validation = []
        for part in parts:
            check(part["train"], part["validation"], parent)
            validation.extend(part["validation"])
        assert len(validation) == len(parent) and set(validation) == parent

    development = set(plan["development"])
    check(plan["development"], plan["test"], usable)
    folds(plan["outer"], development, 5)
    folds(plan["final_tuning"], development, 5)
    for outer in plan["outer"]:
        folds(outer["inner"], set(outer["train"]), 3)


def persist_splits(table, provenance, out, seed):
    plan = split_plan(table, seed)
    plan["feature_fingerprint"] = provenance["fingerprint"]
    write_json(out / "splits.json", plan)
    assignment = table[["row_id", "label", "group_id", "flags"]].copy()
    assignment["partition"] = "excluded"
    assignment.loc[assignment.row_id.isin(plan["development"]), "partition"] = "development"
    assignment.loc[assignment.row_id.isin(plan["test"]), "partition"] = "test"
    assignment["outer_validation_fold"] = 0
    for fold in plan["outer"]:
        assignment.loc[assignment.row_id.isin(fold["validation"]), "outer_validation_fold"] = fold["fold"]
    assignment.to_csv(out / "split_assignments.csv", index=False)
    return plan


def metrics(y, prediction):
    options = dict(labels=LABELS, zero_division=0, average="macro")
    return {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "macro_precision": precision_score(y, prediction, **options),
        "macro_recall": recall_score(y, prediction, **options),
        "macro_f1": f1_score(y, prediction, **options),
        "weighted_f1": f1_score(y, prediction, **{**options, "average": "weighted"}),
    }


def evaluate(rows):
    result = {}
    for model in ("forest", "baseline"):
        y, p = rows.label, rows[model]
        counts = confusion_matrix(y, p, labels=LABELS)
        row_total = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(counts, row_total, out=np.zeros_like(counts, dtype=float),
                               where=row_total != 0)
        result[model] = {
            "metrics": metrics(y, p),
            "classification_report": classification_report(
                y, p, labels=LABELS, target_names=CLASSES, output_dict=True, zero_division=0),
            "confusion_counts": counts.tolist(),
            "confusion_normalized": normalized.tolist(),
        }
    return result


def environment():
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    packages = {name: importlib.metadata.version(name) for name in
                (line.split("==")[0] for line in requirements)}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(), "packages": packages}


def fit_experiment(table, provenance, out, seed=42, workers=4, smoke=False):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise FileExistsError(f"Evaluation already exists in {out}. Choose a new run directory.")
    start = perf_counter()
    plan = persist_splits(table, provenance, out, seed)
    write_json(out / "dataset.json", provenance)
    write_json(out / "class_mapping.json", dict(enumerate(CLASSES)))
    runtime = environment()
    runtime["search_workers"] = workers
    write_json(out / "runtime.json", runtime)
    versions = sorted({(d.metadata["Name"], d.version) for d in importlib.metadata.distributions()})
    (out / "environment.txt").write_text("".join(f"{name}=={version}\n" for name, version in versions))
    data = table.set_index("row_id")
    grid = {**GRID, "forest__n_estimators": [8]} if smoke else GRID
    checkpoint_dir = out / "outer_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_key = {
        "split_sha256": digest(out / "splits.json"),
        "experiment_sha256": digest(ROOT / "experiment.py"),
        "feature_fingerprint": provenance["fingerprint"],
        "package_versions": runtime["packages"], "grid": grid,
    }

    def fit(ids, cv, name):
        started = perf_counter()
        id_to_pos = {row: pos for pos, row in enumerate(ids)}
        paired = [(np.array([id_to_pos[i] for i in fold["train"]]),
                   np.array([id_to_pos[i] for i in fold["validation"]])) for fold in cv]
        pipeline = Pipeline([("forest", RandomForestClassifier(random_state=seed, n_jobs=1))])
        search = GridSearchCV(pipeline, grid, scoring=SCORE, cv=paired, n_jobs=workers,
                              error_score="raise", refit=True)
        training = data.loc[ids]
        x, y = training[list(FEATURES)], training.label
        search.fit(x, y)
        baseline = DummyClassifier(strategy="most_frequent").fit(x, y)
        pd.DataFrame(search.cv_results_).to_csv(out / f"search_{name}.csv", index=False)
        return search, baseline, perf_counter() - started

    def predict(ids, search, baseline, fold=None):
        validation = data.loc[ids]
        x = validation[list(FEATURES)]
        result = validation[["label", "group_id", "flags"]].copy()
        result["forest"] = search.predict(x)
        result["baseline"] = baseline.predict(x)
        probability = search.predict_proba(x)
        for i, label in enumerate(search.classes_):
            result[f"p_{label}"] = probability[:, i]
        if fold is not None:
            result["fold"] = fold
        return result.reset_index()

    def summary(ids, prediction, search, seconds):
        return {"train_size": len(ids), "validation_size": len(prediction),
                "train_counts": distribution(data.loc[ids, "label"]),
                "validation_counts": distribution(prediction.label),
                "params": search.best_params_,
                "fitted_params": search.best_estimator_.named_steps["forest"].get_params(),
                "tuning_and_fit_seconds": seconds, "refit_seconds": search.refit_time_,
                "forest": metrics(prediction.label, prediction.forest),
                "baseline": metrics(prediction.label, prediction.baseline)}

    outer, predictions = [], []
    for split in plan["outer"]:
        number = split["fold"]
        checkpoint = checkpoint_dir / f"fold_{number}.json"
        prediction_path = checkpoint_dir / f"fold_{number}.csv"
        search_path = out / f"search_outer_{number}.csv"
        cached = json.loads(checkpoint.read_text()) if checkpoint.is_file() else None
        if (cached is not None and cached["key"] == checkpoint_key
                and prediction_path.is_file() and search_path.is_file()
                and cached["prediction_sha256"] == digest(prediction_path)
                and cached["search_sha256"] == digest(search_path)):
            prediction = pd.read_csv(prediction_path, keep_default_na=False, float_precision="round_trip")
            fold_result = cached["result"]
            print(f"Reused outer fold {number}/5", flush=True)
        else:
            search, baseline, seconds = fit(split["train"], split["inner"], f"outer_{number}")
            prediction = predict(split["validation"], search, baseline, number)
            fold_result = {"fold": number, **summary(split["train"], prediction, search, seconds)}
            prediction.to_csv(prediction_path, index=False)
            write_json(checkpoint, {"key": checkpoint_key, "result": fold_result,
                                    "prediction_sha256": digest(prediction_path),
                                    "search_sha256": digest(search_path)})
            print(f"Completed outer fold {number}/5", flush=True)
        predictions.append(prediction)
        outer.append(fold_result)
        write_json(out / "folds.json", outer)
    oof = pd.concat(predictions).sort_values("row_id")
    assert oof.row_id.is_unique and set(oof.row_id) == set(plan["development"])
    oof.to_csv(out / "oof_predictions.csv", index=False)
    search, baseline, seconds = fit(plan["development"], plan["final_tuning"], "final")
    test = predict(plan["test"], search, baseline)
    assert test.row_id.is_unique and set(test.row_id) == set(plan["test"])
    test.to_csv(out / "test_predictions.csv", index=False)
    final = summary(plan["development"], test, search, seconds)
    metadata = {
        "mode": provenance["mode"], "feature_order": FEATURES, "settings": provenance["settings"],
        "feature_code_sha256": provenance["code"]["morphology.py"],
        "class_mapping": dict(enumerate(CLASSES)), "seed": seed, "runtime": runtime,
        "provenance": provenance, "selected_parameters": final["fitted_params"],
        "training_partition": "development", "development_size": len(plan["development"]),
        "split_manifest_sha256": digest(out / "splits.json"),
    }
    joblib.dump({**metadata, "pipeline": search.best_estimator_}, out / "model.joblib", compress=3)
    write_json(out / "model_metadata.json", metadata)
    fold_summary = {}
    for model in ("forest", "baseline"):
        fold_summary[model] = {
            name: {"mean": float(np.mean([f[model][name] for f in outer])),
                   "sample_sd": float(np.std([f[model][name] for f in outer], ddof=1))}
            for name in outer[0][model]
        }
    results = {
        "mode": provenance["mode"], "search_space": grid,
        "duration_seconds": perf_counter() - start, "outer": outer,
        "outer_mean_and_sample_sd": fold_summary, "pooled_oof": evaluate(oof),
        "test": evaluate(test), "final_fit": final,
        "prediction_sha256": {name: digest(out / name) for name in
                              ("oof_predictions.csv", "test_predictions.csv")},
        "impurity_importance": dict(zip(FEATURES, search.best_estimator_.named_steps[
            "forest"].feature_importances_)),
    }
    write_json(out / "metrics.json", results)
    validate_predictions(out)
    return results


def validate_predictions(out, table=None):
    result = json.loads((out / "metrics.json").read_text())
    plan = json.loads((out / "splits.json").read_text())
    assert len(result["outer"]) == len(plan["outer"]) == 5
    source = table.set_index("row_id") if table is not None else None
    for key, filename, partition in (("pooled_oof", "oof_predictions.csv", "development"),
                                      ("test", "test_predictions.csv", "test")):
        rows = pd.read_csv(out / filename, float_precision="round_trip")
        assert digest(out / filename) == result["prediction_sha256"][filename]
        assert rows.row_id.is_unique and set(rows.row_id) == set(plan[partition])
        if source is not None:
            for column in ("label", "group_id"):
                assert np.array_equal(rows[column], source.loc[rows.row_id, column])
        assert evaluate(rows) == result[key]
        probabilities = rows[[f"p_{i}" for i in LABELS]].to_numpy()
        assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1)
        assert np.array_equal(probabilities.argmax(axis=1), rows.forest)
        if key == "pooled_oof":
            for split, fold in zip(plan["outer"], result["outer"]):
                group = rows.loc[rows.fold == split["fold"]]
                assert set(group.row_id) == set(split["validation"])
                for model in ("forest", "baseline"):
                    assert metrics(group.label, group[model]) == fold[model]
        else:
            for model in ("forest", "baseline"):
                assert result["final_fit"][model] == result[key][model]["metrics"]
