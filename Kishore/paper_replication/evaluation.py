"""Nested group-aware evaluation of the catalog-based experiment."""
import importlib.metadata
import hashlib
import json
import platform
import fcntl
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, make_scorer)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from common import CLASSES, FEATURES, THRESHOLDS, ROOT, read_json, sha256, write_csv, write_json

MODELS = ("dummy", "dt", "svm", "mlp")
LABELS = (0, 1)
SCORER = make_scorer(f1_score, labels=LABELS, average="macro", zero_division=0)
GRIDS = {
    "dummy": {},
    "dt": {"model__max_depth": [6, 12, None], "model__min_samples_leaf": [1, 10]},
    "svm": {"model__C": [0.5, 2.0]},
    "mlp": {"model__alpha": [0.0001, 0.001]},
}


def model_pipeline(name, seed, smoke=False):
    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "dt": DecisionTreeClassifier(criterion="gini", random_state=seed),
        "svm": LinearSVC(dual=False, max_iter=5000, random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(44, 88, 22), activation="relu",
                             solver="adam", batch_size=32 if smoke else 2048,
                             max_iter=20 if smoke else 60, tol=0.0003,
                             n_iter_no_change=8, random_state=seed),
    }
    steps = [("scaler", StandardScaler())] if name in ("svm", "mlp") else []
    if name == "svm":
        steps.append(("rbf", Nystroem(kernel="rbf", gamma=1/len(FEATURES),
                                      n_components=16 if smoke else 128, random_state=seed)))
    return Pipeline(steps + [("model", models[name])])


def splits(rows, k, seed, stratify=None):
    target = rows.label.to_numpy() if stratify is None else np.asarray(stratify)
    group = rows.group_id.to_numpy()
    unique, inverse = np.unique(group, return_inverse=True)
    group_min = np.full(len(unique), 2, dtype=int)
    group_max = np.full(len(unique), -1, dtype=int)
    np.minimum.at(group_min, inverse, rows.label.to_numpy())
    np.maximum.at(group_max, inverse, rows.label.to_numpy())
    stratified = StratifiedKFold(k, shuffle=True, random_state=seed)
    if len(unique) == len(group):
        cv = list(stratified.split(np.zeros(len(rows)), target))
    elif len(group) - len(unique) <= .001 * len(group) and np.array_equal(group_min, group_max):
        # With sparse repeats of a single label, each astronomical object is
        # one stratification unit. Anchor any repeats to their highest K band.
        strata = np.full(len(unique), -1, dtype=int)
        np.maximum.at(strata, inverse, target)
        assignments = np.empty(len(unique), dtype=int)
        for i, (_, indices) in enumerate(stratified.split(np.zeros(len(unique)), strata)):
            assignments[indices] = i
        expanded = assignments[inverse]
        cv = [(np.flatnonzero(expanded != i), np.flatnonzero(expanded == i)) for i in range(k)]
    else:
        cv = list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(
            np.zeros(len(rows)), target, group))
    for train, val in cv:
        assert not np.intersect1d(group[train], group[val]).size
        assert np.array_equal(np.sort(np.r_[train, val]), np.arange(len(rows)))
        assert rows.iloc[train].label.nunique() == rows.iloc[val].label.nunique() == 2
    assert np.array_equal(np.sort(np.concatenate([b for _, b in cv])), np.arange(len(rows)))
    return cv


def split_plan(table, seed):
    master = table.loc[table.selection_k5.eq("included")].copy()
    stratum = master.label * 3 + np.digitize(master.k, [10, 20])
    train, test = splits(master, 5, seed, stratum)[0]
    dev = master.iloc[train].copy()
    holdout = master.iloc[test].copy()
    for index, (_, val) in enumerate(splits(dev, 5, seed + 1, stratum.iloc[train]), 1):
        dev.loc[dev.index[val], "outer"] = index
    plan = {}
    for k in THRESHOLDS:
        tr = dev.loc[dev.k.ge(k)]
        te = holdout.loc[holdout.k.ge(k)]
        outer = []
        for i in range(1, 6):
            a, b = tr.loc[tr.outer.ne(i)], tr.loc[tr.outer.eq(i)]
            inner = [{"train": a.iloc[u].row_id.tolist(), "validation": a.iloc[v].row_id.tolist()}
                     for u, v in splits(a, 3, seed + 10 + i)]
            outer.append({"fold": i, "train": a.row_id.tolist(), "validation": b.row_id.tolist(),
                          "inner": inner})
        plan[str(k)] = {
            "development": tr.row_id.tolist(), "test": te.row_id.tolist(), "outer": outer,
            "final_tuning": [{"train": tr.iloc[a].row_id.tolist(), "validation": tr.iloc[b].row_id.tolist()}
                             for a, b in splits(tr, 5, seed + 2)],
        }
    verify_plan(table, plan)
    return plan


def verify_plan(table, plan):
    index = table.set_index("row_id")
    master_development = set(plan["5"]["development"])
    master_test = set(plan["5"]["test"])
    master_validation = [set(fold["validation"]) for fold in plan["5"]["outer"]]

    def partition(a, b, eligible):
        assert len(a) == len(set(a)) and len(b) == len(set(b))
        assert set(a).isdisjoint(b) and set(a).union(b) == eligible
        lhs, rhs = index.loc[a], index.loc[b]
        assert set(lhs.group_id).isdisjoint(rhs.group_id)
        assert lhs.label.nunique() == rhs.label.nunique() == 2

    def fold_list(folds, n, rows):
        assert len(folds) == n
        for fold in folds:
            partition(fold["train"], fold["validation"], rows)
        ids = [i for fold in folds for i in fold["validation"]]
        assert len(ids) == len(rows) and set(ids) == rows

    for k in THRESHOLDS:
        selection = plan[str(k)]
        eligible = set(table.loc[table[f"selection_k{k}"].eq("included"), "row_id"])
        a, b = selection["development"], selection["test"]
        partition(a, b, eligible)
        assert set(a) == eligible & master_development and set(b) == eligible & master_test
        fold_list(selection["outer"], 5, set(a))
        fold_list(selection["final_tuning"], 5, set(a))
        for fold, anchor in zip(selection["outer"], master_validation):
            assert set(fold["validation"]) == eligible & anchor
            fold_list(fold["inner"], 3, set(fold["train"]))


def scores(rows):
    y, p = rows.label, rows.prediction
    return {"accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, labels=LABELS, average="macro", zero_division=0)}


def evaluation(rows):
    return {"metrics": scores(rows),
            "class_report": classification_report(
                rows.label, rows.prediction, labels=LABELS, target_names=CLASSES,
                zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(rows.label, rows.prediction, labels=LABELS).tolist()}


def training_environment():
    packages = sorted({(dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()})
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": dict(packages)}


def fit(train, folds, name, seed, workers, smoke):
    start = perf_counter()
    training = train.set_index("row_id", drop=False)
    pos = {row: i for i, row in enumerate(training.row_id)}
    cv = [(np.array([pos[i] for i in fold["train"]]),
           np.array([pos[i] for i in fold["validation"]])) for fold in folds]
    search = GridSearchCV(model_pipeline(name, seed, smoke), GRIDS[name], cv=cv, scoring=SCORER,
                          n_jobs=workers, pre_dispatch="n_jobs", error_score="raise", refit=True)
    search.fit(training[list(FEATURES)], training.label)
    return search, perf_counter() - start


def predict(model, rows, fold=None):
    x = rows[list(FEATURES)]
    answer = rows[["row_id", "objid", "group_id", "label"]].copy()
    answer["prediction"] = model.predict(x)
    if hasattr(model, "predict_proba"):
        answer["spiral_probability"] = model.predict_proba(x)[:, 1]
    else:
        answer["spiral_decision_score"] = model.decision_function(x)
    if fold is not None:
        answer["fold"] = fold
    return answer


def read_predictions(path):
    return pd.read_csv(path, dtype={"objid": str}, float_precision="round_trip")


def record(out, name, inputs, train, valid, folds, seed, workers, smoke, final=False, number=None,
           measurement_policy=None):
    checkpoint = out / f"{name}.json"
    pred_file = out / f"{name}_predictions.csv"
    search_file = out / f"{name}_search.csv"
    model_file = out / f"{name}_model.joblib"
    products = [pred_file, search_file] + ([model_file] if final else [])
    if checkpoint.is_file():
        value = read_json(checkpoint)
        if value["fingerprint"] != inputs or any(
            not p.is_file() or value["sha256"][p.name] != sha256(p) for p in products
        ):
            raise ValueError(f"Checkpoint incompatible or corrupt: {checkpoint}")
        return read_predictions(pred_file), value
    search, seconds = fit(train, folds, name.split("_")[0], seed, workers, smoke)
    result = predict(search.best_estimator_, valid, number)
    write_csv(pred_file, result)
    write_csv(search_file, pd.DataFrame(search.cv_results_))
    if final:
        temp = model_file.with_suffix(".part")
        joblib.dump({"model": search.best_estimator_, "features": FEATURES, "classes": CLASSES,
                     "fingerprint": inputs, "mode": "synthetic" if smoke else "author_catalog",
                     "training_rows": len(train), "seed": seed,
                     "measurement_policy": measurement_policy}, temp, compress=3)
        temp.replace(model_file)
    value = {
        "fingerprint": inputs, "sha256": {p.name: sha256(p) for p in products},
        "train_counts": train.label.value_counts().sort_index().to_dict(),
        "validation_counts": valid.label.value_counts().sort_index().to_dict(),
        "parameters": search.best_params_, "fit_parameters": search.best_estimator_.named_steps["model"].get_params(),
        "seconds": seconds, **evaluation(result),
    }
    fitted = search.best_estimator_.named_steps["model"]
    if isinstance(fitted, MLPClassifier):
        value["optimization"] = {"iterations": fitted.n_iter_, "loss": fitted.loss_,
                                 "loss_curve": fitted.loss_curve_,
                                 "reached_iteration_cap": fitted.n_iter_ >= fitted.max_iter}
    write_json(checkpoint, value)
    return result, value


def validate_saved(out, table):
    manifest = read_json(out / "splits.json")
    verify_plan(table, manifest["thresholds"])
    index = table.set_index("row_id")
    for threshold, split in manifest["thresholds"].items():
        for name in MODELS:
            folder = out / f"k{threshold}" / name
            results = read_json(folder / "results.json")
            for part, ids in (("oof", split["development"]), ("test", split["test"])):
                rows = read_predictions(folder / f"{part}_predictions.csv")
                catalog = index.loc[rows.row_id]
                assert rows.row_id.is_unique and set(rows.row_id) == set(ids)
                for field in ("objid", "group_id", "label"):
                    assert np.array_equal(rows[field], catalog[field])
                assert evaluation(rows) == results[part]
                if part == "oof":
                    for item, fold in zip(results["folds"], split["outer"]):
                        subset = rows.loc[rows.fold.eq(fold["fold"])]
                        assert set(subset.row_id) == set(fold["validation"])
                        assert evaluation(subset)["metrics"] == item["metrics"]
                else:
                    bundle = joblib.load(folder / f"{name}_final_model.joblib")
                    assert bundle["fingerprint"] == manifest["fingerprint"]
                    assert tuple(bundle["features"]) == FEATURES
                    assert tuple(bundle["model"].feature_names_in_) == FEATURES
                    assert np.array_equal(bundle["model"].predict(catalog[list(FEATURES)]), rows.prediction)


def initialize(table, provenance, out, seed, smoke, fixed_plan=None):
    environment = training_environment()
    identity = {"input": provenance, "seed": seed, "smoke": smoke,
                "environment": environment, "grid": GRIDS,
                "svm": {"solver": "LinearSVC primal", "kernel_map": "Nyström RBF",
                        "landmarks": 16 if smoke else 128, "gamma": 1/len(FEATURES)},
                "mlp": {"architecture": [44, 88, 22], "activation": "relu", "solver": "adam",
                        "batch_size": 32 if smoke else 2048, "max_iter": 20 if smoke else 60,
                        "tol": .0003, "n_iter_no_change": 8},
                "code": {name: sha256(ROOT / name) for name in ("common.py", "catalogs.py", "evaluation.py")}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    plan_path = out / "splits.json"
    if plan_path.is_file():
        manifest = read_json(plan_path)
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Output directory belongs to a different run. Use a new directory.")
        plan = manifest["thresholds"]
        verify_plan(table, plan)
    else:
        plan = split_plan(table, seed) if fixed_plan is None else fixed_plan
        verify_plan(table, plan)
        write_json(plan_path, {"fingerprint": fingerprint, "seed": seed, "thresholds": plan})
    if not smoke:
        sample = table.loc[table.selection_k5.eq("included")]
        ellipticals = sample.label.eq(0)
        spirals = sample.label.eq(1)
        assert (ellipticals | spirals).all()
        assert sample.label_provenance.eq("GZ1 Table 2").all()
        assert sample.elliptical.eq(ellipticals.astype(int)).all()
        assert sample.spiral.eq(spirals.astype(int)).all()
        assert sample.uncertain.eq(0).all()
        assert "p_mg" not in sample or not sample.p_mg.ge(.5).any()
        assert all(not col.lower().startswith(("ml2classes", "cnn", "ttype")) for col in table)
        assert np.isfinite(sample[list(FEATURES)]).all().all()
        indexed = table.set_index("row_id")
        checks = {k: {name: indexed.loc[split[name]].label.value_counts().sort_index().to_dict()
                      for name in ("development", "test")} for k, split in plan.items()}
        preflight = {"features": FEATURES, "exact_id_join": provenance["exact_id_join"],
                     "groups": provenance["grouping"],
                     "split_checks": "Group separation asserted at every level and across all K cuts",
                     "class_counts": checks}
        write_json(out / "preflight.json", preflight)
        print(json.dumps(preflight, indent=2), flush=True)
    write_json(out / "provenance.json", identity)
    (out / "environment.txt").write_text("\n".join(f"{p}=={v}" for p, v in environment["packages"].items())+"\n")
    selection_path = out / "sample_manifest.csv"
    if not selection_path.is_file():
        selection = table.drop(columns=list(FEATURES)).copy()
        suffix = plan["5"]
        selection["partition"] = np.select([selection.row_id.isin(suffix["development"]),
                                            selection.row_id.isin(suffix["test"])],
                                           ["development", "test"], default="excluded")
        selection["outer_validation_fold"] = 0
        for fold in suffix["outer"]:
            selection.loc[selection.row_id.isin(fold["validation"]), "outer_validation_fold"] = fold["fold"]
        write_csv(selection_path, selection)
        write_json(out / "manifest_hash.json", {"sha256": sha256(selection_path)})
    if read_json(out / "manifest_hash.json")["sha256"] != sha256(selection_path):
        raise ValueError("Sample manifest checksum mismatch")
    return plan, fingerprint


def run(table, provenance, out, seed=42, workers=2, smoke=False, thresholds=THRESHOLDS,
        fixed_plan=None):
    with (out / ".initialize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan, fingerprint = initialize(table, provenance, out, seed, smoke, fixed_plan)
    index = table.set_index("row_id", drop=False)
    for k in thresholds:
        spec = plan[str(k)]
        for name in MODELS:
            folder = out / f"k{k}" / name
            folder.mkdir(parents=True, exist_ok=True)
            predictions, fold_results = [], []
            for fold in spec["outer"]:
                i = fold["fold"]
                prediction, info = record(folder, f"{name}_outer{i}", fingerprint,
                    index.loc[fold["train"]], index.loc[fold["validation"]], fold["inner"],
                    seed, workers, smoke, number=i, measurement_policy=provenance.get("measurement_policy"))
                predictions.append(prediction)
                fold_results.append({"fold": i, **info})
                print(f"K >= {k} {name} outer {i}/5 saved", flush=True)
            oof = pd.concat(predictions, ignore_index=True).sort_values("row_id")
            held, final_info = record(folder, f"{name}_final", fingerprint, index.loc[spec["development"]],
                index.loc[spec["test"]], spec["final_tuning"], seed, workers, smoke, final=True,
                measurement_policy=provenance.get("measurement_policy"))
            write_csv(folder / "oof_predictions.csv", oof)
            write_csv(folder / "test_predictions.csv", held)
            stats = {}
            for key in scores(oof):
                values = [f["metrics"][key] for f in fold_results]
                stats[key] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
            write_json(folder / "results.json", {"folds": fold_results, "outer_summary": stats,
                       "oof": evaluation(oof), "test": evaluation(held), "final_fit": final_info})
            print(f"K >= {k} {name} held-out artifacts saved", flush=True)
    if set(thresholds) == set(THRESHOLDS):
        validate_saved(out, table)
