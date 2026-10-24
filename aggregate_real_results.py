import json, glob, os
from pathlib import Path
import pandas as pd

root=Path("downloaded")
out=Path("report")
out.mkdir(exist_ok=True)
records=[]
seed_tables=[]
audits=[]
for f in root.rglob("metrics.json"):
    try:
        m=json.loads(f.read_text(encoding="utf-8"))
        records.append({
            "model":m.get("model"),"seed":m.get("seed"),
            "best_epoch":m.get("best_epoch"),
            "test_accuracy":m.get("accuracy"),
            "test_macro_precision":m.get("macro_precision"),
            "test_macro_recall":m.get("macro_recall"),
            "test_macro_f1":m.get("macro_f1"),
            "latency_mean_ms":(m.get("inference_latency") or {}).get("mean_ms"),
            "latency_std_ms":(m.get("inference_latency") or {}).get("std_ms"),
            "source_file":str(f)
        })
    except Exception as e:
        print("METRICS_READ_ERROR",str(f),repr(e))
for f in root.rglob("model_seed_metrics.csv"):
    try:
        df=pd.read_csv(f)
        seed_tables.append(df)
    except Exception as e:
        print("CSV_READ_ERROR",str(f),repr(e))
for f in root.rglob("dataset_audit.json"):
    try:
        audits.append(json.loads(f.read_text(encoding="utf-8")))
    except Exception as e:
        print("AUDIT_READ_ERROR",str(f),repr(e))

if seed_tables:
    all_seed=pd.concat(seed_tables,ignore_index=True).drop_duplicates()
else:
    all_seed=pd.DataFrame()
if not all_seed.empty:
    all_seed.to_csv(out/"all_model_seed_metrics.csv",index=False)
    metric_cols=[c for c in all_seed.columns if c.startswith("test_") and c!="test_n"]
    if not metric_cols:
        metric_cols=[c for c in all_seed.columns if c.startswith("test_")]
    groupcol="model" if "model" in all_seed.columns else all_seed.columns[0]
    summary=all_seed.groupby(groupcol)[metric_cols].agg(["mean","std"]).reset_index()
    summary.columns=["_".join(str(x) for x in col if x).rstrip("_") if isinstance(col,tuple) else str(col) for col in summary.columns]
    summary.to_csv(out/"mean_std_summary.csv",index=False)
if records:
    pd.DataFrame(records).to_csv(out/"metrics_from_json.csv",index=False)
if audits:
    (out/"dataset_audit.json").write_text(json.dumps(audits[0],ensure_ascii=False,indent=2),encoding="utf-8")

# Export and print the actual confusion matrices, training histories, and error samples.
confusion_summary = []
error_summary = []
history_summary = []
for f in root.rglob("confusion_matrix.csv"):
    try:
        df = pd.read_csv(f, index_col=0)
        confusion_summary.append((str(f), df))
    except Exception as e:
        print("CONFUSION_READ_ERROR", str(f), repr(e))
for f in root.rglob("misclassified_samples.csv"):
    try:
        df = pd.read_csv(f)
        error_summary.append((str(f), df))
    except Exception as e:
        print("ERROR_SAMPLES_READ_ERROR", str(f), repr(e))
for f in root.rglob("training_history.csv"):
    try:
        df = pd.read_csv(f)
        history_summary.append((str(f), df))
    except Exception as e:
        print("HISTORY_READ_ERROR", str(f), repr(e))
with (out/"confusion_matrices.txt").open("w",encoding="utf-8") as fp:
    for name, df in confusion_summary:
        fp.write("\\n### "+name+"\\n"+df.to_string()+"\\n")
with (out/"misclassification_summary.txt").open("w",encoding="utf-8") as fp:
    for name, df in error_summary:
        fp.write("\\n### "+name+" | errors="+str(len(df))+"\\n")
        fp.write(df.head(12).to_string(index=False)+"\\n")
with (out/"training_history_summary.txt").open("w",encoding="utf-8") as fp:
    for name, df in history_summary:
        fp.write("\\n### "+name+"\\n"+df.to_string(index=False)+"\\n")
if audits:
    a=audits[0]
    print("=== DATASET AUDIT SUMMARY ===")
    print("source_rows:",a.get("source_rows"),"valid_before_dedup:",a.get("decoded_valid_rows_before_dedup"),
          "duplicates_removed:",a.get("exact_duplicate_rows_removed"),"final_unique_images:",a.get("final_unique_images"))
    print("split_counts:",json.dumps(a.get("split_counts",{}),ensure_ascii=False))
    print("class_counts_by_split:",json.dumps(a.get("class_counts_by_split",{}),ensure_ascii=False))
print("confusion_matrix_files:",len(confusion_summary))
print("misclassification_files:",len(error_summary))
for name,df in confusion_summary:
    print("=== CONFUSION MATRIX:",name,"===")
    print(df.to_string())
for name,df in error_summary:
    print("=== ERROR SUMMARY:",name,"count=",len(df),"===")
    print(df.head(5).to_string(index=False))

print("=== VALIDATION SUMMARY ===")
print("metrics_json_count:",len(records))
print("seed_csv_count:",len(seed_tables))
print("dataset_audit_count:",len(audits))
if not all_seed.empty:
    print("=== ALL SEED METRICS ===")
    print(all_seed.to_string(index=False))
if not records:
    raise SystemExit("No metrics.json files were downloaded; do not treat training as validated.")
if all_seed.empty:
    print("WARNING: no model_seed_metrics.csv; mean/std report unavailable.")

# Trigger artifact aggregation after fixing GitHub Actions expression interpolation.


# Generate a compact, auditable comparison table from the per-seed test results.
if not all_seed.empty:
    required_models = {"resnet18", "mobilenet_v3_small", "efficientnet_v2_s", "efficientnet_v2_s_sk"}
    if "model" not in all_seed.columns or "seed" not in all_seed.columns:
        raise SystemExit("Missing model/seed columns; cannot validate repeated experiments.")
    observed_models = set(all_seed["model"].astype(str))
    if observed_models != required_models:
        raise SystemExit(f"Expected all four models; found {sorted(observed_models)}")
    counts = all_seed.groupby("model")["seed"].nunique().to_dict()
    if any(counts.get(m, 0) != 3 for m in required_models):
        raise SystemExit(f"Expected three distinct seeds per model; found {counts}")
    metric_names = [
        ("test_accuracy", "Accuracy"),
        ("test_macro_precision", "Macro Precision"),
        ("test_macro_recall", "Macro Recall"),
        ("test_macro_f1", "Macro F1"),
    ]
    missing = [col for col, _ in metric_names if col not in all_seed.columns]
    if missing:
        raise SystemExit(f"Missing required test metrics: {missing}")
    rows = ["# Real-data model comparison", "",
            "Values are mean ± sample standard deviation across seeds 42, 43, and 44. Metrics are held-out test-set scores, not validation scores.", "",
            "| Model | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Mean latency (ms) |",
            "|---|---:|---:|---:|---:|---:|"]
    for model in sorted(required_models):
        subset = all_seed[all_seed["model"].astype(str) == model]
        vals = []
        for col, _ in metric_names:
            mean = subset[col].astype(float).mean() * 100
            std = subset[col].astype(float).std(ddof=1) * 100
            vals.append(f"{mean:.2f} ± {std:.2f}")
        latency = (f"{subset['latency_mean_ms'].astype(float).mean():.2f}"
                   if "latency_mean_ms" in subset.columns else "not available")
        rows.append(f"| {model} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {latency} |")
    (out / "model_comparison.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("REPEATED_EXPERIMENT_COUNTS=" + json.dumps(counts, ensure_ascii=False))
    print("COMPARISON_REPORT=" + str(out / "model_comparison.md"))

# Do not mark the report as valid unless confusion matrices, training histories,
# and misclassification lists are available for all 12 model/seed runs.
expected_runs = 4 * 3
if records and len(all_seed) != expected_runs:
    raise SystemExit(f"Expected {expected_runs} model-seed rows, found {len(all_seed)}")
if records and (len(confusion_summary) < expected_runs or len(history_summary) < expected_runs or len(error_summary) < expected_runs):
    raise SystemExit(
        "Incomplete diagnostic artifacts: "
        f"confusion_matrices={len(confusion_summary)}, "
        f"training_histories={len(history_summary)}, "
        f"misclassified_lists={len(error_summary)}; expected at least {expected_runs} each."
    )


# Quantify error patterns from the actual held-out test predictions.
error_pattern_rows = []
for name, df in error_summary:
    if not {"true_label", "predicted_label"}.issubset(df.columns):
        continue
    normalized_name = name.replace("\\\\", "/")
    model = next((m for m in ["resnet18", "mobilenet_v3_small", "efficientnet_v2_s", "efficientnet_v2_s_sk"] if f"/{m}/" in normalized_name), "unknown")
    if df.empty:
        continue
    grouped = df.groupby(["true_label", "predicted_label"], dropna=False)
    for (true_label, predicted_label), part in grouped:
        error_pattern_rows.append({
            "model": model,
            "source_file": name,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": int(len(part)),
            "mean_confidence": float(part["confidence"].mean()) if "confidence" in part.columns else None,
        })
if error_pattern_rows:
    patterns = pd.DataFrame(error_pattern_rows)
    patterns.to_csv(out / "error_pattern_summary.csv", index=False)
    top_patterns = (patterns.groupby(["model", "true_label", "predicted_label"], as_index=False)
                    .agg(error_count=("count", "sum"), mean_confidence=("mean_confidence", "mean"))
                    .sort_values(["model", "error_count"], ascending=[True, False]))
    top_patterns.to_csv(out / "top_misclassification_pairs.csv", index=False)
    print("ERROR_PATTERN_ROWS=" + str(len(patterns)))
