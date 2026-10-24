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
