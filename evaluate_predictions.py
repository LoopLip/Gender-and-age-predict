import argparse
from pathlib import Path
import pandas as pd
import json


def compute_metrics(results_csv: Path, meta_csv: Path):
    res = pd.read_csv(results_csv)
    meta = pd.read_csv(meta_csv)

    # normalize filenames (remove extensions in meta if needed)
    if "file_name" in meta.columns:
        meta = meta.rename(columns={"file_name": "filename"})
    if "filename" not in meta.columns:
        # try common alternatives
        if "img_paths" in meta.columns:
            meta["filename"] = meta["img_paths"].apply(lambda p: Path(p).name)
        elif "image" in meta.columns:
            meta["filename"] = meta["image"].apply(lambda p: Path(p).name)

    # keep only rows where predictions exist
    merged = pd.merge(res, meta, on="filename", how="inner")

    metrics = {}
    # Age MAE (ignore empty predictions)
    merged_age = merged[merged["age"].notna() & (merged["age"] != "")]
    if not merged_age.empty and "ages" in merged_age.columns:
        merged_age["age_gt"] = merged_age["ages"].astype(float)
        merged_age["age_pred"] = merged_age["age"].astype(float)
        metrics["mae_age"] = float(
            (merged_age["age_gt"] - merged_age["age_pred"]).abs().mean()
        )

    # Gender accuracy if available
    if "genders" in merged.columns and merged["gender"].notna().any():
        # ground truth genders stored as 0/1 (confirm mapping: 0==male,1==female)
        merged_gender = merged[merged["gender"].notna() & merged["genders"].notna()]
        if not merged_gender.empty:
            merged_gender["gender_gt"] = merged_gender["genders"].astype(int)
            # predictions are 'M'/'F'
            merged_gender["gender_pred"] = merged_gender["gender"].map({"M": 0, "F": 1})
            metrics["gender_accuracy"] = float(
                (merged_gender["gender_gt"] == merged_gender["gender_pred"]).mean()
            )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions.csv against meta/<db>.csv"
    )
    parser.add_argument("--results", type=str, default="results.csv")
    parser.add_argument(
        "--db",
        type=str,
        default="imdb",
        help="database used to create meta csv in meta/<db>.csv",
    )
    parser.add_argument("--output", type=str, default="metrics.json")
    args = parser.parse_args()

    results_csv = Path(args.results)
    meta_csv = Path("meta").joinpath(f"{args.db}.csv")

    if not results_csv.exists():
        raise SystemExit(f"Results file not found: {results_csv}")
    if not meta_csv.exists():
        raise SystemExit(f"Meta file not found: {meta_csv}")

    metrics = compute_metrics(results_csv, meta_csv)
    print("Metrics:", metrics)
    Path(args.output).write_text(json.dumps(metrics, indent=2))
    print(f"Metrics written to {args.output}")
