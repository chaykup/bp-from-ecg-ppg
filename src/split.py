"""Make the shared train/test split and save it as data/processed/split.csv.

Usage (from the repo root, after build_table.py):
    python src/split.py

Splits by record, never by window: windows from one record are near-copies, so a random window split would put the
same recording in train and test and inflate every score. The split is saved as one row per record_id (not row
indices), so it stays valid if the table is rebuilt in a different row order. split.csv is committed so every model
trains and tests on identical data. pipeline.load_train_test() reads it.
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config
from pipeline import SPLIT, TABLE, TARGETS


def make_split(df: pd.DataFrame) -> pd.DataFrame:
    """One row per record_id with split = "train" or "test", grouped by record_id."""
    gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df["record_id"]))
    split = pd.concat([
        pd.DataFrame({"record_id": df["record_id"].iloc[train_idx].unique(), "split": "train"}),
        pd.DataFrame({"record_id": df["record_id"].iloc[test_idx].unique(), "split": "test"}),
    ])
    shared = set(split.loc[split["split"] == "train", "record_id"]) & set(split.loc[split["split"] == "test", "record_id"])
    assert not shared, f"{len(shared)} records in both train and test"
    return split.sort_values("record_id", key=lambda s: s.str.split("_").map(lambda p: (int(p[1]), int(p[3])))).reset_index(drop=True)


def main():
    df = pd.read_parquet(TABLE)
    split = make_split(df)
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(SPLIT, index=False)

    df = df.merge(split, on="record_id")
    print(f"Wrote {SPLIT.relative_to(SPLIT.parents[2])}")
    print(f"{'':6s} {'records':>8s} {'windows':>9s} {'% windows':>10s}   SBP mean/sd   DBP mean/sd")
    for name, g in df.groupby("split", sort=False):
        print(f"{name:6s} {g['record_id'].nunique():8d} {len(g):9d} {len(g) / len(df):10.1%}   "
              + "   ".join(f"{g[t].mean():5.1f} / {g[t].std():4.1f}" for t in TARGETS))
    print("Records in both sets: 0")
    print("\nWindows per Part:\n", pd.crosstab(df["part"], df["split"], normalize="index").round(3).to_string())


if __name__ == "__main__":
    main()
