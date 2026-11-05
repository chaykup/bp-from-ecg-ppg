"""Ready-to-use train/test data, so every model trains and tests on identical inputs.

Use in a model notebook (run from src/, or add src/ to sys.path):

    from pipeline import load_train_test
    X_train, X_test, y_train, y_test, groups_train = load_train_test()
    model.fit(X_train, y_train)

X is already cleaned, imputed and scaled; y has the two targets, sbp and dbp. To tune, pass the groups so the
cross-validation folds never split a record:

    from sklearn.model_selection import GridSearchCV, GroupKFold
    search = GridSearchCV(model, param_grid, cv=GroupKFold(5), scoring="neg_mean_absolute_error")
    search.fit(X_train, y_train, groups=groups_train)

Preprocessing (settings in config.py):
  1. Drop SDNN (r = 0.98 with RMSSD).
  2. Add pat_missing, 1 where PAT is missing (~1% of windows, which average ~20 mmHg lower SBP).
  3. log1p on RMSSD and clip PIR at PIR_MAX. These are fixed rules; no outlier rows are removed.
  4. Fill missing PAT with the training median.
  5. Standardize every column with training means and SDs (trees don't need it, but it doesn't hurt them).
Steps 4-5 are fitted on the training set only, so nothing from the test set leaks in. They are fitted once on the
whole training set, so CV folds share those statistics; with ~240k training windows the effect is negligible.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import config
from features import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "processed" / "features.parquet"
SPLIT = ROOT / "data" / "processed" / "split.csv"

MODEL_FEATURES = [c for c in FEATURE_COLS if c not in config.DROPPED_FEATURES]
TARGETS = ["sbp", "dbp"]


def load_table() -> pd.DataFrame:
    """The raw feature table joined with the committed train/test assignment (column "split")."""
    df = pd.read_parquet(TABLE)
    df = df.merge(pd.read_csv(SPLIT), on="record_id", how="left", validate="many_to_one")
    assert df["split"].notna().all(), "every record must be in split.csv; rerun split.py after rebuilding the table"
    assert not set(df.loc[df["split"] == "train", "record_id"]) & set(df.loc[df["split"] == "test", "record_id"]), \
        "a record appears in both train and test"
    return df


def fixed_transforms(X: pd.DataFrame) -> pd.DataFrame:
    """Steps 2-3: add pat_missing, log1p the skewed HRV feature, clip PIR. Nothing is learned from data."""
    X = X.copy()
    X["pat_missing"] = X["pat_foot_s"].isna().astype(float)   # PAT foot and peak are always missing together
    for col in config.LOG_FEATURES:
        X[col] = np.log1p(X[col])
    X["pir"] = X["pir"].clip(upper=config.PIR_MAX)
    return X


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Preprocessed X_train, X_test, targets y_train, y_test (sbp, dbp), and the training record_ids for GroupKFold."""
    df = load_table()
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    X_train, X_test = fixed_transforms(train[MODEL_FEATURES]), fixed_transforms(test[MODEL_FEATURES])

    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))

    def prep(X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(scaler.transform(imputer.transform(X)), columns=X.columns, index=X.index)

    return (prep(X_train), prep(X_test),
            train[TARGETS].copy(), test[TARGETS].copy(), train["record_id"].copy())
