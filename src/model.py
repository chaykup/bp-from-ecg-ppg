"""Build and train your model here. Run from the repo root: python src/model.py

X_train, X_test: preprocessed features (pandas DataFrames, 11 columns, already cleaned, imputed and scaled)
y_train, y_test: targets, columns "sbp" and "dbp" in mmHg
groups_train:    record_id of each training row; pass groups=groups_train with GroupKFold when cross-validating
"""
from pipeline import load_train_test

X_train, X_test, y_train, y_test, groups_train = load_train_test()
