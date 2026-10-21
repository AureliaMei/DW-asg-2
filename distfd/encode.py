"""Load a CSV and dictionary-encode every column to integer codes."""
import numpy as np
import pandas as pd


def read_table(path):
    """Read a CSV as strings. Empty cells stay as "" so that null = null, as in TANE."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    empty = [c for c in df.columns if (df[c] == "").all()]
    return df.drop(columns=empty)


def encode(df):
    """Return an (n_rows, n_cols) int32 array of per-column dictionary codes."""
    if df.shape[1] == 0:
        return np.zeros((len(df), 0), dtype=np.int32)
    return np.column_stack([pd.factorize(df[c])[0] for c in df.columns]).astype(np.int32)


def load(path):
    df = read_table(path)
    return list(df.columns), encode(df), df
