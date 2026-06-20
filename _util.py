"""Minimal engine utilities kept on the run path.

The MNN-Join engine's column-pair stage needs exactly two helpers from the
former internal utility modules:

  * ``DataLoader`` — wraps the input DataFrames and exposes ``columns`` /
    ``column_pairs`` / ``tables_with_names`` (the engine builds its candidate
    cross-table column pairs from this).
  * ``filter_pairs_by_threshold`` — drop column pairs below a similarity floor.

Both are pure-Python / pandas; the heavier clustering and q-gram/hashing code
that used to live alongside them (matplotlib / scipy / mmh3 / bitarray) is not
on the released run path and has been dropped.
"""
from __future__ import annotations


def filter_pairs_by_threshold(similarity_dict, threshold):
    """Keep only pairs whose similarity score is >= ``threshold``."""
    return {pair: score for pair, score in similarity_dict.items()
            if score >= threshold}


class DataLoader:
    """Load and pair columns across a list of DataFrames.

    Parameters
    ----------
    tables : list[pandas.DataFrame]
    table_names : list[str], optional
        Defaults to ``Table_0, Table_1, ...``.
    """

    def __init__(self, tables, table_names=None):
        self.tables = tables
        self.table_names = (table_names if table_names
                            else [f"Table_{i}" for i in range(len(tables))])
        self.tables_with_names = dict(zip(self.table_names, self.tables))
        self.columns = self.extract_columns()
        self.column_pairs = self.extract_column_pairs()

    def extract_columns(self):
        """{(table_name, column_name): [values as str, ...]} for every column."""
        columns = {}
        for table_name, df in zip(self.table_names, self.tables):
            for col in df.columns:
                columns[(table_name, col)] = df[col].astype(str).tolist()
        return columns

    def extract_column_pairs(self):
        """All ((t1, c1), (t2, c2)) pairs across DIFFERENT tables."""
        column_keys = list(self.columns.keys())
        column_pairs = []
        for i in range(len(column_keys)):
            for j in range(i + 1, len(column_keys)):
                col1 = column_keys[i]
                col2 = column_keys[j]
                if col1[0] != col2[0]:
                    column_pairs.append((col1, col2))
        return column_pairs
