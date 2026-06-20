"""MNN Join demo: label-free transform-join on a single column pair.

The source has full names; the target has lowercased "first.last" logins. There
is no exact overlap — MNN Join discovers the string transform (lowercase + join
on a delimiter) and aligns the rows, with NO labels and NO LLM.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from mnn_join import mnn_join, mnn_join_tables

src_names  = ["John Smith", "Jane Doe", "Alan Turing", "Grace Hopper", "Ada Lovelace"]
tgt_logins = ["jane.doe", "john.smith", "ada.lovelace", "grace.hopper", "alan.turing"]

print("Source values:", src_names)
print("Target values:", tgt_logins)

# (a) value-list API
matches = mnn_join(src_names, tgt_logins)
print(f"\nmnn_join found {len(matches)} matches:")
for a, b in sorted(matches):
    print(f"  {a!r:<18} -> {b!r}")

# (b) DataFrame API (same result, different entry point)
df_src = pd.DataFrame({"name": src_names})
df_tgt = pd.DataFrame({"login": tgt_logins})
matches2 = mnn_join_tables(df_src, "name", df_tgt, "login")
print(f"\nmnn_join_tables found {len(matches2)} matches.")
