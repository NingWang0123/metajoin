"""MetaJoin demo: the zero-LLM signal router picks ONE tool per column pair and
returns the matches — deterministically, with no endpoint.

Two column pairs that should route differently:
  * an EXACT pair (identical keys)               -> equi-join
  * a TRANSFORM pair (name <-> first.last login) -> MNN Join (transform)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from metajoin import metajoin

# --- Pair 1: exact keys (should route to equi) ---
src1 = pd.DataFrame({"sku": ["A-100", "A-101", "A-102", "A-103", "A-104"]})
tgt1 = pd.DataFrame({"sku": ["A-102", "A-100", "A-104", "A-101", "A-103"]})

# --- Pair 2: name <-> login transform (should route to MNN Join) ---
src2 = pd.DataFrame({"name":  ["John Smith", "Jane Doe", "Alan Turing",
                               "Grace Hopper", "Ada Lovelace"]})
tgt2 = pd.DataFrame({"login": ["jane.doe", "john.smith", "ada.lovelace",
                               "grace.hopper", "alan.turing"]})

for tag, ds, sc, dt, tc in [
    ("exact     ", src1, "sku",  tgt1, "sku"),
    ("transform ", src2, "name", tgt2, "login"),
]:
    pairs, trace = metajoin(ds, sc, dt, tc, return_trace=True)
    print(f"[{tag}] routed -> {trace['tool']:<10s} "
          f"matches={len(pairs)} coverage={trace['coverage']} bijectivity={trace['bijectivity']}")
    for a, b in pairs[:5]:
        print(f"           {a!r:<18} -> {b!r}")
    print()
