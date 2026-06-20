"""Selector demo: rank every (source col, target col) pair and print the top
pick. Two tiny inline tables — one column is an obvious name<->email transform
join, the rest is noise the selector should reject."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from selector import rank_all_pairs

# Source: people with a full name + an unrelated numeric headcount column.
df_src = pd.DataFrame({
    "full_name":  ["John Smith", "Jane Doe", "Alan Turing", "Grace Hopper", "Ada Lovelace"],
    "team_size":  ["12", "7", "31", "4", "19"],
})

# Target: accounts keyed by an email whose local-part encodes the name, plus a
# numeric id column that must NOT be chosen.
df_tgt = pd.DataFrame({
    "account_id": ["1001", "1002", "1003", "1004", "1005"],
    "email":      ["john.smith@corp.com", "jane.doe@corp.com", "alan.turing@corp.com",
                   "grace.hopper@corp.com", "ada.lovelace@corp.com"],
})

ranked = rank_all_pairs(df_src, df_tgt)

_TIERS = {0: "HYBRID", 1: "LEXICAL", 2: "SEMANTIC_RESCUE", 3: "NONE"}
print("Ranked column pairs (best first):")
for ca, cb, ev, mnn, lemb, tier, eff in ranked[:5]:
    print(f"  {ca:>12s} -> {cb:<12s}  tier={_TIERS[tier]:<15s} ev={ev} mnn={mnn} local_emb={lemb} eff={eff}")

top = ranked[0]
print(f"\nTOP PICK: {top[0]} -> {top[1]}  (tier={_TIERS[top[5]]}, effective_anchors={top[6]})")
