# MetaJoin — label-free join components

Three composable, label-free components for joining two tables on a string
column, with no training data and no required LLM. Each is a single flat module
at the repo root.

1. **Selector** (`selector.py`) — a threshold-free, certificate-tier column-pair
   selector. Given two DataFrames it ranks every (source col, target col) pair
   by a logical certificate (HYBRID > LEXICAL > SEMANTIC_RESCUE > NONE) built
   from reciprocal mutual-nearest-neighbour anchors and distinct-anchor
   structure, returning the most joinable pair.

2. **MNN Join** (`mnn_join.py`) — a label-free CPU transform-join engine. It
   searches a catalogue of string-rewrite operators, scores candidate
   alignments with the **idfcos** similarity metric, selects the chain that
   maximises **mutual-NN coverage** (`cov_mnn`), and runs a **self-terminating
   depth-PEAK search** (try 1 then 2 transform steps; keep the deeper chain only
   if it raises mutual-NN coverage). No labels, no LLM, no GPU. (The transform
   operators it searches over are defined inline in this module — they are woven
   through the search and execution paths.)

3. **MetaJoin** (`metajoin.py`) — a label-free, cost-aware router that picks
   ONE join tool per column pair from truth-free signals: equi-join,
   MNN Join (transform), embedding (SBERT top-1 + similarity cut), or entity
   matching. It accepts each tool's output on a **class-appropriate witness** —
   model-free exact-overlap coverage for transforms, sampled pair-judging
   otherwise — and escalates only when that evidence is insufficient.
   **Routing is signal-driven either way; the LLM never drives it.** The
   released default is the **signal-only policy**: it runs deterministically
   with no endpoint, omitting the two LLM-based steps. Enabling the LLM mode
   (off by default here) adds a **content probe** (re-reads raw values when the
   column signals are ambiguous) and **pair-judged repair** (arbitration between
   disagreeing specialists, plus entity-side reciprocity escalation with an
   over-merge veto) — the steps that recover the hardest cases. Signals
   alone and an LLM alone each fall short; this combination is the full MetaJoin.

A shared embedding helper (`_embed.py`, MiniLM via sentence-transformers) backs
the selector's local-embedding tier and MetaJoin's embedding tool; `_util.py`
holds the two small engine utilities kept on the run path.

## Install

```bash
pip install -r requirements.txt
```

Core functionality (selector, MNN Join, signal routing) needs only
`numpy / pandas / scipy / scikit-learn`. `sentence-transformers` + `torch`
enable the embedding paths; `openai` is only needed for MetaJoin's optional LLM
mode.

> **First embedding use downloads the MiniLM weights**
> (`sentence-transformers/all-MiniLM-L6-v2`, ~80 MB). Set `HF_HUB_OFFLINE=1` to
> force offline; if the weights are not cached, the embedding tiers degrade
> gracefully (the selector falls back to its lexical certificate, and the
> embedding tool is simply not selected).

## Examples

```bash
python examples/run_selector.py    # rank column pairs, print the top pick
python examples/run_mnn_join.py    # transform-join two value columns, print matches
python examples/run_metajoin.py    # zero-LLM router over a couple of column pairs
```

## Quick API

```python
import pandas as pd
from selector import rank_all_pairs
from mnn_join import mnn_join, mnn_join_tables
from metajoin import metajoin

# 1. selector
ranked = rank_all_pairs(df_src, df_tgt)          # [(col_a, col_b, ev, mnn, lemb, tier, eff), ...]
best_src, best_tgt = ranked[0][0], ranked[0][1]

# 2. MNN Join on two value lists (or two tables)
matches = mnn_join(["John Smith", "Jane Doe"], ["jsmith", "jdoe"])
matches = mnn_join_tables(df_src, "name", df_tgt, "login")

# 3. MetaJoin zero-LLM router over a chosen column pair
pairs, trace = metajoin(df_src, "name", df_tgt, "login", return_trace=True)
print(trace["tool"], pairs)
```

Run from the repo root so `import mnn_join` / `import selector` / `import
metajoin` resolve (the modules are flat at the top level; the examples add the
repo root to `sys.path` for you).

### MetaJoin LLM mode

The released router defaults to the signal-only policy for zero-dependency
runnability. Enabling the LLM mode (`METAJOIN_LLM=1` + an OpenAI-compatible
endpoint via `VLLM_ENDPOINT`/`CTRL_ENDPOINT`) turns on the content probe and the
pair-judged repair — the full MetaJoin. Routing stays signal-driven in both modes;
with no endpoint those two steps are skipped and the router runs deterministically.
