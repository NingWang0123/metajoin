"""MNN Join — label-free CPU transform-join engine + clean public entry.

MNN Join forms a transform-join between two value columns with NO labels and NO
LLM: it searches a catalogue of string-rewrite operators, scores candidate
alignments with the **idfcos** similarity metric (sparse binary-presence IDF
character-n-gram cosine), selects the chain that maximises **mutual-NN
coverage** (``cov_mnn``), and runs a **self-terminating depth-PEAK search** (try
1 transform step and 2 steps; keep the 2-step chain only if it RAISES the
mutual-NN coverage of the selected alignment, otherwise revert to the 1-step
spine). Everything is CPU-only and deterministic.

Public API (defined at the bottom of this module)
--------------------------------------------------
    mnn_join(src_values, tgt_values, max_steps=2)            -> list[(src, tgt)]
    mnn_join_tables(df_src, src_col, df_tgt, tgt_col, max_steps=2) -> list[...]

Operators
---------
The transform operators MNN Join searches over (``OperatorAction`` + the three
operator lists ``all_operators`` / ``all_operators_without_concate_both`` /
``direct_operators_only``, plus their underlying per-value and per-DataFrame
operator functions) are defined inline in THIS module, around the
``OperatorAction`` class. They are woven through the engine (the search,
sampling, and concat-execution paths all call them directly), so they are kept
here rather than extracted into a separate ``operators.py``.
"""
import re
import random
from collections import defaultdict,Counter
import pandas as pd
import numpy as np
import copy
from bisect import bisect_left
from itertools import permutations
from _util import DataLoader, filter_pairs_by_threshold
from sklearn.cluster import KMeans
import logging
import math
import sys
import os
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as _mp

# Make sibling flat modules (_util, _embed) importable when run from the repo
# root. The off-by-default embedding branches below import ``_embed`` lazily.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# --- MNN Join: fixed released configuration ----------------------------------
# fixed released configuration (MNN Join: idfcos similarity + mutual-NN coverage
# selection/threshold, no learned models). The method runs ONE fixed config;
# these were formerly environment toggles and are now hardcoded module constants.
_METRIC           = 'idfcos'   # sparse binary-presence IDF char-n-gram cosine
_USE_JACCARD      = True        # idfcos lives on the Jaccard-chokepoint path
_COVERAGE_SELECT  = True        # rank candidate chains by coverage, not reward
_SELECT_SIGNAL    = 'cov_mnn'   # coverage * mutual-NN bijectivity
_THRESHOLD_SIGNAL = 'covmnn'    # join cut chosen to maximise cov*mnn
_NO_LEARNED       = True        # no trained meta / stop / TSN / JDN models
_SAFE_CONCAT      = True        # transpose-free (pandas-3-safe) row concatenation
# Exploration epsilon for the eps-greedy operator search. STRING on purpose so
# the original truthiness semantics survive ('0.0' is truthy -> the configured
# greedy rate is used, not the fallback). The MetaJoin router overrides it to
# '0.5' by assigning mnn_join._EPS_SWEEP before driving the engine.
_EPS_SWEEP        = '0.0'
# Dominance-admission 2nd pass: OFF for the standalone MNN Join default. The
# MetaJoin router enables it (tau=1.0, floor=0.0) by assigning these constants.
_DOMINANCE_ADMIT          = False
_DOMINANCE_TAU            = 2.0
_DOMINANCE_FLOOR          = 0.05
_DOMINANCE_COVERAGE_GATE  = False
_DOMINANCE_COVERAGE_MAX   = 0.7
# Per-call hooks set directly by the public entry (_run_depth) instead of env:
_STEP1_SRC_COL     = None   # forced source join column for the current call
_STEP1_TGT_COL     = None   # forced target join column for the current call
_LOG_COMBOS        = False  # capture candidate combos for the depth-PEAK selector
_MAX_STEPS_OVERRIDE = None  # per-call transform depth (1 or 2); set by _run_depth

# --- Adaptive parallelism: detect available cores ---
def _detect_ncores():
    """Auto-detect available CPU cores from SLURM or hardware."""
    for var in ('SLURM_CPUS_PER_TASK', 'SLURM_JOB_CPUS_PER_NODE', 'SLURM_CPUS_ON_NODE'):
        val = os.environ.get(var)
        if val:
            return int(val)
    return max(1, _mp.cpu_count())

NCORES = _detect_ncores()
# Flag to disable internal parallelism when already inside a worker process
_INSIDE_WORKER = False
# Flag to use Jaccard instead of ALCS for all similarity computations
USE_JACCARD = _USE_JACCARD
# Flag to use word-level ALCS as primary similarity (char-level fallback when degenerate)
USE_WORD_ALCS = False
# Flag to use pure SBERT cosine similarity instead of ALCS (V0 ablation E2)
USE_COSINE = False


def _greedy_unique_assignment_score(sim_matrix):
    """Greedy 1-to-1 assignment: pick (i, j) cells in descending sim order,
    each src row and each tgt row used at most once. Returns mean of assigned sims.
    Mirrors the AutoJoin Step 1 / char-ALCS unique-MNN scoring philosophy:
    reward 1-to-1 row alignments, NOT many-src-collapsing-to-one-tgt."""
    import numpy as _np
    n_a, n_b = sim_matrix.shape
    if n_a == 0 or n_b == 0:
        return 0.0
    # Flat-sort all cells by descending sim
    flat_i, flat_j = _np.unravel_index(_np.argsort(-sim_matrix.ravel()), sim_matrix.shape)
    used_a = set(); used_b = set()
    total = 0.0; n_assigned = 0
    cap = min(n_a, n_b)
    for i, j in zip(flat_i, flat_j):
        i = int(i); j = int(j)
        if i in used_a or j in used_b:
            continue
        total += float(sim_matrix[i, j])
        used_a.add(i); used_b.add(j)
        n_assigned += 1
        if n_assigned >= cap:
            break
    return total / max(n_assigned, 1)


def _compute_alcs_reward(sim_matrix):
    """Aggregate ALCS reward for Q-learning operator search.
    Default: mean(row_max) — historical behavior, allows many src rows to
    'match' the same tgt row.
    UNIQUE_ALCS_REWARD=1: greedy 1-to-1 assignment — penalizes
    duplicate-tgt collapses, rewards transformations that produce true
    1-to-1 row alignment (mirrors AutoJoin Step 1 unique-1-to-1 idea)."""
    import numpy as _np
    if sim_matrix.size == 0:
        return 0.0
    if False:
        return _greedy_unique_assignment_score(sim_matrix)
    return float(_np.mean(_np.max(sim_matrix, axis=1)))
# v8: per-pair ALCS variant override set by compute_all_pairs_similarity.
# None  = no override
# 1     = char-level (min_len=1)
# 2     = token-level (min_len=2)
# 'fuzzy'   = alcs_fuzzy_single per-cell similarity
# 'auto_fc' = v9 runtime adaptive: compute BOTH alcs_fuzzy and char-ALCS,
#             pick the one with higher mean_max_ALCS per pair (no trained head)
_V8_FORCED_MATCH_MODE = None
# v7_v2: per-pair embedding-blend weight set by compute_all_pairs_similarity.
# None  = no override (JDN/env still applies)
# float = forced w in [0, 0.5]; consumed in the JOIN execute step where
#         sim = (1-w)*ALCS + w*emb_cos_sim. Reset after each pair finishes.
_V72_FORCED_EMB_WEIGHT = None  # v7_v2: per-pair embedding-blend weight
# Cap fuzzy mode by table size — alcs_fuzzy_single is O(token_count^2 * char_alcs)
# per cell and would explode on tables > ~200 rows. Above this, fall back to token.
_V8_FUZZY_MAX_ROWS = 200
# v9: global env knob — when set, every (col_a, col_b) gets the auto fuzzy-vs-char
# selection at sim-matrix time. Independent from v8's trained head.
if False:
    _V8_FORCED_MATCH_MODE = 'auto_fc'
# Flag to suppress verbose prints (173 print statements = major overhead)
QUIET = False
_DEVNULL = open(os.devnull, 'w') if QUIET else None
# Override threshold for oracle sweep experiments (None = use normal KMeans)
_FORCED_THRESHOLD = None

def set_inside_worker(flag=True):
    """Call from worker processes to disable nested parallelism."""
    global _INSIDE_WORKER
    _INSIDE_WORKER = flag

# Learned (torch) models are NOT part of this release: MNN Join is label-free
# and runs with _NO_LEARNED=1 by default. The optional ``learned_models``
# module is intentionally not shipped, so this import fails and ModelRegistry is
# None — every ``if ModelRegistry is not None`` site below then skips cleanly.
try:
    from learned_models import ModelRegistry  # not shipped -> None
except Exception:
    ModelRegistry = None

_NO_UNIQ_REWARD = False
# Ablation env vars for reward components (H/I/J axes).
# Empty/unset = use per-recipe value (no override).
_MIN_POSITIVE_FRACTION_OVR = None
_AGREEMENT_MULTIPLIER_OVR  = None
_SIM_REWARD_PROFILE_OVR    = None  # "short,medium,long"

@dataclass
class RewardConfig:
    """Consolidated reward/heuristic parameters for the Q-learning agent."""
    reward_factor: float = 10000.0
    # F1 ablation: NO_UNIQ_REWARD=1 zeros uniqueness reward at runtime
    reward_uniqueness_factor: float = (0.0 if _NO_UNIQ_REWARD else 10000.0)
    def __post_init__(self):
        # Force-zero even if a per-recipe JSON override set a non-zero value.
        if _NO_UNIQ_REWARD:
            self.reward_uniqueness_factor = 0.0
        # H ablation: override min_positive_fraction
        if _MIN_POSITIVE_FRACTION_OVR is not None:
            try: self.min_positive_fraction = float(_MIN_POSITIVE_FRACTION_OVR)
            except: pass
        # I ablation: override agreement_multiplier
        if _AGREEMENT_MULTIPLIER_OVR is not None:
            try: self.agreement_multiplier = float(_AGREEMENT_MULTIPLIER_OVR)
            except: pass
        # J ablation: override per-length sim_reward profile (comma-separated short,med,long)
        if _SIM_REWARD_PROFILE_OVR is not None:
            try:
                parts = _SIM_REWARD_PROFILE_OVR.split(',')
                if len(parts) == 3:
                    self.sim_reward_short, self.sim_reward_medium, self.sim_reward_long = \
                        float(parts[0]), float(parts[1]), float(parts[2])
            except: pass
    concate_sim_pos_greedy: float = 1.0
    concate_sim_pos_non_greedy: float = 2.0
    concate_negative_uniq_greedy: float = 1.0
    concate_negative_uniq_non_greedy: float = 0.0
    agreement_multiplier: float = 3.0
    greedy_factor: float = 8.0
    non_greedy_factor: float = 1.0
    sim_reward_short: float = 2.0
    sim_reward_medium: float = 1.0
    sim_reward_long: float = 0.8
    alpha_short: float = -0.05
    alpha_medium: float = -0.025
    alpha_long: float = 0.025
    cluster_fractions_3: tuple = (0.35, 0.05, 0.05)
    cluster_fractions_2: tuple = (0.3, 0.05)
    cluster_fractions_1: tuple = (0.3,)
    join_threshold_cap: float = 0.8
    uniqueness_cap_ratio: float = 0.5  # caps negative uniqueness reward to this fraction of similarity gain
    min_positive_fraction: float = 0.0  # minimum positive_fraction to accept (0=continuous, 0.5=majority gate)
    depth: int = 2  # planning depth for the agent
    # JDN params (KMeans + alpha + cap threshold algorithm). Previously
    # hardcoded at k=7, percentile=75, alpha=length-tiered. Now per-recipe.
    # Falls back to v3_new defaults when recipe config doesn't specify.
    jdn_kmeans_k: int = 7
    jdn_percentile: float = 75.0
    jdn_alpha: float = -0.025   # -1 == use length-tiered (legacy fallback)
    # v6: per-cluster embedding blend weight at JDN step.
    # 0.0 = pure Q-learning sim (r4/v5 behavior); 1.0 = pure embedding.
    embed_weight: float = 0.0
    top_k_percent: float = 0.5  # fraction of sim matrix cells to keep (sparsification)
    # Runtime params — learned per recipe, overridden by env vars if set
    exploration_rate: float = 0.1  # epsilon for Q-learning exploration
    max_steps: int = 5  # Q-learning iterations
    sample_cap: int = 50  # max rows for Q-learning sample
    jaccard_n: int = 2  # n-gram size for Jaccard similarity
    # Shrinkage penalty params (learned via AdamW evolutionary optimization)
    shrink_len_threshold: float = 0.36  # penalize if avg length drops below this fraction of original
    shrink_len_penalty: float = 0.40    # multiplier applied to reward
    shrink_uniq_threshold: float = 0.59 # penalize if unique count drops below this fraction
    shrink_uniq_penalty: float = 0.46
    shrink_tok_threshold: float = 0.36  # penalize if avg token count drops below this fraction
    shrink_tok_penalty: float = 0.41
    # Pair ranking params (learned via AdamW evolutionary optimization)
    rrf_k: float = 72.0                # reciprocal rank fusion constant
    adaptive_topk_gap: float = 0.19    # if top1-top3 gap < this * top1, expand candidates
    adaptive_topk_expand: int = 6      # expand to at least this many candidates

    def get_concate_params(self, greedy: bool):
        if greedy:
            return self.concate_sim_pos_greedy, self.concate_negative_uniq_greedy
        return self.concate_sim_pos_non_greedy, self.concate_negative_uniq_non_greedy

    def get_sim_reward_params(self, min_avg_length: float):
        if min_avg_length < 5:
            return self.sim_reward_short, self.alpha_short
        elif min_avg_length < 10:
            return self.sim_reward_medium, self.alpha_medium
        return self.sim_reward_long, self.alpha_long

    def get_greedy_factor(self, greedy: bool):
        return self.greedy_factor if greedy else self.non_greedy_factor


# ---------------------------------------------------------------------------
# Pre-built configs for different join task types
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Hierarchical RewardConfig wrappers
# Level 1: Concat regime (no_concat / one_side_concat / two_side_concat)
# Level 2: Transform family within each regime
# Trained via grid search on synthetic data (train_reward_configs.py)
# ---------------------------------------------------------------------------

TRAINED_CONFIGS = {
    # --- NO CONCAT regime: single-column joins ---
    # Two-stage learned from 24 comprehensive recipes (Stage A ranking + Stage B F1)
    "no_concat__drastic": RewardConfig(
        reward_factor=14267.1, reward_uniqueness_factor=10000.0,
        agreement_multiplier=7.13, uniqueness_cap_ratio=0.488,
        min_positive_fraction=0.0, join_threshold_cap=0.400,
        alpha_short=0.05, alpha_medium=0.025, alpha_long=-0.025,
        depth=2,
    ),
    "no_concat__reorder": RewardConfig(
        reward_factor=7651.7, reward_uniqueness_factor=10000.0,
        agreement_multiplier=7.40, uniqueness_cap_ratio=0.455,
        min_positive_fraction=0.0, join_threshold_cap=0.550,
        alpha_short=-0.017, alpha_medium=-0.008, alpha_long=0.008,
        depth=2,
    ),
    "no_concat__delimiter": RewardConfig(
        reward_factor=7214.3, reward_uniqueness_factor=10000.0,
        agreement_multiplier=7.05, uniqueness_cap_ratio=0.411,
        min_positive_fraction=0.0, join_threshold_cap=0.400,
        alpha_short=-0.083, alpha_medium=-0.042, alpha_long=0.042,
        depth=2,
    ),
    "no_concat__fuzzy": RewardConfig(
        reward_factor=6965.5, reward_uniqueness_factor=10000.0,
        agreement_multiplier=5.88, uniqueness_cap_ratio=0.465,
        min_positive_fraction=0.0, join_threshold_cap=0.850,
        alpha_short=-0.017, alpha_medium=-0.008, alpha_long=0.008,
        depth=1,
    ),
    "one_concat__merge": RewardConfig(
        reward_factor=9172.5, reward_uniqueness_factor=10000.0,
        agreement_multiplier=7.12, uniqueness_cap_ratio=0.469,
        min_positive_fraction=0.0, join_threshold_cap=0.400,
        alpha_short=-0.083, alpha_medium=-0.042, alpha_long=0.042,
        depth=2,
    ),
    "two_concat__multi_col": RewardConfig(
        reward_factor=4543.5, reward_uniqueness_factor=10000.0,
        agreement_multiplier=5.86, uniqueness_cap_ratio=0.453,
        min_positive_fraction=0.0, join_threshold_cap=0.400,
        alpha_short=-0.15, alpha_medium=-0.075, alpha_long=0.075,
        depth=3,
    ),

    "default": RewardConfig(),
}


class PairDiagnostics:
    """
    Stage 0: Pair diagnostics encoder + learned Stage 1 selector.

    For each candidate pair (c_s, c_t), computes a feature vector z_p,
    then uses KNN soft voting over training data to predict:
    - config distribution (not just one config)
    - operator group priors (conditional on config)
    - concat usefulness (from observed marginal benefit)

    Training data loaded from selector_training_data.json.
    """

    def __init__(self):
        self._training_data = None  # lazy loaded

    def _load_training_data(self):
        """Load training records from JSON (lazy, once)."""
        if self._training_data is not None:
            return
        import json
        path = os.path.join(os.path.dirname(__file__), '..', 'out_put_csv', 'selector_training_data.json')
        try:
            with open(path) as f:
                records = json.load(f)
            self._training_features = np.array([r['features'] for r in records])
            self._training_outcomes = [r['config_outcomes'] for r in records]
            # Normalize features for distance computation
            self._feat_mean = np.mean(self._training_features, axis=0)
            self._feat_std = np.maximum(np.std(self._training_features, axis=0), 1e-8)
            self._training_data = records
        except Exception:
            self._training_data = []
            self._training_features = np.empty((0, 12))
            self._training_outcomes = []
            self._feat_mean = np.zeros(12)
            self._feat_std = np.ones(12)

    def compute_features(self, col_a, col_b, n_sample=15):
        """
        Compute diagnostic feature vector z_p for a column pair.

        Returns dict of features describing the pair's structure.
        """
        sample_a = col_a.head(n_sample)
        sample_b = col_b.head(n_sample)

        # Length statistics
        len_a = sample_a.str.len()
        len_b = sample_b.str.len()
        avg_len_a = float(len_a.mean())
        avg_len_b = float(len_b.mean())
        len_ratio = min(avg_len_a, avg_len_b) / max(avg_len_a, avg_len_b) if max(avg_len_a, avg_len_b) > 0 else 1.0

        # Token statistics
        words_a = sample_a.str.split().str.len()
        words_b = sample_b.str.split().str.len()
        avg_words_a = float(words_a.mean())
        avg_words_b = float(words_b.mean())

        # Character type ratios
        total_chars = max(len_a.sum() + len_b.sum(), 1)
        digit_chars = (
            sample_a.str.replace(r'[^0-9]', '', regex=True).str.len().sum() +
            sample_b.str.replace(r'[^0-9]', '', regex=True).str.len().sum()
        )
        digit_frac = float(digit_chars / total_chars)

        # Delimiter structure
        dash_count_a = float(sample_a.str.count('-').mean())
        dash_count_b = float(sample_b.str.count('-').mean())
        dot_count_a = float(sample_a.str.count(r'\.').mean())
        dot_count_b = float(sample_b.str.count(r'\.').mean())

        # Uppercase / initials pattern
        upper_frac_a = float(sample_a.str.count(r'[A-Z]').sum() / max(len_a.sum(), 1))
        upper_frac_b = float(sample_b.str.count(r'[A-Z]').sum() / max(len_b.sum(), 1))

        # ALCS similarity (one call)
        try:
            init_alcs, init_matrix, penalty, lcs_matrix = get_ALCS_matrix(sample_a, sample_b, True)
            row_maxes = np.max(init_matrix, axis=1)
            alcs_mean = float(np.mean(row_maxes))
            alcs_std = float(np.std(row_maxes))
            good_match_frac = float(np.mean(row_maxes > 0.5))
            penalty_ratio = float(penalty / max(n_sample, 1))
        except Exception:
            alcs_mean, alcs_std, good_match_frac, penalty_ratio = 0.5, 0.2, 0.5, 0.5
            init_alcs, init_matrix, penalty, lcs_matrix = 0.5, None, 0.0, None

        return {
            'avg_len_a': avg_len_a, 'avg_len_b': avg_len_b,
            'len_ratio': len_ratio,
            'avg_words_a': avg_words_a, 'avg_words_b': avg_words_b,
            'digit_frac': digit_frac,
            'dash_count': (dash_count_a + dash_count_b) / 2,
            'dot_count': (dot_count_a + dot_count_b) / 2,
            'upper_frac': (upper_frac_a + upper_frac_b) / 2,
            'alcs_mean': alcs_mean, 'alcs_std': alcs_std,
            'good_match_frac': good_match_frac,
            'penalty_ratio': penalty_ratio,
            # Raw ALCS outputs for reuse
            '_init_alcs': init_alcs, '_init_matrix': init_matrix,
            '_penalty': penalty, '_lcs_matrix': lcs_matrix,
        }

    def predict_priors(self, features, n_source_cols=1, n_target_cols=1, k=7):
        """
        Stage 1: Learned KNN-based prediction.

        Uses training data to predict soft priors via distance-weighted
        voting over K nearest training examples.

        Returns dict with:
        - config_scores: {name: weighted_alcs} soft config distribution
        - operator_group_weights: {group: weight} learned from neighbors
        - concat_gate: float in [0, 1] from observed concat benefit
        """
        self._load_training_data()

        config_names = [n for n in TRAINED_CONFIGS if n != "default"]

        # Build query feature vector (same order as training)
        z_p = np.array([
            features['alcs_mean'], features.get('alcs_std', 0.2),
            features['good_match_frac'], features['penalty_ratio'],
            features['len_ratio'],
            (features['avg_len_a'] + features['avg_len_b']) / 2,
            features['avg_words_a'], features['avg_words_b'],
            features['digit_frac'],
            features['dash_count'], features['dot_count'],
            features['upper_frac'],
        ])

        # --- Learned MetaSelector path (replaces KNN when model is available) ---
        if ModelRegistry is not None and not _learned_off('NO_META'):
            _registry = ModelRegistry()
            _meta_model = _registry.load('meta_selector')
            if _meta_model is not None:
                preds = _meta_model.predict(z_p)
                # Remap config_scores to use actual TRAINED_CONFIGS names
                mapped_scores = {}
                for cname in config_names:
                    mapped_scores[cname] = preds['config_scores'].get(cname, 0.0)
                # If all zeros (config names don't match), distribute uniformly
                if sum(mapped_scores.values()) < 1e-6:
                    mapped_scores = {n: 1.0 / len(config_names) for n in config_names}
                # LOG_RECIPE=path → append (top1, top2, all-scores) per call
                _log_path = None
                if _log_path:
                    import json as _json
                    try:
                        with open(_log_path, 'a') as _lf:
                            _lf.write(_json.dumps(mapped_scores) + '\n')
                    except Exception: pass
                return {
                    'config_scores': mapped_scores,
                    'operator_group_weights': preds['operator_group_weights'],
                    'concat_gate': preds['concat_gate'],
                    'match_mode': preds.get('match_mode', 2),
                }

        # Fallback if no training data
        if len(self._training_data) == 0:
            default_scores = {n: 1.0 for n in config_names}
            return {
                'config_scores': default_scores,
                'operator_group_weights': {g: 1.0 for g in ['substring', 'split_reorder', 'delimiter', 'initials', 'pattern_extract']},
                'concat_gate': 0.5,
            }

        # Normalize and find K nearest neighbors (KNN fallback)
        z_norm = (z_p - self._feat_mean) / self._feat_std
        train_norm = (self._training_features - self._feat_mean) / self._feat_std
        dists = np.sqrt(np.sum((train_norm - z_norm) ** 2, axis=1))
        k_actual = min(k, len(dists))
        neighbor_idx = np.argsort(dists)[:k_actual]
        neighbor_dists = dists[neighbor_idx]

        # Distance-based weights (inverse distance, softmax-like)
        weights = 1.0 / (neighbor_dists + 1e-6)
        weights = weights / weights.sum()

        # --- Soft config voting: weighted average ALCS per config ---
        config_scores = {n: 0.0 for n in config_names}
        for i, idx in enumerate(neighbor_idx):
            outcomes = self._training_outcomes[idx]
            for cname in config_names:
                if cname in outcomes:
                    config_scores[cname] += weights[i] * outcomes[cname]['final_alcs']

        # --- Operator group priors from neighbors (conditional on best config) ---
        op_group_counts = {'substring': 0.0, 'split_reorder': 0.0, 'delimiter': 0.0,
                           'initials': 0.0, 'pattern_extract': 0.0, 'concat': 0.0}
        for i, idx in enumerate(neighbor_idx):
            outcomes = self._training_outcomes[idx]
            # Use ops from the best config for this neighbor
            best_c = max(outcomes, key=lambda c: outcomes[c]['final_alcs'])
            for op_group in outcomes[best_c].get('ops_used', []):
                if op_group in op_group_counts:
                    op_group_counts[op_group] += weights[i]

        # Normalize to [0.2, 1.0] range (never fully suppress)
        max_op = max(op_group_counts.values()) if op_group_counts else 1.0
        operator_group_weights = {
            g: max(0.2, min(1.0, v / max(max_op, 1e-6)))
            for g, v in op_group_counts.items() if g != 'concat'
        }

        # --- Concat gate from observed benefit ---
        concat_benefit = 0.0
        for i, idx in enumerate(neighbor_idx):
            outcomes = self._training_outcomes[idx]
            any_concat = any(o.get('concat_used', False) for o in outcomes.values())
            if any_concat:
                concat_benefit += weights[i]
        concat_gate = min(1.0, concat_benefit * 2)  # scale up since concat is rare in training

        return {
            'config_scores': config_scores,
            'operator_group_weights': operator_group_weights,
            'concat_gate': concat_gate,
        }


class RewardConfigSelector:
    """
    Selects RewardConfig using pair diagnostics (Stage 0 + Stage 1).

    Computes pair features, predicts soft priors, and returns the
    best-scoring config along with operator group weights and concat gate.
    """

    def __init__(self, configs=None):
        self.configs = configs or TRAINED_CONFIGS
        self.diagnostics = PairDiagnostics()
        self.last_priors = None
        self._per_recipe_configs = None  # lazy loaded

    def _load_per_recipe_configs(self):
        """Load per-recipe configs from JSON if available."""
        if self._per_recipe_configs is not None:
            return
        import json
        path = os.path.join(os.path.dirname(__file__), '..', 'out_put_csv', 'per_recipe_configs.json')
        try:
            with open(path) as f:
                data = json.load(f)
            self._per_recipe_configs = {}
            for name, params in data.items():
                self._per_recipe_configs[name] = RewardConfig(
                    reward_factor=params['reward_factor'],
                    reward_uniqueness_factor=params['reward_uniqueness_factor'],
                    agreement_multiplier=params['agreement_multiplier'],
                    uniqueness_cap_ratio=params['uniqueness_cap_ratio'],
                    min_positive_fraction=params['min_positive_fraction'],
                    join_threshold_cap=params['join_threshold_cap'],
                    alpha_short=params['alpha_short'],
                    alpha_medium=params['alpha_medium'],
                    alpha_long=params['alpha_long'],
                    depth=params.get('depth', 2),
                    # r4-style per-recipe JDN params (fall back to defaults
                    # when recipe config predates this augmentation).
                    jdn_kmeans_k=int(params.get('jdn_kmeans_k', 7)),
                    jdn_percentile=float(params.get('jdn_percentile', 75.0)),
                    jdn_alpha=float(params.get('jdn_alpha', -1.0)),
                )
        except Exception:
            self._per_recipe_configs = {}

    def select(self, df_a, column_a, df_b, column_b, pairs=None):
        """
        Two-level selection:
        1. KNN from training data → soft vote over per-recipe configs
        2. Fallback to family configs if no per-recipe match
        """
        col_a = df_a[column_a].astype(str) if column_a in df_a.columns else pd.Series(dtype=str)
        col_b = df_b[column_b].astype(str) if column_b in df_b.columns else pd.Series(dtype=str)

        if col_a.empty or col_b.empty:
            return self.configs.get("default", RewardConfig()), "default"

        features = self.diagnostics.compute_features(col_a, col_b)
        priors = self.diagnostics.predict_priors(features)
        self.last_priors = priors

        # Store raw feature vector for use by TransformSearchNet
        self._last_z_p = np.array([
            features['alcs_mean'], features.get('alcs_std', 0.2),
            features['good_match_frac'], features['penalty_ratio'],
            features['len_ratio'],
            (features['avg_len_a'] + features['avg_len_b']) / 2,
            features['avg_words_a'], features['avg_words_b'],
            features['digit_frac'],
            features['dash_count'], features['dot_count'],
            features['upper_frac'],
        ])

        # Weighted K-NN over per-recipe configs: blend top-K nearest recipes
        self._load_per_recipe_configs()
        self.diagnostics._load_training_data()

        if self._per_recipe_configs and self.diagnostics._training_data:
            z_p = np.array([
                features['alcs_mean'], features.get('alcs_std', 0.2),
                features['good_match_frac'], features['penalty_ratio'],
                features['len_ratio'],
                (features['avg_len_a'] + features['avg_len_b']) / 2,
                features['avg_words_a'], features['avg_words_b'],
                features['digit_frac'],
                features['dash_count'], features['dot_count'],
                features['upper_frac'],
            ])
            z_norm = (z_p - self.diagnostics._feat_mean) / self.diagnostics._feat_std
            train_norm = (self.diagnostics._training_features - self.diagnostics._feat_mean) / self.diagnostics._feat_std
            dists = np.sqrt(np.sum((train_norm - z_norm) ** 2, axis=1))

            # KNN: default K=5 with inverse-distance weighted blend.
            # KNN_K overrides K; NO_BLEND=1 forces K=1 (picks
            # the single nearest recipe — no blending). Blending averages
            # away per-recipe precision when nearest is clearly right.
            _K_override = 0
            _no_blend = False
            if _no_blend:
                K = 1
            elif _K_override > 0:
                K = min(_K_override, len(dists))
            else:
                K = min(5, len(dists))
            top_k_idx = np.argsort(dists)[:K]
            top_k_dists = dists[top_k_idx]
            if K == 1:
                weights = np.array([1.0])
            else:
                weights = 1.0 / (top_k_dists + 1e-6)
                weights = weights / weights.sum()

            # Weighted average of config params from top-K recipes
            blend_rf = 0.0; blend_ruf = 0.0; blend_am = 0.0
            blend_ucr = 0.0; blend_jtc = 0.0; blend_alpha = 0.0; blend_depth = 0.0
            blend_jdn_k = 0.0; blend_jdn_pct = 0.0; blend_jdn_alpha = 0.0
            best_name_parts = []
            for i, idx in enumerate(top_k_idx):
                recipe_name = self.diagnostics._training_data[idx].get('recipe', '')
                rc = self._per_recipe_configs.get(recipe_name)
                if rc is None:
                    continue
                w = weights[i]
                blend_rf += w * rc.reward_factor
                blend_ruf += w * rc.reward_uniqueness_factor
                blend_am += w * rc.agreement_multiplier
                blend_ucr += w * rc.uniqueness_cap_ratio
                blend_jtc += w * rc.join_threshold_cap
                blend_alpha += w * rc.alpha_short
                blend_depth += w * rc.depth
                blend_jdn_k += w * float(rc.jdn_kmeans_k)
                blend_jdn_pct += w * rc.jdn_percentile
                blend_jdn_alpha += w * rc.jdn_alpha
                best_name_parts.append(recipe_name)

            blended = RewardConfig(
                reward_factor=blend_rf,
                reward_uniqueness_factor=blend_ruf,
                agreement_multiplier=blend_am,
                uniqueness_cap_ratio=blend_ucr,
                min_positive_fraction=0.0,
                join_threshold_cap=blend_jtc,
                alpha_short=blend_alpha,
                alpha_medium=blend_alpha * 0.5,
                alpha_long=-blend_alpha * 0.5,
                depth=max(1, round(blend_depth)),
                jdn_kmeans_k=max(2, int(round(blend_jdn_k))),
                jdn_percentile=blend_jdn_pct,
                jdn_alpha=blend_jdn_alpha,
            )
            best_name = best_name_parts[0] if best_name_parts else "default"
            return blended, f"blend({best_name})"

        # Fallback: family-level configs
        scores = priors['config_scores']
        best_name = max(scores, key=scores.get) if scores else "default"
        return self.configs.get(best_name, self.configs.get("default", RewardConfig())), best_name

def insert_column(df, position, position_column_name, column_for_insertion):
    """
    Insert a column at 'front' or 'back' relative to position_column.

    Parameters:
        df: pd.DataFrame
        position: str ('front' or 'back')
        position_column: str (relative column name)
        column_for_insertion: dict {new_column_name: new_column_values}
    """

    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    
    if isinstance(column_for_insertion, dict):
        new_column_name = list(column_for_insertion.keys())[0]
        new_column_values = list(column_for_insertion.values())[0]
    
    elif isinstance(column_for_insertion, pd.Series):
        new_column_name = column_for_insertion.name
        new_column_values = column_for_insertion.values

    # Find the index of the position_column
    col_index = df.columns.get_loc(position_column_name)
    
    # Adjust index based on 'front' or 'back'
    insert_position = col_index if position == 'front' else col_index + 1
    
    # Insert the column
    counter = 1
    while new_column_name in df.columns:
        new_column_name = f"{new_column_name}_{counter}"

    df.insert(insert_position, new_column_name, new_column_values)
    return df

# all operators
def concatenate_front(df, col1, col2):
    df_copy = df.copy()

    df_copy = insert_column(df_copy, 'front', col1, col2)
    
    return df_copy


def concatenate_back(df, col1, col2):
    df_copy = df.copy()
    
    df_copy = insert_column(df_copy, 'back', col1, col2)

    return df_copy

def auto_split_by_operator(value):
    split_values = re.split(r'[^a-zA-Z0-9]', str(value))
    return ' '.join(filter(None, split_values))

def substring_operator(value, start=None, end=None):
    return str(value)[start:end]

def remove_second_char(value):
    value = str(value)
    if len(value) < 2:
        return value  
    return value[0] + value[2:]  

def remove_second_char_backwards(value):
    value = str(value)
    if len(value) < 2:
        return value  
    return value[0:-2] + value[-1]  

def get_pairs(clusters, table_name_a, table_name_b):
    """
    Filters and returns pairs of columns from the specified tables along with their similarity scores.

    Parameters:
    - clusters (dict): A dictionary where keys are tuples of column pairs and values are similarity scores.
    - table_name_a (str): The name of the first table to filter.
    - table_name_b (str): The name of the second table to filter.

    Returns:
    - dict: A dictionary of filtered column pairs with their corresponding similarity scores.
    """
    # Extract all cluster pairs
    clusters_pairs = clusters.values()
    
    # Initialize a dictionary to hold the selected pairs
    selected_clusters = []
    
    # Iterate through each pair in the clusters
    for cluster_pairs in clusters_pairs:
      for pair in cluster_pairs:
        # Each pair consists of two tuples: (table, column)
        (table1, column1), (table2, column2) = pair
        
        if (table1 == table_name_a and table2 == table_name_b):
          selected_clusters.append((column1,column2))
        elif (table1 == table_name_b and table2 == table_name_a):
            selected_clusters.append((column2,column1))
    
    return selected_clusters


def concat_pairs_front(df_a,df_b,insert_cola,insert_colb,space_col_a,space_col_b):
  df_a_copy = df_a.copy()
  df_b_copy = df_b.copy()

  df_a_copy_concated = concatenate_front(df_a_copy, space_col_a, insert_cola)
  df_b_copy_concated = concatenate_front(df_b_copy, space_col_b, insert_colb)

  return df_a_copy_concated,df_b_copy_concated

def concat_pairs_back(df_a,df_b,insert_cola,insert_colb,space_col_a,space_col_b):
  df_a_copy = df_a.copy()
  df_b_copy = df_b.copy()

  df_a_copy_concated = concatenate_back(df_a_copy, space_col_a, insert_cola)
  df_b_copy_concated = concatenate_back(df_b_copy, space_col_b, insert_colb)

  return df_a_copy_concated,df_b_copy_concated


def SelectK_for_separated_reverse(value, reversed_k):
    value = str(value)
    words = value.split()
    words_length = len(words)
    if words_length != 1:
      if words_length == 0:
          return '' 
      elif reversed_k >= words_length:
          return words[-1]
      result = ' '.join(words[reversed_k:words_length])
    else:
      value_length = len(value)
      if reversed_k >= value_length:
          return value[-1]
      result = ''.join(value[value_length - reversed_k:])
    return result

def SelectK_for_separated(value, k):
    value = str(value)
    words = value.split()
    words_length = len(words)
    if words_length != 1:
        if words_length == 0:
            return '' 
        elif k >= words_length:
            return words[0] 
        result = ' '.join(words[:words_length - k])
    else:
        value_length = len(value)
        if k >= value_length:
            return value[0] 
        result = ''.join(value[:value_length - k])

    return result

def shift_1_word_forward(value, shift_val=1):
    value = str(value)
    words = value.split()
    words_length = len(words)
    
    if words_length == 0:
        return '' 
    elif words_length == 1:
        return value  
    else:
        shifted_words = words[1:1+shift_val] +[words[0]]+ words[1+shift_val:]
        return ' '.join(shifted_words)
    
def move_first_to_last(value):
    value = str(value)
    words = value.split()
    words_length = len(words)
    
    if words_length == 0:
        return '' 
    elif words_length == 1:
        return value  
    else:
      shifted_words = words[1:] +[words[0]]
      return ' '.join(shifted_words)
    
def extract_prefix(value):
    parts = re.split(r'[:;]', str(value))
    return parts[0].strip() if parts else ''

def extract_by_delimiter(value, delimiter='-', index=-1):
    """Extract a segment by splitting on delimiter and taking the segment at index."""
    parts = str(value).split(delimiter)
    if not parts or abs(index) > len(parts) or (index >= 0 and index >= len(parts)):
        return str(value)
    return parts[index].strip()

def extract_initials(value, separator='.'):
    """Extract first character of each word, joined by separator."""
    words = str(value).split()
    if not words:
        return str(value)
    return separator.join(w[0].upper() for w in words if w) + separator


def strip_parenthetical(value):
    """Remove parenthetical content and trailing whitespace.
    'Gov. Brown (2003 - 2011)' → 'Gov. Brown'
    'George Washington (1732-1799)' → 'George Washington'
    """
    return re.sub(r'\s*\([^)]*\)\s*', ' ', str(value)).strip()


def strip_numeric_prefix(value):
    """Remove leading number/bullet prefix.
    '1. George Washington' → 'George Washington'
    '23 Abraham Lincoln' → 'Abraham Lincoln'
    '1) Name' → 'Name'
    """
    return re.sub(r'^\s*\d+[\.\)\-\:]?\s+', '', str(value)).strip()


def strip_title_prefix(value):
    """Remove common title prefixes (Gov., Dr., Mr., Mrs., Ms., Prof., Sen., Rep.).
    'Gov. Arnold Schwarzenegger' → 'Arnold Schwarzenegger'
    'Dr. Smith' → 'Smith'
    """
    return re.sub(r'^(Gov\.|Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.|Sen\.|Rep\.)\s+', '', str(value)).strip()


# --- Word-level operators (used when USE_WORD_ALCS=1 or as general tools) ---

def word_sort_alpha(value):
    """Sort words alphabetically. 'Washington George' -> 'George Washington'"""
    words = str(value).split()
    return ' '.join(sorted(words))

def word_dedup(value):
    """Remove duplicate words (case-insensitive). 'New York New York' -> 'New York'"""
    seen = set()
    result = []
    for w in str(value).split():
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            result.append(w)
    return ' '.join(result)

def word_reverse(value):
    """Reverse word order. 'Last First Middle' -> 'Middle First Last'"""
    return ' '.join(str(value).split()[::-1])

def word_keep_alpha_only(value):
    """Remove non-alphabetic tokens. 'John 42 Smith Jr.' -> 'John Smith Jr.'"""
    return ' '.join(w for w in str(value).split() if any(c.isalpha() for c in w))


def extract_after_delimiter_pattern(value, delimiter='-'):
    """Extract from the first word that is immediately followed by delimiter.

    E.g., 'Cornell University Ithaca- NY' with delimiter='-' -> 'Ithaca- NY'
    because 'Ithaca' is the first word followed by '-'.
    Handles variable-length prefixes (university names, titles, etc.)
    """
    value = str(value)
    words = value.split()
    if not words:
        return value
    # Find the first word index where the word (or next char) has delimiter
    for i, w in enumerate(words):
        if w.endswith(delimiter) or (i < len(words) - 1 and words[i + 1].startswith(delimiter)):
            # Check if the word before the delimiter is likely a location start
            # (capitalized, not a common prefix like "of", "the", "and")
            clean = w.rstrip(delimiter).rstrip(',').rstrip(';')
            if clean and clean[0].isupper() and len(clean) > 1:
                return ' '.join(words[i:])
    # Fallback: return from the word before the first delimiter occurrence
    for i, w in enumerate(words):
        if delimiter in w:
            start = max(0, i - 1)
            return ' '.join(words[start:])
    return value

class OperatorAction:
    def __init__(self, name, func, params, operator_type, cumulative=False):
        self.name = name
        self.func = func
        self.params = params.copy()
        self.initial_params = params.copy()
        self.cumulative = cumulative
        self.operator_type = operator_type

    def apply(self, value):
        return self.func(value, **self.params)

    def adjust_params(self):
        # Adjust parameters based on the operator
        if self.name == 'substring_operator_forward' and self.cumulative:
            # Increment start index
            self.params['start'] += 1
            self.params['start'] = max(0, self.params.get('start', 0))
            self.params['end'] = self.params.get('end')
        elif self.name == 'substring_operator_back_ward' and self.cumulative:
            # Decrement end index
            if self.params['end'] is None:
                self.params['end'] = -1
            else:
                self.params['end'] -= 1
            self.params['start'] = max(0, self.params.get('start', 0))
            self.params['end'] = self.params.get('end')

        elif self.name == 'SelectK_for_separated' and self.cumulative:
            self.params['k'] += 1

        elif self.name == 'SelectK_for_separated_reverse' and self.cumulative:
            self.params['reversed_k'] += 1

        elif self.name.startswith('extract_by_delimiter') and self.cumulative:
            if self.params['index'] < 0:
                self.params['index'] -= 1  # -1 -> -2 -> -3
            else:
                self.params['index'] += 1  # 0 -> 1 -> 2

        elif self.name == 'shift_1_word_forward' and self.cumulative:
            self.params['shift_val'] += 1

    def reset_params(self):
        self.params = self.initial_params.copy()



all_operators = [
    OperatorAction('auto_split_by_operator', auto_split_by_operator, {},'direct'),
    OperatorAction('substring_operator_forward', substring_operator, {'start': 1, 'end': None}, 'direct', cumulative=True),
    OperatorAction('substring_operator_back_ward', substring_operator, {'start': 0, 'end': -1}, 'direct' ,cumulative=True),
    OperatorAction('substring_1_forward_constant', substring_operator, {'start': 1, 'end': None},'direct'),
    OperatorAction('substring_1_back_ward_constant', substring_operator, {'start': 0, 'end': -1},'direct'),
    OperatorAction('substring_second_forward_constant', remove_second_char, {},'direct'),
    OperatorAction('substring_second_back_ward_constant', remove_second_char_backwards, {},'direct'),
    OperatorAction('SelectK_for_separated', SelectK_for_separated, {'k':1},'direct_split', cumulative=True),
    OperatorAction('SelectK_for_separated_reverse', SelectK_for_separated_reverse, {'reversed_k': 1}, 'direct_split', cumulative=True),
    OperatorAction('shift_1_word_forward', shift_1_word_forward, {'shift_val':1},'direct_split', cumulative=True),
    OperatorAction('move_first_to_last', move_first_to_last, {},'direct_split'),
    # OperatorAction('extract_prefix', extract_prefix, {},'direct'),
    OperatorAction('extract_by_delimiter_dash_last', extract_by_delimiter, {'delimiter': '-', 'index': -1}, 'direct_split', cumulative=True),
    OperatorAction('extract_by_delimiter_dash_first', extract_by_delimiter, {'delimiter': '-', 'index': 0}, 'direct_split', cumulative=True),
    OperatorAction('extract_by_delimiter_space_last', extract_by_delimiter, {'delimiter': ' ', 'index': -1}, 'direct_split', cumulative=True),
    OperatorAction('extract_by_delimiter_space_first', extract_by_delimiter, {'delimiter': ' ', 'index': 0}, 'direct_split', cumulative=True),
    OperatorAction('extract_initials', extract_initials, {'separator': '.'}, 'direct_split'),
    OperatorAction('extract_after_delimiter_pattern', extract_after_delimiter_pattern, {'delimiter': '-'}, 'direct_split'),
    OperatorAction('strip_parenthetical', strip_parenthetical, {}, 'direct'),
    OperatorAction('strip_numeric_prefix', strip_numeric_prefix, {}, 'direct'),
    OperatorAction('word_sort_alpha', word_sort_alpha, {}, 'direct_split'),
    OperatorAction('word_dedup', word_dedup, {}, 'direct_split'),
    OperatorAction('word_keep_alpha_only', word_keep_alpha_only, {}, 'direct_split'),
    OperatorAction('concatenate_front', concatenate_front, {},'concate'),
    OperatorAction('concatenate_back', concatenate_back, {},'concate'),
    OperatorAction('concat_pairs_front', concat_pairs_front, {},'concate_both'),
    OperatorAction('concat_pairs_back', concat_pairs_back, {},'concate_both')
]

all_operators_without_concate_both = [
    OperatorAction('auto_split_by_operator', auto_split_by_operator, {},'direct'),
    OperatorAction('substring_operator_forward', substring_operator, {'start': 1, 'end': None}, 'direct', cumulative=True),
    OperatorAction('substring_operator_back_ward', substring_operator, {'start': 0, 'end': -1}, 'direct' ,cumulative=True),
    OperatorAction('substring_1_forward_constant', substring_operator, {'start': 1, 'end': None},'direct'),
    OperatorAction('substring_1_back_ward_constant', substring_operator, {'start': 0, 'end': -1},'direct'),
    OperatorAction('substring_second_forward_constant', remove_second_char, {},'direct'),
    OperatorAction('substring_second_back_ward_constant', remove_second_char_backwards, {},'direct'),
    OperatorAction('SelectK_for_separated', SelectK_for_separated, {'k':1},'direct_split', cumulative=True),
    OperatorAction('SelectK_for_separated_reverse', SelectK_for_separated_reverse, {'reversed_k': 1}, 'direct_split', cumulative=True),
    OperatorAction('shift_1_word_forward', shift_1_word_forward, {'shift_val':1},'direct_split'),
    OperatorAction('move_first_to_last', move_first_to_last, {},'direct_split'),
    OperatorAction('extract_prefix', extract_prefix, {},'direct'),
    OperatorAction('extract_by_delimiter_dash_last', extract_by_delimiter, {'delimiter': '-', 'index': -1}, 'direct_split', cumulative=True),
    OperatorAction('extract_by_delimiter_dash_first', extract_by_delimiter, {'delimiter': '-', 'index': 0}, 'direct_split', cumulative=True),
    OperatorAction('extract_by_delimiter_space_last', extract_by_delimiter, {'delimiter': ' ', 'index': -1}, 'direct_split', cumulative=True),
    OperatorAction('extract_by_delimiter_space_first', extract_by_delimiter, {'delimiter': ' ', 'index': 0}, 'direct_split', cumulative=True),
    OperatorAction('extract_initials', extract_initials, {'separator': '.'}, 'direct_split'),
    OperatorAction('extract_after_delimiter_pattern', extract_after_delimiter_pattern, {'delimiter': '-'}, 'direct_split'),
    OperatorAction('strip_parenthetical', strip_parenthetical, {}, 'direct'),
    OperatorAction('strip_numeric_prefix', strip_numeric_prefix, {}, 'direct'),
    OperatorAction('word_sort_alpha', word_sort_alpha, {}, 'direct_split'),
    OperatorAction('word_dedup', word_dedup, {}, 'direct_split'),
    OperatorAction('word_keep_alpha_only', word_keep_alpha_only, {}, 'direct_split'),
    OperatorAction('concatenate_front', concatenate_front, {},'concate'),
    OperatorAction('concatenate_back', concatenate_back, {},'concate'),
]

direct_operators_only = [
    OperatorAction('auto_split_by_operator', auto_split_by_operator, {},'direct'),
    OperatorAction('substring_operator_forward', substring_operator, {'start': 1, 'end': None}, 'direct', cumulative=True),
    OperatorAction('substring_operator_back_ward', substring_operator, {'start': 0, 'end': -1}, 'direct' ,cumulative=True),
    OperatorAction('substring_1_forward_constant', substring_operator, {'start': 1, 'end': None},'direct'),
    OperatorAction('substring_1_back_ward_constant', substring_operator, {'start': 0, 'end': -1},'direct'),
    OperatorAction('substring_second_forward_constant', remove_second_char, {},'direct'),
    OperatorAction('substring_second_back_ward_constant', remove_second_char_backwards, {},'direct'),
    OperatorAction('SelectK_for_separated', SelectK_for_separated, {'k':1},'direct_split', cumulative=True),
    OperatorAction('SelectK_for_separated_reverse', SelectK_for_separated_reverse, {'reversed_k': 1}, 'direct_split', cumulative=True),
    OperatorAction('shift_1_word_forward', shift_1_word_forward, {'shift_val':1},'direct_split'),
    OperatorAction('move_first_to_last', move_first_to_last, {},'direct_split'),
    OperatorAction('extract_prefix', extract_prefix, {},'direct'),
    OperatorAction('extract_by_delimiter_dash_last', extract_by_delimiter, {'delimiter': '-', 'index': -1}, 'direct_split', cumulative=True),
    OperatorAction('extract_by_delimiter_dash_first', extract_by_delimiter, {'delimiter': '-', 'index': 0}, 'direct_split', cumulative=True),
    OperatorAction('extract_by_delimiter_space_last', extract_by_delimiter, {'delimiter': ' ', 'index': -1}, 'direct_split', cumulative=True),
    OperatorAction('extract_by_delimiter_space_first', extract_by_delimiter, {'delimiter': ' ', 'index': 0}, 'direct_split', cumulative=True),
    OperatorAction('extract_initials', extract_initials, {'separator': '.'}, 'direct_split'),
    OperatorAction('extract_after_delimiter_pattern', extract_after_delimiter_pattern, {'delimiter': '-'}, 'direct_split'),
    OperatorAction('strip_parenthetical', strip_parenthetical, {}, 'direct'),
    OperatorAction('strip_numeric_prefix', strip_numeric_prefix, {}, 'direct'),
    OperatorAction('word_sort_alpha', word_sort_alpha, {}, 'direct_split'),
    OperatorAction('word_dedup', word_dedup, {}, 'direct_split'),
    OperatorAction('word_keep_alpha_only', word_keep_alpha_only, {}, 'direct_split'),
]
# strip_title_prefix and word_reverse removed — never selected on AJ/FF/SyGuS
# (258-dataset audit, see commit log).


# ============================================================================
# LABEL-FREE OPERATOR SYNTHESIS  (SYNTH_OPS=1)
# ----------------------------------------------------------------------------
# The hard-coded extract_by_delimiter ops only express delimiters {'-',' '} at
# indices {0,-1}. The prose/DateTime/UserAgent/Number tail needs '/ . : , _ --'
# and middle indices (e.g. 'MFM-5.2.59/xPhone-4.9' -> '4.9';
# '423531' -> length-norm; 'Ducati100' -> 'Ducati'). This synthesizes such ops
# by scanning the SOURCE VALUES ONLY (never truth): it finds the delimiters that
# actually occur (by frequency), instantiates split(delim,index) at the indices
# the data exposes, and adds numeric/strip normalizers. Bounded by a cap.
# ============================================================================

def strip_trailing_digits(value):
    """'Ducati100' -> 'Ducati'; 'v4.9' -> 'v4.'? no: only trailing run of digits.
    Removes a trailing run of digits (and any trailing '.'/space left behind).
    'Ducati100' -> 'Ducati'; 'Model 5' -> 'Model'; 'abc' -> 'abc'."""
    s = str(value)
    out = re.sub(r'\d+\s*$', '', s)
    return out.rstrip(' .-_').strip()

def strip_leading_nondigits(value):
    """Drop a leading run of non-digit chars: 'Ducati100' -> '100'; 'v4.9' -> '4.9';
    'abc' -> 'abc' (no digit -> unchanged)."""
    s = str(value)
    m = re.search(r'\d', s)
    if m is None:
        return s
    return s[m.start():].strip()

def keep_digits_only(value):
    """Keep only the digit characters: 'MFM-5.2.59' -> '5259'; '(404) 555' -> '404555'."""
    s = str(value)
    return ''.join(ch for ch in s if ch.isdigit())

def zero_pad_to(value, width=6):
    """Left-pad a (possibly numeric) string to `width` with zeros.
    '423' -> '000423' (width 6). Non-numeric returned unchanged length-padded too."""
    s = str(value).strip()
    if s == '':
        return s
    return s.rjust(int(width), '0')

def round_number_to(value, ndigits=-5):
    """Round a leading numeric value to 10**(-ndigits) magnitude.
    '423531' with ndigits=-5 -> '400000'.  Non-numeric -> unchanged.
    (Label-free: magnitude chosen from the value's own digit-length, see synth.)"""
    s = str(value).strip()
    m = re.match(r'^[+-]?\d+(?:\.\d+)?', s)
    if not m:
        return s
    try:
        num = float(m.group(0))
        r = round(num, int(ndigits))
        # keep integer formatting when input had no decimal point
        if '.' not in m.group(0):
            r = int(r)
        return str(r)
    except Exception:
        return s


# A delimiter char is a SPLIT candidate; '--' handled as a literal too.
_SYNTH_DELIM_CANDIDATES = ['/', '.', ':', ',', '_', '|', ';', '@', '#', '\\', '--']
# Base ops already cover these (delim,index) combos — don't re-synthesize them.
_BASE_SPLIT_COMBOS = {('-', -1), ('-', 0), (' ', -1), (' ', 0)}
_SYNTH_OP_CAP = 14


def _synthesized_split_op(delim, index):
    """Build a named OperatorAction for extract_by_delimiter(delim, index)."""
    dtag = {'/': 'slash', '.': 'dot', ':': 'colon', ',': 'comma', '_': 'underscore',
            '|': 'pipe', ';': 'semicolon', '@': 'at', '#': 'hash', '\\': 'bslash',
            '--': 'dashdash', '-': 'dash', ' ': 'space'}.get(delim, 'd%d' % ord(delim[0]))
    itag = ('m%d' % abs(index)) if index < 0 else str(index)
    name = 'synth_split_%s_%s' % (dtag, itag)
    return OperatorAction(name, extract_by_delimiter, {'delimiter': delim, 'index': index},
                          'direct_split', cumulative=False)


def synthesize_operators_from_source(df_a, col_a, df_b, col_b):
    """LABEL-FREE op synthesis. Scans SOURCE column values only (col_a in df_a,
    falling back to col_b if col_a absent) — NEVER reads truth/labels — and returns
    a bounded list of new OperatorAction objects:
      * split(delim,index) for each frequently-occurring delimiter, at indices
        0..k-1 and -1 (k = max field count that delimiter produces, capped at 4);
      * strip_trailing_digits / strip_leading_nondigits / keep_digits_only when the
        column is digit-bearing;
      * zero_pad_to(modal width) + round_number_to(magnitude) when the column is
        mostly numeric.
    Capped at _SYNTH_OP_CAP ops total (delimiter ops prioritized by frequency)."""
    # gather source sample values (input only)
    vals = []
    for _df, _c in ((df_a, col_a), (df_b, col_b)):
        if _df is not None and _c in getattr(_df, 'columns', []):
            try:
                vals = [str(v) for v in _df[_c].astype(str).tolist() if str(v).strip() != '']
            except Exception:
                vals = []
            if vals:
                break
    if not vals:
        return []
    vals = vals[:400]  # cap scan cost
    nv = len(vals)

    new_ops = []
    seen_names = set()

    # ---- delimiter discovery: count, per value, which candidate delims appear ----
    delim_doc_freq = {d: 0 for d in _SYNTH_DELIM_CANDIDATES}
    delim_max_fields = {d: 0 for d in _SYNTH_DELIM_CANDIDATES}
    for v in vals:
        for d in _SYNTH_DELIM_CANDIDATES:
            if d in v:
                delim_doc_freq[d] += 1
                nf = len(v.split(d))
                if nf > delim_max_fields[d]:
                    delim_max_fields[d] = nf
    # keep delimiters present in >= 30% of values, ranked by document frequency
    cand = [d for d in _SYNTH_DELIM_CANDIDATES if delim_doc_freq[d] >= max(2, int(0.30 * nv))]
    cand.sort(key=lambda d: -delim_doc_freq[d])

    for d in cand:
        kf = min(delim_max_fields[d], 4)  # cap field index breadth
        # indices: first few forward (0..kf-2) plus last (-1)
        idxs = list(range(0, max(1, kf - 1))) + [-1]
        # de-dup indices while preserving order
        seen_i = set(); idxs = [i for i in idxs if not (i in seen_i or seen_i.add(i))]
        for i in idxs:
            if (d, i) in _BASE_SPLIT_COMBOS:
                continue
            op = _synthesized_split_op(d, i)
            if op.name not in seen_names:
                new_ops.append(op); seen_names.add(op.name)

    # ---- numeric / strip ops: condition on digit content of the column ----
    digit_frac = np.mean([1.0 if any(ch.isdigit() for ch in v) else 0.0 for v in vals])
    mostly_numeric = np.mean([1.0 if re.fullmatch(r'[+-]?\d[\d,\.]*', v.strip()) else 0.0
                              for v in vals]) >= 0.6
    has_trailing_digit = np.mean([1.0 if re.search(r'\d\s*$', v) else 0.0 for v in vals]) >= 0.3
    has_leading_nondigit = np.mean([1.0 if re.match(r'^\D+\d', v) else 0.0 for v in vals]) >= 0.3

    def _add(op):
        if op.name not in seen_names:
            new_ops.append(op); seen_names.add(op.name)

    if has_trailing_digit:
        _add(OperatorAction('synth_strip_trailing_digits', strip_trailing_digits, {}, 'direct'))
    if has_leading_nondigit:
        _add(OperatorAction('synth_strip_leading_nondigits', strip_leading_nondigits, {}, 'direct'))
    if digit_frac >= 0.5:
        _add(OperatorAction('synth_keep_digits_only', keep_digits_only, {}, 'direct'))
    if mostly_numeric:
        # modal int width for zero-pad
        widths = [len(re.sub(r'\D', '', v)) for v in vals if re.sub(r'\D', '', v)]
        if widths:
            w = int(pd.Series(widths).mode().iloc[0])
            if 1 < w <= 12:
                _add(OperatorAction('synth_zero_pad_%d' % w, zero_pad_to, {'width': w}, 'direct'))
        # round to one-significant-figure magnitude (e.g. 6-digit -> round to 1e5)
        intlens = [len(re.match(r'^\d+', v.strip()).group(0)) for v in vals
                   if re.match(r'^\d+', v.strip())]
        if intlens:
            mlen = int(pd.Series(intlens).mode().iloc[0])
            nd = -(mlen - 1)
            if nd < 0:
                _add(OperatorAction('synth_round_1e%d' % (mlen - 1), round_number_to,
                                    {'ndigits': nd}, 'direct'))

    return new_ops[:_SYNTH_OP_CAP]


# ============================================================================
# CONDITION-BASED (state-conditioned) EXPLORATION PRIOR  (COND_EXPLORE=1)
# ----------------------------------------------------------------------------
# The dumb eps-greedy explores by uniform-random sub-sampling of operators.
# This computes a DATA-CONDITIONED prior over operators from the SOURCE column
# structure (input values only, no truth): boost split(d,*) when delimiter d is
# frequent; boost numeric/pad/strip ops when the column is digit-heavy or has
# leading/trailing noise. Seeding the agent's operators_prob_dict with this prior
# makes the exploration sub-sample (random.choices weighted by that dict in
# choose_action) try the operators the DATA STRUCTURE SUGGESTS first, instead of
# uniform-random. Label-free, so it is a legitimate test of "is a state-
# conditioned policy better than context-free eps-greedy under idfcos".
# ============================================================================

# map a delimiter char -> operator-name fragments that act on it
_COND_DELIM_TO_OPFRAG = {
    '-': ['dash'], ' ': ['space'], '/': ['slash'], '.': ['dot'], ':': ['colon'],
    ',': ['comma'], '_': ['underscore'], '|': ['pipe'], ';': ['semicolon'],
    '@': ['at'], '#': ['hash'], '--': ['dashdash'],
}
# split/reorder ops that operate on whitespace-separated tokens
_COND_SPLIT_TOKEN_OPS = {'SelectK_for_separated', 'SelectK_for_separated_reverse',
                         'shift_1_word_forward', 'move_first_to_last',
                         'extract_by_delimiter_space_last', 'extract_by_delimiter_space_first',
                         'word_sort_alpha', 'word_dedup', 'word_keep_alpha_only'}
_COND_NUMERIC_OPS = {'synth_keep_digits_only', 'keep_digits_only'}


def data_conditioned_op_prior(df, col, op_names, df_other=None, col_other=None):
    """Return {op_name: weight} prior conditioned on the SOURCE column structure.
    Label-free (input values only). Weights are multiplicative boosts over a 1.0
    uniform base; callers renormalize. Higher weight => explored/preferred sooner."""
    base = 1.0
    boost = 4.0
    prior = {n: base for n in op_names}

    vals = []
    if df is not None and col in getattr(df, 'columns', []):
        try:
            vals = [str(v) for v in df[col].astype(str).tolist() if str(v).strip() != '']
        except Exception:
            vals = []
    if not vals:
        return prior
    vals = vals[:400]
    nv = len(vals)

    # ---- delimiter frequencies -> boost matching split ops ----
    all_delims = list(_COND_DELIM_TO_OPFRAG.keys())
    for d in all_delims:
        present = sum(1 for v in vals if d in v)
        frac = present / nv
        if frac >= 0.30:
            scale = boost * min(1.0, frac)  # stronger when more frequent
            frags = _COND_DELIM_TO_OPFRAG[d]
            for n in op_names:
                ln = n.lower()
                if any(fr in ln for fr in frags):
                    prior[n] = max(prior[n], base + scale)

    # whitespace token structure -> boost token/split/reorder ops
    multiword_frac = np.mean([1.0 if len(v.split()) > 1 else 0.0 for v in vals])
    if multiword_frac >= 0.4:
        for n in op_names:
            if n in _COND_SPLIT_TOKEN_OPS:
                prior[n] = max(prior[n], base + boost * 0.6)

    # ---- numeric structure -> boost numeric/pad/round/strip ops ----
    digit_frac = np.mean([1.0 if any(ch.isdigit() for ch in v) else 0.0 for v in vals])
    mostly_numeric = np.mean([1.0 if re.fullmatch(r'[+-]?\d[\d,\.]*', v.strip()) else 0.0
                              for v in vals]) >= 0.6
    has_trailing_digit = np.mean([1.0 if re.search(r'\d\s*$', v) else 0.0 for v in vals]) >= 0.3
    has_leading_nondigit = np.mean([1.0 if re.match(r'^\D+\d', v) else 0.0 for v in vals]) >= 0.3

    if digit_frac >= 0.5:
        for n in op_names:
            ln = n.lower()
            if ('keep_digits' in ln) or ('zero_pad' in ln) or ('round_' in ln):
                prior[n] = max(prior[n], base + boost * (1.0 if mostly_numeric else 0.5))
    if has_trailing_digit:
        for n in op_names:
            if 'strip_trailing_digits' in n.lower():
                prior[n] = max(prior[n], base + boost)
    if has_leading_nondigit:
        for n in op_names:
            ln = n.lower()
            if ('strip_leading_nondigits' in ln) or ('strip_numeric_prefix' in ln):
                prior[n] = max(prior[n], base + boost)

    # ---- parenthetical / title noise -> boost strip ops ----
    paren_frac = np.mean([1.0 if '(' in v else 0.0 for v in vals])
    if paren_frac >= 0.3:
        for n in op_names:
            if 'strip_parenthetical' in n.lower():
                prior[n] = max(prior[n], base + boost * 0.7)

    return prior


def apply_conditioned_prior_to_agent(agent, df, col, df_other=None, col_other=None):
    """Seed an agent's operators_prob_dict with the data-conditioned prior so that
    BOTH the explore-branch sub-sampling (random.choices weighted by this dict) AND
    exploit ordering favor the data-suggested ops. Normalized to a probability dist.

    operators_prob_dict is a defaultdict whose factory produces a uniform per-column
    dict on first access of any column key. We REPLACE that factory so EVERY column
    (including ones created lazily later in the chain) gets the conditioned prior,
    and re-snapshot initial_operators_prob_dict so reset_params_prob preserves it."""
    try:
        op_names = [op.name for op in agent.operators]
    except Exception:
        return
    if not op_names:
        return
    prior = data_conditioned_op_prior(df, col, op_names, df_other, col_other)
    tot = sum(prior.values())
    if tot <= 0:
        return
    norm = {n: prior[n] / tot for n in op_names}
    agent._cond_prior_norm = norm
    # replace the default factory so any newly-seen column key gets the prior
    agent.operators_prob_dict.default_factory = (lambda nm=dict(norm): dict(nm))
    # overwrite any already-materialized column keys
    for c in list(agent.operators_prob_dict.keys()):
        agent.operators_prob_dict[c] = dict(norm)
    # re-snapshot so reset_params_prob keeps the prior instead of going uniform
    agent.initial_operators_prob_dict = copy.deepcopy(agent.operators_prob_dict)
    try:
        agent.initial_operators_prob_dict.default_factory = (lambda nm=dict(norm): dict(nm))
    except Exception:
        pass


# helper function

def apply_actions_to_df(transformed_df,transformed_df_action):
  transformed_column_for_change = transformed_df.copy()
  for col_name, action in transformed_df_action.items():
    if col_name not in transformed_column_for_change.columns:
      continue
    transformed_column = transformed_column_for_change[col_name]
    transformed_column_new = [action.apply(value) for value in transformed_column]
    transformed_column_for_change[col_name] = transformed_column_new
  return transformed_column_for_change


def append_actions_to_df_dict(transformations_dict,transformed_df_action):
  for col_name, action in transformed_df_action.items():
    transformations_dict[col_name].append(action)
  return transformations_dict

def print_dict_params(transformations_dict):
  for col_name, transformations in transformations_dict.items():
    print(f"For column {col_name}: {[(op.name,op.params) for op in transformations]}")


def random_choose_col(df):
  return df[random.choices(df.columns, k=1)]

def noop(value):
    return value


#Hunt-Szymanski

def _lcs_length_only_hs(s1, s2):
    """
    Fast LCS LENGTH using Hunt-Szymanski — skips string reconstruction.
    ~2x faster than find_common_string_hs since we skip predecessors tracking
    and backtracking.
    """
    char_to_indices = defaultdict(list)
    for idx, char in enumerate(s2):
        char_to_indices[char].append(idx)

    active = []
    for char in s1:
        if char not in char_to_indices:
            continue
        for pos in reversed(char_to_indices[char]):
            idx = bisect_left(active, pos)
            if idx == len(active):
                active.append(pos)
            else:
                active[idx] = pos
    return len(active)


def find_common_string_hs(s1, s2):
    """
    Finds the Longest Common Subsequence (LCS) between s1 and s2 using the Hunt-Szymanski algorithm.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string.

    Returns:
        str: The LCS string.
    """
    # Step 1: Create inverted index for s2
    char_to_indices = defaultdict(list)
    for idx, char in enumerate(s2):
        char_to_indices[char].append(idx)

    # Step 2: Iterate through s1 and build the LCS using positions in s2
    active = []
    predecessors = []

    for char in s1:
        if char not in char_to_indices:
            continue
        # Get positions in s2 in reverse order to maintain order when inserting
        for pos in reversed(char_to_indices[char]):
            # Find the insertion point using binary search
            idx = bisect_left(active, pos)
            if idx == len(active):
                active.append(pos)
                predecessors.append((pos, idx))
            else:
                active[idx] = pos
                predecessors.append((pos, idx))

    # Step 3: Reconstruct the LCS from active
    if not active:
        return ""

    lcs_length = len(active)
    lcs = [''] * lcs_length
    current_length = lcs_length - 1
    last_pos = active[-1]

    # Backtrack to find the LCS characters
    for pos, idx in reversed(predecessors):
        if idx == current_length and pos <= last_pos:
            lcs[current_length] = s2[pos]
            last_pos = pos
            current_length -= 1
            if current_length < 0:
                break

    return ''.join(lcs)


def lcs_count(s1, s2, min_len, ignore_chars={' '}):
    """
    Returns (lcs_percentage, lcs_len, common_string).
    If the LCS length > min_len, lcs_percentage = lcs_len / max_len_of_(s1_filtered, s2_filtered).
    Else returns (0, 0, '').
    """
    # Ensure string inputs (handles NaN/float from pandas)
    s1 = str(s1) if not isinstance(s1, str) else s1
    s2 = str(s2) if not isinstance(s2, str) else s2
    # Remove ignorable characters
    s1_filtered = ''.join(char for char in s1 if char not in ignore_chars)
    s2_filtered = ''.join(char for char in s2 if char not in ignore_chars)

    # Find LCS
    common_string= find_common_string_hs(s1_filtered, s2_filtered)
    lcs_len = len(common_string)
    adj_total_len = (len(s1_filtered)+len(s2_filtered))/2

    if lcs_len > min_len:
        lcs_percentage = lcs_len / adj_total_len
        return lcs_percentage, lcs_len, common_string
    else:
        return 0, 0, ''


# def get_overlap_counts(arrays):
#     flattened = [number for array in arrays for number in array]
#     frequency = Counter(flattened)
#     overlap_counts = [sum(frequency[number] for number in array) for array in arrays]
#     return overlap_counts


# ============================================================
# alcs_fuzzy: Token-structured alignment with fuzzy token scoring.
#
# Architecture:
#   1. Tokenize: normalize to word tokens (lowercase, split on non-alnum)
#   2. Token-to-token scoring: exact match (O(1)) → char-ALCS (local)
#   3. Sequence alignment: weighted LCS over token sequences
#   4. Normalize by average token count
#
# Char-level is NOT a separate global objective. It is a LOCAL
# explainer inside token-to-token matching. This avoids:
#   - zero overlap on fuzzy words (apple vs saaaple → 0.67)
#   - junk suffix matching from raw char ALCS
#   - double counting (each token matched at most once via LCS)
#
# "apple cider" vs "saaaple gmail com":
#   token sim matrix:
#     apple→saaaple: 0.67 (char-ALCS explains it)
#     apple→gmail:   0.00
#     apple→com:     0.00
#     cider→saaaple: 0.00
#     cider→gmail:   0.00
#     cider→com:     0.00
#   weighted LCS: [(apple, saaaple, 0.67)] → sim = 0.67/2.5 = 0.27
# ============================================================

_FUZZY_TOKEN_THRESHOLD = 0.5  # min char-ALCS to count tokens as matching


def _normalize_to_tokens(s):
    """Canonical tokenization: lowercase, replace non-alnum separators
    with spaces (preserving word boundaries), split, filter empty."""
    s = str(s).lower()
    # Replace non-alphanumeric with space (preserves token boundaries)
    # "apple-cider" → "apple cider", "apple@gmail.com" → "apple gmail com"
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return [t for t in s.split() if t]


def _token_sim(w1, w2):
    """Token-to-token fuzzy similarity. Char-level is the local explainer.
    Fast path: exact match → 1.0 (O(1)).
    Slow path: char-ALCS via Hunt-Szymanski (O(c1*c2), cheap for 3-15 char words)."""
    if w1 == w2:
        return 1.0
    if not w1 or not w2:
        return 0.0
    # Length ratio early exit: if too different, can't exceed threshold
    ratio = min(len(w1), len(w2)) / max(len(w1), len(w2))
    if ratio < _FUZZY_TOKEN_THRESHOLD:
        return 0.0
    # Char-level ALCS as local explainer
    common = find_common_string_hs(w1, w2)
    if len(common) < 2:
        return 0.0
    avg_len = (len(w1) + len(w2)) / 2.0
    return len(common) / avg_len


def alcs_fuzzy_single(s1, s2, threshold=_FUZZY_TOKEN_THRESHOLD):
    """Token-structured fuzzy ALCS between two strings.

    Returns (similarity, n_matched_tokens).

    Alignment is at the TOKEN level. Token-to-token scoring uses
    exact match first, then char-ALCS as local explainer.
    No double counting: LCS ensures each token matched at most once.
    Similarity normalized by average token count, in [0, 1]."""
    tokens1 = _normalize_to_tokens(s1)
    tokens2 = _normalize_to_tokens(s2)
    m, n = len(tokens1), len(tokens2)
    if m == 0 or n == 0:
        return 0.0, 0

    # Build token-to-token similarity matrix
    # Exact match first (O(1)), char-ALCS only when needed
    tsim = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            tsim[i][j] = _token_sim(tokens1[i], tokens2[j])

    # Weighted LCS DP over token sequences
    # dp[i][j] = max total weight aligning tokens1[:i] with tokens2[:j]
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    dp_cnt = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Option 1: match tokens i-1 and j-1 (if above threshold)
            cs = tsim[i - 1][j - 1]
            if cs >= threshold:
                w = dp[i - 1][j - 1] + cs
                c = dp_cnt[i - 1][j - 1] + 1
                if w > dp[i][j]:
                    dp[i][j] = w
                    dp_cnt[i][j] = c
            # Option 2: skip token from s1
            if dp[i - 1][j] > dp[i][j]:
                dp[i][j] = dp[i - 1][j]
                dp_cnt[i][j] = dp_cnt[i - 1][j]
            # Option 3: skip token from s2
            if dp[i][j - 1] > dp[i][j]:
                dp[i][j] = dp[i][j - 1]
                dp_cnt[i][j] = dp_cnt[i][j - 1]

    total_weight = dp[m][n]
    n_matched = dp_cnt[m][n]
    if n_matched == 0:
        return 0.0, 0
    avg_len = (m + n) / 2.0
    sim = total_weight / avg_len if avg_len > 0 else 0.0
    return min(sim, 1.0), n_matched


# Keep backward-compatible names
def word_alcs_single(s1, s2, min_words=1):
    """Backward-compatible wrapper for alcs_fuzzy_single."""
    sim, n_matched = alcs_fuzzy_single(s1, s2)
    if n_matched < min_words:
        return 0.0, 0
    return sim, n_matched


# v9 helpers (alcs_fuzzy_sim_matrix, auto_fc_pick) moved to v3_dd/r4/v9_autofc.py
# to keep this file lighter. Lazy-imported inside the get_ALCS_matrix* branches.


def get_word_ALCS_matrix(list1, list2, min_words=1, greedy=True):
    """Compute word-level ALCS similarity matrix between two lists of strings.
    Same interface as get_ALCS_matrix but operates on word tokens.

    Returns (mean_max_sim, sim_matrix, freq_penalty, lcs_count_matrix).

    Use this for long-text columns where character-level ALCS is too slow
    or too noisy (e.g., governor biographies, descriptions, addresses).
    Complexity: O(m * n * w1 * w2) where w1,w2 are avg word counts.
    For 15-word strings: ~225 ops per pair vs ~2500 for 50-char strings."""
    str_list1 = [str(s) for s in list1]
    str_list2 = [str(s) for s in list2]
    m, n = len(str_list1), len(str_list2)

    if m == 0 or n == 0:
        return 0.0, np.zeros((max(m, 1), max(n, 1))), 0, np.zeros((max(m, 1), max(n, 1)))

    sim_matrix = np.zeros((m, n))
    lcs_matrix = np.zeros((m, n))

    # Top-k word-overlap filtering for large matrices
    TOP_K = min(n, max(5, int(n * 0.15)))
    use_topk = m * n > 400 and n > TOP_K

    if use_topk:
        for i in range(m):
            # Cheap word-set overlap to find top-k candidates
            words_i = set(str_list1[i].lower().split())
            if not words_i:
                continue
            cheap = []
            for j in range(n):
                words_j = set(str_list2[j].lower().split())
                if not words_j:
                    cheap.append((j, 0.0))
                    continue
                overlap = len(words_i & words_j) / max(len(words_i | words_j), 1)
                cheap.append((j, overlap))
            cheap.sort(key=lambda x: x[1], reverse=True)
            for j, _ in cheap[:TOP_K]:
                sim, lcs_len = word_alcs_single(str_list1[i], str_list2[j], min_words)
                sim_matrix[i, j] = sim
                lcs_matrix[i, j] = lcs_len
    else:
        for i in range(m):
            for j in range(n):
                sim, lcs_len = word_alcs_single(str_list1[i], str_list2[j], min_words)
                sim_matrix[i, j] = sim
                lcs_matrix[i, j] = lcs_len

    # Frequency penalty (greedy vs non-greedy, matching char-level behavior)
    mask = sim_matrix == np.max(sim_matrix, axis=1, keepdims=True)
    result_indices = [np.where(row)[0] for row in mask]
    if greedy:
        freq_penalty = get_penalty_overlap_counts(result_indices)
    else:
        mask2 = sim_matrix.T == np.max(sim_matrix.T, axis=1, keepdims=True)
        result_indices2 = [np.where(row)[0] for row in mask2]
        freq_penalty = (get_penalty_overlap_counts_non_greedy(result_indices) +
                        get_penalty_overlap_counts_non_greedy(result_indices2))

    mean_max = float(np.mean(np.max(sim_matrix, axis=1))) if m > 0 else 0.0
    return mean_max, sim_matrix, freq_penalty, lcs_matrix


def get_penalty_overlap_counts(arrays):
    """
    Calculate overlap penalties for each array based on the frequency of their elements.
    Parameters:
    arrays (List[List[int]]): A list of arrays containing numerical values.
    Returns:
    List[int]: A list of penalty scores corresponding to each input array.
    """
    flattened = [number for array in arrays for number in array]
    frequency = Counter(flattened)
    penalty = [
        sum(max(0, frequency[number] - 1) for number in array)
        for array in arrays
    ]

    final_penalty =  np.sum(penalty)/(3*len(flattened)**2)
    
    return final_penalty

# update penalty one

def get_penalty_overlap_counts_non_greedy(arrays):
    """
    Calculate overlap penalties for each array based on the frequency of their elements.
    Parameters:
    arrays (List[List[int]]): A list of arrays containing numerical values.
    Returns:
    List[int]: A list of penalty scores corresponding to each input array.
    """
    flattened = [number for array in arrays for number in array]
    frequency = Counter(flattened)
    penalty = [
        sum(max(0, frequency[number] - 1) for number in array)
        for array in arrays
    ]

    final_penalty = 3 * np.sum(penalty)/len(flattened)**(3/2)
    
    return final_penalty


###
###
###
###
###

def get_top_k_percent(matrix,k):
  matrix = matrix.copy()
  m,n = matrix.shape
  k_percentile = (1-k)*100
  threshold_lst = []
  for i in range(m):
    threshold_lst.append(np.percentile(matrix[i,:], k_percentile))

  for i in range(m):
    for j in range(n):
      if matrix[i,j]>=threshold_lst[i]:
         matrix[i,j] = 1
      else:
        matrix[i,j] = 0
  return matrix



def _compute_lcs_row(args):
    """Worker: compute one row of the ALCS matrix."""
    i, s1, str_list2, min_len, ignore_chars = args
    n = len(str_list2)
    sim_row = np.zeros(n)
    lcs_row = np.zeros(n)
    cs_row = [None] * n
    for j in range(n):
        sim, lcs, cs = lcs_count(s1, str_list2[j], min_len, ignore_chars)
        sim_row[j] = sim
        lcs_row[j] = lcs
        cs_row[j] = cs
    return i, sim_row, lcs_row, cs_row


def _compute_lcs_pairs(args):
    """Worker: compute a batch of unique (s1, s2) pairs."""
    pairs_batch, min_len, ignore_chars = args
    results = {}
    for s1, s2 in pairs_batch:
        sim, lcs, cs = lcs_count(s1, s2, min_len, ignore_chars)
        results[(s1, s2)] = (sim, lcs, cs)
    return results


# Minimum matrix size to justify parallel overhead
_PARALLEL_THRESHOLD = 500  # m*n must exceed this


def _cheap_sim(s1, s2):
    """Ultra-fast similarity estimate using character set overlap. O(len) not O(len^2)."""
    if not s1 or not s2:
        return 0.0
    set1 = set(s1.lower())
    set2 = set(s2.lower())
    if not set1 or not set2:
        return 0.0
    overlap = len(set1 & set2)
    return overlap / max(len(set1), len(set2))


def jaccard_ngram_sim(s1, s2, n=2):
    """Jaccard similarity on character n-grams. Fast token-based similarity.
    Distilled from data discovery version's jaccard_via_inverted_index."""
    s1, s2 = str(s1).lower(), str(s2).lower()
    if not s1 or not s2:
        return 0.0
    ngrams1 = set(s1[i:i+n] for i in range(max(1, len(s1)-n+1)))
    ngrams2 = set(s2[i:i+n] for i in range(max(1, len(s2)-n+1)))
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union > 0 else 0.0


def jaccard_matrix_fast(list1, list2, n=2):
    """Fast Jaccard similarity matrix using pre-computed n-gram sets.
    Distilled from data discovery version. Much faster than full LCS for pair ranking."""
    sets1 = []
    for s in list1:
        s = str(s).lower()
        sets1.append(set(s[i:i+n] for i in range(max(1, len(s)-n+1))))
    sets2 = []
    for s in list2:
        s = str(s).lower()
        sets2.append(set(s[i:i+n] for i in range(max(1, len(s)-n+1))))

    m, nn = len(sets1), len(sets2)
    sim_matrix = np.zeros((m, nn))
    for i in range(m):
        if not sets1[i]:
            continue
        for j in range(nn):
            if not sets2[j]:
                continue
            intersection = len(sets1[i] & sets2[j])
            union = len(sets1[i] | sets2[j])
            sim_matrix[i, j] = intersection / union if union > 0 else 0.0

    mean_max = float(np.mean(np.max(sim_matrix, axis=1))) if m > 0 else 0.0
    return mean_max, sim_matrix


def jaccard_matrix_adaptive(list1, list2, mode='adaptive'):
    """Cheap ALCS-like adaptivity with NO hand-set thresholds: q-gram Jaccard that adapts to
    the data so it handles short AND long strings without the O(n*m) alignment DP. O(n) set-ops,
    fully vectorizable.
      mode='adaptive'    -> per-dataset q in {2,3,4} chosen by the label-free decisiveness of its
                            match matrix (mean top-1 - top-2 margin) -- the same separation signal
                            the cov_mnn selector trusts; parameter-free, no length cutoffs.
      mode='blend'       -> mean Jaccard over q in {2,3,4} (parameter-free, no selection at all)
      mode='containment' -> overlap coeff |A&B|/min(|A|,|B|) at q=2 (handles transform length asymmetry)"""
    s1 = [str(s).lower() for s in list1]
    s2 = [str(s).lower() for s in list2]
    m, nn = len(s1), len(s2)
    if m == 0 or nn == 0:
        return 0.0, np.zeros((max(m,1), max(nn,1)))
    if mode == 'adaptive':
        # pick q by how decisively its matrix separates the best match from the runner-up
        best = None
        for q in (2, 3, 4):
            _, mat = jaccard_matrix_fast(s1, s2, n=q)
            if mat.shape[1] >= 2:
                p = np.partition(mat, -2, axis=1)
                sep = float(np.mean(p[:, -1] - p[:, -2]))   # label-free: top1 - top2 margin
            else:
                sep = float(np.mean(mat))
            if best is None or sep > best[0]:
                best = (sep, mat)
        sim = best[1]
    elif mode == 'blend':
        sim = np.mean([jaccard_matrix_fast(s1, s2, n=q)[1] for q in (2, 3, 4)], axis=0)
    else:  # containment (overlap coefficient, q=2)
        def grams(s, q): return set(s[i:i+q] for i in range(max(1, len(s)-q+1)))
        a = [grams(s, 2) for s in s1]; b = [grams(s, 2) for s in s2]
        sim = np.zeros((m, nn))
        for i in range(m):
            if not a[i]: continue
            for j in range(nn):
                if not b[j]: continue
                mn = min(len(a[i]), len(b[j])); sim[i, j] = len(a[i] & b[j]) / mn if mn else 0.0
    mean_max = float(np.mean(np.max(sim, axis=1)))
    return mean_max, sim


# ============================================================================
# Parameter-free terminal similarity metrics (drop-in replacements for ALCS).
# Each takes two string lists and returns a [m x n] similarity matrix in [0,1].
# NONE reads truth/labels and NONE tunes any constant against F1 -- they are a
# pure function of the input strings only. Selected via _METRIC.
# ============================================================================
def _ncd_matrix(str1, str2):
    """Normalized Compression Distance similarity (Cilibrasi & Vitanyi 2005).
        sim = 1 - (C(xy) - min(C(x),C(y))) / max(C(x),C(y))
    with C = compressed byte length under zlib (Lempel-Ziv, gzip family).
    PARAMETER-FREE: no tuned constant. The only choice is the compressor itself,
    which is part of the published NCD definition (we use zlib at its standard
    default level; FLAG: the compressor family is a definitional choice, not a
    perf-tuned number). Universal by construction -- the LZ dictionary adapts to
    short, long AND dispersed strings with no per-dataset tuning. O(n+m) per pair
    (linear-time compression), strictly cheaper than ALCS's O(n*m) DP."""
    import zlib
    enc = [str(s).encode('utf-8', 'ignore') for s in str1]
    dec = [str(s).encode('utf-8', 'ignore') for s in str2]
    cx = [len(zlib.compress(b)) for b in enc]
    cy = [len(zlib.compress(b)) for b in dec]
    m, n = len(enc), len(dec)
    sim = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        bi, ci = enc[i], cx[i]
        for j in range(n):
            cxy = len(zlib.compress(bi + dec[j]))
            mx = cx[i] if cx[i] >= cy[j] else cy[j]
            mn = cx[i] if cx[i] <= cy[j] else cy[j]
            d = (cxy - mn) / mx if mx > 0 else 0.0
            v = 1.0 - d
            sim[i, j] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    return sim


def _rapidfuzz_cdist(str1, str2, scorer, scale):
    """Vectorized C-speed similarity matrix via rapidfuzz.process.cdist with
    multi-thread workers. Used by parameter-free Jaro and Ratcliff-Obershelp.
    Both scorers are O(n) / near-linear and far cheaper than ALCS's O(n*m) DP."""
    from rapidfuzz import process
    a = [str(s) for s in str1]
    b = [str(s) for s in str2]
    M = process.cdist(a, b, scorer=scorer, workers=-1)
    return np.asarray(M, dtype=np.float64) / scale


def _jaro_matrix(str1, str2):
    """Plain Jaro similarity (Jaro 1989). PARAMETER-FREE: the match window
    floor(max(|a|,|b|)/2)-1 is part of Jaro's definition, NOT a tuned number;
    no prefix bonus (that would be Jaro-Winkler, which adds one constant -- we
    deliberately use plain Jaro to stay strictly parameter-free). O(|a|+|b|) per
    pair, vectorized + threaded via rapidfuzz.cdist -> faster than ALCS."""
    from rapidfuzz.distance import Jaro
    return _rapidfuzz_cdist(str1, str2, Jaro.normalized_similarity, 1.0)


def _ro_matrix(str1, str2):
    """Ratcliff-Obershelp / Gestalt similarity = 2*M/(|a|+|b|) over matching
    blocks (the same ratio as difflib.SequenceMatcher.ratio). PARAMETER-FREE:
    no q, no threshold, no tuned constant. Computed with rapidfuzz.fuzz.ratio,
    the C implementation of the Gestalt ratio (validated == difflib on toy
    strings), so it is FASTER than ALCS despite being alignment-style."""
    from rapidfuzz import fuzz
    return _rapidfuzz_cdist(str1, str2, fuzz.ratio, 100.0)


def _idfgram_matrix(str1, str2):
    """IDF-weighted all-q-gram cosine. Uses ALL q-grams q=1..L (L=longest string
    in the two columns) weighted by IDF computed FROM THE TWO INPUT COLUMNS
    THEMSELVES (data-driven document frequency over the source+target values).
    This is the principled answer to 'Jaccard needs a tuned q': q is NOT fixed,
    it spans every length, and the weighting is learned from the inputs (never
    from truth). PARAMETER-FREE in the perf sense -- no constant is tuned against
    F1. FLAG: q is capped at L = the longest observed string (a natural, data-set
    cap, not a hand-picked number); IDF uses the standard log((N+1)/(df+1))+1
    smoothing from the published TF-IDF definition. O(L^2) grams per string but
    L is small for these benches (and prose is sentence-length, still tractable)."""
    a = [str(s).lower() for s in str1]
    b = [str(s).lower() for s in str2]
    docs = a + b
    N = len(docs)
    Lmax = max((len(s) for s in docs), default=0)

    # COMPUTE GUARD (flagged): cap q-gram length at QMAX. This is NOT tuned for
    # F1 -- it is a quadratic-cost guard so prose-length strings stay tractable.
    # q-grams longer than ~12 chars are almost all unique (IDF-saturated) and add
    # negligible cosine signal; the default keeps the metric effectively "all q".
    QMAX = 12

    def grams(s):
        # ALL q-grams q=1..min(len(s),QMAX): cosine over the binary presence vector
        # with IDF weights is scale-free and (modulo the guard) parameter-free.
        out = set()
        ls = len(s)
        qhi = ls if ls < QMAX else QMAX
        for q in range(1, qhi + 1):
            for i in range(ls - q + 1):
                out.add(s[i:i + q])
        return out

    gdocs = [grams(s) for s in docs]
    df = Counter()
    for g in gdocs:
        for tok in g:
            df[tok] += 1
    # standard smoothed IDF (definitional, not tuned)
    idf = {tok: math.log((N + 1.0) / (c + 1.0)) + 1.0 for tok, c in df.items()}
    # precompute IDF-weighted L2 norms over the binary presence vectors
    ga = gdocs[:len(a)]
    gb = gdocs[len(a):]
    na = [math.sqrt(sum(idf[t] * idf[t] for t in g)) for g in ga]
    nb = [math.sqrt(sum(idf[t] * idf[t] for t in g)) for g in gb]
    m, n = len(a), len(b)
    sim = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        gi, ni = ga[i], na[i]
        if ni == 0.0:
            continue
        # iterate the smaller set for the dot product
        for j in range(n):
            gj, nj = gb[j], nb[j]
            if nj == 0.0:
                continue
            if len(gi) <= len(gj):
                dot = sum(idf[t] * idf[t] for t in gi if t in gj)
            else:
                dot = sum(idf[t] * idf[t] for t in gj if t in gi)
            sim[i, j] = dot / (ni * nj)
    return sim


def _idfcos_matrix(str1, str2):
    """FAST vectorized idfgram: IDF-weighted BINARY char-gram cosine via ONE sparse matmul
    -- identical metric to _idfgram_matrix (binary presence, data-driven IDF, all q=1..L) but
    3-8x FASTER than ALCS instead of ~4x slower (sparse Xs @ Xt^T, not per-pair). Input-only,
    parameter-free; QMAX caps gram length (a compute guard, NOT tuned for F1)."""
    from scipy import sparse as _sp
    QMAX = 12
    a = [str(s).lower() for s in str1]; b = [str(s).lower() for s in str2]
    docs = a + b; N = len(docs)
    def gset(s):
        ls = len(s); hi = ls if (QMAX <= 0 or ls < QMAX) else QMAX
        return set(s[i:i+q] for q in range(1, hi+1) for i in range(ls-q+1))
    gd = [gset(s) for s in docs]
    vocab = {}; df = {}
    for g in gd:
        for t in g:
            if t not in vocab: vocab[t] = len(vocab); df[t] = 0
            df[t] += 1
    V = len(vocab)
    if V == 0: return np.zeros((len(a), len(b)))
    idf = np.empty(V)
    for t, i in vocab.items(): idf[i] = math.log((N + 1.0) / (df[t] + 1.0)) + 1.0
    def build(gl):
        r = []; c = []; v = []
        for ri, g in enumerate(gl):
            for t in g:
                j = vocab[t]; r.append(ri); c.append(j); v.append(idf[j])  # binary presence x idf
        X = _sp.csr_matrix((v, (r, c)), shape=(len(gl), V))
        nr = np.sqrt(X.multiply(X).sum(axis=1)).A1; nr[nr == 0] = 1.0
        return _sp.diags(1.0/nr) @ X
    return (build(gd[:len(a)]) @ build(gd[len(a):]).T).toarray()


# Registry: _METRIC name -> matrix builder (input strings only, no truth).
_METRIC_BUILDERS = {
    'ncd': _ncd_matrix,
    'jaro': _jaro_matrix,
    'ro': _ro_matrix,
    'idfgram': _idfgram_matrix,
    'idfcos': _idfcos_matrix,
}


def _matrix_to_edit_tuple(sim_mat):
    """Shared tail used by every metric: derive (mean_max, sim, freq_penalty, sim)
    from a similarity matrix exactly as _jaccard_matrix_as_edit_dist does, reusing
    the production frequency-overlap penalty. No truth, no tuning."""
    mask = sim_mat == np.max(sim_mat, axis=1, keepdims=True)
    ri = [np.where(row)[0] for row in mask]
    cm = sim_mat.T == np.max(sim_mat.T, axis=1, keepdims=True)
    ri2 = [np.where(c)[0] for c in cm]
    fp = np.sum(get_penalty_overlap_counts(ri)) + np.sum(get_penalty_overlap_counts(ri2))
    mm = float(np.mean(np.max(sim_mat, axis=1)))
    return mm, sim_mat, fp, sim_mat


def _jaccard_matrix_as_edit_dist(list1, list2, check='max_alcs'):
    """Drop-in replacement: Jaccard q-gram similarity instead of LCS.
    JACCARD_MODE in {fixed, adaptive, blend, containment} (fixed = single q=_JACCARD_N).
    When _METRIC names a parameter-free metric (ncd/jaro/ro/idfgram) this single
    chokepoint dispatches to it instead -- all USE_JACCARD branches + the final join
    route through here, so this is the only edit needed.
    Returns same (mean_max, sim_matrix, freq_penalty, lcs_matrix) tuple."""
    str1 = [str(s) for s in list1]
    str2 = [str(s) for s in list2]
    m, n = len(str1), len(str2)
    if m == 0 or n == 0:
        return 0.0, np.zeros((max(m,1), max(n,1))), 0, np.zeros((max(m,1), max(n,1)))
    builder = _METRIC_BUILDERS.get(_TERMINAL_METRIC)
    if builder is not None:
        return _matrix_to_edit_tuple(builder(str1, str2))
    if _JACCARD_MODE == 'fixed':
        _, jac_mat = jaccard_matrix_fast(str1, str2, n=_JACCARD_N)
    else:
        _, jac_mat = jaccard_matrix_adaptive(str1, str2, mode=_JACCARD_MODE)
    mask = jac_mat == np.max(jac_mat, axis=1, keepdims=True)
    ri = [np.where(row)[0] for row in mask]
    cm = jac_mat.T == np.max(jac_mat.T, axis=1, keepdims=True)
    ri2 = [np.where(c)[0] for c in cm]
    fp = np.sum(get_penalty_overlap_counts(ri)) + np.sum(get_penalty_overlap_counts(ri2))
    mm = float(np.mean(np.max(jac_mat, axis=1)))
    return mm, jac_mat, fp, jac_mat


def _cosine_matrix_as_edit_dist(list1, list2, check='max_alcs'):
    """Drop-in replacement: SBERT cosine similarity instead of LCS.
    Returns same (mean_max, sim_matrix, freq_penalty, lcs_matrix) tuple.
    Used by V0 ablation E2 (USE_COSINE=1)."""
    str1 = [str(s) for s in list1]
    str2 = [str(s) for s in list2]
    m, n = len(str1), len(str2)
    if m == 0 or n == 0:
        return 0.0, np.zeros((max(m, 1), max(n, 1))), 0, np.zeros((max(m, 1), max(n, 1)))
    try:
        from _embed import compute_embedding_similarity
        _, cos_mat = compute_embedding_similarity(str1, str2)
        cos_mat = np.clip(cos_mat, 0.0, 1.0)
    except Exception:
        # Fall back to Jaccard if embeddings unavailable
        _, cos_mat = jaccard_matrix_fast(str1, str2, n=_JACCARD_N)
    mask = cos_mat == np.max(cos_mat, axis=1, keepdims=True)
    ri = [np.where(row)[0] for row in mask]
    cm = cos_mat.T == np.max(cos_mat.T, axis=1, keepdims=True)
    ri2 = [np.where(c)[0] for c in cm]
    fp = np.sum(get_penalty_overlap_counts(ri)) + np.sum(get_penalty_overlap_counts(ri2))
    mm = float(np.mean(np.max(cos_mat, axis=1)))
    return mm, cos_mat, fp, cos_mat


def _word_alcs_matrix_as_edit_dist(list1, list2, greedy=True):
    """Drop-in replacement: word-level ALCS instead of char-level LCS.
    Returns same (mean_max, sim_matrix, freq_penalty, lcs_matrix) tuple."""
    str1 = [str(s) for s in list1]
    str2 = [str(s) for s in list2]
    m, n = len(str1), len(str2)
    if m == 0 or n == 0:
        return 0.0, np.zeros((max(m, 1), max(n, 1))), 0, np.zeros((max(m, 1), max(n, 1)))
    return get_word_ALCS_matrix(str1, str2, min_words=1, greedy=greedy)


def get_edit_dist_matrix(list1, list2, min_len=2, ignore_chars={' '},check = 'max_alcs'):
    """
    Computes the edit distance and LCS matrices between two lists of strings.
    When USE_JACCARD=True, uses fast Jaccard bigram similarity instead.
    When USE_WORD_ALCS=True, uses word-level ALCS instead of char-level.
    When USE_COSINE=True, uses SBERT cosine similarity instead (V0 ablation E2).
    """
    if USE_COSINE:
        return _cosine_matrix_as_edit_dist(list1, list2, check)
    if USE_JACCARD:
        return _jaccard_matrix_as_edit_dist(list1, list2, check)
    if USE_WORD_ALCS:
        return _word_alcs_matrix_as_edit_dist(list1, list2, greedy=True)

    m, n = len(list1), len(list2)
    sim_matrix = np.zeros((m, n))
    lcs_matrix = np.zeros((m, n))
    cs_matrix = np.empty((m, n), dtype=object)

    str_list1 = [str(list1[i]) for i in range(m)]
    str_list2 = [str(list2[j]) for j in range(n)]

    # Top-k filtering: only compute full LCS for top 10% most promising targets per row
    # Uses cheap char-set overlap to filter, then full LCS on survivors
    TOP_K = min(n, max(5, int(n * 0.10)))  # 10% of targets, min 5
    use_topk = m * n > 400 and n > TOP_K

    if use_topk:
        # Phase 1: cheap char-set similarity to find top-k targets per source row
        for i in range(m):
            # Compute cheap similarity to all targets
            cheap_sims = [(j, _cheap_sim(str_list1[i], str_list2[j])) for j in range(n)]
            # Sort by cheap sim, take top-k
            cheap_sims.sort(key=lambda x: x[1], reverse=True)
            top_indices = [j for j, _ in cheap_sims[:TOP_K]]

            # Phase 2: full LCS only on top-k
            for j in top_indices:
                sim, lcs, cs = lcs_count(str_list1[i], str_list2[j], min_len, ignore_chars)
                sim_matrix[i, j] = sim
                lcs_matrix[i, j] = lcs
                cs_matrix[i, j] = cs
    else:
        # Small matrix — compute all pairs directly
        for i in range(m):
            for j in range(n):
                sim, lcs, cs = lcs_count(str_list1[i], str_list2[j], min_len, ignore_chars)
                sim_matrix[i, j] = sim
                lcs_matrix[i, j] = lcs
                cs_matrix[i, j] = cs


    if check == 'max_lcs':
        # Step 1: Mask for Max LCS
        mask = lcs_matrix == np.max(lcs_matrix, axis=1, keepdims=True)
        result_indices = [np.where(row)[0] for row in mask]

        # Mask for maximum LCS per column (transpose-based)
        mask2 = lcs_matrix.T == np.max(lcs_matrix.T, axis=1, keepdims=True)
        result_indices2 = [np.where(row)[0] for row in mask2]

    elif check == 'max_alcs':
        # Step 1: Mask for Max LCS
        mask = sim_matrix == np.max(sim_matrix, axis=1, keepdims=True)

        # Step 2: Indices of max LCS
        row_max_indices = [np.where(row)[0] for row in mask]

        # Step 3: Filtered distance list
        ALCS_matrix_with_max_lcs = [
            lcs_matrix[i, row_max_indices[i]] for i in range(len(row_max_indices))
        ]

        # Step 4: Mask for max sim
        max_ALCS_matrix_with_max_lcs_mask = [
            row == np.max(row) for row in ALCS_matrix_with_max_lcs
        ]

        # Step 5: Result indices
        result_indices = [
            row_max_indices[i][max_ALCS_matrix_with_max_lcs_mask[i]] for i in range(len(row_max_indices))
        ]

        col_mask = sim_matrix.T == np.max(sim_matrix.T, axis=1, keepdims=True)

        # Step 2 (Cols): Indices of those maximum values (still row-based in the transposed view)
        col_max_indices = [np.where(c)[0] for c in col_mask]


        # Step 3 (Cols): Gather the corresponding LCS values for each column’s max rows
        ALCS_matrix_col_subset = [
            lcs_matrix[col_max_indices[i], i] for i in range(len(col_max_indices))
        ]

        max_ALCS_matrix_col_subset_mask = [
            col_vals == np.max(col_vals) for col_vals in ALCS_matrix_col_subset
        ]

        result_indices2 = [
            col_max_indices[i][max_ALCS_matrix_col_subset_mask[i]]
            for i in range(len(col_max_indices))
        ]


    elif check == 'alcs':
        mask = sim_matrix == np.max(sim_matrix, axis=1, keepdims=True)
        result_indices = [np.where(row)[0] for row in mask]

        mask2 = sim_matrix.T == np.max(sim_matrix.T, axis=1, keepdims=True)
        result_indices2= [np.where(row)[0] for row in mask2]

    # get the totaly penalty for freq counts
    freq_counts_penalty  = np.sum(get_penalty_overlap_counts(result_indices))+np.sum(get_penalty_overlap_counts(result_indices2))

    # Aggregate ALCS reward — defaults to mean(row_max); set
    # UNIQUE_ALCS_REWARD=1 to use greedy 1-to-1 assignment instead.
    mean_max_ALCS = _compute_alcs_reward(sim_matrix)

    return mean_max_ALCS, sim_matrix, freq_counts_penalty,lcs_matrix

def _compute_masked_row_topk(args):
    """Worker: compute one masked row of the ALCS matrix."""
    i, s1, list2_local, mask_row, ml, ic = args
    nn = len(list2_local)
    sr, lr, cr = np.zeros(nn), np.zeros(nn), [""] * nn
    for j in range(nn):
        if mask_row[j] == 1:
            sim, lcs, cs = lcs_count(s1, list2_local[j], ml, ic)
            sr[j], lr[j], cr[j] = sim, lcs, cs
    return i, sr, lr, cr


def get_edit_dist_matrix_with_top_k(list1, list2,prev_matrix_indicies, min_len=2, ignore_chars={' '},check = 'max_alcs'):
    """Computes edit distance/LCS matrices with dedup + top-k per row.
    Uses Jaccard when USE_JACCARD=True. Uses word-level ALCS when USE_WORD_ALCS=True.
    Uses SBERT cosine when USE_COSINE=True (V0 ablation E2)."""
    if USE_COSINE:
        s1 = list1.tolist() if hasattr(list1, 'tolist') else list(list1)
        s2 = list2.tolist() if hasattr(list2, 'tolist') else list(list2)
        return _cosine_matrix_as_edit_dist(s1, s2, check)
    if USE_JACCARD:
        s1 = list1.tolist() if hasattr(list1, 'tolist') else list(list1)
        s2 = list2.tolist() if hasattr(list2, 'tolist') else list(list2)
        return _jaccard_matrix_as_edit_dist(s1, s2, check)
    if USE_WORD_ALCS:
        s1 = list1.tolist() if hasattr(list1, 'tolist') else list(list1)
        s2 = list2.tolist() if hasattr(list2, 'tolist') else list(list2)
        return _word_alcs_matrix_as_edit_dist(s1, s2, greedy=True)
    m, n = len(list1), len(list2)
    sim_matrix = np.zeros((m, n))
    lcs_matrix = np.zeros((m, n))
    cs_matrix = np.empty((m, n), dtype=object)

    list1 = list1.to_list()
    list2 = list2.to_list()

    str_list1 = [str(v) for v in list1]
    str_list2 = [str(v) for v in list2]
    TOP_K_PER_ROW = min(n, max(20, int(n * 0.10)))

    # Step 1: Collect all unique (s1, s2) pairs that need computing
    # Uses cheap_sim top-k filtering to minimize pairs
    pairs_to_compute = set()
    row_pair_map = {}  # i -> list of (j, s1, s2)
    for i in range(m):
        s1 = str_list1[i]
        active_js = [j for j in range(n) if prev_matrix_indicies[i, j] == 1]
        if len(active_js) > TOP_K_PER_ROW:
            scored = [(j, _cheap_sim(s1, str_list2[j])) for j in active_js]
            scored.sort(key=lambda x: x[1], reverse=True)
            active_js = [j for j, _ in scored[:TOP_K_PER_ROW]]
        row_pair_map[i] = [(j, s1, str_list2[j]) for j in active_js]
        for j in active_js:
            pairs_to_compute.add((s1, str_list2[j]))

    # Step 2: Compute all unique pairs in parallel batches
    unique_pairs = list(pairs_to_compute)
    cache = {}

    if NCORES > 1 and len(unique_pairs) > 100 and not _INSIDE_WORKER:
        # Parallel: split unique pairs into batches, one per worker
        batch_size = max(1, len(unique_pairs) // NCORES)
        batches = [unique_pairs[i:i+batch_size] for i in range(0, len(unique_pairs), batch_size)]
        work = [(batch, min_len, ignore_chars) for batch in batches]
        with ProcessPoolExecutor(max_workers=NCORES) as pool:
            for partial in pool.map(_compute_lcs_pairs, work):
                cache.update(partial)
    else:
        for s1, s2 in unique_pairs:
            cache[(s1, s2)] = lcs_count(s1, s2, min_len, ignore_chars)

    # Step 3: Scatter cached results into matrix
    for i, pairs in row_pair_map.items():
        for j, s1, s2 in pairs:
            sim, lcs, cs = cache[(s1, s2)]
            sim_matrix[i, j] = sim
            lcs_matrix[i, j] = lcs
            cs_matrix[i, j] = cs

    if check == 'max_lcs':
        # Step 1: Mask for Max LCS
        mask = lcs_matrix == np.max(lcs_matrix, axis=1, keepdims=True)
        result_indices = [np.where(row)[0] for row in mask]

        # Mask for maximum LCS per column (transpose-based)
        mask2 = lcs_matrix.T == np.max(lcs_matrix.T, axis=1, keepdims=True)
        result_indices2 = [np.where(row)[0] for row in mask2]

    elif check == 'max_alcs':
        # Step 1: Mask for Max LCS
        mask = sim_matrix == np.max(sim_matrix, axis=1, keepdims=True)

        # Step 2: Indices of max LCS
        row_max_indices = [np.where(row)[0] for row in mask]

        # Step 3: Filtered distance list
        ALCS_matrix_with_max_lcs = [
            lcs_matrix[i, row_max_indices[i]] for i in range(len(row_max_indices))
        ]

        # Step 4: Mask for max sim
        max_ALCS_matrix_with_max_lcs_mask = [
            row == np.max(row) for row in ALCS_matrix_with_max_lcs
        ]

        # Step 5: Result indices
        result_indices = [
            row_max_indices[i][max_ALCS_matrix_with_max_lcs_mask[i]] for i in range(len(row_max_indices))
        ]

        col_mask = sim_matrix.T == np.max(sim_matrix.T, axis=1, keepdims=True)

        # Step 2 (Cols): Indices of those maximum values (still row-based in the transposed view)
        col_max_indices = [np.where(c)[0] for c in col_mask]


        # Step 3 (Cols): Gather the corresponding LCS values for each column’s max rows
        ALCS_matrix_col_subset = [
            lcs_matrix[col_max_indices[i], i] for i in range(len(col_max_indices))
        ]

        max_ALCS_matrix_col_subset_mask = [
            col_vals == np.max(col_vals) for col_vals in ALCS_matrix_col_subset
        ]

        result_indices2 = [
            col_max_indices[i][max_ALCS_matrix_col_subset_mask[i]]
            for i in range(len(col_max_indices))
        ]

    elif check == 'alcs':
        mask = sim_matrix == np.max(sim_matrix, axis=1, keepdims=True)
        result_indices = [np.where(row)[0] for row in mask]

        mask2 = sim_matrix.T == np.max(sim_matrix.T, axis=1, keepdims=True)
        result_indices2= [np.where(row)[0] for row in mask2]

    # get the totaly penalty for freq counts
    freq_counts_penalty  = np.sum(get_penalty_overlap_counts(result_indices))+np.sum(get_penalty_overlap_counts(result_indices2))

    # Aggregate ALCS reward — defaults to mean(row_max); set
    # UNIQUE_ALCS_REWARD=1 to use greedy 1-to-1 assignment instead.
    mean_max_ALCS = _compute_alcs_reward(sim_matrix)

    return mean_max_ALCS, sim_matrix, freq_counts_penalty,lcs_matrix


def get_edit_dist_matrix_non_greedy(list1, list2, min_len=2, ignore_chars={' '},check = 'max_alcs'):
    """Computes edit distance/LCS matrices (non-greedy). Uses Jaccard when USE_JACCARD=True.
    Uses SBERT cosine when USE_COSINE=True (V0 ablation E2)."""
    if USE_COSINE:
        return _cosine_matrix_as_edit_dist(list1, list2, check)
    if USE_JACCARD:
        return _jaccard_matrix_as_edit_dist(list1, list2, check)
    if USE_WORD_ALCS:
        return _word_alcs_matrix_as_edit_dist(list1, list2, greedy=False)
    """
    Additionally, it returns the count of minimum edit distances per row.

    Parameters:
    - list1 (List[str]): The first list of strings.
    - list2 (List[str]): The second list of strings.
    - min_len (int): Minimum length for LCS consideration.
    - ignore_chars (Set[str]): Characters to ignore during comparison.

    Returns:
    - mean_min_edit_dist (float): Mean of the minimum edit distances per row.
    - distance_matrix (np.ndarray): Matrix of edit distances.
    - mean_max_lcs_len (float): Mean of the maximum LCS lengths per row.
    - row_min_count (np.ndarray): Counts of minimum edit distances per row.
    """
    m, n = len(list1), len(list2)
    sim_matrix = np.zeros((m, n))
    lcs_matrix = np.zeros((m, n))
    cs_matrix = np.empty((m, n), dtype=object)

    # Deduplicate: compute LCS only for unique (s1, s2) pairs, then scatter
    str_list1 = [str(list1[i]) for i in range(m)]
    str_list2 = [str(list2[j]) for j in range(n)]
    unique_vals1 = list(dict.fromkeys(str_list1))
    unique_vals2 = list(dict.fromkeys(str_list2))

    if len(unique_vals1) * len(unique_vals2) < m * n:
        cache = {}
        for uv1 in unique_vals1:
            for uv2 in unique_vals2:
                sim, lcs, cs = lcs_count(uv1, uv2, min_len, ignore_chars)
                cache[(uv1, uv2)] = (sim, lcs, cs)
        for i in range(m):
            for j in range(n):
                sim, lcs, cs = cache[(str_list1[i], str_list2[j])]
                sim_matrix[i, j] = sim
                lcs_matrix[i, j] = lcs
                cs_matrix[i, j] = cs
    else:
        for i in range(m):
            for j in range(n):
                sim, lcs,cs = lcs_count(list1[i], list2[j], min_len, ignore_chars)
                sim_matrix[i, j] = sim
                lcs_matrix[i, j] = lcs
                cs_matrix[i, j] = cs


    if check == 'max_lcs':
        # Step 1: Mask for Max LCS
        mask = lcs_matrix == np.max(lcs_matrix, axis=1, keepdims=True)
        result_indices = [np.where(row)[0] for row in mask]

        # Mask for maximum LCS per column (transpose-based)
        mask2 = lcs_matrix.T == np.max(lcs_matrix.T, axis=1, keepdims=True)
        result_indices2 = [np.where(row)[0] for row in mask2]

    elif check == 'max_alcs':
        # Step 1: Mask for Max LCS
        mask = sim_matrix == np.max(sim_matrix, axis=1, keepdims=True)

        # Step 2: Indices of max LCS
        row_max_indices = [np.where(row)[0] for row in mask]

        # Step 3: Filtered distance list
        ALCS_matrix_with_max_lcs = [
            lcs_matrix[i, row_max_indices[i]] for i in range(len(row_max_indices))
        ]

        # Step 4: Mask for max sim
        max_ALCS_matrix_with_max_lcs_mask = [
            row == np.max(row) for row in ALCS_matrix_with_max_lcs
        ]

        # Step 5: Result indices
        result_indices = [
            row_max_indices[i][max_ALCS_matrix_with_max_lcs_mask[i]] for i in range(len(row_max_indices))
        ]

        col_mask = sim_matrix.T == np.max(sim_matrix.T, axis=1, keepdims=True)

        # Step 2 (Cols): Indices of those maximum values (still row-based in the transposed view)
        col_max_indices = [np.where(c)[0] for c in col_mask]


        # Step 3 (Cols): Gather the corresponding LCS values for each column’s max rows
        ALCS_matrix_col_subset = [
            lcs_matrix[col_max_indices[i], i] for i in range(len(col_max_indices))
        ]

        max_ALCS_matrix_col_subset_mask = [
            col_vals == np.max(col_vals) for col_vals in ALCS_matrix_col_subset
        ]

        result_indices2 = [
            col_max_indices[i][max_ALCS_matrix_col_subset_mask[i]]
            for i in range(len(col_max_indices))
        ]


    elif check == 'alcs':
        mask = sim_matrix == np.max(sim_matrix, axis=1, keepdims=True)
        result_indices = [np.where(row)[0] for row in mask]

        mask2 = sim_matrix.T == np.max(sim_matrix.T, axis=1, keepdims=True)
        result_indices2= [np.where(row)[0] for row in mask2]

    # get the totaly penalty for freq counts
    freq_counts_penalty  = np.sum(get_penalty_overlap_counts_non_greedy(result_indices))+np.sum(get_penalty_overlap_counts_non_greedy(result_indices2))

    # Aggregate ALCS reward — defaults to mean(row_max); set
    # UNIQUE_ALCS_REWARD=1 to use greedy 1-to-1 assignment instead.
    mean_max_ALCS = _compute_alcs_reward(sim_matrix)

    return mean_max_ALCS, sim_matrix, freq_counts_penalty,lcs_matrix


def get_edit_dist_matrix_with_top_k_non_greedy(list1, list2,prev_matrix_indicies, min_len=2, ignore_chars={' '},check = 'max_alcs'):
    """Computes edit distance/LCS matrices with dedup + top-k per row.
    Uses SBERT cosine when USE_COSINE=True (V0 ablation E2)."""
    if USE_COSINE:
        s1 = list1.tolist() if hasattr(list1, 'tolist') else list(list1)
        s2 = list2.tolist() if hasattr(list2, 'tolist') else list(list2)
        return _cosine_matrix_as_edit_dist(s1, s2, check)
    if USE_JACCARD:
        s1 = list1.tolist() if hasattr(list1, 'tolist') else list(list1)
        s2 = list2.tolist() if hasattr(list2, 'tolist') else list(list2)
        return _jaccard_matrix_as_edit_dist(s1, s2, check)
    if USE_WORD_ALCS:
        s1 = list1.tolist() if hasattr(list1, 'tolist') else list(list1)
        s2 = list2.tolist() if hasattr(list2, 'tolist') else list(list2)
        return _word_alcs_matrix_as_edit_dist(s1, s2, greedy=False)
    m, n = len(list1), len(list2)
    sim_matrix = np.zeros((m, n))
    lcs_matrix = np.zeros((m, n))
    cs_matrix = np.empty((m, n), dtype=object)

    list1 = list1.to_list()
    list2 = list2.to_list()

    str_list1 = [str(v) for v in list1]
    str_list2 = [str(v) for v in list2]
    TOP_K_PER_ROW = min(n, max(20, int(n * 0.10)))

    # Collect unique pairs with top-k filtering
    pairs_to_compute = set()
    row_pair_map = {}
    for i in range(m):
        s1_val = str_list1[i]
        active_js = [j for j in range(n) if prev_matrix_indicies[i, j] == 1]
        if len(active_js) > TOP_K_PER_ROW:
            scored = [(j, _cheap_sim(s1_val, str_list2[j])) for j in active_js]
            scored.sort(key=lambda x: x[1], reverse=True)
            active_js = [j for j, _ in scored[:TOP_K_PER_ROW]]
        row_pair_map[i] = [(j, s1_val, str_list2[j]) for j in active_js]
        for j in active_js:
            pairs_to_compute.add((s1_val, str_list2[j]))

    # Parallel unique pair computation
    unique_pairs = list(pairs_to_compute)
    cache = {}
    if NCORES > 1 and len(unique_pairs) > 100 and not _INSIDE_WORKER:
        batch_size = max(1, len(unique_pairs) // NCORES)
        batches = [unique_pairs[i:i+batch_size] for i in range(0, len(unique_pairs), batch_size)]
        work = [(batch, min_len, ignore_chars) for batch in batches]
        with ProcessPoolExecutor(max_workers=NCORES) as pool:
            for partial in pool.map(_compute_lcs_pairs, work):
                cache.update(partial)
    else:
        for s1_val, s2_val in unique_pairs:
            cache[(s1_val, s2_val)] = lcs_count(s1_val, s2_val, min_len, ignore_chars)

    # Scatter
    for i, pairs in row_pair_map.items():
        for j, s1_val, s2_val in pairs:
            sim, lcs, cs = cache[(s1_val, s2_val)]
            sim_matrix[i, j] = sim
            lcs_matrix[i, j] = lcs
            cs_matrix[i, j] = cs

    if check == 'max_lcs':
        # Step 1: Mask for Max LCS
        mask = lcs_matrix == np.max(lcs_matrix, axis=1, keepdims=True)
        result_indices = [np.where(row)[0] for row in mask]

        # Mask for maximum LCS per column (transpose-based)
        mask2 = lcs_matrix.T == np.max(lcs_matrix.T, axis=1, keepdims=True)
        result_indices2 = [np.where(row)[0] for row in mask2]

    elif check == 'max_alcs':
        # Step 1: Mask for Max LCS
        mask = sim_matrix == np.max(sim_matrix, axis=1, keepdims=True)

        # Step 2: Indices of max LCS
        row_max_indices = [np.where(row)[0] for row in mask]

        # Step 3: Filtered distance list
        ALCS_matrix_with_max_lcs = [
            lcs_matrix[i, row_max_indices[i]] for i in range(len(row_max_indices))
        ]

        # Step 4: Mask for max sim
        max_ALCS_matrix_with_max_lcs_mask = [
            row == np.max(row) for row in ALCS_matrix_with_max_lcs
        ]

        # Step 5: Result indices
        result_indices = [
            row_max_indices[i][max_ALCS_matrix_with_max_lcs_mask[i]] for i in range(len(row_max_indices))
        ]

        col_mask = sim_matrix.T == np.max(sim_matrix.T, axis=1, keepdims=True)

        # Step 2 (Cols): Indices of those maximum values (still row-based in the transposed view)
        col_max_indices = [np.where(c)[0] for c in col_mask]


        # Step 3 (Cols): Gather the corresponding LCS values for each column’s max rows
        ALCS_matrix_col_subset = [
            lcs_matrix[col_max_indices[i], i] for i in range(len(col_max_indices))
        ]

        max_ALCS_matrix_col_subset_mask = [
            col_vals == np.max(col_vals) for col_vals in ALCS_matrix_col_subset
        ]

        result_indices2 = [
            col_max_indices[i][max_ALCS_matrix_col_subset_mask[i]]
            for i in range(len(col_max_indices))
        ]

    elif check == 'alcs':
        mask = sim_matrix == np.max(sim_matrix, axis=1, keepdims=True)
        result_indices = [np.where(row)[0] for row in mask]

        mask2 = sim_matrix.T == np.max(sim_matrix.T, axis=1, keepdims=True)
        result_indices2= [np.where(row)[0] for row in mask2]

    # get the totaly penalty for freq counts
    freq_counts_penalty  = np.sum(get_penalty_overlap_counts_non_greedy(result_indices))+np.sum(get_penalty_overlap_counts_non_greedy(result_indices2))

    # Aggregate ALCS reward — defaults to mean(row_max); set
    # UNIQUE_ALCS_REWARD=1 to use greedy 1-to-1 assignment instead.
    mean_max_ALCS = _compute_alcs_reward(sim_matrix)

    return mean_max_ALCS, sim_matrix, freq_counts_penalty,lcs_matrix


def get_ALCS_matrix_with_top_k(list1, list2,prev_matrix_indicies,greedy, min_len=2, ignore_chars={' '},check = 'max_alcs', match_mode_override=None):
    global USE_WORD_ALCS
    # v8 forced mode (set by compute_all_pairs_similarity per candidate pair)
    if match_mode_override is None and _V8_FORCED_MATCH_MODE is not None:
        match_mode_override = _V8_FORCED_MATCH_MODE
    # The lazy fuzzy / auto_fc similarity variants are NOT part of this release
    # (the default metric is idfcos + cov_mnn). If those modes are explicitly
    # requested but the optional module is unavailable, degrade to standard
    # char/token ALCS rather than failing.
    if match_mode_override == 'fuzzy':
        match_mode_override = 2  # token-level char ALCS
    if match_mode_override == 'auto_fc':
        match_mode_override = 1  # char-level ALCS
    # Learned match mode takes priority, then adaptive heuristic
    if match_mode_override is not None:
        min_len = match_mode_override
    else:
        avg_len1 = np.mean([len(str(x)) for x in list1]) if len(list1) > 0 else 0
        avg_len2 = np.mean([len(str(x)) for x in list2]) if len(list2) > 0 else 0
        if min(avg_len1, avg_len2) < 4:
            min_len = 1

    if greedy:
        mean_max_ALCS, sim_matrix, freq_counts_penalty,lcs_matrix = get_edit_dist_matrix_with_top_k(list1, list2,prev_matrix_indicies,
                                                                                                    min_len, ignore_chars,check)
    else:
        mean_max_ALCS, sim_matrix, freq_counts_penalty,lcs_matrix = get_edit_dist_matrix_with_top_k_non_greedy(list1, list2,prev_matrix_indicies,
                                                                                                     min_len, ignore_chars,check)
    # Word-ALCS fallback: if degenerate, retry char-level
    if USE_WORD_ALCS and (mean_max_ALCS < 0.05 or
            (mean_max_ALCS > 0.98 and np.std(np.max(sim_matrix, axis=1)) < 0.01)):
        _saved = USE_WORD_ALCS
        USE_WORD_ALCS = False
        if greedy:
            mean_max_ALCS, sim_matrix, freq_counts_penalty, lcs_matrix = get_edit_dist_matrix_with_top_k(list1, list2, prev_matrix_indicies, min_len, ignore_chars, check)
        else:
            mean_max_ALCS, sim_matrix, freq_counts_penalty, lcs_matrix = get_edit_dist_matrix_with_top_k_non_greedy(list1, list2, prev_matrix_indicies, min_len, ignore_chars, check)
        USE_WORD_ALCS = _saved
    return  mean_max_ALCS, sim_matrix, freq_counts_penalty,lcs_matrix


# --- ALCS matrix cache (per-process; helps when multiple rollouts visit same state) ---
# Bounded LRU eviction at 2048 entries to avoid memory blowup on large benchmarks.
# Enable via ALCS_CACHE=1. Safe because each worker process handles one dataset.
_ALCS_CACHE = {}
_ALCS_CACHE_MAX = 2048
_ALCS_CACHE_HITS = 0
_ALCS_CACHE_MISSES = 0

# Q2 (SHARE_SAMPLE_PER_DIRECTION=1): module-level cache of diverse-sample
# results keyed by (id(df_a), column_a_name, sample_size). Cleared at the entry
# of each variations loop so different datasets don't collide. All rollouts of
# the same direction reuse the same df_a_sampled — makes the ALCS cache effective
# for O8's many-seed design.
_SHARED_SAMPLE_CACHE = {}

# _LOG_COMBOS=1: captures every per-rollout combo (in addition to the selected
# winner) so a downstream probe can compute per-combo F1 and verify whether
# max(edit_dist) really equals max(F1). Cleared at variations-loop entry; the
# runner reads this dict after each chain call.
_PROBE_COMBOS = {'rollouts': []}

# LOG_PAIR_SIMS=1: captures (sim_matrix, threshold, concated_df_a, concated_df_b,
# alcs_sim) per chain call so a downstream probe can label each (i, j) cell as
# truth-positive / false-positive / true-negative / false-negative and plot the
# similarity-score distribution. Appended (not cleared) so all rollouts of all
# datasets accumulate; the runner reads + clears between datasets.
_PROBE_PAIR_SIMS = {'calls': []}

# _LOG_COMBOS=1: captures every CANDIDATE combo (not just the selected one)
# during pair/chain selection, so a downstream analysis can test whether a clear
# structural signal (mutual-NN, margin, separation, coverage, uniqueness) would
# pick the highest-true-F1 combo better than ALCS. Raw per-combo data only
# (alcs, freq_penalty, sim_matrix, matched df, transform) — no truth here; the
# runner computes structural signals + scoring F1. Cleared per dataset by runner.
_COMBO_LOG = []

# Operator-safety: the operator currently being evaluated. If a per-dataset
# time budget fires (SIGALRM) mid-evaluation, this holds the culprit op so the
# runner can block it (OP_REMOVE) and restart the search without it.
_CURRENT_OP = None
_LAST_EPS = None       # exploration rate actually used (last search call)
_LAST_TSN_EPS = None   # raw TSN-predicted exploration rate (what the learned model picks)
_LAST_MAXSTEPS = None  # max_steps actually used (TSN may reduce it)
_LAST_THRESHOLD = None # join cutoff actually used (JDN / heuristic / signal)
_LAST_GMM_VALLEY = None # GMM-valley cut on same row_max (paired vs JDN, when logged)

def _learned_off(flag):
    """True if a given trained model should be disabled — either via its own
    env flag (e.g. NO_META) or the master _NO_LEARNED=1 switch.
    Lets us ablate each trained component (meta_selector, stop_continue, TSN,
    JDN, post_transform_policy, override_gain_net) to see which can be turned
    off with no harm, and run T7 fully config-free with _NO_LEARNED=1."""
    # _NO_LEARNED is hardcoded True in the released config, so every learned
    # component (meta / stop / TSN / JDN) is disabled unconditionally.
    return _NO_LEARNED or False

def _signal_threshold(row_max, method='gmm', alcs_sim=0.0):
    """DATA-DRIVEN, NON-HEURISTIC join threshold from the row_max similarity
    distribution. No tuned constants, NO ground truth (row_max is the per-row
    best similarity from the transformed-column sim matrix — input only, never
    labels). Method: fit 1- vs 2-component Gaussian mixtures and let BIC (a
    principled model-selection criterion, no tuned threshold) decide bimodality.
    If 2 components fit better -> the data genuinely separates into match /
    non-match, so cut at the FITTED decision boundary (where the high-mean
    component takes over). If 1 wins -> unimodal (e.g. all rows match) -> admit
    all. Replaces both JDN and the k=7/pct=75/alpha heuristic."""
    import numpy as _np
    v = _np.asarray(row_max, dtype=float).flatten()
    v = v[_np.isfinite(v)]
    if v.size < 4 or _np.unique(v).size < 2:
        return (float(v.min()) - 1e-6) if v.size else 0.5
    X = v.reshape(-1, 1)
    try:
        from sklearn.mixture import GaussianMixture
        if method in ('cond', 'condf'):
            # CONDITION-SWITCHED: fit GMM (auto-k via BIC), then switch on the
            # SEPARATION QUALITY of the valley (d' = standardized gap between the
            # two modes around the dominant valley). Cleanly separated (d'>=2, a
            # 2-sigma effect-size convention) => trust the valley (recall-limited
            # / prose). Weakly separated => conservative cut at the top mode's
            # lower boundary (mu_hi - sd_hi) to avoid over-admission on
            # high-precision / low-separation distributions (autojoin/initials).
            kmax = int(min(7, _np.unique(v).size, max(2, v.size // 3)))
            models = {}
            for k in range(1, kmax + 1):
                try:
                    models[k] = GaussianMixture(k, random_state=0).fit(X)
                except Exception:
                    pass
            if not models:
                return float(v.min()) - 1e-6
            kbest = min(models, key=lambda k: models[k].bic(X))
            if kbest == 1:
                # condf: unimodal/uninformative -> robust ALCS floor (mimics
                # JDN's stable cut) instead of admit-all, which over-admits when
                # alcs is moderate (the initials-long failure). cond: admit-all.
                if method == 'condf':
                    return float(alcs_sim)
                return float(v.min()) - 1e-6
            g = models[kbest]
            mu = g.means_.flatten()
            sd = _np.sqrt(_np.maximum(g.covariances_.flatten(), 1e-12))
            o = _np.argsort(mu)
            mu_s, sd_s = mu[o], sd[o]
            j = int(_np.argmax(_np.diff(mu_s)))   # dominant valley: comp j..j+1
            dprime = (mu_s[j + 1] - mu_s[j]) / _np.sqrt(sd_s[j] ** 2 + sd_s[j + 1] ** 2)
            if dprime >= 2.0:                     # cleanly separated -> valley
                return float((mu_s[j] + mu_s[j + 1]) / 2.0)
            return float(mu_s[-1] - sd_s[-1])     # weak -> conservative top-mode boundary
        if method == 'gmmk':
            # AUTO-k: let BIC choose the number of similarity modes (1..Kmax),
            # capturing multi-cluster structure the heuristic's k=7 was after
            # (and letting occupancy decide — BIC won't keep empty modes). Then
            # cut at the LARGEST gap between adjacent fitted component means (the
            # dominant valley). k* and the cut both come from the data — no 7/75.
            kmax = int(min(7, _np.unique(v).size, max(2, v.size // 3)))
            models = {}
            for k in range(1, kmax + 1):
                try:
                    models[k] = GaussianMixture(k, random_state=0).fit(X)
                except Exception:
                    pass
            if not models:
                return float(v.min()) - 1e-6
            kbest = min(models, key=lambda k: models[k].bic(X))
            if kbest == 1:
                return float(v.min()) - 1e-6   # unimodal -> admit all
            means = _np.sort(models[kbest].means_.flatten())
            j = int(_np.argmax(_np.diff(means)))   # dominant valley between modes
            return float((means[j] + means[j + 1]) / 2.0)
        # default 'gmm': 1 vs 2 components
        g1 = GaussianMixture(1, random_state=0).fit(X)
        g2 = GaussianMixture(2, random_state=0).fit(X)
        if g2.bic(X) < g1.bic(X):  # genuinely bimodal
            mu = g2.means_.flatten()
            lo_m, hi_m, hi_c = float(mu.min()), float(mu.max()), int(_np.argmax(mu))
            xs = _np.linspace(lo_m, hi_m, 256).reshape(-1, 1)
            sw = xs.flatten()[g2.predict(xs) == hi_c]
            return float(sw.min()) if sw.size else (lo_m + hi_m) / 2.0
        return float(v.min()) - 1e-6   # unimodal -> admit all
    except Exception:
        return float(v.min()) - 1e-6

def _kmeans_k_auto(row_max, kmax=7):
    """DATA-DRIVEN replacement for JDN's LEARNED KMeans k. Keeps JDN's threshold
    machinery intact (KMeans on row_max -> percentile-band cut), but chooses the
    number of similarity bands k by the SILHOUETTE score of the 1D row_max
    clustering instead of a trained net. No labels, no tuned constant — k is the
    partition the data's own cluster separation prefers, capped at JDN's range
    (<=7).

    Why this mimics what makes JDN work: JDN's robustness on unimodal /
    uninformative distributions comes from the clustering-quantile structure
    (always returns a sensible high-band cut), NOT from valley detection — so we
    only need to supply k. On a clean multi-modal row_max silhouette finds the
    natural k; on a unimodal blob it returns the smallest k (coarse bands ->
    conservative high cut), reproducing JDN's stable behaviour without the net."""
    import numpy as _np
    v = _np.asarray(row_max, dtype=float).reshape(-1, 1)
    n = v.shape[0]
    uniq = int(_np.unique(v).size)
    if n < 4 or uniq < 3:
        return min(2, uniq) if uniq >= 2 else 1
    kmax = int(min(kmax, uniq - 1, max(2, n // 2)))
    if kmax < 2:
        return 1
    try:
        from sklearn.metrics import silhouette_score
        best_k, best_s = 2, -2.0
        for k in range(2, kmax + 1):
            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(v)
                if len(set(km.labels_)) < 2:
                    continue
                s = silhouette_score(v, km.labels_)
                if s > best_s:
                    best_s, best_k = s, k
            except Exception:
                pass
        return best_k
    except Exception:
        return min(kmax, 7)

def _jdn_like_threshold(row_max, min_avg_length):
    """FULLY DATA-DRIVEN simulation of JDN's threshold (no net, no labels).
    Reproduces JDN's whole machine -- KMeans-band the row_max distribution, cut
    at the match/non-match boundary, then apply the alpha nudge / cap / ALCS-floor
    -- but derives EVERY parameter JDN learned from the distribution itself:

      k     : silhouette-optimal #bands (granularity from cluster separation),
              replacing JDN's learned kmeans_k.
      band  : the MATCH group = clusters above the LARGEST inter-center gap (the
              dominant valley in clustered space), replacing the fixed pct=75 --
              this is what decides which bands count as matches.
      cut   : the match band's MEAN row_max (JDN cuts at a cluster mean too).
      alpha : the match band's STD, so the finalize step (round(cut) - alpha)
              drops the cut to the band's LOWER EDGE -> admits the WHOLE match
              cluster, not just its top half (fixes the pct=75 under-admit).
              Replaces the length-tiered -0.05/-0.025/+0.025 constants.
      cap   : max(row_max) -- never clips a real cut, replacing the fixed 0.8.

    The caller still applies the ALCS floor (max(alcs_sim, cut)) and the 0.2
    clamp, so those knobs stay intact. Returns (cut, alpha, cap).

    Honest limit: on a UNIMODAL blob there is no real valley, so the largest gap
    is an artifact and this lands at the top sub-band floored by ALCS -- the same
    place JDN's machine lands MINUS the learned per-instance offset that only
    labels could supply. That residual is the irreducible no-label gap."""
    import numpy as _np
    v = _np.asarray(row_max, dtype=float).flatten()
    v = v[_np.isfinite(v)]
    cap = float(v.max()) if v.size else 0.8
    if v.size < 4 or _np.unique(v).size < 3:
        return (float(v.min()) if v.size else 0.5), 0.0, cap
    k = _kmeans_k_auto(v, kmax=7)
    if k < 2:
        return float(_np.mean(v)), 0.0, cap
    try:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(v.reshape(-1, 1))
        centers = km.cluster_centers_.flatten()
        order = _np.argsort(centers)
        cs = centers[order]
        j = int(_np.argmax(_np.diff(cs)))           # dominant valley: bands j | j+1
        match_lbls = order[j + 1:]                   # clusters above the valley = matches
        match_mask = _np.isin(km.labels_, match_lbls)
        if not match_mask.any():
            return float(_np.mean(v)), 0.0, cap
        mv = v[match_mask]
        return float(mv.mean()), float(mv.std()), cap
    except Exception:
        return float(_np.mean(v)), 0.0, cap

def _combo_cov(df_matches_multi, sim_matrix):
    """Coverage = matched rows / source rows — a truth-free recall proxy.
    Used as the candidate-combo selector under _COVERAGE_SELECT=1
    (validated to beat ALCS argmax: ALCS prefers clean-but-partial transforms;
    coverage prefers the transform that joined more rows). 1-to-many can exceed
    1.0 (n_pairs > n_src) — still ranks correctly."""
    try:
        n = sim_matrix.shape[0]
        if n and df_matches_multi is not None:
            return float(len(df_matches_multi)) / float(n)
    except Exception:
        pass
    return -1.0

def _combo_mutual_nn(sim_matrix):
    """Bijectivity / mutual-NN rate: fraction of source rows whose top-1 target
    also picks that source as ITS top-1. Truth-free PRECISION proxy. Pairing it
    with coverage (cov*mnn) stops the coverage selector from picking
    high-coverage / low-precision (over-admitting) chains. Neutral (1.0) on
    failure so it never penalizes when uncomputable."""
    try:
        import numpy as _np
        M = _np.asarray(sim_matrix, dtype=float)
        if M.ndim != 2 or M.shape[0] == 0 or M.shape[1] == 0:
            return 1.0
        n = M.shape[0]
        ra = M.argmax(axis=1)   # each src row's best target
        ca = M.argmax(axis=0)   # each target's best src row
        return float((_np.arange(n) == ca[ra]).mean())
    except Exception:
        return 1.0

def _combo_uniqueness(df_matches_multi):
    """Distinct-target rate of the matched pairs: distinct target rows /
    total matched pairs. Truth-free PRECISION proxy that penalizes
    many-source-to-one-target collisions (over-admission). Mirrors the
    runner's `uniq = len(set(t for _,t in pairs)) / len(pairs)`.

    The matched pairs live in df_matches_multi, where the target-row identity
    is the 'target-<tgt_col>' column (col name forced via _STEP1_TGT_COL,
    matching baselines/v0_ablation_runner.py). Falls back to any 'target-'
    prefixed column. Neutral (1.0) on failure/empty so it never penalizes when
    uncomputable, like _combo_mutual_nn."""
    try:
        if df_matches_multi is None:
            return 1.0
        n_pairs = len(df_matches_multi)
        if not n_pairs:
            return 1.0
        tgt_col = _STEP1_TGT_COL
        if not tgt_col or tgt_col not in df_matches_multi.columns:
            tgt_cands = [c for c in df_matches_multi.columns if str(c).startswith('target-')]
            if not tgt_cands:
                return 1.0
            tgt_col = tgt_cands[0]
        n_distinct = df_matches_multi[tgt_col].astype(str).str.strip().nunique()
        return float(n_distinct) / float(n_pairs)
    except Exception:
        return 1.0

def _threshold_by_covmnn(sim_matrix, floor=0.2):
    """SEARCH OVER CUTS for the join threshold: pick the row_max cut whose
    admitted set best agrees with the mutual-NN (bijective) set, instead of a
    fixed rule (condf / JDN / k=7-pct=75). Unifies the threshold with the SAME
    truth-free quantities that drive chain selection (coverage + bijectivity).

    Treat 'source row i is mutually-NN' (its top-1 target picks i back) as a
    truth-free pseudo-label for 'should be matched'. For a candidate cut t the
    admitted set is {i : row_max_i >= t}. Score the cut by F1 between admitted
    and bijective:
      precision(t) = |admitted & bijective| / |admitted|   (truth-free precision)
      recall(t)    = |admitted & bijective| / |bijective|  (truth-free coverage)
      F1(t)        = 2PR/(P+R)
    NOTE on why F1, not the raw cov*mnn product: cov(t)*mnn_rate(t) =
    (|adm|/n)*(|adm&bij|/|adm|) = |adm&bij|/n, where |adm| CANCELS -> the product
    is monotone in t and degenerates to admit-all (verified). Making recall
    relative to |bijective| instead of n breaks the cancellation and restores a
    real interior optimum: F1=1 exactly when the cut admits the bijective set
    and nothing else. LOW cut on clean data (the bijective core all has high
    row_max -> admit it in full: fixes univ under-admit), HIGH cut on noisy data
    (non-bijective rows whose row_max is high by chance get excluded -> fixes
    initials over-admit). Truth-free (row_max + bijectivity, never labels) and
    non-heuristic (argmax of F1 over the data's own row_max levels, no tuned
    constant). Returns the chosen cut, clamped to >= floor."""
    try:
        import numpy as _np
        M = _np.asarray(sim_matrix, dtype=float)
        if M.ndim != 2 or M.shape[0] == 0 or M.shape[1] == 0:
            return floor
        n = M.shape[0]
        row_max = M.max(axis=1)
        ra = M.argmax(axis=1)
        ca = M.argmax(axis=0)
        bij = (_np.arange(n) == ca[ra])          # truth-free 'should match' set
        B = int(bij.sum())
        if B == 0:                               # no bijective signal -> admit all
            return floor
        cuts = _np.unique(row_max)
        best_t, best_f1 = float(cuts[0]), -1.0
        for t in cuts:
            adm = row_max >= t
            k = int(adm.sum())
            if k == 0:
                continue
            tp = int((adm & bij).sum())
            if tp == 0:
                continue
            prec = tp / float(k)
            rec = tp / float(B)
            f1 = 2.0 * prec * rec / (prec + rec)
            if f1 > best_f1:                      # ties keep the LOWER cut
                best_f1, best_t = f1, float(t)
        return max(floor, best_t)
    except Exception:
        return floor

def _alcs_cache_key(list1, list2, greedy, min_len, check, match_mode_override):
    # Fast hash: tuple of values (lists may be long; tuple is hashable)
    return (tuple(str(v) for v in list1), tuple(str(v) for v in list2),
            bool(greedy), int(min_len), check, match_mode_override,
            USE_JACCARD, USE_WORD_ALCS, USE_COSINE,
            _V8_FORCED_MATCH_MODE)


def get_ALCS_matrix(list1, list2,greedy, min_len=2, ignore_chars={' '},check = 'max_alcs', match_mode_override=None):
    global USE_WORD_ALCS, _ALCS_CACHE_HITS, _ALCS_CACHE_MISSES
    # Cache check (gated by env var so the default behavior is unchanged)
    _cache_on = False
    if _cache_on:
        try:
            _key = _alcs_cache_key(list1, list2, greedy, min_len, check, match_mode_override)
            if _key in _ALCS_CACHE:
                _ALCS_CACHE_HITS += 1
                return _ALCS_CACHE[_key]
            _ALCS_CACHE_MISSES += 1
        except Exception:
            _key = None
    else:
        _key = None
    # v8 forced mode (set by compute_all_pairs_similarity per candidate pair)
    if match_mode_override is None and _V8_FORCED_MATCH_MODE is not None:
        match_mode_override = _V8_FORCED_MATCH_MODE
    # The lazy fuzzy / auto_fc similarity variants are NOT part of this release;
    # degrade an explicit request to standard char/token ALCS (see the matching
    # comment in get_ALCS_matrix_with_top_k).
    if match_mode_override == 'fuzzy':
        match_mode_override = 2  # token-level char ALCS
    if match_mode_override == 'auto_fc':
        match_mode_override = 1  # char-level ALCS
    # Learned match mode takes priority, then adaptive heuristic
    if match_mode_override is not None:
        min_len = match_mode_override
    else:
        avg_len1 = np.mean([len(str(x)) for x in list1]) if len(list1) > 0 else 0
        avg_len2 = np.mean([len(str(x)) for x in list2]) if len(list2) > 0 else 0
        if min(avg_len1, avg_len2) < 4:
            min_len = 1
    if greedy:
        mean_max_ALCS, sim_matrix, freq_counts_penalty,lcs_matrix = get_edit_dist_matrix(list1, list2,min_len,ignore_chars,check)
    else:
        mean_max_ALCS, sim_matrix, freq_counts_penalty,lcs_matrix = get_edit_dist_matrix_non_greedy(list1,list2,min_len,ignore_chars,check)
    # Word-ALCS fallback: if degenerate (near-zero or saturated), retry char-level
    if USE_WORD_ALCS and (mean_max_ALCS < 0.05 or
            (mean_max_ALCS > 0.98 and np.std(np.max(sim_matrix, axis=1)) < 0.01)):
        _saved = USE_WORD_ALCS
        USE_WORD_ALCS = False
        if greedy:
            mean_max_ALCS, sim_matrix, freq_counts_penalty, lcs_matrix = get_edit_dist_matrix(list1, list2, min_len, ignore_chars, check)
        else:
            mean_max_ALCS, sim_matrix, freq_counts_penalty, lcs_matrix = get_edit_dist_matrix_non_greedy(list1, list2, min_len, ignore_chars, check)
        USE_WORD_ALCS = _saved
    _result = (mean_max_ALCS, sim_matrix, freq_counts_penalty, lcs_matrix)
    if _cache_on and _key is not None:
        # LRU-ish: drop oldest when full
        if len(_ALCS_CACHE) >= _ALCS_CACHE_MAX:
            try: _ALCS_CACHE.pop(next(iter(_ALCS_CACHE)))
            except StopIteration: pass
        _ALCS_CACHE[_key] = _result
    return _result

# #for discovery



def concatenate_with_order(df,concated_col_name):
    df_copy = df.copy()

    columns_using = [column for column in df_copy.columns if column != concated_col_name]

    # _SAFE_CONCAT=1: transpose-free row concatenation. The default path
    # below uses DataFrame.agg(join, axis=1), which internally TRANSPOSES the
    # frame and crashes on some intermediate shapes under pandas 3.0 (observed
    # as CHAIN_ERROR, e.g. 'duke cs profs'). Reducing column-wise with Series
    # '+' is vectorized, never transposes, and is output-identical otherwise.
    if _SAFE_CONCAT:
        _sep = ' ' if USE_WORD_ALCS else ''
        _parts = [df_copy[c].fillna('').astype(str) for c in columns_using]
        if _parts:
            _joined = _parts[0]
            for _p in _parts[1:]:
                _joined = _joined + _sep + _p
        else:
            _joined = pd.Series([''] * len(df_copy), index=df_copy.index)
        df_copy[concated_col_name] = _joined
        if USE_WORD_ALCS:
            df_copy[concated_col_name] = df_copy[concated_col_name].str.replace(r'\s+', ' ', regex=True).str.strip()
    elif USE_WORD_ALCS:
        # Word-level mode: join with space to preserve word boundaries
        df_copy[concated_col_name] = df_copy[columns_using].fillna('').astype(str).agg(' '.join, axis=1)
        df_copy[concated_col_name] = df_copy[concated_col_name].str.replace(r'\s+', ' ', regex=True).str.strip()
    else:
        df_copy[concated_col_name] = df_copy[columns_using].fillna('').astype(str).agg(''.join, axis=1)

    return df_copy


def get_max_counts(matrix):
  max_values = np.max(matrix, axis=1)

  # Count occurrences of the maximum value in each row
  max_counts = np.sum(matrix == max_values[:, None], axis=1)
  return max_counts

def get_min_counts(matrix):
  min_values = np.min(matrix, axis=1)

  # Count occurrences of the maximum value in each row
  min_counts = np.sum(matrix == min_values[:, None], axis=1)
  return min_counts

def unique_len(row_max):
    # Flatten the list of arrays and find unique elements
    flattened = np.concatenate(row_max)
    unique_elements = np.unique(flattened)
    return len(unique_elements)


def get_overlap_counts(arrays):
    flattened = [number for array in arrays for number in array]
    frequency = Counter(flattened)
    overlap_counts = [sum(frequency[number] for number in array) for array in arrays]
    return overlap_counts

def check_one_in_another(arrays_1, arrays_2):
    """
    check if all elements from array_2 are in array_1 
    """
    for array_1, array_2 in zip(arrays_1, arrays_2):
        if not all(element in array_1 for element in array_2):
            return False
    return True


def more_positive_than_negative(arr,perct=1.0):
    """
    Check if the number of positive values in the array is greater than 
    the number of negative values, excluding zeros.
    
    Parameters:
        arr (numpy.ndarray): Input array.
    
    Returns:
        bool: True if positive values are in majority, False otherwise.
    """
    arr = np.array(arr) 
    positive_count = np.sum(arr > 0)*perct
    negative_count = np.sum(arr < 0)  # Exclude zeros

    return positive_count > negative_count

def more_negative_than_positive(arr):
    """
    Check if the number of positive values in the array is greater than 
    the number of non-positive (zero or negative) values.
    
    Parameters:
        arr (numpy.ndarray): Input array.
    
    Returns:
        bool: True if positive values are in majority, False otherwise.
    """
    arr = np.array(arr)  # Ensure input is a NumPy array
    positive_count = np.sum(arr > 0)
    non_positive_count = arr.size - positive_count
    return positive_count > non_positive_count


def check_words_length_greater_1(values):
  values = values.apply(str)
  for val in values:
    if len(val.split()) >1:
      return True
  return False

class QLearningAgent_edit_dist_modified_for_multi_opt:
    def __init__(
        self,
        operators,
        df,
        agreement_percentage=0.5,
        epsilon=0.1,
        learning_rate=0.1,
        discount_factor=1,
        reward_uniqueness_factor=10000,
        exclude_exist_cols=True,
        reused_operators_prob_dict=None,
        max_concate_num=3,
        depth=2,
        reward_config=None,
    ):
        self.operators = operators
        self.initial_operators = copy.deepcopy(operators)  # reset params
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(float)

        self.operator_indices = {op.name: idx for idx, op in enumerate(operators)}
        # Empty-catalog case (worktree feature): per-side OP_ONLY_*
        # can request "no ops on this side". The agent then has no actions
        # to take and downstream code treats it as identity transform.
        _n_ops = len(operators)
        if _n_ops == 0:
            self.operators_prob = []
            self.operator_dict = defaultdict(list)
            self.operators_prob_dict = defaultdict(dict)
        else:
            self.operators_prob = [1 / _n_ops] * _n_ops
            self.operator_dict = defaultdict(lambda: copy.deepcopy(operators))
            self.operators_prob_dict = defaultdict(lambda: {op.name: 1 / _n_ops for op in operators})
        self.initial_operators_prob_dict = copy.deepcopy(self.operators_prob_dict)

        self.agreement_percentage = agreement_percentage
        self.reward_uniqueness_factor = reward_uniqueness_factor
        self.exclude_exist_cols = exclude_exist_cols
        self.max_concate_num = max_concate_num

        self.reward_config = reward_config or RewardConfig()
        self.depth = max(0, int(depth))

        self.df = df
        self.reused_operators_prob_dict = reused_operators_prob_dict

        # Learned operator selection: set by find_transformed_df_opt from MetaSelector
        # selected_op_names = operators from groups the model predicts are relevant
        # unselected_op_names = operators from groups the model predicts are irrelevant
        self.selected_op_names = None   # None = use all (fallback)
        self.unselected_op_names = None

    def rename_column_key(self, old_name, new_name):
        if old_name in self.operator_dict:
            self.operator_dict[new_name] = self.operator_dict.pop(old_name)
        if old_name in self.operators_prob_dict:
            self.operators_prob_dict[new_name] = self.operators_prob_dict.pop(old_name)

    def update_cum_params(self, transformed_df_action):
        for col_name, action in transformed_df_action.items():
            print(f"start adjusting {action.name} for {col_name}")
            for op in self.operator_dict[col_name]:
                if op.name == action.name and op.name != "shift_1_word_forward":
                    op.adjust_params()
                    print(op.params)
                    break

    def reset_params(self):
        self.operators = copy.deepcopy(self.initial_operators)
        self.operator_dict = {col: copy.deepcopy(self.operators) for col in self.transformed_df.columns}

    def reset_params_prob(self):
        self.operators_prob_dict = copy.deepcopy(self.initial_operators_prob_dict)

    def update_the_operator_probs(self, col_name, action, reward):
        print(f"start adjusting {action.name} for {col_name}")
        if action.name != "Restart":
            prob = self.operators_prob_dict[col_name][action.name]
            print(f"The original prob: {prob}")
            if reward > 0:
                print("Increasing the probability for the operator")
                self.operators_prob_dict[col_name][action.name] += self.learning_rate * (1 - prob)
            elif reward < 0:
                print("Decreasing the probability for the operator")
                self.operators_prob_dict[col_name][action.name] -= self.learning_rate * prob

            self.operators_prob_dict[col_name][action.name] = max(0, self.operators_prob_dict[col_name][action.name])

            total_prob = sum(self.operators_prob_dict[col_name].values())
            if total_prob <= 0:
                # safety: reset uniform
                n = len(self.operators_prob_dict[col_name])
                for op_name in self.operators_prob_dict[col_name]:
                    self.operators_prob_dict[col_name][op_name] = 1.0 / max(1, n)
            else:
                for op_name in self.operators_prob_dict[col_name]:
                    self.operators_prob_dict[col_name][op_name] /= total_prob

            print(f"After adjusting prob: {self.operators_prob_dict[col_name][action.name]}")
            print(f"The adjusted prob operators: {list(self.operators_prob_dict[col_name].items())[:5]}...")
            print("---------------------------------------------------------------------------")

    # ---------------------------------------------------------------------
    # IMPORTANT helper: apply the returned first-step actions to regenerate
    # transformed_df / transformed_df_other exactly like your external code.
    # ---------------------------------------------------------------------
    def _apply_action_snapshot(
        self,
        *,
        df,
        df_other,
        transformed_df,
        transformed_df_other,
        transformed_df_action,
        col_for_action,
        col_other_for_action,
        col_for_col_col_other,
    ):
        transformed_df_new = transformed_df.copy()
        transformed_df_other_new = transformed_df_other.copy()

        for col_name, action in transformed_df_action.items():
            if col_name not in transformed_df_new.columns:
                continue
            if action.operator_type in ["direct", "direct_split"]:
                transformed_df_new[col_name] = [action.apply(v) for v in transformed_df_new[col_name]]

            elif action.operator_type == "concate":
                # col_for_action[col_name] stores the df column to concatenate
                df_col = col_for_action[col_name]
                transformed_df_new = action.func(transformed_df_new, col_name, df[df_col])

            elif action.operator_type == "concate_both":
                insertion_pos = col_for_col_col_other[col_name]  # which col on other side
                df_col_a = col_for_action[col_name]
                df_col_b = col_other_for_action[insertion_pos]
                transformed_df_new, transformed_df_other_new = action.func(
                    transformed_df_new,
                    transformed_df_other_new,
                    df[df_col_a],
                    df_other[df_col_b],
                    col_name,
                    insertion_pos,
                )

        return transformed_df_new, transformed_df_other_new

    # ---------------------------------------------------------------------
    # Your choose_action: SAME reward logic; ONLY fixes candidate storage.
    # It still returns:
    #   transformed_df_action, col_for_action, col_other_for_action,
    #   col_for_col_col_other, max_q, (front, back), all_details
    # ---------------------------------------------------------------------
    def choose_action(
        self,
        df,
        transformed_df,
        transformed_col_name,
        other_column,
        df_other,
        transformed_df_other,
        pairs,
        reversd_order,
        previous_edit_dist,
        prev_freq_counts_penalty,
        prev_lsc_len_mat_full,
        prev_edit_matrix,
        prev_matrix_indicies,
        exclude_cols_dict_front,
        exclude_cols_dict_back,
        exclude_cols_lst,
        greedy,
        *,
        update_probs=True,          # NEW: allow disabling during depth rollouts
        record_candidates=True,     # NEW: store candidates for depth search
    ):
        concate_sim_pos, concate_negative_uniq = self.reward_config.get_concate_params(greedy)

        # store candidates (deep copies) + next metrics for rollouts
        all_transformed_df_details = {}

        df_copy = df.copy()
        transformed_df_action = {}
        col_for_action = {}
        col_other_for_action = {}
        col_for_col_col_other = {}

        all_columns = set(df.columns) | set(transformed_df.columns)
        new_exclude_cols_dict_front = {col: [] for col in all_columns}
        new_exclude_cols_dict_back = {col: [] for col in all_columns}

        transformed_df_copy_best = transformed_df.copy()
        best_previous_edit_dist = previous_edit_dist
        best_lsc_len_mat_full = prev_lsc_len_mat_full
        best_freq_counts_penalty = prev_freq_counts_penalty
        best_edit_matrix = prev_edit_matrix

        reward_factor = self.reward_config.reward_factor
        reward_uniqueness_factor = self.reward_uniqueness_factor
        exclude_exist_cols = self.exclude_exist_cols

        concated_nums = len(transformed_df.columns)

        if exclude_exist_cols:
            available_columns = [
                col for col in df.columns
                if col not in transformed_df.columns and col not in exclude_cols_lst
            ]
            available_columns_df_other = [
                col for col in df_other.columns
                if col not in transformed_df_other.columns
            ]
            available_pairs = [
                (col_a, col_b) for (col_a, col_b) in pairs
                if col_a in available_columns and col_b in available_columns_df_other
            ]
        else:
            df_cols = list(df.columns)
            available_columns = [col for col in df_cols if col not in exclude_cols_lst]
            df_other_cols = list(df_other.columns)
            available_columns_df_other = df_other_cols.copy()
            available_pairs = pairs.copy()

        concate_cols_dict_front = {}
        concate_cols_dict_back = {}
        trans_col_index = {}
        i = 0
        prev_col = None

        for trans_col in transformed_df.columns:
            if trans_col not in exclude_cols_dict_front:
                exclude_cols_dict_front[trans_col] = []
                exclude_cols_dict_back[trans_col] = []

            concate_cols_dict_front[trans_col] = [c for c in available_columns if c not in exclude_cols_dict_front[trans_col]]
            concate_cols_dict_back[trans_col] = [c for c in available_columns if c not in exclude_cols_dict_back[trans_col]]

            trans_col_index[trans_col] = i
            i += 1

        # M1 (Stage-1 adaptive epsilon): boost eps when chain is stuck.
        # Heuristic: eps = max(self.epsilon, 0.5 - best_previous_edit_dist)
        # Low best_previous_edit_dist (≈ chain isn't matching anything) → ramp eps.
        _eff_eps = self.epsilon
        if False:
            try:
                _eff_eps = max(self.epsilon, 0.5 - float(best_previous_edit_dist))
            except Exception: pass
        explore = (random.uniform(0, 1) < _eff_eps)

        # ---------------------------
        # EXPLORE / EXPLOIT share most logic
        # ---------------------------
        action_rewards_dict = {}
        action_rewards_dict_except_concat_both = {}
        max_q = 0

        for col in transformed_df.columns:
            transformed_df_copy = transformed_df_copy_best.copy()
            print(f"Do the {'Exploration' if explore else 'Exploitation'} for {col}")
            transformed_column = transformed_df[col]

            # handle overlap constraints
            if trans_col_index[col] != 0 and prev_col is not None:
                if prev_col in transformed_df_action:
                    operator_prev = transformed_df_action[prev_col]
                    if operator_prev.operator_type == "concate" and "back" in operator_prev.name:
                        concate_cols_dict_front[col] = [c for c in available_columns]
                    else:
                        included_cols = concate_cols_dict_front[col]
                        concate_cols_dict_front[col] = [
                            c for c in included_cols
                            if c not in new_exclude_cols_dict_back[prev_col]
                        ]
            prev_col = col

            need_to_consider_split_ops = check_words_length_greater_1(transformed_column)
            direct_opts = ["direct", "direct_split"] if need_to_consider_split_ops else ["direct"]

            # operator list — learned operator selection
            # Exploitation: try ALL operators from MetaSelector's SELECTED groups (focused, high confidence)
            # Exploration: sample from UNSELECTED groups (discover new patterns)
            if self.selected_op_names is not None and self.unselected_op_names is not None:
                all_ops = self.operator_dict[col]
                # Always-keep operators (concat, auto_split) go into exploitation
                _always_keep = {'auto_split_by_operator', 'concatenate_front', 'concatenate_back',
                                'concat_pairs_front', 'concat_pairs_back'}
                selected_ops = [op for op in all_ops if op.name in self.selected_op_names or op.name in _always_keep]
                unselected_ops = [op for op in all_ops if op.name in self.unselected_op_names and op.name not in _always_keep]

                if explore and unselected_ops:
                    # Exploration: sample from unselected groups
                    probabilities = [self.operators_prob_dict[col].get(op.name, 1.0) for op in unselected_ops]
                    # N1 (Stage-1 softmax sampling): replace raw-weight random.choices with
                    # softmax(probs / τ) sampling. Lower τ = sharper (top-1-ish), higher τ = smoother.
                    _tau = None
                    if _tau:
                        try:
                            import numpy as _np
                            _arr = _np.array(probabilities, dtype=float)
                            _w = _np.exp(_arr / max(float(_tau), 1e-6))
                            _w = _w / _w.sum() if _w.sum() > 0 else None
                            probabilities = list(_w) if _w is not None else probabilities
                        except Exception: pass
                    # U-series: EXPLORE_K overrides the subsample size (default 3).
                    _k_explore = 3
                    operator_list = random.choices(unselected_ops, weights=probabilities, k=min(_k_explore, len(unselected_ops)))
                elif selected_ops:
                    # Exploitation: try ALL selected operators
                    operator_list = selected_ops
                else:
                    # Fallback: use all operators
                    operator_list = all_ops
            else:
                # No MetaSelector info — original behavior
                if explore:
                    full_operator_list = self.operator_dict[col]
                    probabilities = [self.operators_prob_dict[col][op.name] for op in full_operator_list]
                    _k_explore = 3
                    operator_list = random.choices(full_operator_list, weights=probabilities, k=min(_k_explore, len(full_operator_list)))
                else:
                    operator_list = self.operator_dict[col]

            previous_edit_dist_local = best_previous_edit_dist
            prev_freq_counts_penalty_local = best_freq_counts_penalty
            prev_lsc_len_mat_full_local = best_lsc_len_mat_full
            prev_edit_matrix_local = best_edit_matrix

            # Pre-compute base concatenation once (avoid recomputing per operator)
            _base_concat_values = concatenate_with_order(transformed_df_copy_best, transformed_col_name)[transformed_col_name]

            # --- Progressive early rejection setup ---
            # Sample p% of rows (stratified by similarity). If negative, stop.
            # If positive, compute remaining (1-p)% and combine — no work wasted.
            _n_rows = len(transformed_column)
            _sample_frac = 0.15
            _sample_n = max(5, int(_n_rows * _sample_frac))
            _use_early_reject = (_sample_n < _n_rows - 2
                                 and True)

            # Pre-compute stratified sample + remainder indices
            _sample_idx = None
            _remain_idx = None
            if _use_early_reject:
                if prev_edit_matrix_local is not None and len(prev_edit_matrix_local) > _sample_n:
                    _row_maxes = np.max(prev_edit_matrix_local, axis=1)
                    _sorted_idx = np.argsort(_row_maxes)
                    _pick = np.linspace(0, len(_sorted_idx) - 1, _sample_n, dtype=int)
                    _sample_idx = sorted(_sorted_idx[_pick].tolist())
                else:
                    _step = max(1, _n_rows // _sample_n)
                    _sample_idx = list(range(0, _n_rows, _step))[:_sample_n]
                _sample_set = set(_sample_idx)
                _remain_idx = sorted([i for i in range(_n_rows) if i not in _sample_set])

            for op in operator_list:
                globals()['_CURRENT_OP'] = getattr(op, 'name', None)  # op-safety culprit tracker
                globals()['_OP_EVAL_COUNT'] = globals().get('_OP_EVAL_COUNT', 0) + 1  # search-cost counter
                # ---------------- direct / direct_split ----------------
                if op.operator_type in direct_opts:
                    new_values = [op.apply(v) for v in transformed_column]
                    # Skip operators that don't change any values
                    if all(str(nv) == str(ov) for nv, ov in zip(new_values, transformed_column)):
                        continue
                    # Build concatenated column without full DataFrame copy
                    if len(transformed_df_copy_best.columns) == 1:
                        col_for_metrics = pd.Series(new_values)
                    else:
                        transformed_df_tmp = transformed_df_copy_best.copy()
                        transformed_df_tmp[col] = new_values
                        col_for_metrics = concatenate_with_order(transformed_df_tmp, transformed_col_name)[transformed_col_name]

                    # --- Progressive early rejection ---
                    # Phase 1: compute ALCS on stratified sample (p%)
                    # Phase 2: if positive, compute remaining (1-p%) and combine
                    # The sample is never wasted — it's always part of the final result.
                    _used_progressive = False
                    if _use_early_reject and _sample_idx is not None and _remain_idx:
                        try:
                            _col_sample = col_for_metrics.iloc[_sample_idx].reset_index(drop=True)

                            if reversd_order:
                                _s_dist, _s_mat, _s_freq, _s_lcs = get_ALCS_matrix(
                                    other_column, _col_sample, greedy)
                            else:
                                _s_dist, _s_mat, _s_freq, _s_lcs = get_ALCS_matrix(
                                    _col_sample, other_column, greedy)

                            if _s_mat is not None and len(_s_mat) >= 3:
                                _row_max_sample = np.max(_s_mat, axis=1)
                                _s_mean = float(np.mean(_row_max_sample))
                                _s_std = float(np.std(_row_max_sample, ddof=1))
                                _s_n = len(_row_max_sample)
                                _se = _s_std / max(np.sqrt(_s_n), 1e-8)
                                _df = max(_s_n - 1, 1)
                                _t_crit = 1.96 + 2.4 / _df + 0.2 / (_df * _df)
                                _upper_95 = _s_mean + _t_crit * _se

                                # Reject: upper bound of CI is below current ALCS
                                if _upper_95 < previous_edit_dist_local:
                                    q_value = reward_factor * (_s_mean - previous_edit_dist_local)
                                    key = col + op.name
                                    action_rewards_dict[key] = q_value
                                    action_rewards_dict_except_concat_both[key] = q_value
                                    if update_probs:
                                        self.update_the_operator_probs(col, op, q_value)
                                    continue

                                # Accept: compute remaining rows and combine
                                _col_remain = col_for_metrics.iloc[_remain_idx].reset_index(drop=True)
                                if reversd_order:
                                    _r_dist, _r_mat, _r_freq, _r_lcs = get_ALCS_matrix(
                                        other_column, _col_remain, greedy)
                                else:
                                    _r_dist, _r_mat, _r_freq, _r_lcs = get_ALCS_matrix(
                                        _col_remain, other_column, greedy)

                                if _r_mat is not None:
                                    # Reconstruct full matrix in original row order
                                    n_cols_mat = _s_mat.shape[1]
                                    edit_matrix = np.zeros((_n_rows, n_cols_mat))
                                    lsc_len_mat_full = np.zeros((_n_rows, n_cols_mat))
                                    for si, orig_i in enumerate(_sample_idx):
                                        edit_matrix[orig_i] = _s_mat[si]
                                        lsc_len_mat_full[orig_i] = _s_lcs[si]
                                    for ri, orig_i in enumerate(_remain_idx):
                                        edit_matrix[orig_i] = _r_mat[ri]
                                        lsc_len_mat_full[orig_i] = _r_lcs[ri]

                                    edit_dist = float(np.mean(np.max(edit_matrix, axis=1)))
                                    freq_counts_penalty = _s_freq + _r_freq
                                    _used_progressive = True
                        except Exception:
                            pass  # fall through to full computation

                    if not _used_progressive:
                        if reversd_order:
                            edit_dist, edit_matrix, freq_counts_penalty, lsc_len_mat_full = get_ALCS_matrix_with_top_k(
                                other_column, col_for_metrics, prev_matrix_indicies, greedy
                            )
                        else:
                            edit_dist, edit_matrix, freq_counts_penalty, lsc_len_mat_full = get_ALCS_matrix_with_top_k(
                                col_for_metrics, other_column, prev_matrix_indicies, greedy
                            )

                    edit_dist_dif = edit_dist - previous_edit_dist_local
                    reward = 0

                    if prev_freq_counts_penalty_local != 0:
                        reward = -(
                            (freq_counts_penalty - prev_freq_counts_penalty_local)
                            / np.max([freq_counts_penalty, prev_freq_counts_penalty_local])
                        ) * reward_uniqueness_factor

                        lcs_difs = np.max(lsc_len_mat_full, axis=1) - np.max(prev_lsc_len_mat_full_local, axis=1)
                        if reward > 0:
                            n_changed = max(np.sum(lcs_difs != 0), 1)
                            positive_fraction = np.sum(lcs_difs > 0) / n_changed
                            if positive_fraction < self.reward_config.min_positive_fraction:
                                reward = 0
                            else:
                                reward = reward * positive_fraction

                    if edit_dist_dif > 0:
                        eidit_difs = np.max(edit_matrix, axis=1) - np.max(prev_edit_matrix_local, axis=1)
                        n_changed = max(np.sum(eidit_difs != 0), 1)
                        positive_fraction = np.sum(eidit_difs > 0) / n_changed
                        if positive_fraction < self.reward_config.min_positive_fraction:
                            edit_dist_dif = 0
                        else:
                            edit_dist_dif = edit_dist_dif * positive_fraction * self.reward_config.agreement_multiplier

                    sim_component = reward_factor * edit_dist_dif
                    # Cap negative uniqueness reward: don't let it overwhelm a positive similarity gain
                    if sim_component > 0 and reward < 0:
                        reward = max(reward, -sim_component * self.reward_config.uniqueness_cap_ratio)
                    q_value = sim_component + reward

                    # --- Shrinkage / destructive-operator penalties (learned from RewardConfig) ---
                    _rc = self.reward_config
                    if q_value > 0:
                        _orig_vals = transformed_column.astype(str)
                        _new_vals_str = pd.Series(new_values).astype(str)
                        # Length shrinkage
                        _orig_avg_len = _orig_vals.str.len().mean()
                        if _orig_avg_len > 0:
                            _new_avg_len = _new_vals_str.str.len().mean()
                            if _new_avg_len < _rc.shrink_len_threshold * _orig_avg_len:
                                q_value *= _rc.shrink_len_penalty
                        # Uniqueness collapse
                        _orig_nuniq = _orig_vals.nunique()
                        if _orig_nuniq > 0:
                            _new_nuniq = _new_vals_str.nunique()
                            if _new_nuniq < _rc.shrink_uniq_threshold * _orig_nuniq:
                                q_value *= _rc.shrink_uniq_penalty
                        # Token loss
                        _orig_avg_tok = _orig_vals.str.split().str.len().mean()
                        if _orig_avg_tok is not None and _orig_avg_tok > 0:
                            _new_avg_tok = _new_vals_str.str.split().str.len().mean()
                            if _new_avg_tok is not None and _new_avg_tok < _rc.shrink_tok_threshold * _orig_avg_tok:
                                q_value *= _rc.shrink_tok_penalty
                    # --- End shrinkage penalties ---

                    key = col + op.name

                    action_rewards_dict[key] = q_value
                    action_rewards_dict_except_concat_both[key] = q_value

                    print(f"Reward is {q_value} for taking action {op.name}")

                    if update_probs:
                        self.update_the_operator_probs(col, op, q_value)

                    if op.name == "shift_1_word_forward" and q_value <= 0:
                        print("increase the shift value for shift_1_word_forward")
                        for ops in self.operator_dict[col]:
                            if ops.name == "shift_1_word_forward":
                                ops.adjust_params()
                                print(ops.params)

                    if q_value > max_q:
                        max_q = q_value
                        transformed_df_action[col] = op
                        if len(transformed_df_copy_best.columns) == 1:
                            transformed_df_copy_best = pd.DataFrame({col: new_values})
                        else:
                            transformed_df_copy_best = transformed_df_tmp
                        best_previous_edit_dist = edit_dist
                        best_lsc_len_mat_full = lsc_len_mat_full
                        best_freq_counts_penalty = freq_counts_penalty
                        best_edit_matrix = edit_matrix

                    # ✅ FIX: store DEEP COPIES and store next metrics
                    if record_candidates:
                        all_transformed_df_details[key] = {
                            "transformed_df_action": copy.deepcopy(transformed_df_action),
                            "col_for_action": copy.deepcopy(col_for_action),
                            "col_other_for_action": copy.deepcopy(col_other_for_action),
                            "col_for_col_col_other": copy.deepcopy(col_for_col_col_other),
                            "q_value": float(q_value),
                            "exclude": (
                                copy.deepcopy(new_exclude_cols_dict_front),
                                copy.deepcopy(new_exclude_cols_dict_back),
                            ),
                            "next_prev_edit_dist": float(edit_dist),
                            "next_prev_freq_counts_penalty": float(freq_counts_penalty),
                            "next_prev_lsc_len_mat_full": lsc_len_mat_full,
                            "next_prev_edit_matrix": edit_matrix,
                            "edit_dist": float(edit_dist),
                        }

                # ---------------- concate ----------------
                elif op.operator_type == "concate" and concated_nums < self.max_concate_num:
                    if "front" in op.name:
                        concate_cols_dict = concate_cols_dict_front
                        concate_opt = "front"
                    else:
                        concate_cols_dict = concate_cols_dict_back
                        concate_opt = "back"

                    for df_col in concate_cols_dict[col]:
                        transformed_df_tmp = transformed_df.copy()
                        transformed_df_tmp = op.func(transformed_df_tmp, col, df_copy[df_col])
                        transformed_column_new = concatenate_with_order(transformed_df_tmp, transformed_col_name)[transformed_col_name]

                        if reversd_order:
                            edit_dist, edit_matrix, freq_counts_penalty, lsc_len_mat_full = get_ALCS_matrix_with_top_k(
                                other_column, transformed_column_new, prev_matrix_indicies, greedy
                            )
                        else:
                            edit_dist, edit_matrix, freq_counts_penalty, lsc_len_mat_full = get_ALCS_matrix_with_top_k(
                                transformed_column_new, other_column, prev_matrix_indicies, greedy
                            )

                        edit_dist_dif = edit_dist - previous_edit_dist_local
                        reward = 0

                        if prev_freq_counts_penalty_local != 0:
                            lcs_difs = np.max(lsc_len_mat_full, axis=1) - np.max(prev_lsc_len_mat_full_local, axis=1)
                            reward = -(
                                (freq_counts_penalty - prev_freq_counts_penalty_local)
                                / np.max([freq_counts_penalty, prev_freq_counts_penalty_local])
                            ) * reward_factor
                            if reward > 0:
                                n_changed = max(np.sum(lcs_difs != 0), 1)
                                positive_fraction = np.sum(lcs_difs > 0) / n_changed
                                reward = reward * positive_fraction
                            else:
                                reward = reward * concate_negative_uniq

                        if edit_dist_dif > 0:
                            eidit_difs = np.max(edit_matrix, axis=1) - np.max(prev_edit_matrix_local, axis=1)
                            n_changed = max(np.sum(eidit_difs != 0), 1)
                            positive_fraction = np.sum(eidit_difs > 0) / n_changed
                            if positive_fraction < self.reward_config.min_positive_fraction:
                                edit_dist_dif = 0
                            else:
                                edit_dist_dif = edit_dist_dif * positive_fraction * concate_sim_pos

                        sim_component = reward_factor * edit_dist_dif
                        if sim_component > 0 and reward < 0:
                            reward = max(reward, -sim_component * self.reward_config.uniqueness_cap_ratio)
                        q_value = sim_component + reward
                        key = col + op.name + df_col

                        action_rewards_dict[key] = q_value
                        action_rewards_dict_except_concat_both[key] = q_value

                        print(f"Reward is {q_value} for taking action {op.name} with {df_col}")

                        if update_probs:
                            self.update_the_operator_probs(col, op, q_value)

                        # negative cutoff updates (same as your code)
                        if q_value < 0:
                            if concate_opt == "front":
                                new_exclude_cols_dict_front[col].append(df_col)
                            else:
                                new_exclude_cols_dict_back[col].append(df_col)

                        if q_value > max_q:
                            max_q = q_value
                            col_for_action[col] = df_col
                            transformed_df_action[col] = op
                            transformed_df_copy_best = transformed_df_tmp
                            best_previous_edit_dist = edit_dist
                            best_lsc_len_mat_full = lsc_len_mat_full
                            best_freq_counts_penalty = freq_counts_penalty
                            best_edit_matrix = edit_matrix

                        # ✅ FIX: deep copies + next metrics
                        if record_candidates:
                            # ensure col_for_action includes this mapping for apply()
                            tmp_col_for_action = copy.deepcopy(col_for_action)
                            tmp_col_for_action[col] = df_col

                            all_transformed_df_details[key] = {
                                "transformed_df_action": copy.deepcopy(transformed_df_action),
                                "col_for_action": tmp_col_for_action,
                                "col_other_for_action": copy.deepcopy(col_other_for_action),
                                "col_for_col_col_other": copy.deepcopy(col_for_col_col_other),
                                "q_value": float(q_value),
                                "exclude": (
                                    copy.deepcopy(new_exclude_cols_dict_front),
                                    copy.deepcopy(new_exclude_cols_dict_back),
                                ),
                                "next_prev_edit_dist": float(edit_dist),
                                "next_prev_freq_counts_penalty": float(freq_counts_penalty),
                                "next_prev_lsc_len_mat_full": lsc_len_mat_full,
                                "next_prev_edit_matrix": edit_matrix,
                                "edit_dist": float(edit_dist),
                            }

                # NOTE: your concate_both block exists in your full code;
                # apply the SAME candidate-storage fix there:
                # store deep copies + mappings + next metrics,
                # and make sure col_for_action/col_other_for_action/col_for_col_col_other
                # are correctly stored for apply().

        if len(action_rewards_dict) == 0 or np.all(np.array(list(action_rewards_dict.values())) <= 0.0):
            if explore:
                print("The exploration operators cannot increase the reward")
                for c in transformed_df.columns:
                    transformed_df_action[c] = OperatorAction("Noop", noop, {}, "direct")
            else:
                print("The operators cannot further increase the similarity, restart")
                for c in transformed_df.columns:
                    transformed_df_action[c] = OperatorAction("Restart", noop, {}, "direct")

        # SWAP_RATE: if set, overrides _eff_eps for the swap blocks only
        # (decouples post-hoc argmax-swap from the subsample-explore branch).
        # Enables "eps=0 over all 25 ops + small swap probability" (O6).
        _swap_rate = float(_eff_eps)
        # SWAP_DISABLED: set per-iteration by the variations loop under
        # REINFORCE_SWAP=1 (O8) so rollout 0 is forced pure argmax while
        # rollout 1 has swap active. Best-of-N ALCS selection then picks the
        # higher trajectory — i.e., we only "reinforce" the swap path if it wins.
        if False:
            _swap_rate = 0.0

        # P1: TRUE eps-greedy on Q-values. After loop completes (all ops evaluated),
        # with prob `_swap_rate` swap argmax → 2nd-best from all_transformed_df_details.
        # Tests "deliberately pick non-top op" mechanism (distinct from V0's subsample).
        if False and len(all_transformed_df_details) >= 2:
            try:
                if random.uniform(0, 1) < _swap_rate:
                    # Sort candidates by q_value descending and pick the 2nd
                    _sorted_keys = sorted(all_transformed_df_details.keys(),
                                           key=lambda k: -all_transformed_df_details[k]['q_value'])
                    if len(_sorted_keys) >= 2:
                        _swap_key = _sorted_keys[1]
                        _swap_d = all_transformed_df_details[_swap_key]
                        transformed_df_action = _swap_d['transformed_df_action']
                        col_for_action = _swap_d['col_for_action']
                        col_other_for_action = _swap_d['col_other_for_action']
                        col_for_col_col_other = _swap_d['col_for_col_col_other']
                        max_q = _swap_d['q_value']
            except Exception:
                pass

        # P2: like P1 but swap to a RANDOM positive-gain op (not just 2nd-best).
        # Tests "any good-enough non-top op might unlock better trajectory".
        if False and len(all_transformed_df_details) >= 2:
            try:
                if random.uniform(0, 1) < _swap_rate:
                    _pos_keys = [k for k, d in all_transformed_df_details.items()
                                  if d.get('q_value', 0) > 0]
                    # Exclude the current argmax key (we want to SWAP away from it)
                    _argmax_key = max(all_transformed_df_details,
                                       key=lambda k: all_transformed_df_details[k]['q_value'])
                    _candidates = [k for k in _pos_keys if k != _argmax_key]
                    if _candidates:
                        _swap_key = random.choice(_candidates)
                        _swap_d = all_transformed_df_details[_swap_key]
                        transformed_df_action = _swap_d['transformed_df_action']
                        col_for_action = _swap_d['col_for_action']
                        col_other_for_action = _swap_d['col_other_for_action']
                        col_for_col_col_other = _swap_d['col_for_col_col_other']
                        max_q = _swap_d['q_value']
            except Exception:
                pass

        return (
            transformed_df_action,
            col_for_action,
            col_other_for_action,
            col_for_col_col_other,
            max_q,
            (new_exclude_cols_dict_front, new_exclude_cols_dict_back),
            all_transformed_df_details,
        )
    

    def choose_action_plan_with_depth(
        self,
        df,
        transformed_df,
        transformed_col_name,
        other_column,
        df_other,
        transformed_df_other,
        pairs,
        reversd_order,
        previous_edit_dist,
        prev_freq_counts_penalty,
        prev_lsc_len_mat_full,
        prev_edit_matrix,
        prev_matrix_indicies,
        exclude_cols_dict_front,
        exclude_cols_dict_back,
        exclude_cols_lst,
        greedy,
        *,
        depth=2,
        branch_limit=20,
        rollout_update_probs=False,   # planning should not mutate probs
    ):
        """
        Returns:
        best_plan_steps: list of step dicts in order.
            each step dict contains:
            - transformed_df_action
            - col_for_action
            - col_other_for_action
            - col_for_col_col_other
            - q_reward
            - exclude_dict
        best_discounted_return: float
        """

        depth = max(0, int(depth))
        branch_limit = max(1, int(branch_limit))
        gamma = float(self.discount_factor)

        # You need this helper in your class (I used it before).
        # It must apply one "step action" to produce next transformed_df states.
        # If you already have _apply_action_snapshot(), reuse it.
        def _apply_action_snapshot_local(
            tdf, tdf_other,
            step_action, col_for_action, col_other_for_action, col_for_col_col_other
        ):
            tdf_new = tdf.copy()
            tdf_other_new = tdf_other.copy()

            for col_name, action in step_action.items():
                if action.operator_type in ["direct", "direct_split"]:
                    tdf_new[col_name] = [action.apply(v) for v in tdf_new[col_name]]

                elif action.operator_type == "concate":
                    df_col = col_for_action[col_name]
                    tdf_new = action.func(tdf_new, col_name, df[df_col])

                elif action.operator_type == "concate_both":
                    insertion_pos = col_for_col_col_other[col_name]
                    df_col_a = col_for_action[col_name]
                    df_col_b = col_other_for_action[insertion_pos]
                    tdf_new, tdf_other_new = action.func(
                        tdf_new, tdf_other_new,
                        df[df_col_a], df_other[df_col_b],
                        col_name, insertion_pos,
                    )
            return tdf_new, tdf_other_new

        # Core DFS with pruning that returns (best_value, best_steps)
        _stop_model = None
        if ModelRegistry is not None and not _learned_off('NO_STOP'):
            _stop_model = ModelRegistry().load('stop_continue')

        def _search(
            tdf, tdf_other,
            prev_edit_dist_, prev_freq_pen_, prev_lcs_full_, prev_edit_mat_,
            ex_front_, ex_back_,
            depth_left,
            gamma_pow,
            prev_alcs_for_stop=0.0,
        ):
            if depth_left <= 0:
                return 0.0, []

            # Learned stop/continue check
            if _stop_model is not None and depth_left < depth:
                steps_done = depth - depth_left
                delta_alcs = prev_edit_dist_ - prev_alcs_for_stop if prev_alcs_for_stop > 0 else 0.0
                if not _stop_model.should_continue(
                    current_alcs=prev_edit_dist_,
                    delta_alcs=delta_alcs,
                    step_ratio=steps_done / max(depth, 1),
                    n_candidates=len(self.operators_prob_dict),
                    best_q=gamma_pow,
                    positive_fraction=max(0.0, prev_edit_dist_),
                ):
                    return 0.0, []

            # run choose_action ONCE to generate candidate pool at this state
            (
                _best_action, _best_col_for_action, _best_col_other_for_action,
                _best_col_for_col_other, _best_q,
                _best_exclude, all_details
            ) = self.choose_action(
                df,
                tdf,
                transformed_col_name,
                other_column,            # NOTE: if your other_column changes due to concate_both, you must recompute it outside
                df_other,
                tdf_other,
                pairs,
                reversd_order,
                prev_edit_dist_,
                prev_freq_pen_,
                prev_lcs_full_,
                prev_edit_mat_,
                prev_matrix_indicies,
                ex_front_,
                ex_back_,
                exclude_cols_lst,
                greedy,
                update_probs=rollout_update_probs,
                record_candidates=True,
            )

            # print(f'all_details is {all_details[0]}')

            if not all_details:
                return 0.0, []

            # prune candidates by q_value (immediate reward) — branch limit
            candidates = list(all_details.values())
            candidates.sort(key=lambda info: float(info["q_value"]), reverse=True)
            candidates = candidates[:branch_limit]

            best_total = -float("inf")
            best_steps = []

            for info in candidates:
                q0 = float(info["q_value"])
                alcs0 = float(info.get("edit_dist", 0.0))

                # Expand only if q>0 OR ALCS positive (your rule)
                expand = (q0 > 0.0) or (alcs0 > 0.0)

                # This is the step record the OUTSIDE will apply
                step_rec = {
                    "transformed_df_action": info["transformed_df_action"],
                    "col_for_action": info["col_for_action"],
                    "col_other_for_action": info["col_other_for_action"],
                    "col_for_col_col_other": info["col_for_col_col_other"],
                    "q_reward": q0,
                    "exclude_dict": info["exclude"],   # (front, back)
                }

                if not expand or depth_left == 1:
                    total = gamma_pow * q0
                    if total > best_total:
                        best_total = total
                        best_steps = [step_rec]
                    continue

                # Apply this candidate to get the NEXT state (tdf, tdf_other)
                next_tdf, next_tdf_other = _apply_action_snapshot_local(
                    tdf, tdf_other,
                    info["transformed_df_action"],
                    info["col_for_action"],
                    info["col_other_for_action"],
                    info["col_for_col_col_other"],
                )

                # Use stored "next previous" metrics computed inside choose_action candidate evaluation
                next_prev_edit_dist = float(info["next_prev_edit_dist"])
                next_prev_freq_pen = float(info["next_prev_freq_counts_penalty"])
                next_prev_lcs_full = info["next_prev_lsc_len_mat_full"]
                next_prev_edit_mat = info["next_prev_edit_matrix"]
                next_ex_front, next_ex_back = info["exclude"]

                future_val, future_steps = _search(
                    next_tdf, next_tdf_other,
                    next_prev_edit_dist, next_prev_freq_pen, next_prev_lcs_full, next_prev_edit_mat,
                    next_ex_front, next_ex_back,
                    depth_left - 1,
                    gamma_pow * gamma,
                    prev_alcs_for_stop=prev_edit_dist_,
                )

                total = gamma_pow * q0 + future_val

                if total > best_total:
                    best_total = total
                    best_steps = [step_rec] + future_steps

            if best_total == -float("inf"):
                return 0.0, []

            return best_total, best_steps

        if depth == 0:
            # No future reward - just pick the best single-step action
            (
                best_action, best_col_for_action, best_col_other_for_action,
                best_col_for_col_other, best_q,
                best_exclude, _all_details
            ) = self.choose_action(
                df, transformed_df, transformed_col_name, other_column,
                df_other, transformed_df_other, pairs, reversd_order,
                previous_edit_dist, prev_freq_counts_penalty, prev_lsc_len_mat_full,
                prev_edit_matrix, prev_matrix_indicies, exclude_cols_dict_front,
                exclude_cols_dict_back, exclude_cols_lst, greedy,
                update_probs=True, record_candidates=False,
            )
            step = (
                best_action, best_col_for_action, best_col_other_for_action,
                best_col_for_col_other, best_q, best_exclude,
            )
            return [step], float(best_q)

        best_val, best_plan_steps = _search(
            transformed_df, transformed_df_other,
            previous_edit_dist, prev_freq_counts_penalty, prev_lsc_len_mat_full, prev_edit_matrix,
            exclude_cols_dict_front, exclude_cols_dict_back,
            depth,
            1.0,
            prev_alcs_for_stop=0.0,
        )

        plan_steps_tuples = [
        (
            s["transformed_df_action"],
            s["col_for_action"],
            s["col_other_for_action"],
            s["col_for_col_col_other"],
            s["q_reward"],
            s["exclude_dict"],
        )
        for s in best_plan_steps
    ]
        return plan_steps_tuples, float(best_val)



def optimize_transformations_both_edit_dist_opt(max_steps, df_a, df_b,column_a_name,column_b_name, agent_table_a,agent_table_b,pairs, edit_dist_threshold,greedy = True,
                                                transformed_columns_for_a_to_b= None,transformed_colums_for_b_to_a =None,reusing = False,k=None):
    # make all the strings in lower case

    df_a = df_a.loc[:, df_a.isna().mean() < 0.8].astype(str).apply(lambda col: col.str.lower())
    df_b = df_b.loc[:, df_b.isna().mean() < 0.8].astype(str).apply(lambda col: col.str.lower())

    # get reversed pairs
    print(f'pairs are {pairs}')
    reversed_pairs  = [(y, x) for x, y in pairs]

    # Initialize variables for the transformation process for column
    if column_a_name not in df_a.columns or column_b_name not in df_b.columns:

        return None, None, None,None, None
    
    transformations_table_a = defaultdict(list)
    transformations_names_table_a = defaultdict(list)
    transformed_df_a_to_b = df_a[[column_a_name]]

    transformations_table_b = defaultdict(list)
    transformations_names_table_b = defaultdict(list)
    transformed_df_b_to_a = df_b[[column_b_name]]

    # determine the ALCS for col a and col b 
    initial_edit_dist_a_to_b,edit_dist_matrix_a_to_b,ini_freq_counts_penalty,ini_lcs_matrix_full= get_ALCS_matrix(df_a[column_a_name], df_b[column_b_name],greedy)

    print(f"Initial ALCS from table a to b: {initial_edit_dist_a_to_b}")

    # Check if initial edit distance meets the threshold
    if initial_edit_dist_a_to_b >= edit_dist_threshold and ini_freq_counts_penalty == 0:
        print("Initial ALCS meets the threshold. Skipping transformations.")
        # Return empty transformations and the initial similarity matrix
        return transformations_table_a,transformed_df_a_to_b.columns,transformations_table_b,transformed_df_b_to_a.columns,edit_dist_matrix_a_to_b


    # clustering based on max alcs for each row
    # add randomness
    rand_seed = random.randint(1, 10000)
    max_alcs_for_each_row = np.max(edit_dist_matrix_a_to_b, axis=1, keepdims=True)

    num_clusters = 3

    # PERCENTILE_SAMPLE=1 swaps KMeans+fixed-fractions with
    # percentile-stratified (linspace over sorted promise scores). No
    # hyperparameters; adapts to noise distribution per dataset. Default
    # off — original KMeans behavior preserved.
    if False:
        _target_n = int(max(20, len(max_alcs_for_each_row) // 5))
        scores_flat = max_alcs_for_each_row.flatten()
        n_total = len(scores_flat)
        if n_total > _target_n:
            sorted_idx = np.argsort(scores_flat)
            pick = np.linspace(0, n_total - 1, _target_n, dtype=int)
            selected_rows_index = sorted(np.unique(sorted_idx[pick]).tolist())
        else:
            selected_rows_index = list(range(n_total))
    else:
        n_clusters_to_use = min(num_clusters, len(max_alcs_for_each_row))
        kmeans = KMeans(n_clusters=n_clusters_to_use, random_state=rand_seed)
        kmeans.fit(max_alcs_for_each_row)

        df_max_alcs_for_each_row = pd.DataFrame(max_alcs_for_each_row, columns=["value"])
        df_max_alcs_for_each_row['row_number'] = df_max_alcs_for_each_row.index
        df_max_alcs_for_each_row['cluster'] = kmeans.labels_

        df_cluster_means = df_max_alcs_for_each_row.groupby('cluster')['value'].mean()

        df_cluster_means_sorted = df_cluster_means.sort_values(ascending=False)
        unique_clusters = df_cluster_means_sorted.index

        _cf3 = getattr(agent_table_a.reward_config, 'cluster_fractions_3', (0.35, 0.05, 0.05))
        _cf2 = getattr(agent_table_a.reward_config, 'cluster_fractions_2', (0.3, 0.05))
        _cf1 = getattr(agent_table_a.reward_config, 'cluster_fractions_1', (0.3,))

        if len(unique_clusters) == 3:
            highest_cluster = unique_clusters[0]
            medium_cluster = unique_clusters[1]
            lowest_cluster = unique_clusters[2]
            cluster_sample_fractions = {
                highest_cluster: _cf3[0],
                medium_cluster: _cf3[1],
                lowest_cluster: _cf3[2]
            }
        elif len(unique_clusters) == 2:
            highest_cluster = unique_clusters[0]
            lowest_cluster = unique_clusters[1]
            cluster_sample_fractions = {
                highest_cluster: _cf2[0],
                lowest_cluster: _cf2[1]
            }
        elif len(unique_clusters) == 1:
            the_only_cluster = unique_clusters[0]
            cluster_sample_fractions = {the_only_cluster: _cf1[0]}

        def sample_cluster(group):
            c = group.name
            frac = cluster_sample_fractions[c]
            n = max(1, int(frac * len(group)))
            return group.sample(n=n, random_state=rand_seed)

        sampled_df_max_alcs_for_each_row = (
            df_max_alcs_for_each_row
            .groupby('cluster', group_keys=False)
            .apply(sample_cluster)
        )

        selected_rows = sampled_df_max_alcs_for_each_row['row_number'].tolist()
        selected_rows_index = sorted(selected_rows)

    transformed_df_a_to_b = transformed_df_a_to_b.loc[selected_rows_index]
    df_a = df_a.loc[selected_rows_index]


    new_sim_matrix = edit_dist_matrix_a_to_b[selected_rows_index,:]
    new_ini_lcs_matrix_full = ini_lcs_matrix_full[selected_rows_index,:]

    sim_matrix_max_vals = np.max(new_sim_matrix, axis=1, keepdims=True)

    new_mean_alcs = np.mean(sim_matrix_max_vals)

    mask = new_sim_matrix == sim_matrix_max_vals

    modified_result_indices = [np.where(row)[0] for row in mask]

    new_ini_freq_counts_penalty  = np.sum(get_penalty_overlap_counts(modified_result_indices))


    previous_edit_dist_a_to_b = new_mean_alcs
    previous_freq_counts_penalty = new_ini_freq_counts_penalty
    previous_lsc_len_mat_full = new_ini_lcs_matrix_full
    prev_edit_dist_matrix_a_to_b = new_sim_matrix
    # Use learned top_k_percent from RewardConfig, fallback to function param or 0.5
    _k = k if k is not None else getattr(agent_table_a.reward_config, 'top_k_percent', 0.5)
    prev_matrix_indicies = get_top_k_percent(prev_edit_dist_matrix_a_to_b, _k)

    current_best_edit_dist_table_a_to_b = new_mean_alcs
    current_best_transformations_a_to_b = []
    transformed_col_a_b_name = 'transformed_column_a_to_b'

    if transformed_columns_for_a_to_b:
        current_best_transformed_column_a_to_b = pd.Series(transformed_columns_for_a_to_b.copy(),name=transformed_col_a_b_name)
        transformed_column_a_to_b = pd.Series(transformed_columns_for_a_to_b.copy(),name=transformed_col_a_b_name)
        current_best_transformed_df_a_to_b = transformed_columns_for_a_to_b

    else:
        current_best_transformed_column_a_to_b = pd.Series(df_a[column_a_name].copy(),name=transformed_col_a_b_name)
        transformed_column_a_to_b = pd.Series(df_a[column_a_name].copy(),name=transformed_col_a_b_name)
        current_best_transformed_df_a_to_b = df_a[[column_a_name]]

    transformed_column_a_to_b_new = None
    wait_to_restart_a_b = False


    current_best_transformations_b_to_a = []
    transformed_col_b_a_name = 'transformed_column_b_to_a'

    if transformed_colums_for_b_to_a:
        current_best_transformed_column_b_to_a = pd.Series(transformed_colums_for_b_to_a.copy(),name=transformed_col_b_a_name)
        transformed_column_b_to_a = pd.Series(transformed_colums_for_b_to_a.copy(),name=transformed_col_b_a_name)
        current_best_transformed_df_b_to_a = transformed_colums_for_b_to_a

    else:
        current_best_transformed_column_b_to_a = pd.Series(df_b[column_b_name].copy(),name=transformed_col_b_a_name)
        transformed_column_b_to_a = pd.Series(df_b[column_b_name].copy(),name=transformed_col_b_a_name)
        current_best_transformed_df_b_to_a =df_b[[column_b_name]]

    transformed_column_b_to_a_new = None
    wait_to_restart_b_a = False

    agreement_percentage = agent_table_a.agreement_percentage

    transformed_df_action_a_to_b = None
    transformed_df_action_b_to_a = None

    transformed_df_action_a_to_b_col = None
    transformed_df_action_b_to_a_col = None

    transformed_df_action_a_to_b_col_both = None
    transformed_df_action_b_to_a_col_both = None

    exclude_cols_dict_front_a_to_b_org = {col: [] for col in df_a.columns}
    exclude_cols_dict_back_a_to_b_org = {col: [] for col in df_a.columns}

    exclude_cols_dict_front_b_to_a_org = {col: [] for col in df_b.columns}
    exclude_cols_dict_back_b_to_a_org = {col: [] for col in df_b.columns}

    exclude_cols_dict_front_a_to_b = exclude_cols_dict_front_a_to_b_org.copy()
    exclude_cols_dict_back_a_to_b = exclude_cols_dict_back_a_to_b_org.copy()

    exclude_cols_dict_front_b_to_a = exclude_cols_dict_front_b_to_a_org.copy()
    exclude_cols_dict_back_b_to_a = exclude_cols_dict_back_b_to_a_org.copy()


    ## exclude the all numbers cols
    exclude_cols_lst_a_to_b_arg = [col for col in df_a.columns if all_numbers(df_a[col])]
    exclude_cols_lst_b_to_a_arg = [col for col in df_b.columns if all_numbers(df_b[col])]

    exclude_cols_lst_a_to_b = df_a[exclude_cols_lst_a_to_b_arg]
    exclude_cols_lst_b_to_a = df_b[exclude_cols_lst_b_to_a_arg]

    print(f'exclude cols a to b:{exclude_cols_lst_a_to_b_arg}')
    print(f'exclude cols b to a:{exclude_cols_lst_b_to_a_arg}')

    # Step-level early stopping: track ALCS history and stop when improvement flatlines
    _alcs_history = [new_mean_alcs]

    for iters in range(max_steps):
      # Early stop: if recent ALCS deltas show no improvement trend, break.
      # Uses the last 3 deltas — if all are <= 0, the search has stalled.
      # Only for longer runs (max_steps > 3) to avoid cutting short wrapper phase.
      if len(_alcs_history) >= 4 and max_steps > 3:
          _recent_deltas = [_alcs_history[i] - _alcs_history[i-1] for i in range(-3, 0)]
          _all_flat = all(d <= 0.001 for d in _recent_deltas)
      else:
          _all_flat = False
      if _all_flat and iters >= 3:
          print(f"Early stopping at step {iters}: flat ALCS trend, ALCS={previous_edit_dist_a_to_b:.4f}")
          break

      print(f'The iteration is {iters}')
      # find action for a to b
      print('Finding action for table A')
      print('##############################################')

      if wait_to_restart_a_b == True and wait_to_restart_b_a == True:
        print('Starting to Restart')
        table_a_to_b_cols = current_best_transformed_df_a_to_b.columns
        table_b_to_a_cols = current_best_transformed_df_b_to_a.columns
        # no actions can future increase the reward and not meet the threshold, begin to restart
        final_edit_dist, edit_dist_matrix,freq_counts_penalty,lcs_matrix_full = get_ALCS_matrix_with_top_k(current_best_transformed_column_a_to_b, current_best_transformed_column_b_to_a,prev_matrix_indicies,greedy)

        return current_best_transformations_a_to_b, table_a_to_b_cols, current_best_transformations_b_to_a,table_b_to_a_cols, edit_dist_matrix
      


    #   transformed_df_action_a_to_b,transformed_df_action_a_to_b_col,transformed_df_action_b_to_a_col_both,col_for_col_col_other,q_reward,exclude_dict_a_to_b = agent_table_a.choose_action(df_a,
    #     transformed_df_a_to_b,
    #     transformed_col_a_b_name,
    #     transformed_column_b_to_a,
    #     df_b,
    #     transformed_df_b_to_a,
    #     pairs,
    #     False,
    #     previous_edit_dist_a_to_b,
    #     previous_freq_counts_penalty,
    #     previous_lsc_len_mat_full,
    #     prev_edit_dist_matrix_a_to_b,
    #     prev_matrix_indicies,
    #     exclude_cols_dict_front_a_to_b,
    #     exclude_cols_dict_back_a_to_b,
    #     exclude_cols_lst_a_to_b,
    #     greedy
    #     )
      

      base_depth = agent_table_a.depth
      plan_steps, plan_q = agent_table_a.choose_action_plan_with_depth(
        df_a, transformed_df_a_to_b, transformed_col_a_b_name,
        transformed_column_b_to_a, df_b, transformed_df_b_to_a,
        pairs, False,
        previous_edit_dist_a_to_b, previous_freq_counts_penalty, previous_lsc_len_mat_full,
        prev_edit_dist_matrix_a_to_b, prev_matrix_indicies,
        exclude_cols_dict_front_a_to_b, exclude_cols_dict_back_a_to_b,
        exclude_cols_lst_a_to_b, greedy,
        depth=base_depth,
       )

      # Depth is learned per config from two-stage training (Stage B grid search over depth)
      # No runtime escalation heuristic — the trained depth is used directly

      print(plan_steps)

      for transformed_df_action_a_to_b, transformed_df_action_a_to_b_col, \
          transformed_df_action_b_to_a_col_both, col_for_col_col_other, \
            q_reward, exclude_dict_a_to_b  in plan_steps:
        
        print(exclude_dict_a_to_b)

      
        new_exclude_cols_dict_front_a_to_b,new_exclude_cols_dict_back_a_to_b = exclude_dict_a_to_b 


            
        if not transformed_df_action_a_to_b:
            wait_to_restart_a_b = True
        else:
            first_key = list(transformed_df_action_a_to_b.keys())[0]
            first_value = transformed_df_action_a_to_b[first_key]
            if first_value.name == 'Restart':
                wait_to_restart_a_b = True
            else:
                wait_to_restart_a_b = False


        # update the transformed df
        transformed_df_a_to_b_new = transformed_df_a_to_b.copy()
        transformed_df_b_to_a_new = transformed_df_b_to_a.copy()

        # apply transformations to the transformed df 
        contain_concate_opt_a_to_b = False
        for col_name,action in transformed_df_action_a_to_b.items():
            if col_name not in transformed_df_a_to_b_new.columns:
                continue
            if action.operator_type in ['direct','direct_split']:
                transformed_df_a_to_b_new[col_name] = [action.apply(value) for value in transformed_df_a_to_b_new[col_name]]
            elif action.operator_type == 'concate':
                transformed_df_a_to_b_new = action.func(transformed_df_a_to_b_new,col_name,df_a[transformed_df_action_a_to_b_col[col_name]])
                # the chosen action contains the concate opts
                contain_concate_opt_a_to_b = True
            elif action.operator_type == 'concate_both':
                insertion_pos = col_for_col_col_other[col_name]
                transformed_df_a_to_b_new,transformed_df_b_to_a_new = action.func(transformed_df_a_to_b_new,transformed_df_b_to_a_new,
                                                                                    df_a[transformed_df_action_a_to_b_col[col_name]],
                                                                                    df_b[transformed_df_action_b_to_a_col_both[insertion_pos]],
                                                                                    col_name,insertion_pos, )
            

            # concate the transformed df to get the transformed column 
            # edit_dist, edit_matrix,freq_counts_penalty, lsc_len_mat_full, result_values_list, result_indices
            transformed_column_a_to_b_new = concatenate_with_order(transformed_df_a_to_b_new,transformed_col_a_b_name)[transformed_col_a_b_name]
            edit_dist,new_edit_dist_matrix_a_to_b,freq_counts_penalty,lcs_matrix_full = get_ALCS_matrix_with_top_k(transformed_column_a_to_b_new, transformed_column_b_to_a,prev_matrix_indicies,greedy)
            print(f"New ALCS is {edit_dist}")

            if q_reward > 0:
                # Accept transformation
                action_for_transformation_a_to_b = copy.deepcopy(transformed_df_action_a_to_b)
                # append the results
                for col_name, action in action_for_transformation_a_to_b.items():
                    transformations_names_table_a[col_name].append(action.name)
                    transformations_table_a[col_name].append(action)

                    #update the transformed df and concated column
                    transformed_df_a_to_b = transformed_df_a_to_b_new
                    transformed_column_a_to_b = transformed_column_a_to_b_new
                    transformed_df_b_to_a = transformed_df_b_to_a_new
                    #update the edit dist and edit dist matrix
                    previous_edit_dist_a_to_b = edit_dist
                    previous_lsc_len_mat_full= lcs_matrix_full
                    previous_freq_counts_penalty = freq_counts_penalty
                    prev_edit_dist_matrix_a_to_b = new_edit_dist_matrix_a_to_b

                    exclude_cols_dict_front_a_to_b = new_exclude_cols_dict_front_a_to_b
                    exclude_cols_dict_back_a_to_b = new_exclude_cols_dict_back_a_to_b 


                # Update current best transformations if statisfy the cond
                if greedy:
                    if edit_dist > current_best_edit_dist_table_a_to_b :
                        current_best_edit_dist_table_a_to_b = edit_dist
                        current_best_transformations_a_to_b = copy.deepcopy(transformations_table_a)
                        current_best_transformed_column_a_to_b = transformed_column_a_to_b.copy()
                        current_best_transformed_df_a_to_b = transformed_df_a_to_b.copy()
                
                else:
                    current_best_edit_dist_table_a_to_b = edit_dist
                    current_best_transformations_a_to_b = copy.deepcopy(transformations_table_a)
                    current_best_transformed_column_a_to_b = transformed_column_a_to_b.copy()
                    current_best_transformed_df_a_to_b = transformed_df_a_to_b.copy()


                print(f'Q value is {q_reward}')

                for col,every_action in transformed_df_action_a_to_b.items():
                    if every_action.operator_type in ['direct','direct_split']:
                        print(f"Accept the transformation {every_action.name} for {col}")
                    elif every_action.operator_type == 'concate':
                        print(f"Accept the transformation {every_action.name} for {col} with {transformed_df_action_a_to_b_col[col]}")
                    elif every_action.operator_type == 'concate_both':
                        insertion_pos = col_for_col_col_other[col]
                        print(f"Accept the transformation {every_action.name} for {col} with {transformed_df_action_a_to_b_col[col]}")
                        print(f"Accept the transformation {every_action.name} for {insertion_pos} with {transformed_df_action_b_to_a_col_both[insertion_pos]}")
                    
                    print(f'the current action params {every_action.params}')
                    print('==================================================')

                # Check if Edit distance threshold is met and all unique
                if edit_dist >= edit_dist_threshold and freq_counts_penalty == 0:
                    print("ALCS threshold met.")
                    break
                #Adjust parameters if cumulative
                agent_table_a.update_cum_params(action_for_transformation_a_to_b)
                print('--------------------------------------------------------------------')

                # updates exclude cols to empty
                if contain_concate_opt_a_to_b:
                    exclude_cols_dict_front_b_to_a = exclude_cols_dict_front_b_to_a_org.copy()
                    exclude_cols_dict_back_b_to_a = exclude_cols_dict_back_b_to_a_org.copy()

                # update the edit dist if a to b action is accepted

            else:
                # Reject transformation
                # Decrease Q-value of this operator
                print(f'Q value is {q_reward}')
                for col,every_action in transformed_df_action_a_to_b.items():
                    print(f"Reject the transformation {every_action.name} for {col}")
                    print(f'the current action params {every_action.params}')
                    print('==================================================')
                    print('--------------------------------------------------------------------')
      
     # update action for b to a
     #####################################
     #####################################
     #####################################
     #####################################
     #####################################
      print('Finding action for table B')
      print('##############################################')
    #   transformed_df_action_b_to_a, transformed_df_action_b_to_a_col,transformed_df_action_a_to_b_col_both,col_for_col_col_other,q_reward,exclude_dict_b_to_a = agent_table_b.choose_action(df_b,
    #       transformed_df_b_to_a,
    #       transformed_col_b_a_name,
    #       transformed_column_a_to_b,
    #       df_a,
    #       transformed_df_a_to_b,
    #       reversed_pairs,
    #       True,
    #       previous_edit_dist_a_to_b,
    #       previous_freq_counts_penalty,
    #       previous_lsc_len_mat_full,
    #       prev_edit_dist_matrix_a_to_b,
    #       prev_matrix_indicies,
    #       exclude_cols_dict_front_b_to_a,
    #       exclude_cols_dict_back_b_to_a,
    #       exclude_cols_lst_b_to_a,
    #       greedy
    #   )

      plan_steps,_ = agent_table_b.choose_action_plan_with_depth(
        df_b,
        transformed_df_b_to_a,
        transformed_col_b_a_name,
        transformed_column_a_to_b,      # other_column
        df_a,
        transformed_df_a_to_b,          # transformed_df_other
        reversed_pairs,
        True,                           # reversd_order
        previous_edit_dist_a_to_b,       
        previous_freq_counts_penalty,   
        previous_lsc_len_mat_full,      
        prev_edit_dist_matrix_a_to_b,    
        prev_matrix_indicies,
        exclude_cols_dict_front_b_to_a,
        exclude_cols_dict_back_b_to_a,
        exclude_cols_lst_b_to_a,
        greedy,
        depth=2,                         # pick your depth
        )
      

      for transformed_df_action_b_to_a, transformed_df_action_b_to_a_col, \
          transformed_df_action_a_to_b_col_both, col_for_col_col_other, \
          q_reward, exclude_dict_b_to_a in plan_steps:

            new_exclude_cols_dict_front_b_to_a,new_exclude_cols_dict_back_b_to_a = exclude_dict_b_to_a

            print(transformed_df_action_b_to_a)

            if not transformed_df_action_b_to_a:
                wait_to_restart_b_a = True
            else:
                first_key = list(transformed_df_action_b_to_a.keys())[0]
                first_value = transformed_df_action_b_to_a[first_key]
                if first_value.name == 'Restart':
                    wait_to_restart_b_a = True
                else:
                    wait_to_restart_b_a = False

            # update the transformed df
            transformed_df_b_to_a_new = transformed_df_b_to_a.copy()
            transformed_df_a_to_b_new = transformed_df_a_to_b.copy()

            # apply transformations to the transformed df 
            contain_concate_opt_b_to_a = False
            for col_name,action in transformed_df_action_b_to_a.items():
                if col_name not in transformed_df_b_to_a_new.columns:
                    continue
                print(f'Applying {action.name} for {col_name}')
                if action.operator_type in ['direct','direct_split']:
                    transformed_df_b_to_a_new[col_name] = [action.apply(value) for value in transformed_df_b_to_a_new[col_name]]
                elif action.operator_type == 'concate':
                    transformed_df_b_to_a_new = action.func(transformed_df_b_to_a_new,col_name,df_b[transformed_df_action_b_to_a_col[col_name]])
                    # update the contain the concate operators
                    contain_concate_opt_b_to_a = True
                elif action.operator_type == 'concate_both':
                    insertion_pos = col_for_col_col_other[col_name]
                    transformed_df_b_to_a_new,transformed_df_a_to_b_new = action.func(transformed_df_b_to_a_new,transformed_df_a_to_b_new,
                                                                                        df_b[transformed_df_action_b_to_a_col[col_name]],
                                                                                        df_a[transformed_df_action_a_to_b_col_both[insertion_pos]],
                                                                                        col_name,insertion_pos)

            transformed_column_b_to_a_new = concatenate_with_order(transformed_df_b_to_a_new,transformed_col_b_a_name)[transformed_col_b_a_name]
      


            edit_dist,new_edit_dist_matrix_a_to_b,freq_counts_penalty,lcs_matrix_full = get_ALCS_matrix_with_top_k(transformed_column_a_to_b, transformed_column_b_to_a_new,prev_matrix_indicies,greedy)
            print(f"New ALCS is {edit_dist}")
            print(f'the q reward is {q_reward}')

            if q_reward > 0:
                # Accept transformation
                action_for_transformation_b_to_a = copy.deepcopy(transformed_df_action_b_to_a)

                # append the results
                for col_name, action in action_for_transformation_b_to_a.items():
                    transformations_names_table_b[col_name].append(action.name)
                    transformations_table_b[col_name].append(action)

                #update the transformed df and concated column
                transformed_df_b_to_a = transformed_df_b_to_a_new  
                transformed_column_b_to_a = transformed_column_b_to_a_new
                transformed_df_a_to_b = transformed_df_a_to_b_new
                previous_edit_dist_a_to_b = edit_dist
                previous_lsc_len_mat_full= lcs_matrix_full
                previous_freq_counts_penalty = freq_counts_penalty
                prev_edit_dist_matrix_a_to_b = new_edit_dist_matrix_a_to_b
                exclude_cols_dict_front_b_to_a = new_exclude_cols_dict_front_b_to_a
                exclude_cols_dict_back_b_to_a = new_exclude_cols_dict_back_b_to_a

                # Update current best transformations if condition is satisfied
                if greedy:
                    if edit_dist > current_best_edit_dist_table_a_to_b :
                        current_best_edit_dist_table_a_to_b = edit_dist
                        current_best_transformations_b_to_a = copy.deepcopy(transformations_table_b)
                        current_best_transformed_df_b_to_a = transformed_df_b_to_a.copy()
                        current_best_transformed_column_b_to_a = transformed_column_b_to_a.copy()
                else:
                        current_best_edit_dist_table_a_to_b = edit_dist
                        current_best_transformations_b_to_a = copy.deepcopy(transformations_table_b)
                        current_best_transformed_df_b_to_a = transformed_df_b_to_a.copy()
                        current_best_transformed_column_b_to_a = transformed_column_b_to_a.copy()
                    

                print(f'Q value is {q_reward}')
                for col,every_action in transformed_df_action_b_to_a.items():
                    if every_action.operator_type in ['direct','direct_split']:
                        print(f"Accept the transformation {every_action.name} for {col}")
                    elif every_action.operator_type == 'concate':
                        print(f"Accept the transformation {every_action.name} for {col} with {transformed_df_action_b_to_a_col[col]}")
                    elif every_action.operator_type == 'concate_both':
                        insertion_pos = col_for_col_col_other[col]
                        print(f"Accept the transformation {every_action.name} for {col} with {transformed_df_action_b_to_a_col[col]}")
                        print(f"Accept the transformation {every_action.name} for {insertion_pos} with {transformed_df_action_a_to_b_col_both[insertion_pos]}")

                    print(f'the current action params {every_action.params}')
                    print('==================================================')

                # Check if Edit distance threshold is met
                if edit_dist >= edit_dist_threshold and freq_counts_penalty == 0:
                    print("ALCS threshold met for B to A transformations.")
                    break

                # Adjust parameters if cumulative
                agent_table_b.update_cum_params(action_for_transformation_b_to_a)
                print('Adjusted cumulative parameters for B to A transformation.')
                print('--------------------------------------------------------------------')

                if contain_concate_opt_b_to_a:
                    exclude_cols_dict_front_a_to_b = exclude_cols_dict_front_a_to_b_org.copy()
                    exclude_cols_dict_back_a_to_b = exclude_cols_dict_back_a_to_b_org.copy() 

            else:
                # Reject transformation
                print(f'Q value is {q_reward}')
                for col,every_action in transformed_df_action_b_to_a.items():
                    print(f"Reject the transformation {every_action.name} for {col}")
                    print(f'the current action params {every_action.params}')
                    print('==================================================')
                print('--------------------------------------------------------------------')

      # Record ALCS after this step for trend analysis
      _alcs_history.append(previous_edit_dist_a_to_b)


    # Recalculate final Edit distance using the transformed columns

    final_edit_dist, edit_dist_matrix,freq_counts_penalty,lcs_matrix_full= get_ALCS_matrix_with_top_k(current_best_transformed_column_a_to_b, current_best_transformed_column_b_to_a,prev_matrix_indicies,greedy)

    if final_edit_dist!= 1 and freq_counts_penalty!=0:
        if final_edit_dist <= 0.8:
            if all_numbers(current_best_transformed_column_a_to_b) or all_numbers(current_best_transformed_column_b_to_a):
                print('Lowering similarity by one of col is numbers only')
                final_edit_dist = final_edit_dist/2



    print(f'Final ALCS similarity is {final_edit_dist}')

    print(f'The best transformed column a to b is {current_best_transformed_column_a_to_b}')
    print(f'The best transformed column b to a is {current_best_transformed_column_b_to_a}')

    table_a_to_b_cols = current_best_transformed_df_a_to_b.columns
    table_b_to_a_cols = current_best_transformed_df_b_to_a.columns

    if current_best_transformations_a_to_b:
        for cols, actions in current_best_transformations_a_to_b.items():
            print(cols)
            for action in actions:
                print(action.name)
                print(action.params)

    if current_best_transformations_b_to_a:
        for cols, actions in current_best_transformations_b_to_a.items():
            print(cols)
            for action in actions:
                print(action.name)
                print(action.params)

    return current_best_transformations_a_to_b, table_a_to_b_cols, current_best_transformations_b_to_a,table_b_to_a_cols, edit_dist_matrix


#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#################  with reusing 

def optimize_transformations_both_edit_dist_opt_with_reusing(max_steps, df_a, df_b,column_a_name,column_b_name, agent_table_a,agent_table_b,pairs, edit_dist_threshold,greedy = True,
                                                transformed_columns_for_a_to_b= None,transformed_colums_for_b_to_a =None,k=0.5):
    
    # make all the strings in lower case

    df_a = df_a.loc[:, df_a.isna().mean() < 0.8].astype(str).apply(lambda col: col.str.lower())
    df_b = df_b.loc[:, df_b.isna().mean() < 0.8].astype(str).apply(lambda col: col.str.lower())

    # Initialize variables for the transformation process for column
    if column_a_name not in df_a.columns or column_b_name not in df_b.columns:

        return None, None, None, None, None, None, None, None

    # make all the strings in lower case
    df_a = df_a.astype(str).apply(lambda col: col.str.lower())
    df_b = df_b.astype(str).apply(lambda col: col.str.lower())

    # get reversed pairs
    print(f'pairs are {pairs}')
    reversed_pairs  = [(y, x) for x, y in pairs]

    # Initialize variables for the transformation process for column
    transformations_table_a = defaultdict(list)
    transformations_names_table_a = defaultdict(list)
    # transformed_df_a_to_b = df_a[[column_a_name]]

    transformations_table_b = defaultdict(list)
    transformations_names_table_b = defaultdict(list)
    # transformed_df_b_to_a = df_b[[column_b_name]]

    transformed_col_a_b_name = 'transformed_column_a_to_b'
    if transformed_columns_for_a_to_b is not None and isinstance(transformed_columns_for_a_to_b, list) and transformed_columns_for_a_to_b:

        current_best_transformed_df_a_to_b = df_a[transformed_columns_for_a_to_b]

        current_best_transformed_column_a_to_b = concatenate_with_order(current_best_transformed_df_a_to_b,transformed_col_a_b_name)[transformed_col_a_b_name]

        transformed_column_a_to_b = current_best_transformed_column_a_to_b.copy()

        transformed_df_a_to_b = current_best_transformed_df_a_to_b.copy()

    else:
        current_best_transformed_column_a_to_b = pd.Series(df_a[column_a_name].copy(),name=transformed_col_a_b_name)
        transformed_column_a_to_b = pd.Series(df_a[column_a_name].copy(),name=transformed_col_a_b_name)
        current_best_transformed_df_a_to_b = df_a[[column_a_name]]
        transformed_df_a_to_b = current_best_transformed_df_a_to_b.copy()

    transformed_column_a_to_b_new = None
    wait_to_restart_a_b = False


    current_best_transformations_b_to_a = []
    transformed_col_b_a_name = 'transformed_column_b_to_a'

    if transformed_colums_for_b_to_a is not None and isinstance(transformed_colums_for_b_to_a, list) and transformed_colums_for_b_to_a:

        current_best_transformed_df_b_to_a = df_b[transformed_colums_for_b_to_a]

        current_best_transformed_column_b_to_a = concatenate_with_order(current_best_transformed_df_b_to_a,transformed_col_b_a_name)[transformed_col_b_a_name]

        transformed_column_b_to_a = current_best_transformed_column_b_to_a.copy()

        transformed_df_b_to_a = current_best_transformed_df_b_to_a.copy()

    else:
        current_best_transformed_column_b_to_a = pd.Series(df_b[column_b_name].copy(),name=transformed_col_b_a_name)

        transformed_column_b_to_a = pd.Series(df_b[column_b_name].copy(),name=transformed_col_b_a_name)

        current_best_transformed_df_b_to_a =df_b[[column_b_name]]

        transformed_df_b_to_a = current_best_transformed_df_b_to_a.copy()

    transformed_column_b_to_a_new = None
    wait_to_restart_b_a = False

    # determine the ALCS for col a and col b 
    initial_edit_dist_a_to_b,edit_dist_matrix_a_to_b,ini_freq_counts_penalty,ini_lcs_matrix_full= get_ALCS_matrix(df_a[column_a_name], df_b[column_b_name],greedy)

    print(f"Initial ALCS from table a to b: {initial_edit_dist_a_to_b}")

    # Check if initial edit distance meets the threshold
    if initial_edit_dist_a_to_b >= edit_dist_threshold and ini_freq_counts_penalty == 0:
        print("Initial ALCS meets the threshold. Skipping transformations.")
        # Return empty transformations and the initial similarity matrix
        return transformations_table_a,transformed_df_a_to_b.columns,transformations_table_b,transformed_df_b_to_a.columns,edit_dist_matrix_a_to_b,agent_table_a,agent_table_b


    # clustering based on max alcs for each row
    # add randomness
    rand_seed = random.randint(1, 10000)
    max_alcs_for_each_row = np.max(edit_dist_matrix_a_to_b, axis=1, keepdims=True)

    num_clusters = 3

    n_clusters_to_use = min(num_clusters, len(max_alcs_for_each_row))
    kmeans = KMeans(n_clusters=n_clusters_to_use, random_state=rand_seed)
    kmeans.fit(max_alcs_for_each_row)

    df_max_alcs_for_each_row = pd.DataFrame(max_alcs_for_each_row, columns=["value"])
    df_max_alcs_for_each_row['row_number'] = df_max_alcs_for_each_row.index
    df_max_alcs_for_each_row['cluster'] = kmeans.labels_

    df_cluster_means = df_max_alcs_for_each_row.groupby('cluster')['value'].mean()

    df_cluster_means_sorted = df_cluster_means.sort_values(ascending=False)
    unique_clusters = df_cluster_means_sorted.index

    # Use learned cluster fractions from RewardConfig
    _cf3 = getattr(agent_table_a.reward_config, 'cluster_fractions_3', (0.35, 0.05, 0.05))
    _cf2 = getattr(agent_table_a.reward_config, 'cluster_fractions_2', (0.3, 0.05))
    _cf1 = getattr(agent_table_a.reward_config, 'cluster_fractions_1', (0.3,))

    if len(unique_clusters) == 3:
        highest_cluster = unique_clusters[0]
        medium_cluster = unique_clusters[1]
        lowest_cluster = unique_clusters[2]

        cluster_sample_fractions = {
            highest_cluster: _cf3[0],
            medium_cluster: _cf3[1],
            lowest_cluster: _cf3[2]
        }

    elif len(unique_clusters) == 2:
        highest_cluster = unique_clusters[0]
        lowest_cluster = unique_clusters[1]

        cluster_sample_fractions = {
            highest_cluster: _cf2[0],
            lowest_cluster: _cf2[1]
        }

    elif len(unique_clusters) == 1:
        the_only_cluster = unique_clusters[0]
        cluster_sample_fractions = {
            the_only_cluster: _cf1[0]
        }

    def sample_cluster(group):
        c = group.name  # cluster label
        frac = cluster_sample_fractions[c]
        # number of rows to sample; ensure at least 1
        n = max(1, int(frac * len(group)))
        return group.sample(n=n, random_state=rand_seed)

    sampled_df_max_alcs_for_each_row = (
        df_max_alcs_for_each_row
        .groupby('cluster', group_keys=False)
        .apply(sample_cluster)
    )

    selected_rows = sampled_df_max_alcs_for_each_row['row_number'].tolist()
    selected_rows_index = sorted(selected_rows)

    transformed_df_a_to_b = transformed_df_a_to_b.loc[selected_rows_index]
    df_a = df_a.loc[selected_rows_index]

    current_best_transformed_column_a_to_b = current_best_transformed_column_a_to_b.loc[selected_rows_index]
    transformed_column_a_to_b = transformed_column_a_to_b.loc[selected_rows_index]


    new_sim_matrix = edit_dist_matrix_a_to_b[selected_rows_index,:]
    new_ini_lcs_matrix_full = ini_lcs_matrix_full[selected_rows_index,:]

    sim_matrix_max_vals = np.max(new_sim_matrix, axis=1, keepdims=True)

    new_mean_alcs = np.mean(sim_matrix_max_vals)

    mask = new_sim_matrix == sim_matrix_max_vals

    modified_result_indices = [np.where(row)[0] for row in mask]

    new_ini_freq_counts_penalty  = np.sum(get_penalty_overlap_counts(modified_result_indices))


    previous_edit_dist_a_to_b = new_mean_alcs
    previous_freq_counts_penalty = new_ini_freq_counts_penalty
    previous_lsc_len_mat_full = new_ini_lcs_matrix_full
    prev_edit_dist_matrix_a_to_b = new_sim_matrix
    # Use learned top_k_percent from RewardConfig, fallback to function param or 0.5
    _k = k if k is not None else getattr(agent_table_a.reward_config, 'top_k_percent', 0.5)
    prev_matrix_indicies = get_top_k_percent(prev_edit_dist_matrix_a_to_b, _k)

    current_best_edit_dist_table_a_to_b = new_mean_alcs
    current_best_transformations_a_to_b = []

    agreement_percentage = agent_table_a.agreement_percentage

    transformed_df_action_a_to_b = None
    transformed_df_action_b_to_a = None

    transformed_df_action_a_to_b_col = None
    transformed_df_action_b_to_a_col = None

    transformed_df_action_a_to_b_col_both = None
    transformed_df_action_b_to_a_col_both = None

    exclude_cols_dict_front_a_to_b_org = {col: [] for col in df_a.columns}
    exclude_cols_dict_back_a_to_b_org = {col: [] for col in df_a.columns}

    exclude_cols_dict_front_b_to_a_org = {col: [] for col in df_b.columns}
    exclude_cols_dict_back_b_to_a_org = {col: [] for col in df_b.columns}

    exclude_cols_dict_front_a_to_b = exclude_cols_dict_front_a_to_b_org.copy()
    exclude_cols_dict_back_a_to_b = exclude_cols_dict_back_a_to_b_org.copy()

    exclude_cols_dict_front_b_to_a = exclude_cols_dict_front_b_to_a_org.copy()
    exclude_cols_dict_back_b_to_a = exclude_cols_dict_back_b_to_a_org.copy()


    ## exclude the all numbers cols
    exclude_cols_lst_a_to_b_arg = [col for col in df_a.columns if all_numbers(df_a[col])]
    exclude_cols_lst_b_to_a_arg = [col for col in df_b.columns if all_numbers(df_b[col])]

    exclude_cols_lst_a_to_b = df_a[exclude_cols_lst_a_to_b_arg]
    exclude_cols_lst_b_to_a = df_b[exclude_cols_lst_b_to_a_arg]

    print(f'exclude cols a to b:{exclude_cols_lst_a_to_b_arg}')
    print(f'exclude cols b to a:{exclude_cols_lst_b_to_a_arg}')

    # Step-level early stopping: track ALCS history and stop when improvement flatlines
    _alcs_history = [new_mean_alcs]

    for iters in range(max_steps):
      # Early stop: if recent ALCS deltas show no improvement trend, break.
      # Uses the last 3 deltas — if all are <= 0, the search has stalled.
      # Only for longer runs (max_steps > 3) to avoid cutting short wrapper phase.
      if len(_alcs_history) >= 4 and max_steps > 3:
          _recent_deltas = [_alcs_history[i] - _alcs_history[i-1] for i in range(-3, 0)]
          _all_flat = all(d <= 0.001 for d in _recent_deltas)
      else:
          _all_flat = False
      if _all_flat and iters >= 3:
          print(f"Early stopping at step {iters}: flat ALCS trend, ALCS={previous_edit_dist_a_to_b:.4f}")
          break

      print(f'The iteration is {iters}')
      # find action for a to b
      print('Finding action for table A')
      print('##############################################')

      if wait_to_restart_a_b == True and wait_to_restart_b_a == True:
        print('Starting to Restart')
        table_a_to_b_cols = current_best_transformed_df_a_to_b.columns
        table_b_to_a_cols = current_best_transformed_df_b_to_a.columns
        # no actions can future increase the reward and not meet the threshold, begin to restart
        final_edit_dist, edit_dist_matrix,freq_counts_penalty,lcs_matrix_full = get_ALCS_matrix_with_top_k(current_best_transformed_column_a_to_b, current_best_transformed_column_b_to_a,prev_matrix_indicies,greedy)

        return current_best_transformations_a_to_b, table_a_to_b_cols, current_best_transformations_b_to_a,table_b_to_a_cols, edit_dist_matrix,agent_table_a,agent_table_b
      
    #   print(f'transformed_df_a_to_b has {len(transformed_df_a_to_b)}')
    #   print(f'transformed_df_b_to_a has {len(transformed_df_b_to_a)}')

      plan_steps_a, _ = agent_table_a.choose_action_plan_with_depth(
        df_a,
        transformed_df_a_to_b,
        transformed_col_a_b_name,
        transformed_column_b_to_a,
        df_b,
        transformed_df_b_to_a,
        pairs,
        False,
        previous_edit_dist_a_to_b,
        previous_freq_counts_penalty,
        previous_lsc_len_mat_full,
        prev_edit_dist_matrix_a_to_b,
        prev_matrix_indicies,
        exclude_cols_dict_front_a_to_b,
        exclude_cols_dict_back_a_to_b,
        exclude_cols_lst_a_to_b,
        greedy,
        depth=0,
        )
      if not plan_steps_a:
        break
      transformed_df_action_a_to_b, transformed_df_action_a_to_b_col, \
        transformed_df_action_b_to_a_col_both, col_for_col_col_other, \
        q_reward, exclude_dict_a_to_b = plan_steps_a[0]
      
      new_exclude_cols_dict_front_a_to_b,new_exclude_cols_dict_back_a_to_b = exclude_dict_a_to_b 


           
      if not transformed_df_action_a_to_b:
         wait_to_restart_a_b = True
      else:
         first_key = list(transformed_df_action_a_to_b.keys())[0]
         first_value = transformed_df_action_a_to_b[first_key]
         if first_value.name == 'Restart':
            wait_to_restart_a_b = True
         else:
            wait_to_restart_a_b = False

      # update the transformed df
      transformed_df_a_to_b_new = transformed_df_a_to_b.copy()
      transformed_df_b_to_a_new = transformed_df_b_to_a.copy()

      # apply transformations to the transformed df 
      contain_concate_opt_a_to_b = False
      for col_name,action in transformed_df_action_a_to_b.items():
        if col_name not in transformed_df_a_to_b_new.columns:
          continue
        if action.operator_type in ['direct','direct_split']:
          transformed_df_a_to_b_new[col_name] = [action.apply(value) for value in transformed_df_a_to_b_new[col_name]]
        elif action.operator_type == 'concate':
          transformed_df_a_to_b_new = action.func(transformed_df_a_to_b_new,col_name,df_a[transformed_df_action_a_to_b_col[col_name]])
          # the chosen action contains the concate opts
          contain_concate_opt_a_to_b = True
        elif action.operator_type == 'concate_both':
          insertion_pos = col_for_col_col_other[col_name]
          transformed_df_a_to_b_new,transformed_df_b_to_a_new = action.func(transformed_df_a_to_b_new,transformed_df_b_to_a_new,
                                                                            df_a[transformed_df_action_a_to_b_col[col_name]],
                                                                            df_b[transformed_df_action_b_to_a_col_both[insertion_pos]],
                                                                            col_name,insertion_pos, )
          


      # concate the transformed df to get the transformed column 
      # edit_dist, edit_matrix,freq_counts_penalty, lsc_len_mat_full, result_values_list, result_indices
      transformed_column_a_to_b_new = concatenate_with_order(transformed_df_a_to_b_new,transformed_col_a_b_name)[transformed_col_a_b_name]
      edit_dist,new_edit_dist_matrix_a_to_b,freq_counts_penalty,lcs_matrix_full = get_edit_dist_matrix_with_top_k(transformed_column_a_to_b_new, transformed_column_b_to_a,prev_matrix_indicies)
      print(f"New ALCS is {edit_dist}")

      if q_reward > 0:
        # Accept transformation
        action_for_transformation_a_to_b = copy.deepcopy(transformed_df_action_a_to_b)
        # append the results
        for col_name, action in action_for_transformation_a_to_b.items():
          transformations_names_table_a[col_name].append(action.name)
          transformations_table_a[col_name].append(action)

        #update the transformed df and concated column
        transformed_df_a_to_b = transformed_df_a_to_b_new
        transformed_column_a_to_b = transformed_column_a_to_b_new
        transformed_df_b_to_a = transformed_df_b_to_a_new
        #update the edit dist and edit dist matrix
        previous_edit_dist_a_to_b = edit_dist
        previous_lsc_len_mat_full= lcs_matrix_full
        previous_freq_counts_penalty = freq_counts_penalty
        prev_edit_dist_matrix_a_to_b = new_edit_dist_matrix_a_to_b

        exclude_cols_dict_front_a_to_b = new_exclude_cols_dict_front_a_to_b
        exclude_cols_dict_back_a_to_b = new_exclude_cols_dict_back_a_to_b 


        # Update current best transformations if statisfy the cond
        if greedy:
            if edit_dist > current_best_edit_dist_table_a_to_b :
                current_best_edit_dist_table_a_to_b = edit_dist
                current_best_transformations_a_to_b = copy.deepcopy(transformations_table_a)
                current_best_transformed_column_a_to_b = transformed_column_a_to_b.copy()
                current_best_transformed_df_a_to_b = transformed_df_a_to_b.copy()
        
        else:
            current_best_edit_dist_table_a_to_b = edit_dist
            current_best_transformations_a_to_b = copy.deepcopy(transformations_table_a)
            current_best_transformed_column_a_to_b = transformed_column_a_to_b.copy()
            current_best_transformed_df_a_to_b = transformed_df_a_to_b.copy()


        print(f'Q value is {q_reward}')

        for col,every_action in transformed_df_action_a_to_b.items():
            if every_action.operator_type in ['direct','direct_split']:
                print(f"Accept the transformation {every_action.name} for {col}")
            elif every_action.operator_type == 'concate':
                print(f"Accept the transformation {every_action.name} for {col} with {transformed_df_action_a_to_b_col[col]}")
            elif every_action.operator_type == 'concate_both':
                insertion_pos = col_for_col_col_other[col]
                print(f"Accept the transformation {every_action.name} for {col} with {transformed_df_action_a_to_b_col[col]}")
                print(f"Accept the transformation {every_action.name} for {insertion_pos} with {transformed_df_action_b_to_a_col_both[insertion_pos]}")
              
            print(f'the current action params {every_action.params}')
            print('==================================================')

        # Check if Edit distance threshold is met and all unique
        if edit_dist >= edit_dist_threshold and freq_counts_penalty == 0:
            print("ALCS threshold met.")
            break
        #Adjust parameters if cumulative
        agent_table_a.update_cum_params(action_for_transformation_a_to_b)
        print('--------------------------------------------------------------------')

        # updates exclude cols to empty
        if contain_concate_opt_a_to_b:
            exclude_cols_dict_front_b_to_a = exclude_cols_dict_front_b_to_a_org.copy()
            exclude_cols_dict_back_b_to_a = exclude_cols_dict_back_b_to_a_org.copy()

        # update the edit dist if a to b action is accepted

      else:
        # Reject transformation
        # Decrease Q-value of this operator
        print(f'Q value is {q_reward}')
        for col,every_action in transformed_df_action_a_to_b.items():
          print(f"Reject the transformation {every_action.name} for {col}")
          print(f'the current action params {every_action.params}')
          print('==================================================')
        print('--------------------------------------------------------------------')




    
      
     # update action for b to a
     #####################################
     #####################################
     #####################################
     #####################################
     #####################################
    #   print(f'transformed_df_a_to_b has {len(transformed_df_a_to_b)} and df_a has {len(df_a)}')
    #   print(f'transformed_df_b_to_a has {len(transformed_df_b_to_a)} and df_b has {len(df_b)}')
      print('Finding action for table B')
      print('##############################################')
      plan_steps_b, _ = agent_table_b.choose_action_plan_with_depth(
          df_b,
          transformed_df_b_to_a,
          transformed_col_b_a_name,
          transformed_column_a_to_b,
          df_a,
          transformed_df_a_to_b,
          reversed_pairs,
          True,
          previous_edit_dist_a_to_b,
          previous_freq_counts_penalty,
          previous_lsc_len_mat_full,
          prev_edit_dist_matrix_a_to_b,
          prev_matrix_indicies,
          exclude_cols_dict_front_b_to_a,
          exclude_cols_dict_back_b_to_a,
          exclude_cols_lst_b_to_a,
          greedy,
          depth=0,
      )
      if not plan_steps_b:
        break
      transformed_df_action_b_to_a, transformed_df_action_b_to_a_col, \
        transformed_df_action_a_to_b_col_both, col_for_col_col_other, \
        q_reward, exclude_dict_b_to_a = plan_steps_b[0]

      new_exclude_cols_dict_front_b_to_a,new_exclude_cols_dict_back_b_to_a = exclude_dict_b_to_a

      print(transformed_df_action_b_to_a)

      if not transformed_df_action_b_to_a:
         wait_to_restart_b_a = True
      else:
         first_key = list(transformed_df_action_b_to_a.keys())[0]
         first_value = transformed_df_action_b_to_a[first_key]
         if first_value.name == 'Restart':
            wait_to_restart_b_a = True
         else:
            wait_to_restart_b_a = False

      # update the transformed df
      transformed_df_b_to_a_new = transformed_df_b_to_a.copy()
      transformed_df_a_to_b_new = transformed_df_a_to_b.copy()

      # apply transformations to the transformed df 
      contain_concate_opt_b_to_a = False
      for col_name,action in transformed_df_action_b_to_a.items():
        if col_name not in transformed_df_b_to_a_new.columns:
          continue
        print(f'Applying {action.name} for {col_name}')
        if action.operator_type in ['direct','direct_split']:
          transformed_df_b_to_a_new[col_name] = [action.apply(value) for value in transformed_df_b_to_a_new[col_name]]
        elif action.operator_type == 'concate':
          transformed_df_b_to_a_new = action.func(transformed_df_b_to_a_new,col_name,df_b[transformed_df_action_b_to_a_col[col_name]])
          # update the contain the concate operators
          contain_concate_opt_b_to_a = True
        elif action.operator_type == 'concate_both':
          insertion_pos = col_for_col_col_other[col_name]
          transformed_df_b_to_a_new,transformed_df_a_to_b_new = action.func(transformed_df_b_to_a_new,transformed_df_a_to_b_new,
                                                                            df_b[transformed_df_action_b_to_a_col[col_name]],
                                                                            df_a[transformed_df_action_a_to_b_col_both[insertion_pos]],
                                                                            col_name,insertion_pos)

      transformed_column_b_to_a_new = concatenate_with_order(transformed_df_b_to_a_new,transformed_col_b_a_name)[transformed_col_b_a_name]
      


      edit_dist,new_edit_dist_matrix_a_to_b,freq_counts_penalty,lcs_matrix_full = get_edit_dist_matrix_with_top_k(transformed_column_a_to_b, transformed_column_b_to_a_new,prev_matrix_indicies)
      print(f"New ALCS is {edit_dist}")
      print(f'the q reward is {q_reward}')

      if q_reward > 0:
          # Accept transformation
          action_for_transformation_b_to_a = copy.deepcopy(transformed_df_action_b_to_a)

          # append the results
          for col_name, action in action_for_transformation_b_to_a.items():
            transformations_names_table_b[col_name].append(action.name)
            transformations_table_b[col_name].append(action)

          #update the transformed df and concated column
          transformed_df_b_to_a = transformed_df_b_to_a_new  
          transformed_column_b_to_a = transformed_column_b_to_a_new
          transformed_df_a_to_b = transformed_df_a_to_b_new
          previous_edit_dist_a_to_b = edit_dist
          previous_lsc_len_mat_full= lcs_matrix_full
          previous_freq_counts_penalty = freq_counts_penalty
          prev_edit_dist_matrix_a_to_b = new_edit_dist_matrix_a_to_b
          exclude_cols_dict_front_b_to_a = new_exclude_cols_dict_front_b_to_a
          exclude_cols_dict_back_b_to_a = new_exclude_cols_dict_back_b_to_a

          # Update current best transformations if condition is satisfied
          if greedy:
              if edit_dist > current_best_edit_dist_table_a_to_b :
                current_best_edit_dist_table_a_to_b = edit_dist
                current_best_transformations_b_to_a = copy.deepcopy(transformations_table_b)
                current_best_transformed_df_b_to_a = transformed_df_b_to_a.copy()
                current_best_transformed_column_b_to_a = transformed_column_b_to_a.copy()
          else:
                current_best_edit_dist_table_a_to_b = edit_dist
                current_best_transformations_b_to_a = copy.deepcopy(transformations_table_b)
                current_best_transformed_df_b_to_a = transformed_df_b_to_a.copy()
                current_best_transformed_column_b_to_a = transformed_column_b_to_a.copy()
              

          print(f'Q value is {q_reward}')
          for col,every_action in transformed_df_action_b_to_a.items():
            if every_action.operator_type in ['direct','direct_split']:
                print(f"Accept the transformation {every_action.name} for {col}")
            elif every_action.operator_type == 'concate':
                print(f"Accept the transformation {every_action.name} for {col} with {transformed_df_action_b_to_a_col[col]}")
            elif every_action.operator_type == 'concate_both':
                insertion_pos = col_for_col_col_other[col]
                print(f"Accept the transformation {every_action.name} for {col} with {transformed_df_action_b_to_a_col[col]}")
                print(f"Accept the transformation {every_action.name} for {insertion_pos} with {transformed_df_action_a_to_b_col_both[insertion_pos]}")

            print(f'the current action params {every_action.params}')
            print('==================================================')

          # Check if Edit distance threshold is met
          if edit_dist >= edit_dist_threshold and freq_counts_penalty == 0:
              print("ALCS threshold met for B to A transformations.")
              break

          # Adjust parameters if cumulative
          agent_table_b.update_cum_params(action_for_transformation_b_to_a)
          print('Adjusted cumulative parameters for B to A transformation.')
          print('--------------------------------------------------------------------')

          if contain_concate_opt_b_to_a:
            exclude_cols_dict_front_a_to_b = exclude_cols_dict_front_a_to_b_org.copy()
            exclude_cols_dict_back_a_to_b = exclude_cols_dict_back_a_to_b_org.copy() 

      else:
          # Reject transformation
          print(f'Q value is {q_reward}')
          for col,every_action in transformed_df_action_b_to_a.items():
            print(f"Reject the transformation {every_action.name} for {col}")
            print(f'the current action params {every_action.params}')
            print('==================================================')
          print('--------------------------------------------------------------------')



    # Recalculate final Edit distance using the transformed columns

    final_edit_dist, edit_dist_matrix,freq_counts_penalty,lcs_matrix_full= get_edit_dist_matrix_with_top_k(current_best_transformed_column_a_to_b, current_best_transformed_column_b_to_a,prev_matrix_indicies)

    if final_edit_dist!= 1 and freq_counts_penalty!=0:
        if final_edit_dist <= 0.8:
            if all_numbers(current_best_transformed_column_a_to_b) or all_numbers(current_best_transformed_column_b_to_a):
                print('Lowering similarity by one of col is numbers only')
                final_edit_dist = final_edit_dist/2



    print(f'Final ALCS similarity is {final_edit_dist}')

    print(f'The best transformed column a to b is {current_best_transformed_column_a_to_b}')
    print(f'The best transformed column b to a is {current_best_transformed_column_b_to_a}')

    table_a_to_b_cols = current_best_transformed_df_a_to_b.columns
    table_b_to_a_cols = current_best_transformed_df_b_to_a.columns

    if current_best_transformations_a_to_b:
        for cols, actions in current_best_transformations_a_to_b.items():
            print(cols)
            for action in actions:
                print(action.name)
                print(action.params)

    if current_best_transformations_b_to_a:
        for cols, actions in current_best_transformations_b_to_a.items():
            print(cols)
            for action in actions:
                print(action.name)
                print(action.params)

    return current_best_transformations_a_to_b, table_a_to_b_cols, current_best_transformations_b_to_a,table_b_to_a_cols, edit_dist_matrix,agent_table_a,agent_table_b


def build_dataframe_in_order(df, column_order):
    """
    Build a new DataFrame with the given column order.
    
    Parameters:
        df (pd.DataFrame): Original DataFrame.
        column_order (list): Desired column order.
    
    Returns:
        pd.DataFrame: New DataFrame with columns in the specified order.
    """
    # 1) If they passed None → bail
    if column_order is None:
        return df.copy()

    # 2) If it's a pandas Index, convert to list
    if isinstance(column_order, pd.Index):
        column_order = list(column_order)

    # 3) If it's empty → bail
    if len(column_order) == 0:
        return df.copy()
    
    df_copy = df.copy()
    
    # Iterate through the desired column order
    for col in column_order:
        # If the column is not already in the DataFrame, create it based on the source column
        if col not in df_copy.columns:
            # Extract the base column name before any suffix (e.g., "B" from "B_1_1")
            base_col = col.split('_')[0]
            if base_col in df.columns:
                # Assign the values of the base column to the new column
                df_copy[col] = df_copy[base_col]
    
    # Reorder the DataFrame columns to match the desired order
    df_copy = df_copy[column_order]
    
    return df_copy

def apply_all_actions_to_df(transformed_df,transformed_df_action):
  transformed_column_for_change = transformed_df.copy()
  transformed_df_cols = transformed_df.columns
  if transformed_df_action:
    for col_name, actions in transformed_df_action.items():
        if col_name not in transformed_column_for_change.columns:
            continue
        for action in actions:
            if action.operator_type in ['direct','direct_split']:
                transformed_column = transformed_column_for_change[col_name]
                transformed_column_new = [action.apply(value) for value in transformed_column]
                transformed_column_for_change[col_name] = transformed_column_new
  return transformed_column_for_change

def _diverse_sample_by_alcs(df_a, df_b, column_a_name, column_b_name, sample_size):
    """ALCS-score-guided diverse sampling.

    1. Compute quick ALCS on a small pilot sample to get per-row similarity scores
    2. Stratify rows into 3 buckets: high/medium/low similarity
    3. Sample equally from each bucket — gives the Q-learning agent diverse examples

    This is faster than full-data AND better than random sampling because it ensures
    the agent sees easy matches, hard matches, and mismatches.

    Q1 (DETERMINISTIC_SAMPLE=1): seed RNG from content hash so identical
    (df_a, col_a, df_b, col_b, sample_size) inputs always produce the SAME sample.
    Eliminates the per-rollout sample drift that breaks the ALCS cache on O8.
    """
    n_a = len(df_a)
    if sample_size >= n_a:
        return df_a.copy()

    col_a = df_a[column_a_name].fillna('').astype(str) if column_a_name in df_a.columns else df_a.iloc[:, 0].fillna('').astype(str)
    col_b = df_b[column_b_name].fillna('').astype(str) if column_b_name in df_b.columns else df_b.iloc[:, 0].fillna('').astype(str)

    # Q1: content-hash seed; otherwise fall back to global random state (default).
    if False:
        import hashlib
        _h = hashlib.md5()
        _h.update(repr(col_a.tolist()).encode('utf-8', errors='replace'))
        _h.update(repr(col_b.tolist()).encode('utf-8', errors='replace'))
        _h.update(str(sample_size).encode())
        _seed = int(_h.hexdigest()[:8], 16)
        _rng = random.Random(_seed)
        _pd_seed = _seed
    else:
        _rng = random
        _pd_seed = None

    try:
        # Quick pilot: compute ALCS for each source row against a small target sample
        pilot_b = col_b.sample(min(15, len(col_b)), random_state=_pd_seed).tolist()
        row_scores = np.zeros(n_a)
        for i, val_a in enumerate(col_a):
            best = 0.0
            for val_b in pilot_b:
                sim, _, _ = lcs_count(str(val_a), str(val_b), 2)
                if sim > best:
                    best = sim
            row_scores[i] = best

        # Stratify into 3 buckets by score
        q33 = np.percentile(row_scores, 33)
        q66 = np.percentile(row_scores, 66)
        low = [i for i in range(n_a) if row_scores[i] <= q33]
        mid = [i for i in range(n_a) if q33 < row_scores[i] <= q66]
        high = [i for i in range(n_a) if row_scores[i] > q66]

        per_bucket = max(1, sample_size // 3)
        sampled = []
        for bucket in [high, mid, low]:  # prioritize high-sim rows
            if bucket:
                sampled.extend(_rng.sample(bucket, min(per_bucket, len(bucket))))
        remaining = sample_size - len(sampled)
        if remaining > 0:
            leftover = [i for i in range(n_a) if i not in sampled]
            if leftover:
                sampled.extend(_rng.sample(leftover, min(remaining, len(leftover))))

        return df_a.iloc[sampled].reset_index(drop=True)
    except Exception:
        return df_a.sample(min(sample_size, n_a), random_state=_pd_seed).reset_index(drop=True)


def find_transformed_df_opt(sample_proportion,max_steps, df_a, df_b,column_a_name,column_b_name,all_operators,pairs, edit_dist_threshold,exploration_rate,agreement_percentage,greedy):
   min_size = min(int( df_a.shape[0]* sample_proportion),int(df_b.shape[0]* sample_proportion))
   sample_size = max(min(10, df_a.shape[0]), min_size)  # min 10 or all rows if < 10
   # Cap Q-learning sample. Full data for final join.
   sample_size = min(sample_size, _SAMPLE_CAP)

   # v9 PPO bypass: when OPT=ppo, route around RewardConfigSelector,
   # TransformSearchNet, diverse-sample, MetaSelector, and the per-pair
   # Q-learning agents — straight to the pretrained policy.
   if False:  # OPT=ppo path removed from the released config
       from v3_dd.r4.ppo_optimizer import optimize_transformations_ppo
       df_a_sampled = (df_a.sample(min(sample_size, df_a.shape[0])).reset_index(drop=True)
                       if sample_size < df_a.shape[0] else df_a.copy())
       df_b_sampled = df_b.copy()
       trans_a_to_b, table_a_to_b_cols, trans_b_to_a, table_b_to_a_cols, _ = \
           optimize_transformations_ppo(
               max_steps, df_a_sampled, df_b_sampled,
               column_a_name, column_b_name, pairs)
       df_a_concated = build_dataframe_in_order(df_a, table_a_to_b_cols)
       df_b_concated = build_dataframe_in_order(df_b, table_b_to_a_cols)
       df_a_concated_applied = apply_all_actions_to_df(
           df_a_concated, trans_a_to_b).astype(str).map(str.lower)
       df_b_concated_applied = apply_all_actions_to_df(
           df_b_concated, trans_b_to_a).astype(str).map(str.lower)
       return (df_a_concated_applied, df_b_concated_applied,
               trans_a_to_b, table_a_to_b_cols,
               trans_b_to_a, table_b_to_a_cols)

   # ALCS-score-guided diverse sampling for source.
   # NO_DIVERSE_SAMPLE=1 skips this: it does N×15 ALCS calls per
   # find_transformed_df_opt call (huge per-pair overhead on small datasets).
   # Falls back to random sample.
   # Q2: SHARE_SAMPLE_PER_DIRECTION=1 reuses the same sample across
   # rollouts of the same direction (cache keyed by id(df_a)+col+size).
   _share_sample = False
   _share_key = (id(df_a), column_a_name, sample_size) if _share_sample else None
   if sample_size < df_a.shape[0]:
       if _share_sample and _share_key in _SHARED_SAMPLE_CACHE:
           df_a_sampled = _SHARED_SAMPLE_CACHE[_share_key].copy()
       elif False:
           df_a_sampled = df_a.sample(min(sample_size, df_a.shape[0])).reset_index(drop=True)
           if _share_sample:
               _SHARED_SAMPLE_CACHE[_share_key] = df_a_sampled.copy()
       else:
           df_a_sampled = _diverse_sample_by_alcs(df_a, df_b, column_a_name, column_b_name, sample_size)
           if _share_sample:
               _SHARED_SAMPLE_CACHE[_share_key] = df_a_sampled.copy()
   else:
       df_a_sampled = df_a.copy()
   df_b_sampled = df_b.copy()

   # Select reward config via learned MetaSelector (config blending only)
   _selector = RewardConfigSelector()
   _rc, _rc_name = _selector.select(df_a_sampled, column_a_name, df_b_sampled, column_b_name)

   # Try TransformSearchNet for learned explore_rate, max_steps, config
   _tsn_pred = None
   if ModelRegistry is not None and not _learned_off('NO_TSN'):
       try:
           _tsn = ModelRegistry().load('transform_search')
           if _tsn is not None and _selector.last_priors:
               # Use the same 12-dim feature vector as MetaSelector
               _tsn_pred = _tsn.predict(np.array(_selector._last_z_p, dtype=np.float32))
       except Exception:
           pass

   # Use config-level runtime params (learned per recipe), with env-var override.
   # _EPS_SWEEP takes top priority and survives _apply_ablation_env's reset
   # (EXPLORATION_RATE does not) — so an eps sweep actually overrides the
   # TSN-predicted exploration rate instead of being silently ignored.
   if _EPS_SWEEP:
       _eps = float(_EPS_SWEEP)
   elif False:
       _eps = 0.1
   elif _tsn_pred is not None:
       _eps = max(0.05, _tsn_pred['exploration_rate'])
   else:
       _eps = getattr(_rc, 'exploration_rate', exploration_rate)
   # Capture what the LEARNED model (TSN) actually picked, so the runner can log
   # it: the raw TSN eps prediction + the eps actually used + TSN max_steps.
   globals()['_LAST_EPS'] = float(_eps)
   globals()['_LAST_TSN_EPS'] = (float(_tsn_pred['exploration_rate']) if _tsn_pred is not None else float('nan'))

   # Override max_steps from TransformSearchNet if available and no env override
   # Cap at the default to prevent timeouts — TSN can reduce steps but not exceed default
   _default_max_steps = max_steps  # preserve the caller's value (usually 5)
   if _tsn_pred is not None and not _MAX_STEPS_OVERRIDE:
       _tsn_steps = max(2, int(round(_tsn_pred['max_steps'])))
       max_steps = min(_tsn_steps, _default_max_steps)
   globals()['_LAST_MAXSTEPS'] = int(max_steps)

   # Detect concat signal: source has more columns than matched target
   n_src_cols = len(df_a_sampled.columns)
   n_tgt_cols = len(df_b_sampled.columns)
   concat_boost = 1.0
   if n_src_cols > n_tgt_cols:
       concat_boost = 2.0 + (n_src_cols - n_tgt_cols)
   if _selector.last_priors and _selector.last_priors.get('concat_gate', 0) > 0.5:
       concat_boost = max(concat_boost, 2.0)
   # TransformSearchNet concat gate override
   if _tsn_pred is not None and _tsn_pred.get('concat_gate', 0) > 0.5:
       concat_boost = max(concat_boost, 2.0)

   # Operator selection: disabled for now (model not accurate enough yet).
   # Infrastructure preserved in choose_action — enable via OP_SELECTION=1
   _ops_to_use = copy.deepcopy(all_operators)

   # ---- PART 1: LABEL-FREE OPERATOR SYNTHESIS (SYNTH_OPS=1) ----
   # Scan SOURCE values only (input, never truth) and append data-derived
   # split/numeric/strip operators the hard-coded catalog cannot express.
   if False:
       try:
           _existing_names = {getattr(op, 'name', None) for op in _ops_to_use}
           _synth = synthesize_operators_from_source(
               df_a_sampled, column_a_name, df_b_sampled, column_b_name)
           _synth = [op for op in _synth if op.name not in _existing_names]
           _ops_to_use = _ops_to_use + _synth
           globals()['_LAST_SYNTH_OPS'] = [op.name for op in _synth]
       except Exception:
           globals()['_LAST_SYNTH_OPS'] = []
   else:
       globals()['_LAST_SYNTH_OPS'] = []

   # EXPERIMENTAL (worktree only): verifier-driven op-catalog mutation.
   # Three independent modes (highest priority first):
   #
   # 1) OP_ONLY  (REPLACE mode): comma-separated names. If set,
   #    the catalog is REPLACED with EXACTLY these ops in this order.
   #    Q-learning's catalog has nothing else to consider, so it must
   #    pick from this list. Use when the verifier knows the exact op.
   #
   # 2) OP_REMOVE (EXPLORE mode): drop named ops from the catalog.
   #    Q-learning still runs but cannot pick the dropped ones.
   #
   # 3) OP_PREFER (EXPLORE mode): re-order so prefered ops are
   #    tried first by Q-learning. Catalog otherwise unchanged.
   #
   # ONLY takes precedence over REMOVE+PREFER (REPLACE wins).
   # EXPERIMENTAL (worktree only): verifier-driven CATALOG mutation.
   # The chain's Q-learning still does the selection — the verifier just
   # curates the menu.
   #
   # OP_REMOVE: comma-separated names to drop from the catalog
   #   (Q-learning won't be able to pick them).
   # OP_PREFER: comma-separated names to prepend so the agent's
   #   exploration sees them first (still kept in catalog otherwise).
   #
   # No effect when env vars are unset.
   _op_remove = ''
   _op_prefer = ''
   if _op_remove:
       _remove_set = {n.strip() for n in _op_remove.split(',') if n.strip()}
       _ops_to_use = [op for op in _ops_to_use if getattr(op, 'name', None) not in _remove_set]
   if _op_prefer:
       _prefer_list = [n.strip() for n in _op_prefer.split(',') if n.strip()]
       _by_name = {getattr(op, 'name', None): op for op in _ops_to_use}
       _front = [_by_name[n] for n in _prefer_list if n in _by_name]
       _rest = [op for op in _ops_to_use if getattr(op, 'name', None) not in set(_prefer_list)]
       _ops_to_use = _front + _rest
   _ops_a = _ops_to_use
   _ops_b = copy.deepcopy(_ops_to_use)

   agent_table_a = QLearningAgent_edit_dist_modified_for_multi_opt(_ops_a, df_a_sampled, agreement_percentage, _eps, reward_config=_rc, depth=0)
   agent_table_b = QLearningAgent_edit_dist_modified_for_multi_opt(_ops_b, df_b_sampled, agreement_percentage, _eps, reward_config=_rc, depth=0)

   # ---- PART 2: CONDITION-BASED EXPLORATION (COND_EXPLORE=1) ----
   # Seed each agent's per-column operator prior from the SOURCE column structure
   # (label-free). The explore-branch in choose_action sub-samples ops via
   # random.choices weighted by operators_prob_dict; seeding it with the data-
   # conditioned prior makes exploration try the data-suggested operators first
   # instead of uniform-random — the state-conditioned policy under test.
   if False:
       try:
           apply_conditioned_prior_to_agent(agent_table_a, df_a_sampled, column_a_name,
                                             df_b_sampled, column_b_name)
           apply_conditioned_prior_to_agent(agent_table_b, df_b_sampled, column_b_name,
                                             df_a_sampled, column_a_name)
       except Exception:
           pass

   # Learned operator selection (opt-in via env var until model is accurate)
   if False and _selector.last_priors and _selector.last_priors.get('operator_group_weights'):
       _ogw = _selector.last_priors['operator_group_weights']
       _op_group_map = {
           'substring': ['substring_operator_forward', 'substring_operator_back_ward',
                         'substring_1_forward_constant', 'substring_1_back_ward_constant',
                         'substring_second_forward_constant', 'substring_second_back_ward_constant'],
           'split_reorder': ['SelectK_for_separated', 'SelectK_for_separated_reverse',
                             'shift_1_word_forward', 'move_first_to_last'],
           'delimiter': ['extract_by_delimiter_dash_last', 'extract_by_delimiter_dash_first',
                         'extract_by_delimiter_space_last', 'extract_by_delimiter_space_first',
                         'extract_after_delimiter_pattern', 'extract_prefix'],
           'initials': ['extract_initials'],
           'pattern_extract': ['strip_parenthetical', 'strip_numeric_prefix'],
       }
       _selected_op_names = set()
       _unselected_op_names = set()
       for group, op_names in _op_group_map.items():
           if _ogw.get(group, 0.0) > 0.3:
               _selected_op_names.update(op_names)
           else:
               _unselected_op_names.update(op_names)
       if len(_selected_op_names) > 0 and len(_unselected_op_names) > 0:
           agent_table_a.selected_op_names = _selected_op_names
           agent_table_a.unselected_op_names = _unselected_op_names
           agent_table_b.selected_op_names = _selected_op_names
           agent_table_b.unselected_op_names = _unselected_op_names

   # Boost concat operator probabilities if concat signal detected
   if concat_boost > 1.0:
       for agent in [agent_table_a, agent_table_b]:
           for col in list(agent.operators_prob_dict.keys()):
               probs = agent.operators_prob_dict[col]
               for op_name in probs:
                   if 'concat' in op_name.lower() or 'concatenate' in op_name.lower():
                       probs[op_name] *= concat_boost
               # Renormalize
               total = sum(probs.values())
               if total > 0:
                   for op_name in probs:
                       probs[op_name] /= total


   trans_a_to_b, table_a_to_b_cols, trans_b_to_a,table_b_to_a_cols, edit_dist_matrix = optimize_transformations_both_edit_dist_opt(max_steps, df_a_sampled, df_b_sampled,column_a_name,column_b_name,
                                                                                                                                                                              agent_table_a,agent_table_b,
                                                                                                                                                                               pairs, edit_dist_threshold,greedy)
   df_a_concated = build_dataframe_in_order(df_a, table_a_to_b_cols)
   df_b_concated = build_dataframe_in_order(df_b, table_b_to_a_cols)
   
   df_a_concated_applied = apply_all_actions_to_df(df_a_concated,trans_a_to_b).astype(str).apply(lambda col: col.str.lower())
   df_b_concated_applied =  apply_all_actions_to_df(df_b_concated,trans_b_to_a).astype(str).apply(lambda col: col.str.lower())

   return df_a_concated_applied,df_b_concated_applied,trans_a_to_b, table_a_to_b_cols, trans_b_to_a,table_b_to_a_cols


def find_transformed_df_opt_with_reusing(sample_proportion,max_steps, df_a, df_b,column_a_name,column_b_name,all_operators,pairs,
                                          edit_dist_threshold,exploration_rate,agreement_percentage,greedy,
                                          transcols_a_b = None,
                                          transcols_b_a = None,
                                          agenta = None,
                                          agentb = None):
   min_size = min(int( df_a.shape[0]* sample_proportion),int(df_b.shape[0]* sample_proportion))
   sample_size = max(1, min_size)

   # Diverse sampling: cluster by string length then sample from each cluster
   if sample_size < df_a.shape[0]:
       join_col = df_a[column_a_name].astype(str) if column_a_name in df_a.columns else df_a.iloc[:, 0].astype(str)
       lengths = join_col.str.len()
       try:
           q33 = lengths.quantile(0.33)
           q66 = lengths.quantile(0.66)
           buckets = pd.cut(lengths, bins=[-1, q33, q66, float('inf')], labels=[0, 1, 2])
           samples_per_bucket = max(1, sample_size // 3)
           sampled_indices = []
           for bucket_id in [0, 1, 2]:
               bucket_rows = df_a.index[buckets == bucket_id].tolist()
               if bucket_rows:
                   n_take = min(samples_per_bucket, len(bucket_rows))
                   sampled_indices.extend(random.sample(bucket_rows, n_take))
           remaining = sample_size - len(sampled_indices)
           if remaining > 0:
               leftover = [i for i in df_a.index if i not in sampled_indices]
               sampled_indices.extend(random.sample(leftover, min(remaining, len(leftover))))
           df_a_sampled = df_a.loc[sampled_indices].reset_index(drop=True)
       except Exception:
           df_a_sampled = df_a.sample(sample_size).reset_index(drop=True)
   else:
       df_a_sampled = df_a.copy()
   df_b_sampled = df_b.copy()

   if agenta:
       agent_table_a = agenta

   else:
       agent_table_a = QLearningAgent_edit_dist_modified_for_multi_opt(all_operators,df_a_sampled,agreement_percentage,exploration_rate)

   if agentb:
       agent_table_b = agentb
   else:
       agent_table_b = QLearningAgent_edit_dist_modified_for_multi_opt(all_operators,df_b_sampled,agreement_percentage,exploration_rate)


   trans_a_to_b, table_a_to_b_cols, trans_b_to_a,table_b_to_a_cols, edit_dist_matrix,agent_a,agent_b = optimize_transformations_both_edit_dist_opt_with_reusing(max_steps, df_a_sampled, df_b_sampled,
                                                                                                                                   column_a_name,column_b_name,
                                                                                                                                    agent_table_a,agent_table_b,
                                                                                                                                    pairs, edit_dist_threshold,greedy,
                                                                                                                                    transcols_a_b,transcols_b_a)                                                                                                                                  
   df_a_concated = build_dataframe_in_order(df_a, table_a_to_b_cols)
   df_b_concated = build_dataframe_in_order(df_b, table_b_to_a_cols)
   
   df_a_concated_applied = apply_all_actions_to_df(df_a_concated,trans_a_to_b).astype(str).apply(lambda col: col.str.lower())
   df_b_concated_applied =  apply_all_actions_to_df(df_b_concated,trans_b_to_a).astype(str).apply(lambda col: col.str.lower())

   return df_a_concated_applied,df_b_concated_applied,trans_a_to_b, table_a_to_b_cols, trans_b_to_a,table_b_to_a_cols,agent_a,agent_b


def perform_join_lcs(
    df_a, 
    df_b, 
    lcs_matrix, 
    threshold=0.6, 
    include_lcs_percentage=False
):
    """
    Greedy one-to-one assignment join: pair rows of df_a with rows of df_b in
    order of decreasing pairwise similarity, keeping each pair whose similarity
    is >= ``threshold`` and never reusing a row on either side.

    NOTE: despite the historical name, this does NOT recompute an LCS/ALCS
    metric. It operates on whatever similarity matrix it is handed. In MNN Join
    that is the SAME idfcos similarity (IDF-weighted binary char-n-gram cosine)
    used to score the transform chain during search, so chain selection and the
    committed join share one metric; ``threshold`` is the label-free cut chosen
    upstream by the covmnn rule.

    Parameters:
    - df_a (pd.DataFrame): The left (source) DataFrame.
    - df_b (pd.DataFrame): The right (target) DataFrame.
    - lcs_matrix (np.ndarray): A 2D similarity matrix where lcs_matrix[i][j] is
      the (idfcos) similarity between df_a.iloc[i] and df_b.iloc[j]. Name kept
      for backward compatibility with call sites.
    - threshold (float, optional): Minimum similarity for a match. Defaults to 0.6.
    - include_lcs_percentage (bool, optional): Append the similarity as a column.

    Returns:
    - pd.DataFrame: The joined rows, one-to-one, greedy by descending similarity.
    """
    
    # Validate input dimensions
    if lcs_matrix.shape != (len(df_a), len(df_b)):
        raise ValueError("lcs_matrix dimensions must match the lengths of df_a and df_b.")
    
    matches = []
    
    # Create a list of all possible (i, j, lcs_score) tuples
    all_matches = [
        (i, j, lcs_matrix[i][j])
        for i in range(lcs_matrix.shape[0])
        for j in range(lcs_matrix.shape[1])
        if lcs_matrix[i][j] >= threshold
    ]
    
    # Sort the matches in descending order of LCS similarity
    all_matches_sorted = sorted(all_matches, key=lambda x: x[2], reverse=True)
    
    # Initialize arrays to track available indices in df_a and df_b
    available_a = np.ones(len(df_a), dtype=bool)
    available_b = np.ones(len(df_b), dtype=bool)
    
    # Convert DataFrames to NumPy arrays for faster access
    a_array = df_a.to_numpy()
    b_array = df_b.to_numpy()
    
    # Precompute column names with suffixes to avoid duplicates
    df_a_columns = df_a.columns.tolist()
    df_b_columns = df_b.columns.tolist()
    
    # Identify overlapping columns
    overlapping_cols = set(df_a_columns).intersection(set(df_b_columns))
    
    # Create new column names with suffixes for df_b to avoid duplication
    if overlapping_cols:
        df_b_columns_renamed = [
            f"{col}_b" if col in overlapping_cols else col for col in df_b.columns
        ]
    else:
        df_b_columns_renamed = df_b_columns.copy()
    
    # Prepare column names for the final DataFrame
    final_columns = df_a_columns + df_b_columns_renamed
    if include_lcs_percentage:
        final_columns += ['lcs_sim']
    
    # Iterate through the sorted matches and assign them
    for i, j, lcs in all_matches_sorted:
        if available_a[i] and available_b[j]:
            # Concatenate the rows from df_a and df_b
            joined_row = np.concatenate((a_array[i], b_array[j]))

            # Append LCS similarity if required
            if include_lcs_percentage:
                joined_row = np.append(joined_row, lcs)

            matches.append(joined_row)

            # Mark the matched indices as unavailable
            available_a[i] = False
            available_b[j] = False

    # T1: _DOMINANCE_ADMIT=1 — second pass admits per-row top-1 pairs
    # whose dominance (top1 / top2) >= τ (default 2.0) and top1 >= floor
    # (default 0.05), even if below `threshold`. Captures pairs that the
    # transformations couldn't lift above the JDN threshold but whose
    # per-row ranking still uniquely identifies them. No truth used.
    if _DOMINANCE_ADMIT:
        try:
            _dom_tau = _DOMINANCE_TAU
            _dom_floor = _DOMINANCE_FLOOR
            # T3: coverage-gated dominance admission. When the gate is enabled,
            # skip the admission pass entirely if base (pre-admission) join
            # coverage is already high — admitting marginal pairs hurts
            # precision on saturated datasets. Input-only: no truth used.
            _cov_skip = False
            if _DOMINANCE_COVERAGE_GATE:
                _cov_max = _DOMINANCE_COVERAGE_MAX
                _total_src = lcs_matrix.shape[0]
                _matched_src = int(sum(1 for _x in available_a if not _x))
                _coverage = (_matched_src / _total_src) if _total_src > 0 else 0.0
                if _coverage >= _cov_max:
                    _cov_skip = True  # base coverage saturated — skip admission
            # Walk unmatched src rows
            for i in (range(lcs_matrix.shape[0]) if not _cov_skip else range(0)):
                if not available_a[i]:
                    continue
                row = lcs_matrix[i]
                if row.size == 0 or float(np.max(row)) < _dom_floor:
                    continue
                # Top-1 and top-2 sims, ignoring already-used targets
                order = np.argsort(-row)
                t1_j = None
                t1_v = 0.0
                for j_cand in order:
                    j_cand = int(j_cand)
                    if available_b[j_cand]:
                        t1_j = j_cand
                        t1_v = float(row[j_cand])
                        break
                if t1_j is None or t1_v < _dom_floor:
                    continue
                # Top-2 sim among AVAILABLE targets (different from t1_j)
                t2_v = 0.0
                for j_cand in order:
                    j_cand = int(j_cand)
                    if j_cand == t1_j:
                        continue
                    if available_b[j_cand]:
                        t2_v = float(row[j_cand])
                        break
                if t1_v / max(t2_v, 1e-6) < _dom_tau:
                    continue
                # Admit dominance pair
                joined_row = np.concatenate((a_array[i], b_array[t1_j]))
                if include_lcs_percentage:
                    joined_row = np.append(joined_row, t1_v)
                matches.append(joined_row)
                available_a[i] = False
                available_b[t1_j] = False
        except Exception:
            pass

    # Convert the list of matches to a NumPy array for efficient DataFrame construction
    if matches:
        matches_array = np.vstack(matches)
    else:
        # If no matches found, create an empty array with appropriate number of columns
        matches_array = np.empty((0, len(final_columns)), dtype=object)
    
    # Create the final DataFrame
    df_matches = pd.DataFrame(matches_array, columns=final_columns)
    
    # Optionally, convert data types back to original types if necessary
    # This step can be customized based on the specific data types in df_a and df_b
    
    return df_matches.reset_index(drop=True)   


### only for auto join benchmarks (source can multiple joins)

def perform_join_lcs_multi(
    df_a, 
    df_b, 
    lcs_matrix, 
    threshold=0.6, 
    include_lcs_percentage=False
):
    """
    Performs a one-to-many join between df_a and df_b based on the Longest Common Subsequence (LCS) distances.
    If no matches are found, the rows are not included in the final output.
    
    Parameters:
    - df_a (pd.DataFrame): The left DataFrame.
    - df_b (pd.DataFrame): The right DataFrame.
    - lcs_matrix (np.ndarray): A 2D NumPy array where lcs_matrix[i][j] represents the LCS distance between df_a.iloc[i] and df_b.iloc[j].
    - threshold (int, optional): The minimum allowable LCS similarity for a match. Defaults to 0.
    - include_lcs_percentage (bool, optional): Whether to include the LCS similarity in the resulting DataFrame. Defaults to False.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the joined rows from df_a and df_b based on the LCS criteria.
    """
    
    # Validate input dimensions
    if lcs_matrix.shape != (len(df_a), len(df_b)):
        raise ValueError("lcs_matrix dimensions must match the lengths of df_a and df_b.")
    
    matches = []
    
    # Create a list of all possible (i, j, lcs_score) tuples
    all_matches = [
        (i, j, lcs_matrix[i][j])
        for i in range(lcs_matrix.shape[0])
        for j in range(lcs_matrix.shape[1])
        if lcs_matrix[i][j] >= threshold
    ]
    
    # Sort the matches in descending order of LCS similarity
    all_matches_sorted = sorted(all_matches, key=lambda x: x[2], reverse=True)
    
    # Initialize arrays to track available indices in df_b
    available_b = np.ones(len(df_b), dtype=bool)

    available_a = np.ones(len(df_a), dtype=bool)

    
    
    # Convert DataFrames to NumPy arrays for faster access
    a_array = df_a.to_numpy()
    b_array = df_b.to_numpy()
    
    # Precompute column names with suffixes to avoid duplicates
    df_a_columns = df_a.columns.tolist()
    df_b_columns = df_b.columns.tolist()
    
    # Identify overlapping columns
    overlapping_cols = set(df_a_columns).intersection(set(df_b_columns))
    
    # Create new column names with suffixes for df_b to avoid duplication
    if overlapping_cols:
        df_b_columns_renamed = [
            f"{col}_b" if col in overlapping_cols else col for col in df_b.columns
        ]
    else:
        df_b_columns_renamed = df_b_columns.copy()
    
    # Prepare column names for the final DataFrame
    final_columns = df_a_columns + df_b_columns_renamed
    if include_lcs_percentage:
        final_columns += ['lcs_sim']
    
    # Iterate through the sorted matches and assign them
    if any('source-' in col for col in df_b.columns):
      for i, j, lcs in all_matches_sorted:
          if available_b[j]:
              # Concatenate the rows from df_a and df_b
              joined_row = np.concatenate((a_array[i], b_array[j]))

              # Append LCS similarity if required
              if include_lcs_percentage:
                  joined_row = np.append(joined_row, lcs)

              matches.append(joined_row)

              # Keep row from df_a available for multiple joins, but mark the matched row from df_b as unavailable
              available_b[j] = False
    else:
      for i, j, lcs in all_matches_sorted:
          if available_a[i]:
              # Concatenate the rows from df_a and df_b
              joined_row = np.concatenate((a_array[i], b_array[j]))

              # Append LCS similarity if required
              if include_lcs_percentage:
                  joined_row = np.append(joined_row, lcs)

              matches.append(joined_row)

              # Keep row from df_b available for multiple joins, but mark the matched row from df_a as unavailable
              available_a[i] = False

    # T1 (multi version): same dominance-admission pass for unmatched src rows.
    # In the multi-join branch (any 'source-' in df_b.columns), src can match
    # multiple targets — we still admit dominance pairs but only for src rows
    # that haven't been matched yet, to avoid over-admission.
    if _DOMINANCE_ADMIT:
        try:
            _dom_tau = _DOMINANCE_TAU
            _dom_floor = _DOMINANCE_FLOOR
            # Track which src rows already got at least one match
            matched_src_rows = set()
            n_a_cols = len(df_a_columns)
            for joined in matches:
                # find which src row this corresponds to — by matching against a_array
                # Simpler: skip this — admit dominance for ANY src that has 0 matches so far
                pass
            # Recount: walk through matches and figure out matched src indices
            # (since we don't have indices stored, approximate by checking a_array equality)
            existing_match_src = set()
            for m in matches:
                src_part = m[:n_a_cols]
                for i in range(len(df_a)):
                    if np.array_equal(a_array[i], src_part):
                        existing_match_src.add(i); break
            # T3: coverage-gated dominance admission (input-only, no truth).
            # Skip the admission pass when base coverage is already high.
            _cov_skip = False
            if _DOMINANCE_COVERAGE_GATE:
                _cov_max = _DOMINANCE_COVERAGE_MAX
                _total_src = lcs_matrix.shape[0]
                _coverage = (len(existing_match_src) / _total_src) if _total_src > 0 else 0.0
                if _coverage >= _cov_max:
                    _cov_skip = True
            for i in (range(lcs_matrix.shape[0]) if not _cov_skip else range(0)):
                if i in existing_match_src:
                    continue
                row = lcs_matrix[i]
                if row.size == 0 or float(np.max(row)) < _dom_floor:
                    continue
                order = np.argsort(-row)
                t1_j = None; t1_v = 0.0
                for j_cand in order:
                    j_cand = int(j_cand)
                    if available_b[j_cand]:
                        t1_j = j_cand; t1_v = float(row[j_cand]); break
                if t1_j is None or t1_v < _dom_floor:
                    continue
                t2_v = 0.0
                for j_cand in order:
                    j_cand = int(j_cand)
                    if j_cand == t1_j: continue
                    if available_b[j_cand]:
                        t2_v = float(row[j_cand]); break
                if t1_v / max(t2_v, 1e-6) < _dom_tau:
                    continue
                joined_row = np.concatenate((a_array[i], b_array[t1_j]))
                if include_lcs_percentage:
                    joined_row = np.append(joined_row, t1_v)
                matches.append(joined_row)
                available_b[t1_j] = False
        except Exception:
            pass

    # If no matches were found, return an empty DataFrame
    if not matches:
        return pd.DataFrame(columns=final_columns)
    
    # Convert the list of matches to a NumPy array for efficient DataFrame construction
    matches_array = np.vstack(matches)
    
    # Create the final DataFrame
    df_matches = pd.DataFrame(matches_array, columns=final_columns)
    
    return df_matches.reset_index(drop=True)


# large threshold push no joins
def custom_round(value):
    if math.isnan(value):  
        return 1000
    if value == 1.0:
        return 1.0
    return math.floor(value * 10) / 10


def all_numbers(col):
    """
    Checks if all values in 'col' (whether it's a list or a Pandas Series)
    are numbers (while ignoring nulls).
    """
    if not isinstance(col, pd.Series):
        col = pd.Series(col)

    non_null = col.dropna()

    return non_null.apply(is_number).all()

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def get_the_join_using_multi_to_multi_greedy_opt(sample_proportion,max_steps, df_a, df_b,column_a_name,column_b_name, edit_dist_threshold,all_operators,pairs,
                                                 exploration_rate,agreement_percentage,greedy,find_transformed_df_opt_func = find_transformed_df_opt):

    df_a_concated_applied,df_b_concated_applied,trans_a_to_b, table_a_to_b_cols, trans_b_to_a,table_b_to_a_cols = find_transformed_df_opt_func(sample_proportion,max_steps, df_a,
                                                                        df_b,column_a_name,column_b_name, all_operators,pairs, edit_dist_threshold,
                                                                        exploration_rate,agreement_percentage,greedy)

    concated_df_a = concatenate_with_order(df_a_concated_applied,'a_to_b')['a_to_b']
    concated_df_b = concatenate_with_order(df_b_concated_applied,'b_to_a')['b_to_a']

    total_length_df_a = sum(len(s) for s in concated_df_a)
    average_length_df_a = total_length_df_a / len(concated_df_a)

    total_length_df_b = sum(len(s) for s in concated_df_b)
    average_length_df_b = total_length_df_b / len(concated_df_b)

    min_avg_length = np.min([average_length_df_a,average_length_df_b])

    # Learned blocking for large tables
    m_rows, n_rows = len(concated_df_a), len(concated_df_b)
    if False and m_rows * n_rows > 50000:
        try:
            from blocking import block_combined
            candidates = block_combined(concated_df_a.tolist(), concated_df_b.tolist())
            _block_matrix = np.zeros((m_rows, n_rows), dtype=int)
            for i, js in candidates:
                for j in js[:20]:
                    _block_matrix[i, j] = 1
            alcs_sim, sim_matrix, freq_counts_penalty, lsc_matrix = get_ALCS_matrix(concated_df_a, concated_df_b, greedy)
            sim_matrix = sim_matrix * _block_matrix
        except:
            alcs_sim, sim_matrix, freq_counts_penalty, lsc_matrix = get_ALCS_matrix(concated_df_a, concated_df_b, greedy)
    else:
        alcs_sim, sim_matrix, freq_counts_penalty, lsc_matrix = get_ALCS_matrix(concated_df_a, concated_df_b, greedy)

    print(f'The alcs percentage is {alcs_sim}')

    # Blend embedding similarity into ALCS matrix — weight learned by JDN
    # (set after JDN prediction below, or env var override)
    _embed_weight = 0.0  # default: no blending unless JDN says to

    df_matches = None
    df_matches_multi = None

    # --- Join threshold: KMeans with learned or default parameters ---
    row_max = np.max(sim_matrix, axis=1, keepdims=True)
    n_samples = row_max.shape[0]

    # Default KMeans params (old v3 values)
    _km_k = min(7, n_samples)
    _km_percentile = 75.0
    _km_alpha = -0.05 if min_avg_length < 5 else (-0.025 if min_avg_length < 10 else 0.025)
    _km_cap = 0.8  # old v3 cap
    sim_reward_factor = 2.0 if min_avg_length < 5 else (1.0 if min_avg_length < 10 else 0.8)

    # Try learned KMeans params from JoinDecisionNet
    _jdn = None
    if ModelRegistry is not None and not _learned_off('NO_JDN'):
        try:
            _jdn = ModelRegistry().load('join_decision')
        except:
            pass

    if _jdn is not None:
        try:
            row_max_flat = row_max.flatten()
            _jdn_stats = np.array([
                float(np.mean(row_max_flat)),
                float(np.std(row_max_flat)),
                float(np.percentile(row_max_flat, 25)),
                float(np.percentile(row_max_flat, 50)),
                float(np.percentile(row_max_flat, 75)),
                float(np.min(row_max_flat)),
                float(np.max(row_max_flat)),
                float(alcs_sim),
                float(min(min_avg_length / 20.0, 1.0)),
            ], dtype=np.float32)
            _jdn_pred = _jdn.predict(_jdn_stats)
            _km_k = min(_jdn_pred['kmeans_k'], n_samples)
            _km_percentile = _jdn_pred['percentile']
            _km_alpha = _jdn_pred['alpha']
            _km_cap = _jdn_pred['cap']
            _embed_weight = _jdn_pred.get('embed_weight', 0.0)
        except Exception:
            pass  # fall through to defaults

    # _THRESHOLD_SIGNAL=otsu|gap: derive the cutoff from the row_max
    # distribution itself (parameter-free) instead of KMeans(k)+percentile —
    # replaces both JDN and the k=7/pct=75 heuristic with a signal.
    _thr_sig = _THRESHOLD_SIGNAL
    if _thr_sig == 'kdata':
        # FULLY DATA-DRIVEN JDN: every param (k, band, alpha, cap) from the
        # distribution (see _jdn_like_threshold); ALCS-floor + clamp applied below
        # keep JDN's remaining knobs intact.
        join_threshold, _km_alpha, _km_cap = _jdn_like_threshold(row_max, min_avg_length)
    elif _thr_sig == 'covmnn':
        # SEARCH OVER CUTS by the cov*mnn objective (see _threshold_by_covmnn).
        join_threshold = _threshold_by_covmnn(sim_matrix)
    elif _thr_sig:
        join_threshold = _signal_threshold(row_max, _thr_sig, alcs_sim)
    # Run KMeans with learned (or default) params
    elif _km_k > 1:
        kmeans = KMeans(n_clusters=_km_k)
        kmeans.fit(row_max)
        centers = kmeans.cluster_centers_.flatten()
        _pctl_val = np.percentile(centers, _km_percentile)
        _selected_idx = np.argmin(np.abs(centers - _pctl_val))
        _selected_rows = np.where(kmeans.labels_ == _selected_idx)[0]
        join_threshold = np.mean(row_max[_selected_rows])
    else:
        join_threshold = float(np.mean(row_max))

    # S1 ablation: NO_ALCS_FLOOR=1 skips the alcs_sim floor so the join
    # threshold is driven purely by JDN/KMeans on row_max distribution. The
    # default floor couples global column similarity to per-row match acceptance
    # — a confounded mechanism that drives both over-matching (Pattern A) and
    # under-matching (Pattern B) failures observed in the ALCS-vs-F1 probe.
    # covmnn already chose the cut to optimize cov*mnn; the alcs floor is what
    # caused the univ under-admit (it lifted the cut above the bijective core),
    # so covmnn bypasses it.
    if True and _thr_sig != 'covmnn':
        join_threshold = np.max([alcs_sim, join_threshold])
    # covmnn: use the optimized cut directly (no cap/alpha — the cap=0.8 also
    # clipped univ). condf: signal cut, no alpha. else: alpha-adjusted KMeans.
    if _thr_sig == 'covmnn':
        _threshold = float(join_threshold)
    elif _thr_sig == 'condf':
        _threshold = min(custom_round(join_threshold), _km_cap)
    else:
        _threshold = min(custom_round(join_threshold) - _km_alpha, _km_cap)
    _threshold = max(0.2, _threshold)
    globals()['_LAST_THRESHOLD'] = float(_threshold)  # capture the join cutoff used
    # LOG_GMM_VALLEY=1: also compute the GMM-valley cut on the SAME row_max
    # (paired comparison: does JDN cut higher than the natural valley?).
    if False:
        try:
            globals()['_LAST_GMM_VALLEY'] = float(_signal_threshold(row_max, 'gmm'))
        except Exception:
            globals()['_LAST_GMM_VALLEY'] = float('nan')

    # Blend embedding similarity into sim_matrix with learned weight.
    # Priority: v7_v2 forced (per-pair) > EMB_WEIGHT env > JDN > 0.
    # v7_v2's emb_weight head is per (col_a, col_b); the env override is
    # global; JDN is per-call but predicts ~0 on most benchmarks.
    if _V72_FORCED_EMB_WEIGHT is not None:
        _embed_weight = _V72_FORCED_EMB_WEIGHT
    else:
        # EMB_WEIGHT env var overrides JDN's prediction (e.g., for testing
        # whether embedding blend helps on benchmarks where JDN predicts ~0).
        # Skipped when v7_v2 has set a per-pair forced weight above.
        _emb_w_env = None
        if _emb_w_env is not None:
            try:
                _embed_weight = float(_emb_w_env)
            except Exception:
                pass
    # Clamp raised to 1.0 when env var is explicitly set (V0 ablation E2 needs pure cosine);
    # otherwise keep the conservative 0.5 cap so JDN predictions stay bounded.
    _emb_w_env_set = False
    _embed_weight = max(0.0, min(1.0 if _emb_w_env_set else 0.5, _embed_weight))
    if _embed_weight > 0.01:
        try:
            from _embed import compute_embedding_similarity
            _, _emb_sim = compute_embedding_similarity(
                concated_df_a.astype(str).tolist(),
                concated_df_b.astype(str).tolist(),
            )
            sim_matrix = (1.0 - _embed_weight) * sim_matrix + _embed_weight * _emb_sim
            alcs_sim = float(np.mean(np.max(sim_matrix, axis=1)))
        except Exception:
            pass

    # Oracle sweep override
    if _FORCED_THRESHOLD is not None:
        _threshold = _FORCED_THRESHOLD

    # Post-transform join policy: learned regime override with confidence gate
    _ptjp_conf_threshold = None
    if _ptjp_conf_threshold is not None and _FORCED_THRESHOLD is None:
        try:
            import torch
            import torch.nn.functional as _F
            _ptjp_path = os.path.join(os.path.dirname(__file__), '..', 'out_put_csv', 'post_transform_policy.pt')
            if os.path.exists(_ptjp_path):
                _ptjp_data = torch.load(_ptjp_path, weights_only=False)
                from post_transform_policy import PostTransformJoinPolicy, compute_post_transform_features, REGIME_NAMES
                _ptjp_model = PostTransformJoinPolicy(_ptjp_data['n_features'])
                _ptjp_model.load_state_dict(_ptjp_data['model_state']); _ptjp_model.eval()
                _ptjp_fm = torch.tensor(_ptjp_data['feat_mean'])
                _ptjp_fs = torch.tensor(_ptjp_data['feat_std'])

                # Compute features from the ALREADY-COMPUTED sim_matrix
                _n_cols_a = len(table_a_to_b_cols) if table_a_to_b_cols is not None else 1
                _n_cols_b = len(table_b_to_a_cols) if table_b_to_a_cols is not None else 1
                _avg_a = float(np.mean([len(str(x)) for x in concatenate_with_order(
                    apply_all_actions_to_df(build_dataframe_in_order(df_a, table_a_to_b_cols), trans_a_to_b), 'a')['a']]))
                _avg_b = float(np.mean([len(str(x)) for x in concatenate_with_order(
                    apply_all_actions_to_df(build_dataframe_in_order(df_b, table_b_to_a_cols), trans_b_to_a), 'b')['b']]))

                _ptjp_feats = compute_post_transform_features(
                    sim_matrix, _n_cols_a, _n_cols_b, _avg_a, _avg_b, alcs_sim)
                _ptjp_x = (torch.tensor([_ptjp_feats], dtype=torch.float32) - _ptjp_fm) / _ptjp_fs

                with torch.no_grad():
                    _ptjp_out = _ptjp_model(_ptjp_x)
                _ptjp_regime = REGIME_NAMES[_F.softmax(_ptjp_out['regime_logits'], dim=-1)[0].argmax().item()]
                _ptjp_conf = _ptjp_out['confidence'][0].item()
                _ptjp_join_t = {'strict': 0.85, 'balanced': 0.70, 'fuzzy': 0.55}

                _tau = float(_ptjp_conf_threshold) if _ptjp_conf_threshold != 'always' else 0.0
                if _ptjp_conf >= _tau or _ptjp_conf_threshold == 'always':
                    _threshold = _ptjp_join_t[_ptjp_regime]
        except Exception:
            pass  # fallback to baseline threshold

    # OverrideGainNet: predict gain from overriding baseline threshold
    if False and _FORCED_THRESHOLD is None:
        try:
            import torch
            _ogn_path = os.path.join(os.path.dirname(__file__), '..', 'out_put_csv', 'override_gain_net.pt')
            if os.path.exists(_ogn_path):
                _ogn_data = torch.load(_ogn_path, weights_only=False)
                from override_gain_net import OverrideGainNet, compute_features_25
                _ogn_model = OverrideGainNet(_ogn_data['n_features'])
                _ogn_model.load_state_dict(_ogn_data['model_state']); _ogn_model.eval()
                _ogn_fm = torch.tensor(_ogn_data['feat_mean'])
                _ogn_fs = torch.tensor(_ogn_data['feat_std'])

                _n_cols_a = len(table_a_to_b_cols) if table_a_to_b_cols is not None else 1
                _n_cols_b = len(table_b_to_a_cols) if table_b_to_a_cols is not None else 1
                _avg_a = float(np.mean([len(str(x)) for x in concatenate_with_order(
                    apply_all_actions_to_df(build_dataframe_in_order(df_a, table_a_to_b_cols), trans_a_to_b), 'a')['a']]))
                _avg_b = float(np.mean([len(str(x)) for x in concatenate_with_order(
                    apply_all_actions_to_df(build_dataframe_in_order(df_b, table_b_to_a_cols), trans_b_to_a), 'b')['b']]))

                _ogn_feats = compute_features_25(
                    sim_matrix, _n_cols_a, _n_cols_b, _avg_a, _avg_b, alcs_sim, _threshold)
                _ogn_x = (torch.tensor([_ogn_feats], dtype=torch.float32) - _ogn_fm) / _ogn_fs

                with torch.no_grad():
                    _ogn_gains = _ogn_model(_ogn_x)[0]  # [gain_strict, gain_fuzzy]
                _gain_strict = _ogn_gains[0].item()
                _gain_fuzzy = _ogn_gains[1].item()

                # Override only if predicted gain is positive
                if _gain_strict > 0.01 and _gain_strict > _gain_fuzzy:
                    _threshold = 0.85
                elif _gain_fuzzy > 0.01:
                    _threshold = 0.55
                # else: keep baseline threshold
        except Exception:
            pass

    # PROBE: capture (sim_matrix, threshold, joined-col values, alcs) so a
    # downstream probe runner can label each cell against truth_pairs.
    if False:
        try:
            _PROBE_PAIR_SIMS['calls'].append({
                'sim_matrix': np.array(sim_matrix, dtype=np.float32),
                'threshold': float(_threshold),
                'alcs_sim': float(alcs_sim),
                'concated_df_a': list(concated_df_a),
                'concated_df_b': list(concated_df_b),
            })
        except Exception:
            pass

    df_matches = perform_join_lcs(df_a, df_b, sim_matrix, _threshold, False)
    df_matches_multi = perform_join_lcs_multi(df_a, df_b, sim_matrix, _threshold, False)


    return alcs_sim,df_matches,df_matches_multi, trans_a_to_b, table_a_to_b_cols, trans_b_to_a,table_b_to_a_cols,freq_counts_penalty,sim_matrix,lsc_matrix,sim_reward_factor


def get_the_join_using_multi_to_multi_greedy_opt_with_reusing(sample_proportion,max_steps, df_a, df_b,column_a_name,column_b_name,
                                                               edit_dist_threshold,all_operators,pairs,exploration_rate,
                                                               agreement_percentage,greedy,
                                                                transcols_a_b,
                                                                transcols_b_a,
                                                                agenta,
                                                                agentb,
                                                                find_transformed_df_opt_func = find_transformed_df_opt_with_reusing):

    df_a_concated_applied,df_b_concated_applied,trans_a_to_b, table_a_to_b_cols,\
          trans_b_to_a,table_b_to_a_cols,agent_a,agent_b = find_transformed_df_opt_func(sample_proportion,max_steps, df_a,
                                                                        df_b,column_a_name,column_b_name, all_operators,pairs, edit_dist_threshold,
                                                                        exploration_rate,agreement_percentage,greedy,
                                                                        transcols_a_b,
                                                                        transcols_b_a,
                                                                        agenta,
                                                                        agentb )

    concated_df_a = concatenate_with_order(df_a_concated_applied,'a_to_b')['a_to_b']
    concated_df_b = concatenate_with_order(df_b_concated_applied,'b_to_a')['b_to_a']

    total_length_df_a = sum(len(s) for s in concated_df_a)
    average_length_df_a = total_length_df_a / len(concated_df_a)

    total_length_df_b = sum(len(s) for s in concated_df_b)
    average_length_df_b = total_length_df_b / len(concated_df_b)

    min_avg_length = np.min([average_length_df_a,average_length_df_b])

    alcs_sim, sim_matrix,freq_counts_penalty, lsc_matrix  = get_ALCS_matrix(concated_df_a, concated_df_b,greedy)

    print(f'The alcs percentage is {alcs_sim}')

    df_matches = None
    df_matches_multi = None

    # --- Join threshold: KMeans + alpha + cap, all params from the
    # selected RewardConfig (trained per recipe via grid search).
    # When jdn_alpha == -1.0 (sentinel), use legacy length-tiered alpha.
    row_max = np.max(sim_matrix, axis=1, keepdims=True)
    _selector = RewardConfigSelector()
    _rc, _rc_name = _selector.select(df_a, column_a_name, df_b, column_b_name)

    k = min(_rc.jdn_kmeans_k, row_max.shape[0])
    _pct = _rc.jdn_percentile
    if _rc.jdn_alpha == -1.0:
        # Legacy length-tiered alpha (for recipes without trained jdn_alpha)
        if min_avg_length < 5:
            alpha = -0.05
        elif min_avg_length < 10:
            alpha = -0.025
        else:
            alpha = 0.025
    else:
        alpha = _rc.jdn_alpha

    if k > 1:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(row_max)
        centers = kmeans.cluster_centers_.flatten()
        pctl_val = np.percentile(centers, _pct)
        median_high_idx = np.argmin(np.abs(centers - pctl_val))
        mh_indices = np.where(kmeans.labels_ == median_high_idx)[0]
        join_threshold = np.mean(row_max[mh_indices])
    else:
        join_threshold = float(np.mean(row_max))

    # sim_reward_factor by 3-tier length (reward shaping unchanged)
    if min_avg_length < 5:
        sim_reward_factor = 2.0
    elif min_avg_length < 10:
        sim_reward_factor = 1.0
    else:
        sim_reward_factor = 0.8

    join_threshold = np.max([alcs_sim, join_threshold])
    _threshold = min(custom_round(join_threshold) - alpha, _rc.join_threshold_cap)
    _threshold = max(0.2, _threshold)

    if _FORCED_THRESHOLD is not None:
        _threshold = _FORCED_THRESHOLD

    # PROBE: capture (sim_matrix, threshold, joined-col values, alcs) so a
    # downstream probe runner can label each cell against truth_pairs.
    if False:
        try:
            _PROBE_PAIR_SIMS['calls'].append({
                'sim_matrix': np.array(sim_matrix, dtype=np.float32),
                'threshold': float(_threshold),
                'alcs_sim': float(alcs_sim),
                'concated_df_a': list(concated_df_a),
                'concated_df_b': list(concated_df_b),
            })
        except Exception:
            pass

    df_matches = perform_join_lcs(df_a, df_b, sim_matrix, _threshold, False)
    df_matches_multi = perform_join_lcs_multi(df_a, df_b, sim_matrix, _threshold, False)


    return alcs_sim,df_matches,df_matches_multi, trans_a_to_b, table_a_to_b_cols, trans_b_to_a,table_b_to_a_cols,freq_counts_penalty,sim_matrix,lsc_matrix,sim_reward_factor,agent_a,agent_b



def get_df_from_table_lst(tables,attr):
    return tables[attr[0]]


class ALCS_sim:

    def __init__(self, data_loader,sample_proportion=1.0,filter_threshold = 0.1):
        """
        Initializes the CombinedQGramSimilarity class.
        
        Parameters:
        - data_loader: instance of DataLoader
            Provides `columns` (dict) and `column_pairs` (list of tuples).
        - q: int (default=2)
            Q-gram size. (No longer used if we are moving to ALCS-based approach only.)
        - sample_proportion: float (default=0.5)
            Proportion of data to sample from each column (0 < p <= 1).
        """
        self.columns = data_loader.columns
        self.column_pairs = data_loader.column_pairs 
        self.sample_proportion = sample_proportion
        
        self.filter_threshold = filter_threshold

        self.sampled_data = {}

        self.create_sampled_data()

        self.tables = data_loader.tables_with_names

    def create_sampled_data(self):
        """Create sampled data for each column."""
        data_length = min(len(data) for data in self.columns.values())
        if data_length == 0:
            raise ValueError("At least one column is empty.")
        sample_size = max(1, int(data_length * self.sample_proportion))
        indices = random.sample(range(data_length), sample_size)
        for col_key, data in self.columns.items():
            self.sampled_data[col_key] = [data[i] for i in indices]

    def compute_all_pairs_similarity(self):
        """
        Compute ALCS-based similarities between column pairs using the
        `get_edit_dist_matrix` function. Returns a dictionary of
        {(col1, col2): mean_max_ALCS}.
        """
        similarities = {}
        penalty = {}
        sim_matrixes = {}
        lcs_matrixes = {}
        reward_dicts = {}

        # Phase 1: Multi-signal cheap scan to rank pairs (Jaccard bigram + token-set + sorted-token + profile)
        # All signals are O(n) per pair — much faster than ALCS O(n^2) for initial ranking
        signal_scores = {}  # {signal_name: {(col1, col2): score}}
        # Pair ranking signals
        signal_names = ['jaccard_bigram', 'token_set_jaccard', 'sorted_token_overlap', 'profile_compat']
        # Embedding signal available but disabled by default — needs PairPosteriorNet (V2 Phase 1)
        # to properly weight it. Raw embedding hurts more datasets than it helps.
        if False:
            signal_names.append('embedding_sim')
        for sn in signal_names:
            signal_scores[sn] = {}

        def _token_set_jaccard(list_a, list_b):
            """Token-set Jaccard: tokenize by spaces, compare as sets (order-invariant)."""
            scores = []
            for a in list_a:
                best = 0.0
                toks_a = set(str(a).lower().split())
                if not toks_a:
                    scores.append(0.0)
                    continue
                for b in list_b:
                    toks_b = set(str(b).lower().split())
                    if not toks_b:
                        continue
                    inter = len(toks_a & toks_b)
                    union = len(toks_a | toks_b)
                    if union > 0:
                        best = max(best, inter / union)
                scores.append(best)
            return sum(scores) / len(scores) if scores else 0.0

        def _sorted_token_overlap(list_a, list_b):
            """Sort tokens alphabetically, rejoin, then Jaccard on bigrams of sorted string."""
            def _sort_str(s):
                return ' '.join(sorted(str(s).lower().split()))
            sorted_a = [_sort_str(v) for v in list_a]
            sorted_b = [_sort_str(v) for v in list_b]
            mean_sc, _ = jaccard_matrix_fast(sorted_a, sorted_b, n=2)
            return float(mean_sc)

        def _profile_compat(vals_a, vals_b):
            """Column profile compatibility: compare avg_token_count, avg_string_length, digit_fraction, unique_ratio."""
            def _profile(vals):
                strs = [str(v) for v in vals]
                n = len(strs) if strs else 1
                avg_tok = sum(len(s.split()) for s in strs) / n
                avg_len = sum(len(s) for s in strs) / n
                total_chars = sum(len(s) for s in strs)
                digit_chars = sum(c.isdigit() for s in strs for c in s)
                digit_frac = digit_chars / total_chars if total_chars > 0 else 0.0
                unique_ratio = len(set(strs)) / n if n > 0 else 0.0
                return avg_tok, avg_len, digit_frac, unique_ratio
            p_a = _profile(vals_a)
            p_b = _profile(vals_b)
            # Similarity: 1 - normalized absolute difference for each feature
            score = 0.0
            for va, vb in zip(p_a, p_b):
                denom = max(abs(va), abs(vb), 1e-9)
                score += 1.0 - min(abs(va - vb) / denom, 1.0)
            return score / 4.0  # average over 4 features

        for col1, col2 in self.column_pairs:
            df_a = get_df_from_table_lst(self.tables, col1)
            df_b = get_df_from_table_lst(self.tables, col2)
            df_a = df_a.loc[:, df_a.isna().mean() < 0.5]
            df_b = df_b.loc[:, df_b.isna().mean() < 0.5]
            if col1[1] in df_a.columns and col2[1] in df_b.columns:
                # Skip purely numeric pairs (distilled from old v3)
                if all_numbers(df_a[col1[1]]) and all_numbers(df_b[col2[1]]):
                    for sn in signal_names:
                        signal_scores[sn][(col1, col2)] = 0.0
                    continue
                try:
                    col_a_vals = df_a[col1[1]].fillna('').astype(str).head(20).tolist()
                    col_b_vals = df_b[col2[1]].fillna('').astype(str).head(20).tolist()
                    # Penalize factor if one side is all-numeric
                    num_pen = 0.5 if (all_numbers(df_a[col1[1]]) or all_numbers(df_b[col2[1]])) else 1.0
                    # Signal 1: Jaccard bigram
                    quick_jac, _ = jaccard_matrix_fast(col_a_vals, col_b_vals, n=2)
                    signal_scores['jaccard_bigram'][(col1, col2)] = float(quick_jac) * num_pen
                    # Signal 2: Token-set Jaccard (order-invariant)
                    signal_scores['token_set_jaccard'][(col1, col2)] = _token_set_jaccard(col_a_vals, col_b_vals) * num_pen
                    # Signal 3: Sorted-token overlap
                    signal_scores['sorted_token_overlap'][(col1, col2)] = _sorted_token_overlap(col_a_vals, col_b_vals) * num_pen
                    # Signal 4: Profile compatibility
                    signal_scores['profile_compat'][(col1, col2)] = _profile_compat(col_a_vals, col_b_vals) * num_pen
                    # Signal 5 (optional): ALCS-guided embedding
                    if 'embedding_sim' in signal_names:
                        try:
                            from _embed import embedding_pair_score
                            signal_scores['embedding_sim'][(col1, col2)] = embedding_pair_score(col_a_vals, col_b_vals) * num_pen
                        except:
                            signal_scores['embedding_sim'][(col1, col2)] = 0.0
                except Exception:
                    for sn in signal_names:
                        signal_scores[sn][(col1, col2)] = 0.0

        # Score pairs: use PairPosteriorNet (V2) if available, else RRF fallback
        all_pairs = list(signal_scores[signal_names[0]].keys())
        fast_scores = {pair: 0.0 for pair in all_pairs}

        # Step 1: Always compute RRF baseline (works for 28/31)
        _rrf_k = 60
        for sn in signal_names:
            ranked = sorted(all_pairs, key=lambda p: signal_scores[sn].get(p, 0.0), reverse=True)
            for rank_idx, pair in enumerate(ranked):
                fast_scores[pair] += 1.0 / (_rrf_k + rank_idx + 1)

        # Step 1b: PAIR_SELECTOR overrides fast_scores with a learned MLP.
        #   'learned' = v7 (pair head only, sigmoid prob in [0,1])
        #   'v8'      = pair head + ALCS-variant head (3-way: char/token/fuzzy);
        #               variant is recorded per pair in self._v8_alcs_for_pair
        #               and consumed by get_ALCS_matrix* via _V8_FORCED_MATCH_MODE.
        # Replaces the multi-signal jaccard RRF as the initial column-pair
        # chooser; downstream pipeline unchanged. Env PAIR_SELECTOR_MODEL
        # points at the .pt (v7 file for 'learned', v8 file for 'v8').
        _ps_mode = None
        if _ps_mode == 'step1':
            # AutoJoin Step 1 forced single-pair: bench wrapper sets
            # _STEP1_SRC_COL / _STEP1_TGT_COL per dataset.
            # We force fast_scores so only that single (col_a, col_b) is
            # passed to the downstream operator search.
            forced_src = _STEP1_SRC_COL
            forced_tgt = _STEP1_TGT_COL
            if forced_src and forced_tgt:
                _learned = {}
                for col1, col2 in all_pairs:
                    if col1[1] == forced_src and col2[1] == forced_tgt:
                        _learned[(col1, col2)] = 1.0
                    else:
                        _learned[(col1, col2)] = 0.0
                # If forced pair not found in candidates (col name mismatch),
                # fall through to RRF — preserves the bench so we don't crash.
                if any(v == 1.0 for v in _learned.values()):
                    fast_scores = _learned
        elif _ps_mode == 'learned':
            try:
                from v3_dd.r4.learned_pair_selector import score_pair_learned
                _learned = {}
                for col1, col2 in all_pairs:
                    df_a = get_df_from_table_lst(self.tables, col1)
                    df_b = get_df_from_table_lst(self.tables, col2)
                    df_a = df_a.loc[:, df_a.isna().mean() < 0.5]
                    df_b = df_b.loc[:, df_b.isna().mean() < 0.5]
                    if col1[1] in df_a.columns and col2[1] in df_b.columns:
                        try:
                            _learned[(col1, col2)] = score_pair_learned(
                                df_a[col1[1]].fillna('').astype(str).tolist(),
                                df_b[col2[1]].fillna('').astype(str).tolist(),
                            )
                        except Exception:
                            _learned[(col1, col2)] = 0.0
                    else:
                        _learned[(col1, col2)] = 0.0
                fast_scores = _learned
            except Exception:
                pass  # fall back to RRF if the learned model can't load
        elif _ps_mode == 'v8':
            try:
                from v3_dd.r4.learned_pair_selector_v8 import _forward as _v8_fwd
                _learned = {}
                _alcs = {}
                for col1, col2 in all_pairs:
                    df_a = get_df_from_table_lst(self.tables, col1)
                    df_b = get_df_from_table_lst(self.tables, col2)
                    df_a = df_a.loc[:, df_a.isna().mean() < 0.5]
                    df_b = df_b.loc[:, df_b.isna().mean() < 0.5]
                    if col1[1] in df_a.columns and col2[1] in df_b.columns:
                        try:
                            out = _v8_fwd(
                                df_a[col1[1]].fillna('').astype(str).tolist(),
                                df_b[col2[1]].fillna('').astype(str).tolist(),
                            )
                            if out is None:
                                _learned[(col1, col2)] = 0.0
                                _alcs[(col1, col2)] = 1  # default token
                            else:
                                _learned[(col1, col2)] = out[0]
                                _alcs[(col1, col2)] = out[1]
                        except Exception:
                            _learned[(col1, col2)] = 0.0
                            _alcs[(col1, col2)] = 1
                    else:
                        _learned[(col1, col2)] = 0.0
                        _alcs[(col1, col2)] = 1
                fast_scores = _learned
                # Stash per-pair variant for the transform loop. Caller
                # (the per-pair transform) sets _V8_FORCED_MATCH_MODE
                # before invoking get_ALCS_matrix*.
                self._v8_alcs_for_pair = _alcs
            except Exception:
                pass
        elif _ps_mode == 'v7_v2':
            # v7_v2: same arch as v7 (binary pair head, sigmoid in [0,1])
            # PLUS a second head outputting per-pair embedding-blend weight
            # in [0, 0.5]. The pair score replaces fast_scores; the
            # emb_weight is stashed per-pair and consumed at JOIN execute
            # time via _V72_FORCED_EMB_WEIGHT (set in the per-pair loop
            # below, BEFORE the call into find_transformed_df_opt).
            try:
                from v3_dd.r4.v7_v2.inference import score_pair_v7v2
                _learned = {}
                _emb_w = {}
                for col1, col2 in all_pairs:
                    df_a = get_df_from_table_lst(self.tables, col1)
                    df_b = get_df_from_table_lst(self.tables, col2)
                    df_a = df_a.loc[:, df_a.isna().mean() < 0.5]
                    df_b = df_b.loc[:, df_b.isna().mean() < 0.5]
                    if col1[1] in df_a.columns and col2[1] in df_b.columns:
                        try:
                            ps, ew = score_pair_v7v2(
                                df_a[col1[1]].fillna('').astype(str).tolist(),
                                df_b[col2[1]].fillna('').astype(str).tolist(),
                            )
                        except Exception:
                            ps, ew = 0.0, 0.0
                        _learned[(col1, col2)] = ps
                        _emb_w[(col1, col2)] = ew
                    else:
                        _learned[(col1, col2)] = 0.0
                        _emb_w[(col1, col2)] = 0.0
                fast_scores = _learned
                self._v72_emb_weight_for_pair = _emb_w
            except Exception:
                pass

        # Step 2: PairPosteriorNet AUGMENTATION
        # Don't change RRF ranking at all. Instead, if PPN's top pair
        # is NOT in RRF's top_k, ADD it to the search set.
        # This preserves all 28/31 baseline pairs and only adds new candidates.
        _ppn_extra_pairs = []
        if (ModelRegistry is not None
                and False):
            try:
                _ppn_v2 = ModelRegistry().load('pair_posterior')
                if _ppn_v2 is not None:
                    import numpy as _np
                    _ppn_scores = {}
                    for pair in all_pairs:
                        col1, col2 = pair
                        _pf = _np.array([
                            signal_scores.get('jaccard_bigram', {}).get(pair, 0.0),
                            signal_scores.get('embedding_sim', {}).get(pair, 0.0) if 'embedding_sim' in signal_names else 0.0,
                            signal_scores.get('token_set_jaccard', {}).get(pair, 0.0),
                            signal_scores.get('sorted_token_overlap', {}).get(pair, 0.0),
                            signal_scores.get('profile_compat', {}).get(pair, 0.0),
                            0.0, 0.0, 0.0, 0.0,
                            len(get_df_from_table_lst(self.tables, col1)),
                            len(get_df_from_table_lst(self.tables, col2)),
                        ], dtype=_np.float32)
                        pred = _ppn_v2.predict(_pf)
                        _ppn_scores[pair] = pred['expected_f1']

                    # Add PPN pairs with POSITIVE utility as extra candidates
                    # Utility = E[F1] - λ·E[cost] — learned, not hardcoded
                    _lambda_cost = 0.001
                    for pair in all_pairs:
                        pred = _ppn_v2.predict(_np.array([
                            signal_scores.get('jaccard_bigram', {}).get(pair, 0.0),
                            signal_scores.get('embedding_sim', {}).get(pair, 0.0) if 'embedding_sim' in signal_names else 0.0,
                            signal_scores.get('token_set_jaccard', {}).get(pair, 0.0),
                            signal_scores.get('sorted_token_overlap', {}).get(pair, 0.0),
                            signal_scores.get('profile_compat', {}).get(pair, 0.0),
                            0.0, 0.0, 0.0, 0.0,
                            len(get_df_from_table_lst(self.tables, pair[0])),
                            len(get_df_from_table_lst(self.tables, pair[1])),
                        ], dtype=_np.float32), lambda_cost=_lambda_cost)
                        if pred['utility'] > 0:
                            _ppn_extra_pairs.append(pair)
            except:
                pass

        # Phase 2: Only run expensive transform on top candidate pairs
        ranked_pairs = sorted(fast_scores.items(), key=lambda x: x[1], reverse=True)

        if _ps_mode in ('learned', 'v8', 'v7_v2', 'step1') and len(ranked_pairs) >= 1:
            # Confidence-interval-style top-K: include candidates whose
            # score lies within z·σ of the top-1 score, where σ is the
            # std of all candidate scores in this dataset. Adapts to the
            # per-dataset score distribution — wide band when scores
            # cluster, tight band when top-1 is clearly separated.
            #   ratio = (top - score_i) / σ; keep if ratio <= z
            # z = 1.0 by default (≈ "1-sigma" inclusion). Override via
            # TOPK_Z env var.
            scores = np.array([s for _, s in ranked_pairs], dtype=np.float64)
            top1_score = float(scores[0])
            sigma = float(np.std(scores)) if len(scores) > 1 else 0.0
            z = 1.0
            if sigma <= 1e-9:
                n_candidates = 1
            else:
                cutoff = top1_score - z * sigma
                n_candidates = max(1, int((scores >= cutoff).sum()))
            # Hard cap so we never explode on flat distributions
            _max_k = 5
            n_candidates = min(n_candidates, _max_k, len(ranked_pairs))
        else:
            # Original RRF top-K logic for non-learned modes.
            n_above = len([p for p in ranked_pairs if p[1] > 0.2])
            n_candidates = max(3, n_above)
            _wrapper_rc = RewardConfig()
            if len(ranked_pairs) >= 3:
                top1_score = ranked_pairs[0][1]
                top3_score = ranked_pairs[2][1]
                if top1_score > 0 and (top1_score - top3_score) < _wrapper_rc.adaptive_topk_gap * top1_score:
                    n_candidates = max(_wrapper_rc.adaptive_topk_expand, n_above, n_candidates)

        top_pairs = ranked_pairs[:n_candidates]

        # Augment with PPN extra pairs (add, never remove from RRF selection)
        _rrf_top_set = {p for p, _ in top_pairs}
        for extra_pair in _ppn_extra_pairs:
            if extra_pair not in _rrf_top_set:
                top_pairs.append((extra_pair, 0.0))  # score 0 but still searched
                _rrf_top_set.add(extra_pair)

        for (col1, col2), fast_score in top_pairs:
            df_a = get_df_from_table_lst(self.tables, col1)
            df_b = get_df_from_table_lst(self.tables, col2)
            df_a = df_a.loc[:, df_a.isna().mean() < 0.5]
            df_b = df_b.loc[:, df_b.isna().mean() < 0.5]

            # v8 ALCS-variant override for this pair (char=0→min_len=1,
            # token=1→2, fuzzy=2→'fuzzy' which routes to alcs_fuzzy_single
            # in get_ALCS_matrix*). Reset after the pair finishes so other
            # code paths see no stale override.
            #
            # v9 (ALCS_MODE=auto_fc) takes PRIORITY: even if the v8 head
            # predicted a class, force runtime auto_fc per pair.
            global _V8_FORCED_MATCH_MODE
            if False:
                _V8_FORCED_MATCH_MODE = 'auto_fc'
            else:
                _V8_FORCED_MATCH_MODE = None
                _v8_alcs_map = getattr(self, '_v8_alcs_for_pair', None)
                if _v8_alcs_map is not None:
                    _v8_cls = _v8_alcs_map.get((col1, col2), 1)
                    if _v8_cls == 0:
                        _V8_FORCED_MATCH_MODE = 1          # char
                    elif _v8_cls == 2:
                        _V8_FORCED_MATCH_MODE = 'fuzzy'    # alcs_fuzzy_single
                    else:
                        _V8_FORCED_MATCH_MODE = 2          # token (default)

            # v7_v2: stash per-pair embedding-blend weight (consumed by
            # the JOIN execute step's _embed_weight blend block via the
            # _V72_FORCED_EMB_WEIGHT global). Reset to None when no
            # v7_v2 map is present so other code paths see no stale w.
            global _V72_FORCED_EMB_WEIGHT
            _V72_FORCED_EMB_WEIGHT = None
            _v72_map = getattr(self, '_v72_emb_weight_for_pair', None)
            if _v72_map is not None:
                _V72_FORCED_EMB_WEIGHT = float(_v72_map.get((col1, col2), 0.0))

            if col1[1] in df_a.columns and col2[1] in df_b.columns:

                df_a_concated_applied,df_b_concated_applied,trans_a_to_b, table_a_to_b_cols, trans_b_to_a,table_b_to_a_cols = \
                    find_transformed_df_opt(sample_proportion = _WRAPPER_SAMPLE ,max_steps =_WRAPPER_STEPS, df_a= df_a, df_b = df_b,
                                                                column_a_name =col1[1] ,column_b_name = col2[1],
                                                                    edit_dist_threshold = 1.0 ,all_operators = direct_operators_only,
                                                                    pairs = {},exploration_rate = 0,agreement_percentage = 0.5,greedy = False)

                if df_a_concated_applied is not None:

                    a_to_b_col = concatenate_with_order(df_a_concated_applied,'a_to_b')['a_to_b']
                    b_to_a_col = concatenate_with_order(df_b_concated_applied,'b_to_a')['b_to_a']


                    mean_max_ALCS, sim_matrix,freq_counts_penalty, lcs_matrix  = get_ALCS_matrix(concatenate_with_order(df_a_concated_applied,'a_to_b')['a_to_b'],
                                        concatenate_with_order(df_b_concated_applied,'b_to_a')['b_to_a'],False)

                    if all_numbers(a_to_b_col) or all_numbers(b_to_a_col):
                        # decay the similarity if all numbers for both columns
                        if mean_max_ALCS !=1 and freq_counts_penalty!=0:
                            mean_max_ALCS = mean_max_ALCS/2


                    similarities[(col1, col2)] = mean_max_ALCS
                    penalty[(col1, col2)] = freq_counts_penalty
                    sim_matrixes[(col1, col2)] = sim_matrix
                    lcs_matrixes[(col1, col2)] = lcs_matrix

        # Reset v8 ALCS override after the per-pair loop so other code
        # paths (e.g. final scoring outside this method) see no override.
        _V8_FORCED_MATCH_MODE = None
        # Same reset for v7_v2 emb-weight override.
        _V72_FORCED_EMB_WEIGHT = None

        # compare with the worst penalty
        prev_penalty = np.max(list(penalty.values()))
        prev_penalty_index = [key for key, value in penalty.items() if value == prev_penalty][0]
        prev_lcs_matrix = lcs_matrixes[prev_penalty_index]


        # compare with the lowest sim
        prev_sim = np.min(list(similarities.values()))
        prev_sim_index = [key for key, value in similarities.items() if value == prev_sim][0]
        prev_sim_matrix = sim_matrixes[prev_sim_index]

        # print(f'len(similarities) is {len(similarities)}')

        if len(similarities)>1:
            similarities = filter_pairs_by_threshold(similarities, self.filter_threshold)

        # print(f'after filter, len(similarities) is {len(similarities)}')


        for col1, col2 in similarities:
            reward = 0
            edit_dist_dif = similarities[(col1, col2)] - prev_sim

            if prev_penalty != 0:
                reward = -(penalty[(col1, col2)] - prev_penalty) / np.max([prev_penalty,penalty[(col1, col2)]])  * 30000
                lcs_difs = np.max(lcs_matrixes[(col1, col2)], axis=1) - np.max(prev_lcs_matrix, axis=1)
                if reward > 0:
                    n_changed = max(np.sum(lcs_difs != 0), 1)
                    positive_fraction = np.sum(lcs_difs > 0) / n_changed
                    reward = reward * positive_fraction

            if edit_dist_dif > 0:
                edit_dist_difs = np.max(sim_matrixes[(col1, col2)], axis=1) - np.max(prev_sim_matrix, axis=1)
                n_changed = max(np.sum(edit_dist_difs != 0), 1)
                positive_fraction = np.sum(edit_dist_difs > 0) / n_changed
                edit_dist_dif = edit_dist_dif * positive_fraction

            sim_component = 10000 * edit_dist_dif
            if sim_component > 0 and reward < 0:
                reward = max(reward, -sim_component * 0.5)
            estimated_reward = sim_component + reward
            reward_dicts[(col1, col2)] = estimated_reward

        # Sort the reward dictionary by value in descending order
        sorted_reward_dicts = dict(sorted(reward_dicts.items(), key=lambda item: item[1], reverse=True))

        sorted_sim_dicts = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))

        print(f'similarities are {sorted_sim_dicts}')
        print(f'sorted_reward_dicts are {sorted_reward_dicts}')


        return sorted_sim_dicts,sorted_reward_dicts


# reusing algo by recording the initial sim
class ALCS_sim_for_reusing_algo:

    def __init__(self, data_loader,sample_proportion=1.0,filter_threshold = 0.1):
        """
        Initializes the CombinedQGramSimilarity class.
        
        Parameters:
        - data_loader: instance of DataLoader
            Provides `columns` (dict) and `column_pairs` (list of tuples).
        - q: int (default=2)
            Q-gram size. (No longer used if we are moving to ALCS-based approach only.)
        - sample_proportion: float (default=0.5)
            Proportion of data to sample from each column (0 < p <= 1).
        """
        self.columns = data_loader.columns
        self.column_pairs = data_loader.column_pairs 
        self.sample_proportion = sample_proportion
        
        self.filter_threshold = filter_threshold

        self.sampled_data = {}

        self.create_sampled_data()

        self.tables = data_loader.tables_with_names

    def create_sampled_data(self):
        """Create sampled data for each column."""
        data_length = min(len(data) for data in self.columns.values())
        if data_length == 0:
            raise ValueError("At least one column is empty.")
        sample_size = max(1, int(data_length * self.sample_proportion))
        indices = random.sample(range(data_length), sample_size)
        for col_key, data in self.columns.items():
            self.sampled_data[col_key] = [data[i] for i in indices]

    def compute_all_pairs_similarity(self):
        """
        Compute ALCS-based similarities between column pairs using the
        `get_edit_dist_matrix` function. Returns a dictionary of
        {(col1, col2): mean_max_ALCS}.
        """
        ini_sim = {}
        similarities = {}
        penalty = {}
        sim_matrixes = {}
        lcs_matrixes = {}
        reward_dicts = {}

        # For each column pair, compute ALCS
        for col1, col2 in self.column_pairs:

            df_a = get_df_from_table_lst(self.tables, col1)
            df_b = get_df_from_table_lst(self.tables, col2)


            df_a = df_a.loc[:, df_a.isna().mean() < 0.5]
            df_b = df_b.loc[:, df_b.isna().mean() < 0.5]
            

            if col1[1] in df_a.columns and col2[1] in df_b.columns:

                df_a_concated_applied,df_b_concated_applied,trans_a_to_b, table_a_to_b_cols, trans_b_to_a,table_b_to_a_cols = \
                    find_transformed_df_opt(sample_proportion = _WRAPPER_SAMPLE ,max_steps =_WRAPPER_STEPS, df_a= df_a, df_b = df_b,
                                                                column_a_name =col1[1] ,column_b_name = col2[1],
                                                                    edit_dist_threshold = 1.0 ,all_operators = direct_operators_only,
                                                                    pairs = {},exploration_rate = 0,agreement_percentage = 0.5,greedy = False)
                
                if df_a_concated_applied is not None:
                
                    a_to_b_col = concatenate_with_order(df_a_concated_applied,'a_to_b')['a_to_b']
                    b_to_a_col = concatenate_with_order(df_b_concated_applied,'b_to_a')['b_to_a']

                    
                    mean_max_ALCS, sim_matrix,freq_counts_penalty, lcs_matrix  = get_ALCS_matrix(concatenate_with_order(df_a_concated_applied,'a_to_b')['a_to_b'], 
                                        concatenate_with_order(df_b_concated_applied,'b_to_a')['b_to_a'],False)
                    
                    ini_mean_max_ALCS, _,_, _  = get_ALCS_matrix(df_a[col1[1]].astype(str).str.lower(),
                                                                 df_b[col2[1]].astype(str).str.lower(),False)
                    
                    if all_numbers(a_to_b_col) or all_numbers(b_to_a_col):
                        # decay the similarity if all numbers for both columns
                        if mean_max_ALCS !=1 and freq_counts_penalty!=0:
                            mean_max_ALCS = mean_max_ALCS/2


                    similarities[(col1, col2)] = mean_max_ALCS
                    penalty[(col1, col2)] = freq_counts_penalty
                    sim_matrixes[(col1, col2)] = sim_matrix
                    lcs_matrixes[(col1, col2)] = lcs_matrix 
                    ini_sim[(col1, col2)] = [ini_mean_max_ALCS,mean_max_ALCS]


        # compare with the worst penalty
        prev_penalty = np.max(list(penalty.values()))
        prev_penalty_index = [key for key, value in penalty.items() if value == prev_penalty][0]
        prev_lcs_matrix = lcs_matrixes[prev_penalty_index]


        # compare with the lowest sim
        prev_sim = np.min(list(similarities.values()))
        prev_sim_index = [key for key, value in similarities.items() if value == prev_sim][0]
        prev_sim_matrix = sim_matrixes[prev_sim_index]

        similarities = filter_pairs_by_threshold(similarities, self.filter_threshold)


        for col1, col2 in similarities:
            reward = 0
            edit_dist_dif = similarities[(col1, col2)] - prev_sim

            if prev_penalty != 0:
                reward = -(penalty[(col1, col2)] - prev_penalty) / np.max([prev_penalty,penalty[(col1, col2)]])  * 30000
                lcs_difs = np.max(lcs_matrixes[(col1, col2)], axis=1) - np.max(prev_lcs_matrix, axis=1)
                if reward > 0:
                    n_changed = max(np.sum(lcs_difs != 0), 1)
                    positive_fraction = np.sum(lcs_difs > 0) / n_changed
                    reward = reward * positive_fraction

            if edit_dist_dif > 0:
                edit_dist_difs = np.max(sim_matrixes[(col1, col2)], axis=1) - np.max(prev_sim_matrix, axis=1)
                n_changed = max(np.sum(edit_dist_difs != 0), 1)
                positive_fraction = np.sum(edit_dist_difs > 0) / n_changed
                edit_dist_dif = edit_dist_dif * positive_fraction

            sim_component = 10000 * edit_dist_dif
            if sim_component > 0 and reward < 0:
                reward = max(reward, -sim_component * 0.5)
            estimated_reward = sim_component + reward
            reward_dicts[(col1, col2)] = estimated_reward

        # Sort the reward dictionary by value in descending order
        sorted_reward_dicts = dict(sorted(reward_dicts.items(), key=lambda item: item[1], reverse=True))

        sorted_sim_dicts = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))

        print(f'similarities are {sorted_sim_dicts}')
        print(f'sorted_reward_dicts are {sorted_reward_dicts}')

        return sorted_sim_dicts,sorted_reward_dicts,ini_sim
    

def learning_restart_greedy_version_opt_multi_results(operator_lsts,      
                     exploration_rates,
                     sample_proportions,
                     max_steps,
                     df_a,
                     df_b,
                     column_a_name,
                     column_b_name,
                     pairs,
                     edit_dist_thresholds,
                     agreement_percentages,
                     greedy,
                     find_transformed_df_opt_func = find_transformed_df_opt):
    """
    Attempts different combinations of operators, edit distance thresholds,
    exploration rates, and sample proportions to find a set of matches
    using get_the_join_using_multi_to_multi. Tries multiple data variations:
      - Original df_a, df_b
      - df_a, df_b with columns dropped
      - Reversed order: df_b, df_a
      - Reversed order with columns dropped
    Returns the first successful match (df_matches, trans_a_to_b, etc.).
    """

    # Create the dropped-column versions

    iteration_times = 0

    reversed_pairs  = [(y, x) for x, y in pairs]

    combos = []


    # Loop over all parameter choices
    for edit_dist_threshold in edit_dist_thresholds:
        print(f'Edit percentage threshold is {edit_dist_threshold}')
        for operators in operator_lsts:
            for exploration_rate in exploration_rates:
                print(f'Exploration rate is {exploration_rate}')
                for sample_proportion in sample_proportions:
                    print(f'Sample proportion is {sample_proportion}')
                    for agreement_percentage in agreement_percentages:
                        print(f'Agreement percentage is {agreement_percentage}')
                        data_variations = [
                            # (DataFrame A, DataFrame B, column_A_name, column_B_name)
                            (df_a, df_b, column_a_name, column_b_name, pairs),
                            # reversed (b -> a direction)
                            (df_b, df_a, column_b_name, column_a_name, reversed_pairs),
                        ]
                        # L1 ablation: NO_REVERSE_DIRECTION=1 skips the b->a direction
                        # to test if bidirectional search matters.
                        if False:
                            data_variations = data_variations[:1]
                        # L2 ablation: DOUBLE_FORWARD=1 runs a->b TWICE (matched
                        # compute budget to bidirectional). Tests if 2x compute alone
                        # (not the reversal itself) is what bidirectional buys us.
                        if False:
                            data_variations = [
                                (df_a, df_b, column_a_name, column_b_name, pairs),
                                (df_a, df_b, column_a_name, column_b_name, pairs),
                            ]
                        # K2 ablation: VARIATIONS_MULTIPLIER=N duplicates data_variations
                        # N times (preserves whatever direction config was set above).
                        # Lets us trade compute for F1 on top of any other ablation.
                        _vm = None
                        if _vm:
                            try:
                                data_variations = data_variations * max(1, int(_vm))
                            except Exception: pass
                        # Q2: clear shared-sample cache at variations loop entry so a
                        # new (df_a, df_b) pair doesn't collide on stale id() keys.
                        if False:
                            _SHARED_SAMPLE_CACHE.clear()
                        # PROBE: clear per-rollout combo log at the start of each
                        # variations loop. Populated below after each combo is created.
                        if _LOG_COMBOS:
                            _PROBE_COMBOS['rollouts'] = []

                        for _var_idx, (dfa, dfb, col_a, col_b, pairs) in enumerate(data_variations):
                            iteration_times += 1
                            print(f'In iteration time {iteration_times}')
                            # O8: REINFORCE_SWAP=1 — first 2 rollouts pure
                            # argmax (one per direction since base data_variations
                            # has 2 entries: a->b and b->a). Remaining rollouts have
                            # swap active. Best-of-N ALCS selection downstream
                            # keeps the higher trajectory. With multiplier=4 this
                            # gives 2 deterministic baselines + 6 swap attempts.
                            # REINFORCE_SWAP is disabled in the released
                            # config, so the per-rollout swap-gating writes below
                            # never fire.
                            if False:
                                pass
                            
                            # Start of the join process
                            print(f"Starting join attempt with sample_proportion={sample_proportion}, max_steps={max_steps}, edit_dist_threshold={edit_dist_threshold}")

                            edit_dist,df_matches,df_matches_multi, trans_a_to_b, table_a_to_b_cols, trans_b_to_a, table_b_to_a_cols,\
                            freq_counts_penalty,sim_matrix,lsc_matrix,sim_reward_factor = \
                                get_the_join_using_multi_to_multi_greedy_opt(
                                    sample_proportion,
                                    max_steps,
                                    dfa,
                                    dfb,
                                    col_a,
                                    col_b,
                                    edit_dist_threshold,
                                    operators,
                                    pairs,
                                    exploration_rate,
                                    agreement_percentage,
                                    greedy,
                                    find_transformed_df_opt_func
                                )
                            
                            combo = (edit_dist,df_matches,df_matches_multi, trans_a_to_b,
                                     table_a_to_b_cols, trans_b_to_a,
                                       table_b_to_a_cols,freq_counts_penalty,sim_matrix,lsc_matrix,sim_reward_factor)

                            combos.append(combo)
                            # PROBE: capture (rollout_idx, edit_dist, df_matches_multi) so
                            # the runner can compute per-combo F1 after the chain returns.
                            if _LOG_COMBOS:
                                _PROBE_COMBOS['rollouts'].append({
                                    'rollout_idx': _var_idx,
                                    'edit_dist': float(edit_dist) if edit_dist is not None else None,
                                    'freq_counts_penalty': int(freq_counts_penalty) if freq_counts_penalty is not None else None,
                                    'df_matches_multi': df_matches_multi,
                                    'trans_a_to_b': trans_a_to_b,
                                    'table_a_to_b_cols': table_a_to_b_cols,
                                    'trans_b_to_a': trans_b_to_a,
                                    'table_b_to_a_cols': table_b_to_a_cols,
                                })
                            
                            # Log completion of the join attempt
                            print(f"Join attempt completed. df_matches: {'Found match' if df_matches is not None else 'No match found'}")

                            # If df_matches meets the threshold
                            if edit_dist>=edit_dist_threshold and freq_counts_penalty == 0:
                                print('Match found! Returning results.')
                                return combos

    # returns the best reuslt
    print('No match found with the given parameters.')
    return combos


def learning_restart_greedy_version_opt_multi_results_with_reusing(operator_lsts,      
                     exploration_rates,
                     sample_proportions,
                     max_steps,
                     df_a,
                     df_b,
                     column_a_name,
                     column_b_name,
                     pairs,
                     edit_dist_thresholds,
                     agreement_percentages,
                     greedy,
                     transformed_columns_for_a_to_b,
                     transformed_columns_for_b_to_a,
                     agent_a,
                     agent_b,
                     find_transformed_df_opt_func = find_transformed_df_opt_with_reusing):
    """
    Attempts different combinations of operators, edit distance thresholds,
    exploration rates, and sample proportions to find a set of matches
    using get_the_join_using_multi_to_multi. Tries multiple data variations:
      - Original df_a, df_b
      - df_a, df_b with columns dropped
      - Reversed order: df_b, df_a
      - Reversed order with columns dropped
    Returns the first successful match (df_matches, trans_a_to_b, etc.).
    """

    # Create the dropped-column versions

    iteration_times = 0

    reversed_pairs  = [(y, x) for x, y in pairs]

    combos = []


    # Loop over all parameter choices
    for edit_dist_threshold in edit_dist_thresholds:
        print(f'Edit percentage threshold is {edit_dist_threshold}')
        for operators in operator_lsts:
            for exploration_rate in exploration_rates:
                print(f'Exploration rate is {exploration_rate}')
                for sample_proportion in sample_proportions:
                    print(f'Sample proportion is {sample_proportion}')
                    for agreement_percentage in agreement_percentages:
                        print(f'Agreement percentage is {agreement_percentage}')
                        data_variations = [
                            # (DataFrame A, DataFrame B, column_A_name, column_B_name)
                            (df_a, df_b, column_a_name, column_b_name, pairs, transformed_columns_for_a_to_b,transformed_columns_for_b_to_a,agent_a,agent_b),
                            # reversed
                            (df_b, df_a, column_b_name, column_a_name, reversed_pairs, transformed_columns_for_b_to_a ,transformed_columns_for_a_to_b,agent_b,agent_a ),
                        ]

                        for dfa, dfb, col_a, col_b, pairs,transcols_a_b,transcols_b_a,agenta,agentb in data_variations:
                            iteration_times += 1
                            print(f'In iteration time {iteration_times}')
                            
                            # Start of the join process
                            print(f"Starting join attempt with sample_proportion={sample_proportion}, max_steps={max_steps}, edit_dist_threshold={edit_dist_threshold}")

                            edit_dist,df_matches,df_matches_multi, trans_a_to_b, table_a_to_b_cols, trans_b_to_a, table_b_to_a_cols,\
                            freq_counts_penalty,sim_matrix,lsc_matrix,sim_reward_factor,agent_for_a,agent_for_b = \
                                get_the_join_using_multi_to_multi_greedy_opt_with_reusing(
                                    sample_proportion,
                                    max_steps,
                                    dfa,
                                    dfb,
                                    col_a,
                                    col_b,
                                    edit_dist_threshold,
                                    operators,
                                    pairs,
                                    exploration_rate,
                                    agreement_percentage,
                                    greedy,
                                    transcols_a_b,
                                    transcols_b_a,
                                    agenta,
                                    agentb,
                                    find_transformed_df_opt_func 
                                )
                            
                            combo = (edit_dist,df_matches,df_matches_multi, trans_a_to_b, 
                                     table_a_to_b_cols, trans_b_to_a,
                                       table_b_to_a_cols,freq_counts_penalty,sim_matrix,lsc_matrix,sim_reward_factor,agent_for_a,agent_for_b)

                            combos.append(combo)
                            
                            # Log completion of the join attempt
                            print(f"Join attempt completed. df_matches: {'Found match' if df_matches is not None else 'No match found'}")

                            # If df_matches meets the threshold
                            if edit_dist>=edit_dist_threshold and freq_counts_penalty == 0:
                                print('Match found! Returning results.')
                                return combos

    # returns the best reuslt
    print('No match found with the given parameters.')
    return combos
    
pairs = {}


def getting_reward(prev_edit_dist,edit_dist,prev_penalty,penalty,greedy,sim_reward_factor,reward_config=None):

    if reward_config is None:
        reward_config = RewardConfig()

    ALCS_sim_dif = edit_dist - prev_edit_dist
    reward =0

    factor = reward_config.get_greedy_factor(greedy)

    ## solve the different column pair issues (different columns have different numbers of rows), also have to consider about the coverage

    # NO_FREQ_PENALTY=1 zeros the freq_counts_penalty contribution
    # — leaves Q-learning reward as PURELY (constant × ΔALCS).
    # Use with UNIQUE_ALCS_REWARD=1 to test the unique-assignment
    # signal in isolation (no double-penalty interaction with freq counts).
    if False:
        reward = 0
    elif ALCS_sim_dif<0.25:
        reward = -(penalty - prev_penalty) / np.max([prev_penalty,penalty]) * reward_config.reward_uniqueness_factor

    else:
        reward = -(penalty - prev_penalty) / np.max([prev_penalty,penalty]) * (reward_config.reward_uniqueness_factor / 10.0)



    estimated_reward = sim_reward_factor*factor*reward_config.reward_factor * ALCS_sim_dif + reward

    print(f'prev penalty is {prev_penalty} and temp penalty is {penalty}')

    print(f'prev ALCS is {prev_edit_dist} and temp ALCS is {edit_dist} with reward factor {sim_reward_factor}')

    print(f'reward form ALCS {sim_reward_factor*factor*reward_config.reward_factor * ALCS_sim_dif}, uiqueness is {reward}')

    return estimated_reward




def process_single_join_for_multi_to_multi_greedy_with_all_best_results_opt_multi_results_with_policy(
    tables, join_order_lst, operator_lsts,
    edit_dist_thresholds, sample_proportions,agreement_percentages ,exploration_rates,max_steps, top_k_pairs = 1,greedy = True,
    find_transformed_df_opt_func = find_transformed_df_opt
):
    best_combos = None
    best_combos_multi = None
    best_edit_dist_multi = 0
    best_cov_multi = -1.0   # _COVERAGE_SELECT: best coverage seen (multi)
    best_cov_reward = -1.0  # _COVERAGE_SELECT: best coverage seen (1to1/reward slot)
    i = 0
    pairs_dict = {f'pair {i}': None for i in range(top_k_pairs)}
    best_reward = 0
    for attr_a, attr_b in join_order_lst:

        print(f'start trying {i+1} pair')

        if i<top_k_pairs:
            attr_a_name = attr_a[1]
            attr_b_name = attr_b[1]

            df_a = get_df_from_table_lst(tables, attr_a)
            df_b = get_df_from_table_lst(tables, attr_b)


            combos = \
                learning_restart_greedy_version_opt_multi_results(
                    operator_lsts, exploration_rates,sample_proportions,max_steps, df_a, df_b,
                    attr_a_name, attr_b_name,
                    pairs, edit_dist_thresholds, agreement_percentages,greedy,find_transformed_df_opt_func
                )

            if i == 0:
                (prev_edit_dist,prev_df_matches,prev_df_matches_multi, prev_trans_a_to_b, prev_table_a_to_b_cols, prev_trans_b_to_a, \
                prev_table_b_to_a_cols,prev_freq_counts_penalty,prev_sim_matrix,prev_lsc_matrix,pre_sim_reward_factor) = combos[0]

                best_combos = (prev_edit_dist,prev_df_matches_multi, prev_trans_a_to_b, prev_table_a_to_b_cols, prev_trans_b_to_a, \
                prev_table_b_to_a_cols,prev_freq_counts_penalty,prev_sim_matrix,prev_lsc_matrix)

                best_combos_multi = (prev_edit_dist,prev_df_matches_multi, prev_trans_a_to_b, prev_table_a_to_b_cols, prev_trans_b_to_a, \
                prev_table_b_to_a_cols,prev_freq_counts_penalty,prev_sim_matrix,prev_lsc_matrix)

            # _LOG_COMBOS: record every candidate (truth-free) before the
            # selection logic (which may early-return) discards the rest.
            if _LOG_COMBOS:
                for _c in combos:
                    try:
                        _COMBO_LOG.append({
                            'pair_i': i,
                            'alcs': float(_c[0]),
                            'freq_penalty': float(_c[7]),
                            'sim_matrix': np.array(_c[8], dtype=np.float32),
                            'df_matches_multi': _c[2],
                            'trans_a_to_b': _c[3],
                        })
                    except Exception:
                        pass

            for combo in combos:

                edit_dist,df_matches,df_matches_multi, trans_a_to_b, table_a_to_b_cols, trans_b_to_a, \
                table_b_to_a_cols,freq_counts_penalty,sim_matrix,lsc_matrix,sim_reward_factor = combo
                # **Check if the join was successful**
                # successfully join for 1 to 1 join
                if edit_dist >= np.max(edit_dist_thresholds) and freq_counts_penalty == 0:
                    best_combos = (edit_dist, df_matches, trans_a_to_b, table_a_to_b_cols, trans_b_to_a, table_b_to_a_cols,freq_counts_penalty,
                                   sim_matrix,lsc_matrix)

                    best_combos_multi = (edit_dist,df_matches_multi, trans_a_to_b, table_a_to_b_cols, trans_b_to_a, table_b_to_a_cols,freq_counts_penalty,
                                         sim_matrix,lsc_matrix)

                    pairs_dict[f'pair {i}'] = (best_combos,best_combos_multi)

                    return pairs_dict

                else:
                    # _COVERAGE_SELECT=1: rank candidate combos by coverage
                    # (matched/source rows), ALCS as tiebreak, instead of ALCS /
                    # reward. ALCS favors clean-but-partial transforms; coverage
                    # prefers the one that joined more rows (recall-bound tasks).
                    _cov_sel = _COVERAGE_SELECT
                    _cov = _combo_cov(df_matches_multi, sim_matrix) if _cov_sel else None
                    # _SELECT_SIGNAL=cov_mnn: rank by coverage * mutual_nn
                    # (recall * precision) so over-admitting low-precision chains
                    # (high coverage, low bijectivity) are down-ranked.
                    if _cov is not None and _SELECT_SIGNAL == 'cov_mnn':
                        _cov = _cov * _combo_mutual_nn(sim_matrix)
                    # _SELECT_SIGNAL=cov_uniq: rank by coverage * uniqueness
                    # (recall * distinct-target rate) — penalizes many-source-to-
                    # one-target collisions instead of bijectivity. Parallel to
                    # cov_mnn; additive, leaves the cov_mnn path untouched.
                    elif _cov is not None and _SELECT_SIGNAL == 'cov_uniq':
                        _cov = _cov * _combo_uniqueness(df_matches_multi)

                    if _cov_sel:
                        if (_cov > best_cov_multi) or (_cov == best_cov_multi and edit_dist > best_edit_dist_multi):
                            best_cov_multi = _cov
                            best_edit_dist_multi = edit_dist
                            best_combos_multi = (edit_dist,df_matches_multi, trans_a_to_b, table_a_to_b_cols, trans_b_to_a, table_b_to_a_cols,freq_counts_penalty,
                                                sim_matrix,lsc_matrix)
                    elif edit_dist>best_edit_dist_multi:
                        best_edit_dist_multi = edit_dist
                        best_combos_multi = (edit_dist,df_matches_multi, trans_a_to_b, table_a_to_b_cols, trans_b_to_a, table_b_to_a_cols,freq_counts_penalty,
                                            sim_matrix,lsc_matrix)

                    if _cov_sel:
                        if (_cov > best_cov_reward) or (_cov == best_cov_reward and edit_dist > prev_edit_dist):
                            best_cov_reward = _cov
                            best_combos = (edit_dist,df_matches_multi, trans_a_to_b, table_a_to_b_cols, trans_b_to_a, table_b_to_a_cols,freq_counts_penalty,
                                                sim_matrix,lsc_matrix)
                            (prev_edit_dist,prev_df_matches_multi, prev_trans_a_to_b, prev_table_a_to_b_cols, prev_trans_b_to_a, \
                                prev_table_b_to_a_cols,prev_freq_counts_penalty,prev_sim_matrix,prev_lsc_matrix) = best_combos
                    else:
                        reward = getting_reward(prev_edit_dist,edit_dist,
                                                prev_freq_counts_penalty,freq_counts_penalty,greedy,sim_reward_factor)
                        if reward > best_reward:
                            best_combos = (edit_dist,df_matches_multi, trans_a_to_b, table_a_to_b_cols, trans_b_to_a, table_b_to_a_cols,freq_counts_penalty,
                                                sim_matrix,lsc_matrix)

                            (prev_edit_dist,prev_df_matches_multi, prev_trans_a_to_b, prev_table_a_to_b_cols, prev_trans_b_to_a, \
                                prev_table_b_to_a_cols,prev_freq_counts_penalty,prev_sim_matrix,prev_lsc_matrix) = best_combos


                pairs_dict[f'pair {i}'] = (best_combos,best_combos_multi)

            i+=1

        else:
            return pairs_dict

    return pairs_dict


def record_combos(combos,dict_val):
    (prev_edit_dist,prev_df_matches,prev_df_matches_multi, prev_trans_a_to_b, prev_table_a_to_b_cols, prev_trans_b_to_a, \
    prev_table_b_to_a_cols,prev_freq_counts_penalty,prev_sim_matrix,prev_lsc_matrix,pre_sim_reward_factor) = combos[0]

    best_combos = (prev_edit_dist,prev_df_matches_multi, prev_trans_a_to_b, prev_table_a_to_b_cols, prev_trans_b_to_a, \
    prev_table_b_to_a_cols,prev_freq_counts_penalty,prev_sim_matrix,prev_lsc_matrix,pre_sim_reward_factor)

    dict_val = best_combos





def process_single_join_for_multi_to_multi_greedy_with_all_best_results_opt_multi_results_with_policy_for_reusing_with_reusing_agent(
    pair_updated_tables, join_order_lst, operator_lsts, 
    edit_dist_thresholds, sample_proportions,agreement_percentages ,exploration_rates,max_steps, top_k_pairs = 1,greedy = True,
    transformed_columns_for_a_to_b_dict= None,transformed_colums_for_b_to_a_dict =None,agent_a_dict = None, agent_b_dict = None,
    find_transformed_df_opt_func = find_transformed_df_opt_with_reusing
):

    i = 0
    # pairs_dict = defaultdict(list) 
    best_reward = 0
    all_transformations_dict = {pairs:[] for pairs in join_order_lst}

    print(pair_updated_tables)

    for pairs in join_order_lst:

        tables = pair_updated_tables[pairs]

        print('tables')

        print(tables)

        attr_a, attr_b = pairs

        if transformed_colums_for_b_to_a_dict:
            transformed_columns_for_a_to_b = transformed_columns_for_a_to_b_dict[pairs]
        else:
            transformed_columns_for_a_to_b = None

        if transformed_colums_for_b_to_a_dict:
            transformed_columns_for_b_to_a = transformed_colums_for_b_to_a_dict[pairs]
        else:
            transformed_columns_for_b_to_a = None

        if agent_a_dict:
            agent_a = agent_a_dict[pairs]
        else:
            agent_a = None

        if agent_b_dict:
            agent_b = agent_b_dict[pairs]
        else:
            agent_b = None


        print(f'start trying {i+1} pair')

        if i<top_k_pairs:
            attr_a_name = attr_a[1]
            attr_b_name = attr_b[1]


            df_a = get_df_from_table_lst(tables, attr_a)
            df_b = get_df_from_table_lst(tables, attr_b)

            # print('tables a')
            # print(df_a)

            # print('tables b')
            # print(df_b)

            combos = \
                learning_restart_greedy_version_opt_multi_results_with_reusing(
                    operator_lsts, exploration_rates,sample_proportions,max_steps, df_a, df_b, 
                    attr_a_name, attr_b_name, 
                    pairs, edit_dist_thresholds, agreement_percentages,greedy,
                    transformed_columns_for_a_to_b,transformed_columns_for_b_to_a,
                    agent_a,agent_b,find_transformed_df_opt_func
                )
                    
                        
            for combo in combos:

                edit_dist,df_matches,df_matches_multi, trans_a_to_b, table_a_to_b_cols, trans_b_to_a, \
                table_b_to_a_cols,freq_counts_penalty,sim_matrix,lsc_matrix,sim_reward_factor,agent_a,agent_b = combo

                best_combo = (edit_dist,df_matches_multi, trans_a_to_b, table_a_to_b_cols,
                               trans_b_to_a, table_b_to_a_cols,freq_counts_penalty,
                                            sim_matrix,lsc_matrix,sim_reward_factor,agent_a,agent_b) 

                all_transformations_dict[(attr_a, attr_b)].append(best_combo)

            i+=1 

        else:
            return all_transformations_dict

    return all_transformations_dict



def get_merged_table_multi_to_multi_greedy_with_all_best_result_opt_multi_results_with_policy(dfs, operator_lsts, edit_dist_thresholds,sample_proportions,
                                                                                              agreement_percentages,exploration_rates,max_steps = 10,top_k_pairs =2,
                                                                                              find_transformed_df_opt_func = find_transformed_df_opt):
    #step 1 InvertedIndexAndDataSamplingQGramSimilarity,Inverted_index_and_data_smaplinhg_QGramSimilarity
    dataloader = DataLoader(dfs)
    comb_sim = ALCS_sim(dataloader, sample_proportion = 1.0)
    sorted_similarities,sorted_reward_dicts = comb_sim.compute_all_pairs_similarity()

    lst = sorted_similarities.values()
    if not lst:
        return {}
    max_val = max(lst)


    high_sim_pairs = np.max([len([val for val in lst if val > 0.55]),len([val for val in lst if val >max_val - 0.25])])


    if high_sim_pairs >=2:
        greedy = False
        # top_k_pair = top_k_pairs
        ap = agreement_percentages
    else:
        greedy = True
        # top_k_pair = 1
        ap =agreement_percentages_non_greedy


    join_order_lst = list(sorted_reward_dicts.keys())

    # Adaptive top_k: use signal score distribution to decide how many pairs to try.
    # If scores are flat (no clear winner), try more pairs.
    # Uses normalized entropy of the score distribution — no hardcoded thresholds.
    _top_k = top_k_pairs
    if len(sorted_similarities) > 2:
        _scores = np.array(list(sorted_similarities.values()))
        _scores_pos = _scores[_scores > 0]
        if len(_scores_pos) >= 2:
            # Normalize to probability distribution
            _probs = _scores_pos / _scores_pos.sum()
            # Entropy: high = flat distribution (need more pairs), low = clear winner
            _entropy = -np.sum(_probs * np.log(_probs + 1e-10))
            _max_entropy = np.log(len(_probs))  # uniform distribution entropy
            _norm_entropy = _entropy / max(_max_entropy, 1e-10)  # [0, 1]
            # Scale top_k: entropy=0 → top_k stays, entropy=1 → top_k * 3
            _top_k = max(top_k_pairs, int(top_k_pairs * (1.0 + 2.0 * _norm_entropy)))
            _top_k = min(_top_k, len(join_order_lst))  # don't exceed available pairs

    # step 3 to 4
    tables= dataloader.tables_with_names

    pairs_dict  = process_single_join_for_multi_to_multi_greedy_with_all_best_results_opt_multi_results_with_policy(tables, join_order_lst, operator_lsts,
    edit_dist_thresholds, sample_proportions,ap ,exploration_rates,max_steps,_top_k,greedy,find_transformed_df_opt_func)

    return pairs_dict




def find_transformed_df_data_discovery_ver(sample_proportion,max_steps, df_a, df_b,column_a_name,column_b_name,all_operators,pairs, edit_dist_threshold,exploration_rate,agreement_percentage,greedy):
   

   # make it at most 20 rows
   min_size = max(int(df_a.shape[0]* sample_proportion),int(df_b.shape[0]* sample_proportion))

   sample_size = max(1, min_size)
   sample_size = min(sample_size,20)

   df_b_sample = min(len(df_b),2*sample_size)

   # Diverse sampling: cluster by string length then sample from each cluster
   if sample_size < df_a.shape[0]:
       join_col = df_a[column_a_name].astype(str) if column_a_name in df_a.columns else df_a.iloc[:, 0].astype(str)
       lengths = join_col.str.len()
       try:
           q33 = lengths.quantile(0.33)
           q66 = lengths.quantile(0.66)
           buckets = pd.cut(lengths, bins=[-1, q33, q66, float('inf')], labels=[0, 1, 2])
           samples_per_bucket = max(1, sample_size // 3)
           sampled_indices = []
           for bucket_id in [0, 1, 2]:
               bucket_rows = df_a.index[buckets == bucket_id].tolist()
               if bucket_rows:
                   n_take = min(samples_per_bucket, len(bucket_rows))
                   sampled_indices.extend(random.sample(bucket_rows, n_take))
           remaining = sample_size - len(sampled_indices)
           if remaining > 0:
               leftover = [i for i in df_a.index if i not in sampled_indices]
               sampled_indices.extend(random.sample(leftover, min(remaining, len(leftover))))
           df_a_sampled = df_a.loc[sampled_indices].reset_index(drop=True)
       except Exception:
           df_a_sampled = df_a.sample(sample_size).reset_index(drop=True)
   else:
       df_a_sampled = df_a.copy()
   df_b_sampled = df_b.sample(df_b_sample).reset_index(drop=True)

   # Select reward config via learned MetaSelector (config blending only)
   _selector = RewardConfigSelector()
   _rc, _rc_name = _selector.select(df_a_sampled, column_a_name, df_b_sampled, column_b_name)

   # Use ALL operators, depth=0 (proven more robust)
   agent_table_a = QLearningAgent_edit_dist_modified_for_multi_opt(all_operators,df_a_sampled,agreement_percentage,exploration_rate, reward_config=_rc, depth=0)
   agent_table_b = QLearningAgent_edit_dist_modified_for_multi_opt(all_operators,df_b_sampled,agreement_percentage,exploration_rate, reward_config=_rc, depth=0)


   trans_a_to_b, table_a_to_b_cols, trans_b_to_a,table_b_to_a_cols, edit_dist_matrix = optimize_transformations_both_edit_dist_opt(max_steps, df_a_sampled, df_b_sampled,column_a_name,column_b_name,
                                                                                                                                                                              agent_table_a,agent_table_b,
                                                                                                                                                                               pairs, edit_dist_threshold,greedy)
   df_a_concated = build_dataframe_in_order(df_a, table_a_to_b_cols)
   df_b_concated = build_dataframe_in_order(df_b, table_b_to_a_cols)
   
   df_a_concated_applied = apply_all_actions_to_df(df_a_concated,trans_a_to_b).astype(str).apply(lambda col: col.str.lower())
   df_b_concated_applied =  apply_all_actions_to_df(df_b_concated,trans_b_to_a).astype(str).apply(lambda col: col.str.lower())

   return df_a_concated_applied,df_b_concated_applied,trans_a_to_b, table_a_to_b_cols, trans_b_to_a,table_b_to_a_cols


def find_transformed_df_opt_with_reusing_data_discovery_ver(sample_proportion,max_steps, df_a, df_b,column_a_name,column_b_name,all_operators,pairs,
                                          edit_dist_threshold,exploration_rate,agreement_percentage,greedy,
                                          transcols_a_b = None,
                                          transcols_b_a = None,
                                          agenta = None,
                                          agentb = None):
   
   # make it at most 20 rows
   min_size = max(int( df_a.shape[0]* sample_proportion),int(df_b.shape[0]* sample_proportion))

   sample_size = max(1, min_size)
   sample_size = min(sample_size,20)

   df_b_sample = min(len(df_b),2*sample_size)

   # Diverse sampling: cluster by string length then sample from each cluster
   if sample_size < df_a.shape[0]:
       join_col = df_a[column_a_name].astype(str) if column_a_name in df_a.columns else df_a.iloc[:, 0].astype(str)
       lengths = join_col.str.len()
       try:
           q33 = lengths.quantile(0.33)
           q66 = lengths.quantile(0.66)
           buckets = pd.cut(lengths, bins=[-1, q33, q66, float('inf')], labels=[0, 1, 2])
           samples_per_bucket = max(1, sample_size // 3)
           sampled_indices = []
           for bucket_id in [0, 1, 2]:
               bucket_rows = df_a.index[buckets == bucket_id].tolist()
               if bucket_rows:
                   n_take = min(samples_per_bucket, len(bucket_rows))
                   sampled_indices.extend(random.sample(bucket_rows, n_take))
           remaining = sample_size - len(sampled_indices)
           if remaining > 0:
               leftover = [i for i in df_a.index if i not in sampled_indices]
               sampled_indices.extend(random.sample(leftover, min(remaining, len(leftover))))
           df_a_sampled = df_a.loc[sampled_indices].reset_index(drop=True)
       except Exception:
           df_a_sampled = df_a.sample(sample_size).reset_index(drop=True)
   else:
       df_a_sampled = df_a.copy()
   df_b_sampled = df_b.sample(df_b_sample).reset_index(drop=True)

   if agenta:
       agent_table_a = agenta

   else:
       agent_table_a = QLearningAgent_edit_dist_modified_for_multi_opt(all_operators,df_a_sampled,agreement_percentage,exploration_rate)

   if agentb:
       agent_table_b = agentb
   else:
       agent_table_b = QLearningAgent_edit_dist_modified_for_multi_opt(all_operators,df_b_sampled,agreement_percentage,exploration_rate)


   trans_a_to_b, table_a_to_b_cols, trans_b_to_a,table_b_to_a_cols, edit_dist_matrix,agent_a,agent_b = optimize_transformations_both_edit_dist_opt_with_reusing(max_steps, df_a_sampled, df_b_sampled,
                                                                                                                                   column_a_name,column_b_name,
                                                                                                                                    agent_table_a,agent_table_b,
                                                                                                                                    pairs, edit_dist_threshold,greedy,
                                                                                                                                    transcols_a_b,transcols_b_a)                                                                                                                                  
   df_a_concated = build_dataframe_in_order(df_a, table_a_to_b_cols)
   df_b_concated = build_dataframe_in_order(df_b, table_b_to_a_cols)
   
   df_a_concated_applied = apply_all_actions_to_df(df_a_concated,trans_a_to_b).astype(str).apply(lambda col: col.str.lower())
   df_b_concated_applied =  apply_all_actions_to_df(df_b_concated,trans_b_to_a).astype(str).apply(lambda col: col.str.lower())

   return df_a_concated_applied,df_b_concated_applied,trans_a_to_b, table_a_to_b_cols, trans_b_to_a,table_b_to_a_cols,agent_a,agent_b



operator_lsts = [all_operators_without_concate_both]

direct_operators_lst = [direct_operators_only]


# --- Runtime parameters (hardcoded in the released fixed config) ---
# All formerly hardcoded — now configurable for optimizer sweeps
exploration_rates = [0.1]
sample_proportions = [1.0]
edit_dist_thresholds = [0.95]
agreement_percentages = [1.0]
agreement_percentages_non_greedy = [2.0]
max_steps = 5
_JACCARD_N = 2  # n-gram size for Jaccard (fixed mode)
_JACCARD_MODE = 'fixed'  # fixed|adaptive|blend|containment
# Parameter-free terminal-similarity metric selector. The released config fixes
# this to idfcos (dispatched at the _jaccard_matrix_as_edit_dist chokepoint). Input-only.
_TERMINAL_METRIC = _METRIC
_SAMPLE_CAP = 50  # max rows for Q-learning
_WRAPPER_STEPS = 3  # steps for pair ranking
_WRAPPER_SAMPLE = 0.1  # sample for pair ranking

pairs = {}


# ===========================================================================
# PUBLIC ENTRY — clean, benchmark-free MNN Join.
#
# Input is two value lists (or two DataFrames + column names); output is the
# deduplicated set of matched (source_value, target_value) pairs. No benchmark
# scaffolding, no ground-truth scoring, no file I/O, no absolute paths. The
# MNN-Join out-of-the-box defaults (idfcos + cov_mnn coverage selection + no
# learned models + safe concat + greedy) are fixed as module constants at the
# TOP of this module, so simply importing it yields MNN-Join behaviour.
# ===========================================================================
from typing import List as _List, Sequence as _Sequence, Tuple as _Tuple
import contextlib as _contextlib
import io as _io


def _selected_mnn(matches, df_matches_out):
    """Mutual-NN coverage (= ``_combo_mutual_nn``) of the SELECTED combo.

    The engine logs each evaluated combo in ``_COMBO_LOG`` (only when
    ``_LOG_COMBOS=1``) together with its sim_matrix. Recover the matrix of
    the combo whose ``df_matches_multi`` is the one returned (by object identity,
    then by content) and apply the engine's truth-free bijectivity proxy. Neutral
    1.0 when unrecoverable — the engine's own convention, so a missing matrix
    never spuriously favours either depth.
    """
    n = len(matches)
    if df_matches_out is None:
        return 1.0, n
    want = id(df_matches_out)
    for e in _COMBO_LOG:
        if id(e.get('df_matches_multi')) == want:
            try:
                return float(_combo_mutual_nn(e['sim_matrix'])), n
            except Exception:
                return 1.0, n
    try:
        for e in _COMBO_LOG:
            dfm = e.get('df_matches_multi')
            if dfm is not None and len(dfm) == n and n > 0:
                return float(_combo_mutual_nn(e['sim_matrix'])), n
    except Exception:
        pass
    return 1.0, n


def _run_depth(df_src, src_col, df_tgt, tgt_col, depth):
    """Run the engine's cov_mnn-selected chain at a fixed transform depth.

    Returns (deduped_match_pairs, selected_combo_mutual_nn). The two columns are
    built from in-memory value lists; the join columns are prefixed so the
    engine's STEP1 column env points resolve.
    """
    src = df_src.copy()
    tgt = df_tgt.copy()
    src.columns = ['source-' + str(c) for c in src.columns]
    tgt.columns = ['target-' + str(c) for c in tgt.columns]
    keyonly_src = 'source-' + str(src_col)
    keyonly_tgt = 'target-' + str(tgt_col)
    global _STEP1_SRC_COL, _STEP1_TGT_COL, _MAX_STEPS_OVERRIDE, _LOG_COMBOS
    _STEP1_SRC_COL = keyonly_src
    _STEP1_TGT_COL = keyonly_tgt
    _MAX_STEPS_OVERRIDE = depth
    _LOG_COMBOS = True

    try:
        set_inside_worker(True)
    except Exception:
        pass
    try:
        _COMBO_LOG.clear()
    except Exception:
        pass

    # exploration rate: the configured eps (MNN Join default '0.0' = greedy; the
    # MetaJoin router may set mnn_join._EPS_SWEEP = '0.5').
    _exploration_rates = [0.1]
    if _EPS_SWEEP:
        _exploration_rates = [float(_EPS_SWEEP)]

    # The engine prints heavily; swallow stdout.
    with _contextlib.redirect_stdout(_io.StringIO()):
        pairs_dict = get_merged_table_multi_to_multi_greedy_with_all_best_result_opt_multi_results_with_policy(
            [src, tgt], operator_lsts, edit_dist_thresholds,
            sample_proportions, agreement_percentages,
            _exploration_rates, depth, 3,
        )

    matches = []
    df_matches_out = None
    _prefer_multi = False
    for pair_key, v in (pairs_dict or {}).items():
        if v is None:
            continue
        try:
            c1, cm = v
        except Exception:
            continue
        _chain_order = ([('multi', cm), ('1to1', c1)] if _prefer_multi
                        else [('1to1', c1), ('multi', cm)])
        for tag, combo in _chain_order:
            if combo is None or len(combo) < 2:
                continue
            df_m = combo[1]
            if df_m is None or len(df_m) == 0:
                continue
            if keyonly_src in df_m.columns and keyonly_tgt in df_m.columns:
                for sv, tv in zip(df_m[keyonly_src].astype(str),
                                  df_m[keyonly_tgt].astype(str)):
                    matches.append((sv.strip(), tv.strip()))
                df_matches_out = df_m
                break
        if matches:
            break

    cand = list(dict.fromkeys(
        (str(a).strip(), str(b).strip()) for a, b in matches
        if str(a).strip() and str(b).strip()))
    mnn, _ = _selected_mnn(matches, df_matches_out)
    return cand, mnn


def mnn_join_tables(df_src, src_col, df_tgt, tgt_col, max_steps: int = 2, seed: int = 0):
    """Run MNN Join on a (src_col) -> (tgt_col) transform join over two tables.

    Returns the deduplicated list of matched ``(source_value, target_value)``
    pairs. Self-terminating depth-PEAK search:
      * always run the 1-step chain;
      * if ``max_steps >= 2``, also run the 2-step chain and keep it ONLY when it
        RAISES the mutual-NN coverage of the selected alignment (label-free peak
        rule); otherwise keep the 1-step spine.

    ``seed`` fixes the RNG so results are reproducible (the engine subsamples
    large columns); pass ``seed=None`` to leave the global RNG state unchanged.
    For bit-identical runs across processes also set ``PYTHONHASHSEED=0``.
    """
    if seed is not None:
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
    cand1, mnn1 = _run_depth(df_src, src_col, df_tgt, tgt_col, 1)
    if max_steps < 2:
        return cand1
    cand2, mnn2 = _run_depth(df_src, src_col, df_tgt, tgt_col, 2)
    # PEAK RULE (truth-free): take the deeper chain iff the 2nd op raised cov_mnn.
    return cand2 if mnn2 > mnn1 else cand1


def mnn_join(src_values, tgt_values, max_steps: int = 2, seed: int = 0):
    """Run MNN Join directly on two value lists (the two join columns).

    Returns the deduplicated list of matched ``(source_value, target_value)``
    pairs. Thin wrapper over :func:`mnn_join_tables` that wraps each list in a
    one-column DataFrame. See :func:`mnn_join_tables` for ``seed``.
    """
    df_src = pd.DataFrame({'key': [str(v) for v in src_values]})
    df_tgt = pd.DataFrame({'key': [str(v) for v in tgt_values]})
    return mnn_join_tables(df_src, 'key', df_tgt, 'key', max_steps=max_steps, seed=seed)