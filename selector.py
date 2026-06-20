"""Threshold-free column-pair selector (certificate tiers).

Given two DataFrames, ``rank_all_pairs`` ranks every (source column, target
column) pair by a logical CERTIFICATE rather than a weighted score. Each pair
gets a tier label:
    HYBRID > LEXICAL > SEMANTIC_RESCUE > NONE
and pairs are sorted lexicographically by (tier_rank, -effective_anchor_count).
There are no tuned thresholds: selection is carried by reciprocal mutual-nearest-
neighbour structure and distinct-anchor counts.

How a pair earns its certificate
---------------------------------
  1. Primary-value filter: a longest-common-substring match counts only when the
     matched span covers enough of the SHORTER cell, rejecting incidental
     long-text mentions.
  2. Local-windowed embedding MNN with hub rejection: encode just the matched
     anchor substrings (not the full row context), apply reciprocal mutual-NN,
     and reject hubs (each side must be the UNIQUE nearest of the other). Uses
     the shared ``_embed`` helper; degrades to the lexical certificate when no
     embedding model is available.
  3. Effective anchor count: distinct surviving anchors per side.
  4. Certificate tiering replaces a scalar score:
       HYBRID          = lexical AND local-embedding evidence both present
       LEXICAL         = lexical evidence only
       SEMANTIC_RESCUE = lexical sparse but local-embedding evidence present
       NONE            = neither

Public API
----------
    rank_all_pairs(df_a, df_b)
        -> [(col_a, col_b, evidence_unique, raw_mnn, local_emb, tier, eff), ...]
"""
from __future__ import annotations
import os
import re
import sys
from difflib import SequenceMatcher
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

MAX_ROWS = 60
# selection is carried by reciprocal mutual-NN + distinct-anchor structure; these are no-op floors
MIN_SIM = 0.0
MIN_CONTIG = 1       # min contiguous common substring (compact-evidence)
MIN_VARIANT_LEN = 3  # variant must be >= 3 chars to be considered
MIN_PRIMARY_FRAC = 0.0  # LCS match must cover >= this fraction of the SHORTER cell

TIER_HYBRID = 0
TIER_LEXICAL = 1
TIER_SEMANTIC_RESCUE = 2
TIER_NONE = 3
_TIER_LABELS = {0: 'HYBRID', 1: 'LEXICAL', 2: 'SEMANTIC_RESCUE', 3: 'NONE'}


# ---------------------------------------------------------------------------
# Cell variants
# ---------------------------------------------------------------------------

_PAREN_RE = re.compile(r'\s*\([^)]*\)\s*')
_COORD_RE = re.compile(r'\d+°[^\s;,]*')


# v3 additions: token closure + residue classification
TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")  # word-like tokens (letters/digits/_/')
TITLECASE_RE = re.compile(r"^[A-Z][a-z']+$|^[A-Z]\.?$")  # word OR initial
NAME_PARTICLES = {'van', 'de', 'del', 'la', 'le', 'of', 'the', 'da', 'di'}
STRONG_DELIM_RE = re.compile(r"[;|\n\r]")  # phrase closure stops at these
SAFE_RESIDUE_RE = re.compile(
    r"^[\s\(\)\[\]\{\}\.,;:!?\-\*'\"/\\\|]*$|"  # pure punctuation
    r"^\s*\(\s*\d+\s*\)\s*$|"                      # "(8)"
    r"^[\s\d°′″NSEWnsew.,;\-+x×*/]+$|"              # coordinates / numeric (case-insensitive directional)
    r"^@[\w\.\-]+$|"                                # email domain "@x.k12.ga.us"
    r"^\.[\w\.\-]+$|"                               # ".k12.ga.us"
    r"^[\.\s]*@[\w\.\-]+$"                          # ".@x.k12.ga.us" (leading dot+@)
)
MAX_LOCAL_GAP = 3       # split ALCS components when char-gap exceeds this
MIN_COMPONENT_LEN = 3
MIN_COMPONENT_DENSITY = 0.6
DANGEROUS_RESIDUE_MIN_WORDS = 4   # >= 4 unmatched word-tokens = dangerous

def _tokenize_with_spans(s: str):
    """Returns list of (token_text, char_start, char_end) for all word-like tokens."""
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(s)]

def _alcs_alignment_path(a: str, b: str):
    """Returns list of (i, j) char-position pairs aligned by SequenceMatcher's
    matching blocks. Each match block of size n contributes n consecutive pairs."""
    sm = SequenceMatcher(None, a, b, autojunk=False)
    path = []
    for blk in sm.get_matching_blocks():
        if blk.size == 0:
            continue
        for k in range(blk.size):
            path.append((blk.a + k, blk.b + k))
    return path

def _split_components(path, max_gap=MAX_LOCAL_GAP):
    """Split alignment path into local components whenever the char-gap on
    EITHER side exceeds max_gap. Filters scattered cross-word LCS noise."""
    if not path:
        return []
    comps = [[path[0]]]
    for prev, curr in zip(path, path[1:]):
        gap_x = curr[0] - prev[0] - 1
        gap_y = curr[1] - prev[1] - 1
        if gap_x > max_gap or gap_y > max_gap:
            comps.append([curr])
        else:
            comps[-1].append(curr)
    return comps

def _component_is_word_like(comp):
    """A component must be long enough and dense enough to be word-like."""
    if len(comp) < MIN_COMPONENT_LEN:
        return False
    span_x = comp[-1][0] - comp[0][0] + 1
    span_y = comp[-1][1] - comp[0][1] + 1
    density = len(comp) / max(span_x, span_y)
    return density >= MIN_COMPONENT_DENSITY

def _supported_char_positions(comps, side='x'):
    """Returns set of char positions that are part of any accepted component."""
    out = set()
    idx = 0 if side == 'x' else 1
    for comp in comps:
        if not _component_is_word_like(comp):
            continue
        for pair in comp:
            out.add(pair[idx])
    return out

def _token_closure(s: str, supported_chars: set):
    """A token is supported if its char-span overlaps the supported-chars set
    by >= 50% OR contains an aligned run >= 2 chars. Returns list of (text,
    start, end, supported)."""
    tokens = _tokenize_with_spans(s)
    out = []
    for tok, st, en in tokens:
        positions = [p for p in supported_chars if st <= p < en]
        if not positions:
            out.append((tok, st, en, False))
            continue
        positions.sort()
        # check max contig run
        run = 1
        max_run = 1
        for a, b in zip(positions, positions[1:]):
            if b == a + 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        frac = len(positions) / max(1, en - st)
        supported = (max_run >= 2) or (frac >= 0.5)
        out.append((tok, st, en, supported))
    return out

def _phrase_closure(tokens, orig_text=None):
    """Expand supported tokens to their containing phrase: a phrase is a
    contiguous run of titlecase/initial tokens or name particles, not split
    by strong delimiters (semicolon/pipe/newline). Tokens already supported
    are kept; their titlecase/particle neighbours become supported too.
    If orig_text is provided, titlecase decisions use the original-cased
    string (since `tokens` may come from a lowercased copy)."""
    if not tokens:
        return tokens
    n = len(tokens)
    supp = [t[3] for t in tokens]
    def _titlecaseish(idx):
        tok = tokens[idx][0]
        if orig_text is not None:
            st, en = tokens[idx][1], tokens[idx][2]
            orig_tok = orig_text[st:en]
        else:
            orig_tok = tok
        return bool(TITLECASE_RE.match(orig_tok)) or tok.lower() in NAME_PARTICLES
    def _gap_blocked(left_idx, right_idx):
        # Strong delimiter (;|\n) in the original-text gap between two adjacent
        # tokens stops phrase closure across it. Audit #1 caught the prior
        # version walking past these.
        if orig_text is None:
            return False
        gap = orig_text[tokens[left_idx][2]:tokens[right_idx][1]]
        return bool(STRONG_DELIM_RE.search(gap))
    for i in range(n):
        if not supp[i]:
            continue
        # expand left
        j = i - 1
        while j >= 0:
            if _gap_blocked(j, j + 1):
                break
            if _titlecaseish(j):
                supp[j] = True
                j -= 1
            else:
                break
        # expand right
        j = i + 1
        while j < n:
            if _gap_blocked(j - 1, j):
                break
            if _titlecaseish(j):
                supp[j] = True
                j += 1
            else:
                break
    return [(t[0], t[1], t[2], supp[i]) for i, t in enumerate(tokens)]

def _render_kept_and_residue(s: str, tokens):
    """Returns (kept_str, residue_str). kept_str = supported tokens joined by
    space. residue_str = the original cell with supported-token chars removed
    (preserves all non-token chars and unsupported tokens — used to classify
    what's left)."""
    kept = ' '.join(t[0] for t in tokens if t[3])
    if not tokens:
        return kept, s
    keep_mask = [True] * len(s)
    for tok, st, en, supported in tokens:
        if supported:
            for k in range(st, en):
                keep_mask[k] = False
    residue = ''.join(c for k, c in enumerate(s) if keep_mask[k]).strip()
    return kept, residue

def _is_dangerous_residue(residue: str) -> bool:
    """Residue is DANGEROUS if it contains >= DANGEROUS_RESIDUE_MIN_WORDS
    word-tokens that are NOT mostly digits/punctuation. Safe residues are
    parens, coords, email domains, single counts."""
    residue = residue.strip()
    if not residue:
        return False
    if SAFE_RESIDUE_RE.match(residue):
        return False
    word_tokens = TOKEN_RE.findall(residue)
    word_tokens = [w for w in word_tokens if not w.isdigit()]
    return len(word_tokens) >= DANGEROUS_RESIDUE_MIN_WORDS

def _normalize_keep_case(s: str) -> str:
    """Same as _normalize but preserves original case (for phrase closure)."""
    return ' '.join(str(s).split())


def _mask_by_alcs(a_orig: str, b_orig: str):
    """Top-level token-closed ALCS masking. Returns
    (kept_a, kept_b, residue_a, residue_b, supported_char_count_x_y)."""
    a_norm = _normalize(a_orig)
    b_norm = _normalize(b_orig)
    if not a_norm or not b_norm:
        return '', '', a_orig, b_orig, 0
    # original-case versions with same whitespace shape as a_norm/b_norm
    a_norm_cased = _normalize_keep_case(a_orig)
    b_norm_cased = _normalize_keep_case(b_orig)
    path = _alcs_alignment_path(a_norm, b_norm)
    comps = _split_components(path)
    sup_x = _supported_char_positions(comps, 'x')
    sup_y = _supported_char_positions(comps, 'y')
    tokens_a = _phrase_closure(_token_closure(a_norm, sup_x), a_norm_cased)
    tokens_b = _phrase_closure(_token_closure(b_norm, sup_y), b_norm_cased)
    kept_a, residue_a = _render_kept_and_residue(a_norm, tokens_a)
    kept_b, residue_b = _render_kept_and_residue(b_norm, tokens_b)
    sup_count = len(sup_x) + len(sup_y)
    return kept_a, kept_b, residue_a, residue_b, sup_count


def _normalize(s: str) -> str:
    return ' '.join(str(s).lower().split())


def cell_variants(s: str) -> List[str]:
    """Conservative deterministic variants. Each variant is a candidate
    'short form' of the cell. Order: raw → paren-stripped → coord-stripped
    → segment splits → email local-part variants. De-duplicated."""
    if s is None:
        return []
    s_str = str(s)
    raw = _normalize(s_str)
    out = [raw]

    # paren stripping: "Alaska (8)" → "alaska"
    no_paren = _normalize(_PAREN_RE.sub(' ', s_str))
    if no_paren and no_paren != raw and len(no_paren) >= MIN_VARIANT_LEN:
        out.append(no_paren)

    # coord/semicolon stripping: "Maine; 44°21′N..." → "maine"
    # Take prefix before first semicolon or coordinate marker.
    cut = re.split(r'[;]|\d+°', s_str, maxsplit=1)[0].strip()
    cut_norm = _normalize(cut)
    if cut_norm and cut_norm != raw and len(cut_norm) >= MIN_VARIANT_LEN:
        out.append(cut_norm)

    # email local-part: "john.smith@district.k12.us" → "john smith", "johnsmith", "jsmith"
    if '@' in s_str:
        local = s_str.split('@', 1)[0]
        out.append(_normalize(local))
        spaced = _normalize(local.replace('.', ' ').replace('_', ' ').replace('-', ' '))
        if spaced != _normalize(local):
            out.append(spaced)
        compact = _normalize(local.replace('.', '').replace('_', '').replace('-', ''))
        if compact and compact != spaced:
            out.append(compact)

    # delimiter segments — only when residue-after-segment looks structured
    # (numeric, parens, or short — guards against long-blurb containment)
    for sep in [' | ', '|', '\n']:
        if sep in s_str:
            for seg in s_str.split(sep):
                seg_norm = _normalize(seg)
                if seg_norm and len(seg_norm) >= MIN_VARIANT_LEN and seg_norm != raw:
                    out.append(seg_norm)

    # Dedupe preserving order
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Cell-pair similarity with compact-evidence requirement
# ---------------------------------------------------------------------------

def _alcs_compact(a: str, b: str) -> Tuple[float, int, str]:
    """v3: token-closed ALCS masking. Returns (sim, supported_char_count,
    anchor_key). anchor_key encodes 'kept_a||kept_b||residue_safe_flag'.
    Returns null tuple if mask is empty or both residues dangerous."""
    kept_a, kept_b, residue_a, residue_b, sup_count = _mask_by_alcs(a, b)
    if not kept_a or not kept_b:
        return 0.0, 0, ''
    if sup_count < MIN_CONTIG:
        return 0.0, 0, ''
    a_norm = _normalize(a); b_norm = _normalize(b)
    if not a_norm or not b_norm:
        return 0.0, 0, ''
    sim = sup_count / (len(a_norm) + len(b_norm))
    # primary-value coverage: kept must cover >= MIN_PRIMARY_FRAC of the
    # SHORTER cell after normalization (so long, safe-residue text doesn't
    # kill an otherwise solid match)
    shorter = min(len(a_norm), len(b_norm))
    if min(len(kept_a), len(kept_b)) < MIN_PRIMARY_FRAC * shorter:
        return 0.0, 0, ''
    danger_a = _is_dangerous_residue(residue_a)
    danger_b = _is_dangerous_residue(residue_b)
    residue_safe = not (danger_a or danger_b)
    flag = '1' if residue_safe else '0'
    anchor_key = f'{kept_a}||{kept_b}||{flag}'
    return float(sim), int(sup_count), anchor_key


def _is_email_shape(vals: Sequence[str], min_frac: float = 0.5) -> bool:
    """True if at least min_frac fraction of vals contain '@'."""
    if not vals: return False
    n_at = sum(1 for v in vals if isinstance(v, str) and '@' in v)
    return (n_at / max(len(vals), 1)) >= min_frac


def _best_variant_match(variants_a: List[str], variants_b: List[str],
                         email_local_only: bool = False
                         ) -> Tuple[float, int, str]:
    """Best (sim, contig, anchor) over all (variant_a, variant_b) cross pairs.
    If email_local_only, restrict to non-raw variants on both sides where
    available — gives email cells a chance to match name-form columns."""
    best = (0.0, 0, '')
    if not variants_a or not variants_b:
        return best
    # Email mode: skip the raw email string for the email side
    if email_local_only:
        # Use variants[1:] if available (skip raw)
        a_use = variants_a[1:] if len(variants_a) > 1 else variants_a
        b_use = variants_b[1:] if len(variants_b) > 1 else variants_b
    else:
        a_use, b_use = variants_a, variants_b
    for va in a_use:
        for vb in b_use:
            sim, contig, anchor = _alcs_compact(va, vb)
            if sim > best[0]:
                best = (sim, contig, anchor)
    return best


# ---------------------------------------------------------------------------
# Pair scoring
# ---------------------------------------------------------------------------

def _build_sim_and_anchors(values_a: Sequence[str], values_b: Sequence[str]
                            ) -> Tuple[np.ndarray, Dict[Tuple[int, int], str]]:
    """Build n_a × n_b sim matrix + anchor key dict for accepted cells."""
    a = list(values_a)[:MAX_ROWS]
    b = list(values_b)[:MAX_ROWS]
    n_a, n_b = len(a), len(b)
    S = np.zeros((n_a, n_b), dtype=np.float32)
    anchors: Dict[Tuple[int, int], str] = {}
    if not a or not b:
        return S, anchors
    # Determine email mode
    a_email = _is_email_shape(a)
    b_email = _is_email_shape(b)
    email_mode = (a_email and not b_email) or (b_email and not a_email)
    # Pre-compute variants per cell
    variants_a = [cell_variants(str(x)) for x in a]
    variants_b = [cell_variants(str(x)) for x in b]
    for i in range(n_a):
        for j in range(n_b):
            sim, contig, anchor = _best_variant_match(
                variants_a[i], variants_b[j], email_local_only=email_mode)
            if sim < MIN_SIM or contig < MIN_CONTIG:
                continue
            S[i, j] = sim
            anchors[(i, j)] = anchor
    return S, anchors


def _compute_metrics(S: np.ndarray, anchors: Dict[Tuple[int, int], str],
                      values_a: Sequence[str], values_b: Sequence[str]
                      ) -> Tuple[int, int, List[Tuple[int, int, str]]]:
    """Returns (raw_mnn_count, evidence_unique_count, mnn_pairs).

    Returns the actual mnn_pairs list so the optional embedding row-context
    rerank step can use them."""
    n_a, n_b = S.shape
    if n_a == 0 or n_b == 0:
        return 0, 0, []
    a2b = S.argmax(axis=1)
    b2a = S.argmax(axis=0)
    a_strict = np.array([(S[i, a2b[i]] > S[i, :]).sum() == n_b - 1
                          for i in range(n_a)])
    b_strict = np.array([(S[b2a[j], j] > S[:, j]).sum() == n_a - 1
                          for j in range(n_b)])
    mnn_pairs: List[Tuple[int, int, str]] = []
    for i in range(n_a):
        j = a2b[i]
        if (a_strict[i] and b_strict[j] and b2a[j] == i
                and S[i, j] >= MIN_SIM and (i, j) in anchors):
            mnn_pairs.append((i, j, anchors[(i, j)]))
    a_strs = [str(x).lower() for x in list(values_a)[:MAX_ROWS]]
    b_strs = [str(x).lower() for x in list(values_b)[:MAX_ROWS]]
    distinct_anchors = set()
    for i, j, anchor in mnn_pairs:
        # v3: anchor is "kept_a||kept_b||safe_flag"; use side-specific kept
        # strings for substring uniqueness checks.
        parts = anchor.split('||')
        kept_a_only = parts[0] if len(parts) >= 1 else ''
        kept_b_only = parts[1] if len(parts) >= 2 else kept_a_only
        a_low = kept_a_only.lower()
        b_low = kept_b_only.lower()
        a_count = sum(1 for s in a_strs if a_low and a_low in s)
        b_count = sum(1 for s in b_strs if b_low and b_low in s)
        if a_count == 1 and b_count == 1:
            distinct_anchors.add(anchor.lower())
    return len(mnn_pairs), len(distinct_anchors), mnn_pairs


def _row_context_str(df: pd.DataFrame, exclude_col: str, row_idx: int) -> str:
    """Concat all OTHER cells in this row (excluding the candidate join col)
    into a single string for SBERT embedding. Limits to first 200 chars per
    cell to keep encoding fast."""
    parts = []
    for c in df.columns:
        if c == exclude_col: continue
        try:
            v = str(df.iloc[row_idx][c])
        except Exception:
            continue
        if v and v != 'nan':
            parts.append(v[:200])
    return ' | '.join(parts) if parts else 'EMPTY'


def _embedding_row_context_mnn(mnn_pairs: List[Tuple[int, int, str]],
                                 df_a: pd.DataFrame, col_a: str,
                                 df_b: pd.DataFrame, col_b: str,
                                 max_pairs: int = 25, min_cos: float = 0.4) -> int:
    """Re-validate ALCS row pairs using SBERT on the REST of each row.
    Returns count of pairs that survive a SECOND mutual-NN check in semantic
    context space. Lower-bounded threshold (0.4) since row contexts are noisy."""
    if len(mnn_pairs) < 2:
        # 0 or 1 pair → can't form mutual-NN; return raw count as "trivially supported"
        return len(mnn_pairs)
    pairs = mnn_pairs[:max_pairs]
    src_idxs = [i for i, _, _ in pairs]
    tgt_idxs = [j for _, j, _ in pairs]
    ctx_a = [_row_context_str(df_a, col_a, i) for i in src_idxs]
    ctx_b = [_row_context_str(df_b, col_b, j) for j in tgt_idxs]
    try:
        from _embed import encode_cached
        emb_a = encode_cached(ctx_a)
        emb_b = encode_cached(ctx_b)
        if emb_a.shape[1] < 2 or emb_b.shape[1] < 2:
            return 0
        S = (emb_a @ emb_b.T).astype(np.float32)
    except Exception:
        return 0
    n = len(pairs)
    a2b = S.argmax(axis=1)
    b2a = S.argmax(axis=0)
    cnt = 0
    for i in range(n):
        j = int(a2b[i])
        if b2a[j] == i and float(S[i, j]) >= min_cos:
            cnt += 1
    return cnt


def _local_anchor_embedding_mnn(mnn_pairs, max_pairs=30, min_cos=0.0):
    if len(mnn_pairs) < 2:
        return len(mnn_pairs)
    pairs = mnn_pairs[:max_pairs]
    anchors_a, anchors_b, safe_flags = [], [], []
    for _, _, anchor in pairs:
        parts = anchor.split('||')
        ka = parts[0] if len(parts) >= 1 else ''
        kb = parts[1] if len(parts) >= 2 else ka
        flag = parts[2] if len(parts) >= 3 else '1'
        anchors_a.append(ka)
        anchors_b.append(kb)
        safe_flags.append(flag == '1')
    try:
        from _embed import encode_cached
        emb_a = encode_cached(anchors_a)
        emb_b = encode_cached(anchors_b)
        if emb_a.shape[1] < 2 or emb_b.shape[1] < 2:
            return 0
        S = (emb_a @ emb_b.T).astype(np.float32)
    except Exception:
        return 0
    n = len(pairs)
    a2b = S.argmax(axis=1)
    b2a = S.argmax(axis=0)
    a2b_counts = np.bincount(a2b, minlength=n)
    b2a_counts = np.bincount(b2a, minlength=n)
    cnt = 0
    for i in range(n):
        j = int(a2b[i])
        # require reciprocal MNN, hub-free, above min_cos, AND residue-safe
        if (b2a[j] == i and a2b_counts[j] == 1 and b2a_counts[i] == 1
                and float(S[i, j]) >= min_cos
                and safe_flags[i] and safe_flags[j]):
            cnt += 1
    return cnt


def alcs_v2_score(values_a: Sequence[str], values_b: Sequence[str]
                    ) -> Tuple[int, int, List[Tuple[int, int, str]]]:
    """Returns (raw_mnn, evidence_unique, mnn_pairs). Lex-sort by (ev, mnn)."""
    S, anchors = _build_sim_and_anchors(values_a, values_b)
    mnn, ev, pairs = _compute_metrics(S, anchors, values_a, values_b)
    return mnn, ev, pairs


def rank_all_pairs(df_a: pd.DataFrame, df_b: pd.DataFrame
                    ) -> List[Tuple[str, str, int, int, int, int, int]]:
    """LGSA certificate-tier ranker.

    Returns [(col_a, col_b, evidence_unique, raw_mnn, local_emb_count,
              tier, effective_count), ...]
    sorted by (tier, -effective_count, -ev, -mnn) ascending on tier.

    Tier semantics:
      0 HYBRID          ev > 0 AND local_emb > 0
      1 LEXICAL         ev > 0
      2 SEMANTIC_RESCUE ev == 0 AND mnn < 3 AND local_emb > 0
      3 NONE            otherwise
    Effective count:
      HYBRID          -> min(ev, local_emb)
      LEXICAL         -> ev
      SEMANTIC_RESCUE -> local_emb
      NONE            -> 0
    """
    pair_data: List[Tuple[str, str, int, int, List[Tuple[int, int, str]]]] = []
    for ca in df_a.columns:
        va = df_a[ca].astype(str).tolist()
        for cb in df_b.columns:
            vb = df_b[cb].astype(str).tolist()
            mnn, ev, pairs = alcs_v2_score(va, vb)
            pair_data.append((ca, cb, ev, mnn, pairs))

    out: List[Tuple[str, str, int, int, int, int, int]] = []
    for ca, cb, ev, mnn, pairs in pair_data:
        # Local-embedding step is ALWAYS on in LGSA (no env gate).
        if pairs:
            local_emb = _local_anchor_embedding_mnn(pairs)
        else:
            local_emb = 0
        if ev > 0 and local_emb > 0:
            tier = TIER_HYBRID
            eff = min(ev, local_emb)
        elif ev > 0:
            tier = TIER_LEXICAL
            eff = ev
        elif ev == 0 and mnn < 3 and local_emb > 0:
            tier = TIER_SEMANTIC_RESCUE
            eff = local_emb
        else:
            tier = TIER_NONE
            eff = 0
        out.append((ca, cb, ev, mnn, local_emb, tier, eff))

    # Lex sort: tier ascending, then effective_count, ev, mnn descending
    out.sort(key=lambda x: (x[5], -x[6], -x[2], -x[3]))
    return out
