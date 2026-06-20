"""Minimal embedding helper (MiniLM / sentence-transformers, lazy + degrades).

Merges the former embedding-similarity module and the per-process embedding
cache into one file. Loads ``sentence-transformers/all-MiniLM-L6-v2`` once
(lazily), L2-normalizes and caches vectors per string, and offers a couple of
small similarity helpers. Everything degrades gracefully when
sentence-transformers / the weights are unavailable (a TF-IDF + SVD fallback is
used for the column-pair similarity helpers; the cache returns a zero matrix so
callers can detect the no-SBERT case via ``rank < 2``).

Public surface used by the shipped package
-------------------------------------------
    sbert_model()                 -> the loaded SentenceTransformer (or fallback)
    encode_cached(strings)        -> (n, D) L2-normalized cached embeddings
    embedding_cos_sim(a, b)       -> n_a x n_b cosine matrix (or None)
    compute_embedding_similarity  -> (mean_row_max, full_matrix)  [engine hook]
    embedding_pair_score          -> ALCS-guided column-pair score [engine hook]

Used by ``selector``'s local-embedding tier and ``metajoin``'s embedding tool,
and referenced (behind try/except) by the off-by-default embedding branches of
the ``mnn_join`` engine.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------
_MODEL = None
_MODEL_TYPE = None  # 'sentence_transformer' or 'tfidf'


def _get_model():
    """Load the embedding model (singleton, lazy). Returns (model, model_type)."""
    global _MODEL, _MODEL_TYPE
    if _MODEL is not None:
        return _MODEL, _MODEL_TYPE

    # Prefer sentence-transformers (all-MiniLM-L6-v2, CPU).
    try:
        from sentence_transformers import SentenceTransformer
        import logging
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        _MODEL = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        _MODEL_TYPE = 'sentence_transformer'
        return _MODEL, _MODEL_TYPE
    except Exception:
        pass

    # Fallback marker — the column-pair helpers below build a TF-IDF + SVD
    # embedding on the fly; the cache treats this as "no SBERT".
    _MODEL = 'tfidf'
    _MODEL_TYPE = 'tfidf'
    return _MODEL, _MODEL_TYPE


def sbert_model():
    """Return the loaded SentenceTransformer (raises if unavailable).

    The cost-aware router's embedding tool calls this directly; callers that
    must degrade gracefully should use :func:`encode_cached` /
    :func:`embedding_cos_sim` instead, which fall back to a zero matrix / None.
    """
    from sentence_transformers import SentenceTransformer  # noqa: F401  (import error surfaced to caller)
    model, model_type = _get_model()
    if model_type != 'sentence_transformer':
        raise RuntimeError('sentence-transformers/all-MiniLM-L6-v2 is unavailable')
    return model


def get_embedding_model():
    """Public wrapper returning just the model object (not the type tuple)."""
    model, _ = _get_model()
    return model


# ---------------------------------------------------------------------------
# Per-process embedding cache (key = raw string, value = L2-normalized vector)
# ---------------------------------------------------------------------------
_EMB_VEC_CACHE: dict[str, np.ndarray] = {}
_CACHE_MAX = 200_000


def encode_cached(strings: list[str]) -> np.ndarray:
    """Return (len(strings), D) matrix of L2-normalized embeddings.

    Only strings not already cached are encoded. If SBERT is unavailable the
    returned matrix has width 1 (a zero column) so callers can detect the
    no-embedding case via ``result.shape[1] < 2``.
    """
    model, mtype = _get_model()
    if mtype != 'sentence_transformer':
        return np.zeros((len(strings), 1), dtype=np.float32)

    requested = set(strings)
    missing = [s for s in requested if s not in _EMB_VEC_CACHE]
    if missing:
        # Evict BEFORE inserting so we never drop a vector this call needs;
        # never evict a key in the current request.
        overflow = len(_EMB_VEC_CACHE) + len(missing) - _CACHE_MAX
        if overflow > 0:
            evicted = 0
            for k in list(_EMB_VEC_CACHE.keys()):
                if k in requested:
                    continue
                _EMB_VEC_CACHE.pop(k, None)
                evicted += 1
                if evicted >= overflow:
                    break
        vecs = model.encode(missing, show_progress_bar=False, batch_size=64)
        for s, v in zip(missing, vecs):
            v = np.asarray(v, dtype=np.float32)
            v /= max(float(np.linalg.norm(v)), 1e-8)
            _EMB_VEC_CACHE[s] = v

    return np.stack([_EMB_VEC_CACHE[s] for s in strings], axis=0)


def embedding_cos_sim(vals_a: list[str], vals_b: list[str]) -> np.ndarray | None:
    """n_a x n_b cosine similarity on cached L2-normalized embeddings.
    Returns None when SBERT is unavailable or either side is empty."""
    if not vals_a or not vals_b:
        return None
    va = encode_cached([str(v) for v in vals_a])
    vb = encode_cached([str(v) for v in vals_b])
    if va.shape[1] < 2:
        return None
    return (va @ vb.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Column-pair similarity helpers (engine hooks; off-by-default paths only).
# These build a TF-IDF + SVD embedding when SBERT is absent so they always
# return a value.
# ---------------------------------------------------------------------------

def build_embeddings(values_a, values_b, n_components=32, ngram_range=(2, 4)):
    """Build (embeddings_a, embeddings_b) for two lists of strings."""
    model, model_type = _get_model()

    if model_type == 'sentence_transformer':
        all_vals = list(values_a) + list(values_b)
        n_a = len(values_a)
        emb = model.encode(all_vals, show_progress_bar=False, batch_size=64)
        return emb[:n_a], emb[n_a:]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    all_vals = list(values_a) + list(values_b)
    n_a = len(values_a)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range,
                                 max_features=5000, lowercase=True)
    tfidf = vectorizer.fit_transform(all_vals)
    n_comp = min(n_components, tfidf.shape[1] - 1, len(all_vals) - 1)
    if n_comp < 2:
        emb = tfidf.toarray()
        return emb[:n_a], emb[n_a:]
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    emb = svd.fit_transform(tfidf)
    return emb[:n_a], emb[n_a:]


def compute_embedding_similarity(values_a, values_b, **kwargs):
    """Return (mean_of_row_max_cosine, full_cosine_matrix)."""
    emb_a, emb_b = build_embeddings(values_a, values_b, **kwargs)
    sim_matrix = cosine_similarity(emb_a, emb_b)
    row_max = np.max(sim_matrix, axis=1)
    return float(np.mean(row_max)), sim_matrix


def embedding_pair_score(col_a_vals, col_b_vals, sample_n=20, top_k_alcs=10):
    """ALCS-guided embedding score for column-pair ranking. For each source
    value, find top-k targets by cheap char overlap, then embed and take the
    max cosine. Mean over sources. 0.0 on any failure."""
    vals_a = [str(v).lower() for v in col_a_vals[:sample_n]]
    vals_b = [str(v).lower() for v in col_b_vals]
    if not vals_a or not vals_b:
        return 0.0
    try:
        def _char_overlap(s1, s2):
            set1, set2 = set(s1), set(s2)
            if not set1 or not set2:
                return 0.0
            return len(set1 & set2) / len(set1 | set2)

        candidates_per_source = []
        for a in vals_a:
            overlaps = [(j, _char_overlap(a, b)) for j, b in enumerate(vals_b)]
            overlaps.sort(key=lambda x: x[1], reverse=True)
            top_j = [vals_b[j] for j, _ in overlaps[:top_k_alcs]]
            candidates_per_source.append((a, top_j))

        all_strings = []
        source_indices = []
        for a, cands in candidates_per_source:
            start = len(all_strings)
            all_strings.append(a)
            all_strings.extend(cands)
            end = len(all_strings)
            source_indices.append((start, start + 1, end))
        if len(all_strings) < 3:
            return 0.0

        model, model_type = _get_model()
        if model_type == 'sentence_transformer':
            emb = model.encode(all_strings, show_progress_bar=False, batch_size=64)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                                         max_features=5000, lowercase=True)
            tfidf = vectorizer.fit_transform(all_strings)
            n_comp = min(32, tfidf.shape[1] - 1, len(all_strings) - 1)
            emb = tfidf.toarray() if n_comp < 2 else TruncatedSVD(
                n_components=n_comp, random_state=42).fit_transform(tfidf)

        scores = []
        for src_start, cand_start, cand_end in source_indices:
            src_emb = emb[src_start:src_start + 1]
            cand_emb = emb[cand_start:cand_end]
            if len(cand_emb) == 0:
                scores.append(0.0)
                continue
            cos_sims = cosine_similarity(src_emb, cand_emb)[0]
            scores.append(float(np.max(cos_sims)))
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0
