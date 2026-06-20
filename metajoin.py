#!/usr/bin/env python
"""MetaJoin — label-free, cost-aware join router.

Given a (source column -> target column) pair, MetaJoin routes the join to ONE
tool chosen from TRUTH-FREE signals — equi-join, MNN Join (transform), embedding
(SBERT top-1 + similarity cut), or entity matching — and accepts that tool's
output on a class-appropriate witness (model-free exact-overlap coverage for the
transform/exact classes; sampled pair-judging otherwise), escalating only when
the evidence is insufficient. ROUTING IS ALWAYS SIGNAL-DRIVEN; the LLM never
drives it.

Public entry
------------
    metajoin(df_src, src_col, df_tgt, tgt_col, return_trace=False)
        -> list[(source_value, target_value)]   (or (pairs, trace) with the flag)

Default = SIGNAL-ONLY policy: deterministic, no endpoint, no prior, no truth.
Setting ``METAJOIN_LLM=1`` (plus an OpenAI-compatible endpoint via
``VLLM_ENDPOINT`` / ``CTRL_ENDPOINT``) enables the optional LLM mode (content
probe + acceptance judge) via the ReAct controller in ``react()``. The OpenAI
client is imported lazily, so the default path needs no openai package, no
endpoint, and no network.

This module is self-contained: the small SBERT model-loader and the
verify / normalise / scoring helpers it needs are folded in below."""
import os, sys, json, glob, re, contextlib, io, time
# Make sibling flat modules (mnn_join, _embed) importable when run from the repo
# root.
_HERE=os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0,_HERE)
# The MNN Join engine hardcodes its fixed config (idfcos + covmnn + cov_mnn +
# safe-concat + no learned models). MetaJoin applies its own ROUTER overrides
# (a non-greedy eps and the dominance-admission second pass) as engine module
# constants immediately after importing it -- see below.
# Default to the ZERO-LLM signal-routing policy: the released MetaJoin runs
# deterministically with no LLM endpoint. The LLM controller path stays reachable
# by setting METAJOIN_LLM=1 (which clears these). AGENT_SIG_ONLY = signal rules
# pick the tool; AGENT_NO_LLM = execute the recommended action with no controller
# call; AGENT_SELECTIVE = the per-tool feedback loop (no force-all).
if os.environ.get('METAJOIN_LLM','0')!='1':
    os.environ.setdefault('AGENT_NO_LLM','1'); os.environ.setdefault('AGENT_SIG_ONLY','1'); os.environ.setdefault('AGENT_SELECTIVE','1')
import numpy as np, pandas as pd
# Benchmark-only scaffolding (reads benchmark CSVs / truth) is NOT shipped.
# tool_mnn calls mnn_join on value lists; R is only touched by the legacy
# benchmark react()/main() paths, which guard for R is None.
try:
    import v0_ablation_runner as R
except Exception:
    R=None
import mnn_join as _mnn_engine
from mnn_join import mnn_join as _mnn_join_values
# MetaJoin router overrides over the engine's standalone defaults:
# a non-greedy exploration rate + the dominance-admission 2nd pass
# (tau=1.0, floor=0.0). Set as engine constants, no environment.
_mnn_engine._EPS_SWEEP = '0.5'
_mnn_engine._DOMINANCE_ADMIT = True
_mnn_engine._DOMINANCE_TAU = 1.0
_mnn_engine._DOMINANCE_FLOOR = 0.0
from collections import Counter

# ---------------------------------------------------------------------------
# Folded-in helpers (previously the separate sbert-tool + scoring/accept
# modules). The released MetaJoin only needs this small surface from them; the
# benchmark CLIs and the LOTUS column selector that used to live alongside are
# not shipped. ``A``/``L`` are aliased to this module so the existing
# ``A.foo`` / ``L.foo`` references throughout resolve here.
# ---------------------------------------------------------------------------
_SBERT=None
def _sbert_model():
    """Lazily load all-MiniLM-L6-v2 (CPU). Raises if sentence-transformers /
    the weights are unavailable — callers on the signal-only path guard for it
    (route_decision falls back to lexical signals when SBERT is missing)."""
    global _SBERT
    if _SBERT is None:
        from sentence_transformers import SentenceTransformer
        _SBERT=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    return _SBERT

_nrm=lambda v: re.sub(r'\s+',' ',str(v)).strip().lower()
VMODEL=os.environ.get('VLLM_MODEL','qwen3-8b-base')
_ep=None; client=None
def _get_client():
    """Lazily build the OpenAI client against VLLM_ENDPOINT. Returns None when no
    endpoint is configured, so verify() degrades to a no-op False vote (the
    signal-only default never calls this)."""
    global _ep, client
    if client is not None:
        return client
    _ep=os.environ.get('VLLM_ENDPOINT')
    if not _ep:
        return None
    from openai import OpenAI
    client=OpenAI(api_key='EMPTY',base_url=_ep)
    return client
_FS=('Decide if SOURCE and TARGET are the SAME record, possibly after ANY string transformation (initials, '
 'abbreviation, reorder, reformat, casing, extract/concatenate parts, expand/contract). True if derivable OR same '
 'entity; False only if clearly unrelated.\n\nSOURCE: "John Smith"  TARGET: "jsmith"  ANSWER: True\n'
 'SOURCE: "2020-01-15"  TARGET: "01/15/2020"  ANSWER: True\nSOURCE: "Apple Inc."  TARGET: "Microsoft"  ANSWER: False\n'
 'SOURCE: "Hebei Province"  TARGET: "Hebei"  ANSWER: True\nSOURCE: "Beijing"  TARGET: "Shanghai"  ANSWER: False\n')
def verify(a,b):   # one LLM True/False; returns (verdict, in_tok, out_tok). Retries so a blip isn't a false no-join.
    cl=_get_client()
    if cl is None:        # no LLM endpoint configured -> no-op False vote (zero tokens)
        return False,0,0
    for att in range(6):
        try:
            r=cl.completions.create(model=VMODEL,prompt=_FS+f'SOURCE: "{a}"  TARGET: "{b}"  ANSWER:',temperature=0.0,max_tokens=3,stop=['\n'])
            t=(r.choices[0].text or '').strip().lower(); return (t.startswith('true') or t.startswith('yes')), r.usage.prompt_tokens, r.usage.completion_tokens
        except Exception: time.sleep(1.0+0.5*att)
    return False,0,0
def _row_truth(name,s,t):
    """Benchmark-only: build the row-level ground-truth pair set from
    'ground truth.csv'. Returns an empty set when the benchmark data root (R) is
    unavailable — the signal-only path never calls this."""
    try: gt=pd.read_csv(f'{R.DATA_ROOTS[BM]}/{name}/ground truth.csv',dtype=str,keep_default_na=False)
    except Exception: return set()
    if any(c.startswith('source-') for c in gt.columns):
        sc=[(c,c[7:]) for c in gt.columns if c.startswith('source-') and c[7:] in s.columns]; tc=[(c,c[7:]) for c in gt.columns if c.startswith('target-') and c[7:] in t.columns]
    else: sc=[(c,c) for c in gt.columns if c in s.columns]; tc=[(c,c) for c in gt.columns if c in t.columns]
    def idx(df,cols):
        m={}
        for i,r in df.iterrows(): m.setdefault(tuple(_nrm(r[c]) for c in cols),[]).append(i)
        return m
    def build(sk,tk,sg,tg):
        sm=idx(s,sk); tm=idx(t,tk); T=set()
        for _,g in gt.iterrows():
            a=tuple(_nrm(g[c]) for c in sg); b=tuple(_nrm(g[c]) for c in tg)
            for i in sm.get(a,[]):
                for j in tm.get(b,[]): T.add((i,j))
        return T
    T=set()
    if sc and tc:
        T=build([x for _,x in sc],[x for _,x in tc],[c for c,_ in sc],[c for c,_ in tc])
        if not T:
            sb=max(sc,key=lambda p:s[p[1]].nunique()); tb=max(tc,key=lambda p:t[p[1]].nunique()); T=build([sb[1]],[tb[1]],[sb[0]],[tb[0]])
    return T
def score_rows(pairs,name,s,t,sc,tc,truth):
    """Benchmark-only F1 of predicted pairs vs the row-level truth set."""
    if not truth or sc not in s.columns or tc not in t.columns: return float('nan')
    s2r={}; [s2r.setdefault(_nrm(v),[]).append(i) for i,v in enumerate(s[sc])]
    t2r={}; [t2r.setdefault(_nrm(v),[]).append(j) for j,v in enumerate(t[tc])]
    pred=set()
    for a,b in pairs:
        for i in s2r.get(_nrm(a),[]):
            for j in t2r.get(_nrm(b),[]): pred.add((i,j))
    tp=len(pred&truth); pr=tp/len(pred) if pred else 0; rc=tp/len(truth) if truth else 0
    return 2*pr*rc/(pr+rc) if (pr+rc) else 0.0
# self-alias: existing A./L. references in this module resolve to the folded-in helpers.
A=sys.modules[__name__]; L=sys.modules[__name__]

BM=os.environ.get('BM','flashfill'); BUDGET=int(os.environ.get('AGENT_BUDGET','-1'))
# DTT = cached transform-by-example pairs (benchmark-only 3rd voter). Loads from
# DTT_PAIRS if present, else empty -> tool_dtt yields nothing. Never needed on
# the signal-only default path.
_DTT_PATH=os.environ.get('DTT_PAIRS',f'out_put_csv/dtt_pairs_qjsel_{BM}.json')
_DTT=json.load(open(_DTT_PATH)) if os.path.exists(_DTT_PATH) else {}
MAXSTEP=int(os.environ.get('MAXSTEP','8')); CTRL=os.environ.get('CTRL_MODEL','qwen3-8b-instruct')
MAX_N=1500   # refuse MNN Join's slow transform search above this table size
SELECTIVE=bool(os.environ.get('AGENT_SELECTIVE'))       # AGENT_SELECTIVE=1: gate the force-all backstop -> genuine
#   signal-driven feedback-loop tool selection (controller's tool choices + early-finish actually stick; only the prior-
#   recommended tool is run as a fallback if the controller ran nothing). Default (unset) = original run-all-and-fuse agent.
_VERIFY_ACCEPT=os.environ.get('AGENT_VERIFY_ACCEPT','1')!='0'   # QUALITY GATE: sample-verify a tool's joined pairs before committing; reject+escalate if low correct-rate
_VK=int(os.environ.get('VERIFY_K','6'))                 # how many joined pairs to sample-verify on accept
_VACC_TAU=float(os.environ.get('VERIFY_ACCEPT_TAU','0.5'))   # min fraction of sampled pairs that must verify correct to commit
_MEM_PATH=os.environ.get('AGENT_MEM_PATH','out_put_csv/agent_tool_memory.csv')   # online memory: (signals, tool, verified_quality) -> augments prior tool-picking
_USE_MEM=os.environ.get('AGENT_USE_MEM','1')!='0'
_NO_LLM=bool(os.environ.get('AGENT_NO_LLM'))   # AGENT_NO_LLM=1 (+SELECTIVE): execute the signal-computed RECOMMENDED action directly, NO controller LLM call -> pure signal-rules router (the LLM-vs-rules ablation)
_SIG_ONLY=bool(os.environ.get('AGENT_SIG_ONLY'))   # ZERO-LABEL SIGNALS-ONLY strawman (pair with AGENT_NO_LLM=1, all LLM flags off): route by the scout-legend rules (no prior, no catalog/probe), rule-accept sbert (no judge) -> no model and no supervision ANYWHERE in the agent; measures "can statistics alone replace the model layers?"
_FUSE_UNION=bool(os.environ.get('AGENT_FUSE_UNION'))   # run-all only: accept the UNION of all tools' pairs (truth-free, no verify) -> the precision-dilution baseline (you can't pick the right tool without truth)
_ESC=float(os.environ.get('AGENT_ESC_MARGIN','0') or 0)   # SELECTIVE: when prior top1-top2 margin < ESC, DEFER the first accept, run the runner-up too, sample-verify both, commit the better (label-free near-tie escalation; the cost/accuracy knob)
_BACKFILL=bool(os.environ.get('AGENT_BACKFILL'))   # V2 recall lever: after commit, propose matches for UNCOVERED source rows from the next-best tool; LLM-judge a sample; commit on majority
_CASCADE=bool(os.environ.get('AGENT_CASCADE'))   # V2 ZERO-OFFLINE-LABELS arm: ignore the learned prior entirely; run tools cheapest-first (equi->dtt->sbert->mnn[->jellyfish]) and let the runtime LLM judge accept/reject each output (use with AGENT_VERIFY_ACCEPT=1). No offline supervision anywhere.
_TYPE_PROBE=bool(os.environ.get('AGENT_TYPE_PROBE'))   # V2 CONTENT probe: the LLM READS aligned sample value pairs and classifies the join type (exact/transform/semantic/entity) -- understanding the signals can't carry; if it disagrees with the prior's pick, arm a judged comparison (no thresholds)
_PROBE_ROUTE=bool(os.environ.get('AGENT_PROBE_ROUTE'))   # V2 FLAGSHIP (zero truth-derived labels ANYWHERE): the type-probe picks the FIRST tool (content); judge + escalation + backfill do the rest. The learned prior is never consulted for routing.
_TOOL_SELECT=bool(os.environ.get('AGENT_TOOL_SELECT'))   # V2 AGENTIC selection (user design, zero labels): the model reads the TOOL CATALOG (capabilities + costs) + raw column samples and judges which KIND of tool the task needs under a minimize-cost constraint, naming PRIMARY + BACKUP; the judge-compare arbitrates them.
_EXP_ROUTE=bool(os.environ.get('AGENT_EXP_ROUTE'))   # V2 EXPERIENCE routing (user design, zero labels, "active-learning"-style): route by the agent's OWN accumulated experience (kNN over judge-verified tool quality from PAST tasks, prequential); where experience is thin, fall back to catalog tool-select + judged trials -- and RECORD the judged outcomes, so picking improves over time. No truth labels anywhere.
_TOOLCOST={'equi':0,'dtt':1,'sbert':2,'mnn':3,'jellyfish':4}   # cost ranks for tie-breaking (cheap first)
_XFORM_TOOLS=('run_mnn','run_dtt','run_equi')   # exact-evidence class: every proposed pair is an EXACT match under an explicit program -> evidence = overlap coverage, no pair-judging
_DISTILL_ROUTES=os.environ.get('DISTILL_ROUTES','')   # V2: csv (bm,dataset,tool) -> route the FIRST tool by the DISTILLED local-model router instead of the kNN prior
_DR={}
if _DISTILL_ROUTES:
    try:
        _drd=pd.read_csv(_DISTILL_ROUTES); _DR={(str(r.bm),str(r.dataset)):str(r.tool) for r in _drd.itertuples()}
    except Exception: pass
_CHAT_EXTRA={'chat_template_kwargs':{'enable_thinking':False}}
# controller (instruct) and verifier (base) live on SEPARATE endpoints: ctrl_client -> instruct, A._get_client() -> base.
# Both are LAZY: the controller client is built only when an LLM action actually fires (METAJOIN_LLM=1 mode), so import
# needs NO openai package install, NO endpoint, NO network.
ctrl_client=None
def _ctrl_client():
    global ctrl_client
    if ctrl_client is not None: return ctrl_client
    _ep=os.environ.get('CTRL_ENDPOINT') or os.environ.get('VLLM_ENDPOINT')
    if not _ep: raise RuntimeError('LLM controller mode requires CTRL_ENDPOINT or VLLM_ENDPOINT (set METAJOIN_LLM=1 + an endpoint).')
    from openai import OpenAI
    ctrl_client=OpenAI(api_key='EMPTY',base_url=_ep)
    return ctrl_client

# ---------- PRIOR over past tasks (validated k-NN; leave-this-task-out at eval) ----------
# OPTIONAL: the flagship zero-LLM signal-routing policy does NOT consult the prior.
# If a prior CSV is present (PRIOR_PATH), the learned-prior routing path is enabled;
# otherwise PRI=None and prior_lookup returns a neutral recommendation.
FCOLS=['exact_sig','char_jac','tok_jac','len_ratio','log_n','uniq']
try:
    PRI=pd.read_csv(os.environ.get('PRIOR_PATH','out_put_csv/prior_retrieval.csv'))
except Exception:
    PRI=None
_RECCOLS=['equi_f1','qj','sb','dtt']; _TOOLN=['equi','mnn','sbert','dtt']
if PRI is not None:
    _PX=PRI[FCOLS].values.astype(float); _MU=_PX.mean(0); _SD=_PX.std(0)+1e-9; _PXn=(_PX-_MU)/_SD
    if 'jelly' in PRI.columns: _RECCOLS.append('jelly'); _TOOLN.append('jellyfish')
    _REC=PRI[_RECCOLS].values; _KEY={(r.bm,r.dataset):i for i,r in PRI.iterrows()}
else:
    import numpy as _np0
    _PX=_np0.zeros((0,len(FCOLS))); _MU=_np0.zeros(len(FCOLS)); _SD=_np0.ones(len(FCOLS)); _PXn=_PX
    _REC=_np0.zeros((0,len(_RECCOLS))); _KEY={}
_LOGO=bool(os.environ.get('AGENT_LOGO'))   # leave-one-GROUP-out: also drop same-benchmark NEAR-DUPLICATE siblings from the prior (truth-free firewall against sibling leakage; audit: bikes/bikes-small/...)
def _stem(nm):
    import re as _re
    s=_re.sub(r'[-_. ]*\d+$','',str(nm)); s=_re.sub(r'(-(small|long|repeat|tiny|big|short))+$','',s)
    return s.rstrip('-_. ')
if PRI is not None:
    _PBM=PRI['bm'].astype(str).values; _PSTEM=np.array([_stem(n) for n in PRI['dataset'].astype(str).values])
else:
    _PBM=np.array([],dtype=object); _PSTEM=np.array([],dtype=object)
# ---------- ONLINE MEMORY: (signals, tool, verified_quality) accumulated as the agent runs; augments tool-picking (truth-free) ----------
_MEM={'rows':[]}
if _USE_MEM:
    try: _MEM['rows']=pd.read_csv(_MEM_PATH).to_dict('records')
    except Exception: pass
def mem_lookup(feats,K=12):
    rows=_MEM['rows']
    if not rows: return {}
    X=np.array([[float(r.get(c,0.0)) for c in FCOLS] for r in rows]); Xn=(X-_MU)/_SD
    x=(np.array([feats[c] for c in FCOLS])-_MU)/_SD; d2=((Xn-x)**2).sum(1); nb=np.argsort(d2)[:min(K,len(rows))]
    out={}
    for t in ['equi','mnn','sbert','dtt','jellyfish']:
        qs=[float(rows[i]['quality']) for i in nb if str(rows[i].get('tool'))==t]
        if qs: out[t]=(float(np.mean(qs)),len(qs))
    ef=[float(rows[i].get('ntools',1) or 1) for i in nb if rows[i].get('ntools')==rows[i].get('ntools')]
    if ef: out['_effort']=(float(np.mean(ef)),len(ef))   # how many tools similar past tasks ENDED UP needing (plan hint)
    out['_d2']=float(np.mean(d2[nb]))                    # how CLOSE the experience actually is (z-space): gate cross-domain pollution
    return out
def mem_record(bm,name,feats,tool,quality,cov,ntools=None):
    r={c:round(float(feats[c]),4) for c in FCOLS}; r.update(bm=bm,dataset=name,tool=tool,quality=round(float(quality),3),coverage=round(float(cov),3),ntools=ntools)
    _MEM['rows'].append(r)
    try: pd.DataFrame(_MEM['rows']).to_csv(_MEM_PATH,index=False)
    except Exception: pass
# class-aware escalation order (v3): when a tool FAILS the judge, try its CLASS sibling first, then SWITCH class.
_CLASS_NEXT={'mnn':['dtt','sbert','jellyfish','equi'],'dtt':['mnn','sbert','jellyfish','equi'],
             'sbert':['jellyfish','mnn','dtt','equi'],'jellyfish':['sbert','dtt','mnn','equi'],
             'equi':['dtt','mnn','sbert','jellyfish']}
def prior_lookup(bm,name,feats,K=15):
    if PRI is None or _PXn.shape[0]==0:   # no learned prior -> neutral recommendation (the flagship zero-LLM router ignores this anyway)
        tools=list(_TOOLN); return dict(zip(tools,[0.0]*len(tools)), rec='mnn', mnn_poor=False, mem_used=False)
    x=(np.array([feats[c] for c in FCOLS])-_MU)/_SD
    d2=((_PXn-x)**2).sum(1)
    if (bm,name) in _KEY: d2[_KEY[(bm,name)]]=1e18
    if _LOGO:                                               # exclude same-bm siblings (near-identical signals OR shared name-stem) -> no sibling leakage
        _sib=(_PBM==str(bm)) & ((d2<0.05) | (_PSTEM==_stem(name)))
        d2[_sib]=1e18
    nb=np.argsort(d2)[:K]; mr=_REC[nb].mean(0).astype(float); tools=list(_TOOLN)
    _mu=False
    if _USE_MEM and _MEM['rows']:                            # blend accumulated VERIFIED quality on similar-signal tasks into the prior
        mq=mem_lookup(feats,K=12)
        for ti,t in enumerate(tools):
            if t in mq and mq[t][1]>=2:
                w=min(0.8, mq[t][1]/8.0); mr[ti]=(1-w)*mr[ti]+w*mq[t][0]; _mu=True
    return dict(zip(tools,[round(float(v),3) for v in mr]), rec=tools[int(mr.argmax())], mnn_poor=bool(mr[1]<0.6), mem_used=_mu)

# ---------- truth-free signals & quality ----------
_nrm=A._nrm
def signals(s,t,sc,tc):
    sv=[x.strip() for x in s[sc] if x.strip()]; tv=[x.strip() for x in t[tc] if x.strip()]
    sn={_nrm(x) for x in sv}; tn={_nrm(x) for x in tv}
    def ng(vals,n=3):
        S=set()
        for v in vals: x=_nrm(v); S.add(x) if len(x)<n else S.update(x[i:i+n] for i in range(len(x)-n+1))
        return S
    A_,B_=ng(sv),ng(tv); ls=np.mean([len(x) for x in sv]); lt=np.mean([len(x) for x in tv])
    _dig=sum(c.isdigit() for v in sv for c in v)/max(1,sum(len(v) for v in sv))            # numeric/phone/date transform -> mnn/dtt
    _mw=float(np.mean([1.0 if len(_nrm(v).split())>1 else 0.0 for v in sv])) if sv else 0.0  # multi-word records -> entity-ish (EM/jellyfish)
    _atk=float(np.mean([len(v.split()) for v in sv])) if sv else 0.0   # RECORD-NESS: multi-field records run 30-52 toks/val vs <=4.6 for names/strings (bimodal, 6x margin)
    return dict(exact_sig=len(sn&tn)/max(1,len(sn)), char_jac=len(A_&B_)/max(1,len(A_|B_)),
                tok_jac=len({w for v in sv for w in _nrm(v).split()}&{w for v in tv for w in _nrm(v).split()})/
                        max(1,len({w for v in sv for w in _nrm(v).split()}|{w for v in tv for w in _nrm(v).split()})),
                len_ratio=min(ls,lt)/max(1e-6,max(ls,lt)), log_n=float(np.log10(len(sv)+1)), uniq=len(sn)/max(1,len(sv)),
                digit_frac=round(_dig,3), multiword=round(_mw,3), card_ratio=round(min(len(tn)/max(1,len(sn)),9.99),3),
                avg_toks=round(_atk,2), avg_chars=round(float(ls),1))
def bijectivity(pairs):
    if not pairs: return 0.0
    return len({b for _,b in pairs})/len(pairs)            # 1.0 = perfectly 1-to-1 (truth-free)
def coverage(pairs,sv): return len({a for a,_ in pairs}&set(sv))/max(1,len(sv))

# ---------- TOOLS (return pairs + truth-free quality) ----------
def tool_equi(s,t,sc,tc,sv,tv):
    tn={}; [tn.setdefault(_nrm(x),x) for x in tv]
    return {(v,tn[_nrm(v)]) for v in sv if _nrm(v) in tn}
def tool_mnn(sv,tv):
    # MNN Join: label-free CPU transform-join over the two value lists
    # (idfcos + mutual-NN coverage + self-terminating depth-PEAK search).
    if not sv or not tv: return set()
    try:
        with contextlib.redirect_stdout(io.StringIO()): m=_mnn_join_values(list(sv),list(tv))
        return {(str(a).strip(),str(b).strip()) for a,b in m if str(a).strip() and str(b).strip()}
    except Exception: return set()
def tool_sbert(sv,tv):
    if not sv or not tv: return set(),0.0,[]
    mdl=L._sbert_model()
    try: mdl=mdl.to('cpu')
    except Exception: pass
    se=mdl.encode(sv,show_progress_bar=False,batch_size=128); te=mdl.encode(tv,show_progress_bar=False,batch_size=128)
    se=se/(np.linalg.norm(se,axis=1,keepdims=True)+1e-8); te=te/(np.linalg.norm(te,axis=1,keepdims=True)+1e-8)
    S=se@te.T; top1=S.argmax(axis=1)
    scored=sorted([(sv[i],tv[int(top1[i])],float(S[i,int(top1[i])])) for i in range(len(sv))],key=lambda x:-x[2])
    sim=float(np.mean([x[2] for x in scored])) if scored else 0.0
    return {(a,b) for a,b,_ in scored}, sim, scored
def tool_dtt(name): return {(str(p[0]).strip(),str(p[1]).strip()) for p in A._DTT.get(name,[]) if p and len(p)>=2}
_JELLY_EP=os.environ.get('JELLY_ENDPOINT'); _jelly_cli=[None]
def _jelly_rediscover():
    """Re-read the Jellyfish endpoint from the environment (recovery hook if the
    pinned endpoint goes dead mid-run). Returns JELLY_ENDPOINT, or None if unset."""
    return os.environ.get('JELLY_ENDPOINT')
def _jelly():
    if _jelly_cli[0] is None and _JELLY_EP:
        from openai import OpenAI as _OA; _jelly_cli[0]=_OA(api_key='EMPTY',base_url=_JELLY_EP)
    return _jelly_cli[0]
def tool_jellyfish(sv,tv,K=5):                              # off-the-shelf EM matcher (no labels): SBERT-block top-K + Jellyfish per-pair Yes/No
    cl=_jelly()
    if not cl or not sv or not tv: return set()
    mdl=L._sbert_model()
    try: mdl=mdl.to('cpu')
    except Exception: pass
    se=mdl.encode(sv,show_progress_bar=False,batch_size=128); te=mdl.encode(tv,show_progress_bar=False,batch_size=128)
    se=se/(np.linalg.norm(se,axis=1,keepdims=True)+1e-8); te=te/(np.linalg.norm(te,axis=1,keepdims=True)+1e-8)
    S=se@te.T; kk=min(K,len(tv)); topk=np.argsort(-S,axis=1)[:,:kk]
    def _emp(a,b): return ("You are tasked with determining whether two records refer to the same real-world entity based on the attributes "
        "provided. Carefully compare the attributes of each record before deciding. Note: missing values should not be the basis of your "
        f"decision.\nRecord A: [{a}]\nRecord B: [{b}]\nAre Record A and Record B the same entity? Choose your answer from: [Yes, No].")
    def _judge(a,b):
        for _ in range(3):
            try:
                r=_jelly_cli[0].chat.completions.create(model=os.environ.get('JELLY_MODEL','jellyfish'),messages=[{'role':'user','content':_emp(a,b)}],temperature=0.0,max_tokens=4)
                return (r.choices[0].message.content or '').strip().lower().startswith('yes')
            except Exception: pass
        return None                                          # None = ENDPOINT FAILURE (distinct from a genuine 'No')
    from concurrent.futures import ThreadPoolExecutor
    jobs=[(i,int(j)) for i in range(len(sv)) for j in topk[i]]
    for _attempt in range(2):
        with ThreadPoolExecutor(max_workers=12) as ex:
            verds=list(ex.map(lambda ij:(ij[0],ij[1],_judge(sv[ij[0]],tv[ij[1]])),jobs))
        _fail=sum(1 for _,_,y in verds if y is None)
        if _fail<=len(jobs)//2: break
        # >half the judgments errored -> the pinned server is dead. Rediscover the newest live endpoint and retry ONCE.
        _new=_jelly_rediscover()
        if not _new: break
        from openai import OpenAI as _OA; _jelly_cli[0]=_OA(api_key='EMPTY',base_url=_new)
        print(f"    [jelly-rediscover] endpoint dead ({_fail}/{len(jobs)} errors) -> switched to {_new}",flush=True)
    return {(sv[i],tv[j]) for i,j,y in verds if y}
def sim_summary(scored):                                    # show the LLM the distribution + examples + tau->count preview
    if not scored: return ''
    sims=[sm for _,_,sm in scored]
    pct={f'p{p}':round(float(np.percentile(sims,p)),2) for p in (99,90,75,50,25,10)}
    ex=[scored[0], scored[len(scored)//2], scored[-1]]     # highest, median, lowest similarity pair
    exs=' ; '.join(f"sim={sm:.2f} {s_[:20]!r}->{t_[:20]!r}" for s_,t_,sm in ex)
    prev=' '.join(f"{tau}:{sum(1 for x in sims if x>=tau)}" for tau in (0.9,0.8,0.7,0.6,0.5))
    return f"sim_pctl={pct} | examples[hi/mid/lo]: {exs} | threshold_preview(tau:#pairs)= {prev}"

def dist_scout(sv,tv):
    """Universal DISTRIBUTION signals up front (truth-free) so the controller can route the FIRST tool:
    exact-match fraction + SBERT top-1 sim percentiles + decision margin (top1-top2). Returns (dict, scored);
    scored is cached so a later run_sbert reuses this same SBERT pass (no double compute). Join-type tells:
    high exact_frac->equi | high sim_p50 & wide margin->semantic(sbert) | low sim but structural->transform(mnn)."""
    sn={_nrm(x) for x in sv}; tn={_nrm(x) for x in tv}
    exact_frac=len(sn&tn)/max(1,len(sn))
    if not sv or not tv: return {'exact_frac':round(exact_frac,3)}, []
    mdl=L._sbert_model()
    try: mdl=mdl.to('cpu')
    except Exception: pass
    se=mdl.encode(sv,show_progress_bar=False,batch_size=128); te=mdl.encode(tv,show_progress_bar=False,batch_size=128)
    se=se/(np.linalg.norm(se,axis=1,keepdims=True)+1e-8); te=te/(np.linalg.norm(te,axis=1,keepdims=True)+1e-8)
    S=se@te.T
    if S.shape[1]>=2:
        idx=np.argpartition(-S,1,axis=1)[:,:2]; t2=np.sort(S[np.arange(S.shape[0])[:,None],idx],axis=1)  # [2nd,1st]
        top1=t2[:,1]; margin=float(np.mean(t2[:,1]-t2[:,0]))
    else:
        top1=S.max(1); margin=float('nan')
    a1=S.argmax(1); scored=sorted([(sv[i],tv[int(a1[i])],float(S[i,int(a1[i])])) for i in range(len(sv))],key=lambda x:-x[2])
    pct={f'sim_p{p}':round(float(np.percentile(top1,p)),2) for p in (90,50,10)}
    d=dict(exact_frac=round(exact_frac,3),sbert_margin=(round(margin,3) if margin==margin else margin),sbert_mean=round(float(top1.mean()),3),**pct)
    return d, scored

# ---------- LLM controller ----------
SYS=("You are a cost-aware JOIN AGENT. Goal: join SOURCE.col to TARGET.col by choosing tools, spending the LEAST "
 "cost for a high-coverage, high-precision join.\n"
 "TOOLS & COST: probe(free) | prior(free; past-task tool recommendation) | run_equi(free,instant; exact matches, "
 "auto-accepted) | run_sbert(free,fast; semantic top-1 + similarity distribution) | run_mnn(free but SLOW transform "
 "search; good for transforms, BAD for semantic) | run_dtt(LLM tokens; hard transforms) | run_jellyfish(LLM; off-the-shelf "
 "ENTITY-MATCHING model; for entity joins where source/target are records that differ but refer to the same entity — "
 "multi-word records, low exact_frac, low char/transform overlap) | accept(commit ALL of the last "
 "tool's pairs) | threshold{tau}(keep only sbert pairs with similarity>=tau) | verify{k}(LLM tokens) | "
 "fuse{k}(COMBINE all tools run so far: pairs >=2 tools agree on are auto-accepted, disagreements are LLM-decided; "
 "use after running 2+ tools to merge them precisely) | finish.\n"
 "STRATEGY: (1) The opening message ALREADY has signals + prior. TRUST the prior: run the highest expected_recall tool. "
 "(2) If prior mnn_poor=True (semantic), SKIP slow run_mnn; use run_sbert. (3) ACCEPTANCE depends on join type: for "
 "a 1-to-1 TRANSFORM join (bijectivity~1.0, every source has a match) ACCEPT all pairs. For a SPARSE ENTITY join "
 "(bijectivity LOW, many sources but few targets -> most sources have NO match) accepting every top-1 is WRONG (low "
 "precision); instead READ the run_sbert similarity distribution (sim_pctl, examples, threshold_preview) and call "
 "threshold{\"tau\":X} at the cut that separates true matches (high sim) from forced non-matches (low sim). (4) The "
 "verifier is UNRELIABLE on entity pairs -- prefer threshold over verify for semantic joins. (5) pairs >=2 tools agree "
 "on are auto-accepted. (6) finish as soon as the join looks good -- do not keep adding tools.\n"
 "Reply ONLY with JSON: {\"thought\":\"...\",\"action\":\"<one action>\",\"args\":{}}.")
if SELECTIVE:
    SYS+=("\nSELECTIVE MODE: ONLY the tools you actually run are used — there is NO automatic fallback that runs the "
     "others. So PICK well. The opening message has 'distribution' + signals (exact_frac, sbert sim percentiles, sbert_margin, "
     "digit_frac, multiword, card_ratio): high exact_frac -> run_equi; high digit_frac (phone/date/numeric) or high char/token "
     "overlap but low sbert -> run_mnn (transform); high sim_p50 with wide sbert_margin -> run_sbert (semantic); ENTITY-ish "
     "(multiword high + low exact_frac + low char overlap, e.g. names/orgs that differ but are the same entity) -> run_jellyfish. "
     "FEEDBACK LOOP: run ONE tool, READ its coverage + bijectivity + "
     "quality; if coverage is high and quality is strong, ACCEPT and finish. Escalate ONLY if coverage or confidence is "
     "low: run a SECOND tool, then call fuse to COMBINE them (e.g. mnn+sbert, or mnn+lotus via the verifier) — fuse "
     "auto-accepts pairs both tools agree on and LLM-decides the rest. Prefer ACCEPT (1 tool) on easy joins; reserve "
     "fuse for hard/ambiguous ones. Minimize tools+cost while keeping the join high-coverage and precise. "
     "IMPORTANT: the STATE includes a 'RECOMMENDED ... ACTION' computed from the data (signals/coverage/similarity) — "
     "COPY that action exactly unless a signal clearly contradicts it; it encodes the correct routing/accept/threshold/fuse/finish choice.")
_CTRL_COMPLETIONS=bool(os.environ.get('CTRL_COMPLETIONS'))   # qwen3-8b-base controller -> completions + few-shot (base+chat unreliable)
_REACT_FS=(  # few-shot: the STATE shows a RECOMMENDED ... ACTION computed from the data -> COPY it (deviate only with a strong reason)
 'STATE: ... RECOMMENDED FIRST ACTION (start here ...): {"thought":"trust the prior","action":"run_mnn","args":{}}\n'
 'JSON: {"thought":"follow the recommended first action","action":"run_mnn","args":{}}\n\n'
 'STATE: ...run_mnn: coverage=1.00 ... -> TRANSFORM tool ... || RECOMMENDED NEXT ACTION: {"action":"accept","args":{}}\n'
 'JSON: {"thought":"transform tool, full coverage -> accept","action":"accept","args":{}}\n\n'
 'STATE: ...run_equi: proposed 0 pairs ... || RECOMMENDED NEXT ACTION: {"action":"run_mnn","args":{}}\n'
 'JSON: {"thought":"equi found nothing -> escalate to mnn","action":"run_mnn","args":{}}\n\n'
 'STATE: ...run_sbert: ... low-sim tail ... || RECOMMENDED NEXT ACTION: {"action":"threshold","args":{"tau":0.6}}\n'
 'JSON: {"thought":"sparse low-sim tail -> threshold","action":"threshold","args":{"tau":0.6}}\n\n'
 'STATE: ...2 tools run ... || RECOMMENDED NEXT ACTION: {"action":"fuse","args":{}}\n'
 'JSON: {"thought":"two tools ran -> fuse them","action":"fuse","args":{}}\n\n'
 'STATE: ...coverage_acc=1.00 ... || RECOMMENDED NEXT ACTION: {"action":"finish","args":{}}\n'
 'JSON: {"thought":"full coverage -> done","action":"finish","args":{}}\n\n')
def llm_act(state):
    for att in range(4):
        try:
            if _CTRL_COMPLETIONS:                            # base: completions + few-shot
                r=_ctrl_client().completions.create(model=CTRL,prompt=SYS+"\n\nExamples:\n"+_REACT_FS+"STATE: "+state+"\nJSON:",
                    temperature=0.0,max_tokens=120,stop=['\n\n'])
                txt=r.choices[0].text or ''
            else:                                            # instruct: chat
                r=_ctrl_client().chat.completions.create(model=CTRL,messages=[{'role':'system','content':SYS},
                    {'role':'user','content':state}],temperature=0.0,max_tokens=160,extra_body=_CHAT_EXTRA)
                txt=r.choices[0].message.content or ''
            m=re.search(r'\{.*\}',txt,re.S)
            if m:
                o=json.loads(m.group(0)); return o.get('action','finish'), o.get('args',{}) or {}, o.get('thought',''), r.usage
        except Exception: pass
    return 'PARSEFAIL',{},'(parse-fail)',None

def llm_threshold(name, sv, scored):
    """ReAct micro-decision for SPARSE entity joins: LLM reads the sbert similarity distribution + examples and
    picks the cut tau that separates true matches from forced non-matches. Returns (tau, usage)."""
    if not scored: return float('nan'), None
    prompt=("SPARSE entity-matching join: most source rows have NO match, so accepting every top-1 is wrong. "
            f"Here is the SBERT top-1 similarity distribution for {name}. Pick a similarity threshold tau (0..1) that "
            "KEEPS true matches (high sim) and DROPS forced non-matches (low sim). Reply ONLY the number.\n"
            f"{sim_summary(scored)}\nTAU:")
    for _ in range(4):
        try:
            if _CTRL_COMPLETIONS:
                r=_ctrl_client().completions.create(model=CTRL,prompt=prompt,temperature=0.0,max_tokens=6,stop=['\n'])
                _t=r.choices[0].text or ''
            else:
                r=_ctrl_client().chat.completions.create(model=CTRL,messages=[{'role':'user','content':prompt}],temperature=0.0,max_tokens=6,extra_body=_CHAT_EXTRA)
                _t=r.choices[0].message.content or ''
            m=re.search(r'0?\.\d+',_t)
            if m: return float(m.group(0)), r.usage
        except Exception: pass
    return 0.7, None

_TP_MAP={'exact':'equi','transform':'mnn','semantic':'sbert','entity':'jellyfish'}
def type_probe(scored,sv,tv):
    """CONTENT understanding (what signals can't carry): the local model READS raw samples of BOTH columns
    (UNALIGNED -- alignments from the scout are wrong exactly on hard transform tasks) and names the join type."""
    s_ex='; '.join(repr(x[:40]) for x in sv[:5]); t_ex='; '.join(repr(x[:40]) for x in tv[:5])
    fs=("SOURCE: '2020-01-01'; '1999-03-04'; '2007-11-30'\nTARGET: 'Mar 4, 1999'; 'Jan 1, 2020'; 'Nov 30, 2007'\nTYPE: transform\n\n"
        "SOURCE: 'NYC'; 'UK'; 'LA'\nTARGET: 'United Kingdom'; 'New York City'; 'Los Angeles'\nTYPE: semantic\n\n"
        "SOURCE: 'iPhone 12 64GB black | Apple | 799'; 'Galaxy S21 | Samsung | 699'\nTARGET: 'Apple iPhone12 (64 GB) Black | smartphone'; 'Samsung Galaxy S-21 5G'\nTYPE: entity\n\n"
        "SOURCE: 'alpha'; 'beta'; 'gamma'\nTARGET: 'beta'; 'alpha'; 'gamma'\nTYPE: exact\n\n"
        "SOURCE: 'John Smith'; 'Mary Jones'\nTARGET: 'J.S.'; 'M.J.'\nTYPE: transform\n\n"
        "SOURCE: '(607) 255-4321'; '(415) 555-0199'\nTARGET: '6072554321'; '4155550199'\nTYPE: transform\n\n")
    prompt=("Two table columns are to be joined. From RAW SAMPLES of each column (not aligned), classify how matching "
            "values relate: exact (identical strings), transform (one column is a systematic syntactic reformatting of "
            "the other), semantic (paraphrase/synonym), entity (records differing in fields/wording but describing the "
            "same real-world entity).\n\n"+fs+f"SOURCE: {s_ex}\nTARGET: {t_ex}\nTYPE:")
    for _ in range(3):
        try:
            r=_ctrl_client().completions.create(model=CTRL,prompt=prompt,temperature=0.0,max_tokens=4,stop=['\n'])
            t=(r.choices[0].text or '').strip().lower()
            for k in _TP_MAP:
                if k in t: return k, r.usage
        except Exception: pass
    return None, None

def _sig_gloss(feats,scout):
    """Render the label-free signals WITH plain-language glosses so the selector can reason over statistics
    it could not compute itself (user design: 'this is jaccard sim and the sim is ..., then let the agent think')."""
    if not feats: return ''
    g=[]
    e=feats['exact_sig']; g.append(f"exact-match={e:.2f}"+(" (values never match exactly)" if e<0.05 else " (mostly identical values)" if e>0.5 else ""))
    c=feats['char_jac']; g.append(f"char-3gram-jaccard={c:.2f}"+(" (almost no shared character patterns)" if c<0.1 else " (strong character overlap: likely reformatting)" if c>0.3 else ""))
    g.append(f"token-jaccard={feats['tok_jac']:.2f}")
    d=feats['digit_frac']; g.append(f"digit-fraction={d:.2f}"+(" (numeric/date/code heavy)" if d>0.3 else ""))
    a=feats.get('avg_toks',0); g.append(f"avg-tokens-per-value={a}"+(" (multi-field records)" if a>=8 else ""))
    if scout:
        p50=scout.get('sim_p50'); mg=scout.get('sbert_margin')
        if p50 is not None: g.append(f"embedding-top1-sim-p50={p50}"+(" (embeddings align these well)" if isinstance(p50,(int,float)) and p50>=0.6 else " (embeddings cannot align these)" if isinstance(p50,(int,float)) and p50<0.4 else ""))
        if mg is not None and mg==mg: g.append(f"embedding-margin={mg}")
    return "SIGNALS: "+"; ".join(g)+"\n"

def tool_select(sv,tv,feats=None,scout=None):
    """AGENTIC tool selection (zero labels): the model reads the TOOL CATALOG (capability + cost descriptions),
    RAW SAMPLES of both columns, AND the glossed signals, and judges which kind of tool this join needs under a
    minimize-cost constraint. Returns (primary, backup, usage). Extensible: a new tool = a new catalog line."""
    s_ex='; '.join(repr(x[:40]) for x in sv[:5]); t_ex='; '.join(repr(x[:40]) for x in tv[:5])
    cat=("TOOLS:\n"
         "- equi: exact hash join (free, instant). Matches identical strings only.\n"
         "- mnn: transformation-search join (CPU-heavy search). Discovers a systematic syntactic transformation "
         "(substring, split/concat, reorder, date/code/number reformatting) mapping source values onto target values, then joins exactly.\n"
         "- dtt: model-based transformation join (LLM cost). Learns hard or irregular transformations that operator search misses.\n"
         "- sbert: embedding semantic join (fast, cheap). Matches values that are paraphrases, synonyms, or abbreviations of the same meaning.\n"
         "- jellyfish: entity-matching model (LLM cost per candidate pair). Judges whether two multi-field records describe the same real-world entity.\n")
    fs=("SOURCE: '2020-01-01'; '1999-03-04'\nTARGET: 'Jan 1, 2020'; 'Mar 4, 1999'\n"
        "SIGNALS: exact-match=0.00 (values never match exactly); char-3gram-jaccard=0.06 (almost no shared character patterns); digit-fraction=0.52 (numeric/date/code heavy); embedding-top1-sim-p50=0.44\n"
        "PRIMARY: mnn BACKUP: dtt\n\n"
        "SOURCE: 'NYC'; 'UK'\nTARGET: 'New York City'; 'United Kingdom'\n"
        "SIGNALS: exact-match=0.00 (values never match exactly); char-3gram-jaccard=0.09; embedding-top1-sim-p50=0.78 (embeddings align these well); embedding-margin=0.21\n"
        "PRIMARY: sbert BACKUP: jellyfish\n\n"
        "SOURCE: 'iPhone 12 64GB black | Apple | 799'; 'Galaxy S21 | Samsung | 699'\nTARGET: 'Apple iPhone12 (64 GB) Black | smartphone'\n"
        "SIGNALS: exact-match=0.00; avg-tokens-per-value=9.5 (multi-field records); embedding-top1-sim-p50=0.62\n"
        "PRIMARY: jellyfish BACKUP: sbert\n\n"
        "SOURCE: 'alpha'; 'beta'; 'gamma'\nTARGET: 'beta'; 'gamma'; 'alpha'\n"
        "SIGNALS: exact-match=1.00 (mostly identical values)\n"
        "PRIMARY: equi BACKUP: sbert\n\n"
        "SOURCE: 'John Smith'; 'Mary Jones'\nTARGET: 'J.S.'; 'M.J.'\n"
        "SIGNALS: exact-match=0.00; char-3gram-jaccard=0.04 (almost no shared character patterns); embedding-top1-sim-p50=0.33 (embeddings cannot align these)\n"
        "PRIMARY: mnn BACKUP: dtt\n\n"
        "SOURCE: '(607) 255-4321'; '(415) 555-0199'\nTARGET: '6072554321'; '4155550199'\n"
        "SIGNALS: exact-match=0.00; char-3gram-jaccard=0.41 (strong character overlap: likely reformatting); digit-fraction=0.83 (numeric/date/code heavy)\n"
        "PRIMARY: mnn BACKUP: dtt\n\n")
    sig=_sig_gloss(feats,scout)
    prompt=("You must join two table columns. Read the tool catalog, the RAW SAMPLES of each column (unaligned), and "
            "the measured SIGNALS (each explained), then judge which KIND of tool this join needs. CONSTRAINT: "
            "minimize cost -- pick the cheapest tool whose capability matches the evidence; also name a BACKUP of a "
            "plausible alternative kind.\n\n"
            +cat+"\n"+fs+f"SOURCE: {s_ex}\nTARGET: {t_ex}\n{sig}PRIMARY:")
    for _ in range(3):
        try:
            r=_ctrl_client().completions.create(model=CTRL,prompt=prompt,temperature=0.0,max_tokens=12,stop=['\n'])
            t_=(r.choices[0].text or '').strip().lower()
            m=re.findall(r'(equi|mnn|dtt|sbert|jellyfish)',t_)
            if m: return m[0],(m[1] if len(m)>1 else None),r.usage
        except Exception: pass
    return None,None,None


# ===========================================================================
# PUBLIC ENTRY — clean, benchmark-free MetaJoin (zero-LLM signal router).
# Runs the deployed AGENT_SIG_ONLY scout-legend routing rules end-to-end on two
# in-memory DataFrames: compute truth-free signals + SBERT distribution scout,
# route to ONE tool by the legend rules (equi / MNN-Join / embedding / entity),
# run it, and rule-accept its pairs (heuristic similarity cut for the embedding
# tool, accept-all for the exact-evidence tools). No prior, no LLM, no truth.
# ===========================================================================
def route_decision(sv, tv):
    """Return (tool_name, signals_dict, scout_dict) for the zero-LLM router.

    ``tool_name`` is one of 'equi' | 'mnn' | 'sbert' | 'jellyfish'. Mirrors the
    AGENT_SIG_ONLY legend rules in react() exactly (jellyfish is only chosen when
    a JELLY_ENDPOINT is configured; otherwise its branches fall through to the
    transform/embedding tools)."""
    # signals() reads s[sc]/t[tc]; build minimal 1-col frames so we can reuse it.
    s = pd.DataFrame({'k': [str(x) for x in sv]})
    t = pd.DataFrame({'k': [str(x) for x in tv]})
    feats = signals(s, t, 'k', 'k')
    try:
        ds, scored = dist_scout([str(x) for x in sv], [str(x) for x in tv])
    except Exception:
        # No SBERT weights available (offline + uncached) -> route on lexical
        # signals only (exact_frac from the data; the sbert branches go quiet).
        _sn = {_nrm(x) for x in sv}; _tn = {_nrm(x) for x in tv}
        ds = {'exact_frac': round(len(_sn & _tn) / max(1, len(_sn)), 3)}
    _p50 = ds.get('sim_p50', 0.0); _mgn = ds.get('sbert_margin', float('nan'))
    _jelly_on = _jelly() is not None
    if ds.get('exact_frac', 0) > 0.5:
        tool = 'equi'
    elif (feats['avg_toks'] >= 8 or feats['avg_chars'] >= 60) and _jelly_on:
        tool = 'jellyfish'
    elif _p50 >= 0.6 and not (_mgn == _mgn and _mgn < 0.10):
        tool = 'sbert'
    elif feats['digit_frac'] > 0.3 or feats['char_jac'] > 0.3:
        tool = 'mnn'
    elif feats['multiword'] > 0.8 and feats['exact_sig'] < 0.05 and feats['char_jac'] < 0.15 and _jelly_on:
        tool = 'jellyfish'
    else:
        tool = 'mnn'
    if tool == 'mnn' and len(sv) > MAX_N:   # cost guard (not a quality judgment): too large for the slow transform search
        tool = 'sbert'
    return tool, feats, ds


def metajoin(df_src, src_col, df_tgt, tgt_col, return_trace=False):
    """Label-free cost-aware MetaJoin over a (src_col)->(tgt_col) column pair.

    Routes to ONE tool by the zero-LLM signal rules and returns the accepted
    ``(source_value, target_value)`` match pairs (a list). With
    ``return_trace=True`` returns ``(pairs, trace_dict)`` where the trace records
    the routed tool + signals. Deterministic; no LLM, no endpoint, no prior."""
    sv = list(dict.fromkeys([str(x).strip() for x in df_src[src_col] if str(x).strip()]))
    tv = list(dict.fromkeys([str(x).strip() for x in df_tgt[tgt_col] if str(x).strip()]))
    if not sv or not tv:
        return ([], {'tool': None, 'reason': 'empty column'}) if return_trace else []
    tool, feats, ds = route_decision(sv, tv)
    if tool == 'equi':
        pairs = tool_equi(None, None, None, None, sv, tv)
    elif tool == 'mnn':
        pairs = tool_mnn(sv, tv)
    elif tool == 'jellyfish':
        pairs = tool_jellyfish(sv, tv)
    else:  # sbert (embedding): SIGNALS-ONLY rule-accept (faithful to react()'s
           # _SIG_ONLY path) — commit the top-1 pairs; no judge, no threshold.
        P, _sim, _scored = tool_sbert(sv, tv)
        pairs = set(P)
    pairs = sorted(pairs)
    if return_trace:
        return pairs, {'tool': tool, 'n_pairs': len(pairs),
                       'signals': {k: feats[k] for k in ('exact_sig', 'char_jac', 'digit_frac', 'multiword', 'avg_toks')},
                       'scout': {k: ds.get(k) for k in ('exact_frac', 'sim_p50', 'sbert_margin')},
                       'coverage': round(coverage(pairs, sv), 3), 'bijectivity': round(bijectivity(pairs), 3)}
    return pairs


def react(name, art, budget):
    sc,tc=art['picked_pair']['src_col'],art['picked_pair']['tgt_col']
    s=pd.read_csv(f'{R.DATA_ROOTS[BM]}/{name}/source.csv',dtype=str,keep_default_na=False)
    t=pd.read_csv(f'{R.DATA_ROOTS[BM]}/{name}/target.csv',dtype=str,keep_default_na=False)
    sv=list(dict.fromkeys([x.strip() for x in s[sc] if x.strip()])); tv=list(dict.fromkeys([x.strip() for x in t[tc] if x.strip()]))
    truth=A._row_truth(name,s,t)
    feats=signals(s,t,sc,tc); votes=Counter(); ran=set(); calls=0; tin=tout=0; trace=[]
    accepted=set(); last_pairs=[set()]; sbert_scored=[[]]; chosen_tau=[float('nan')]  # accepted only GROWS; holders
    last_tool=['']; bad_tools=set(); tool_q={}              # last tool run; tools that FAILED the accept-verify gate; {tool: verified-quality} for the memory
    rec_holder=[{'action':'finish','args':{}}]              # NO-LLM mode: the current signal-computed RECOMMENDED action to execute (= what the LLM is told to copy)
    _esc={'armed':False,'second':None,'stash':None}         # near-tie escalation state (AGENT_ESC_MARGIN)
    _rej_best=[None]                                        # best gate-rejected proposal (coverage, pairs): NEVER end a task empty
    def add_consensus():
        accepted.update(p for p,c in votes.items() if c>=2)   # >=2 tools agree -> free-accept
    pl=prior_lookup(BM,name,feats)                          # pre-seed probe + prior so routing info is always present
    obs=(f"TASK {BM}/{name}: |source|={len(sv)} |target|={len(tv)}. sample_src={sv[:3]} sample_tgt={tv[:3]}.\n"
         f"signals={ {k:round(v,3) for k,v in feats.items()} }\n"
         f"prior(past tasks): expected_recall={ {k:pl[k] for k in ['equi','mnn','sbert','dtt']} } recommend={pl['rec']} mnn_poor={pl['mnn_poor']}")
    if SELECTIVE:                                            # universal DISTRIBUTION up front for routing (caches the sbert pass)
        _ds,_scs=dist_scout(sv,tv); sbert_scored[0]=_scs
        obs+=f"\ndistribution={_ds}  (route: high exact_frac->run_equi; high digit_frac/char-overlap->run_mnn; high sim_p50+wide sbert_margin->run_sbert; multiword+low exact+low char-overlap (entity)->run_jellyfish)"
        # ONE routing mechanism for ALL tools (v2 unified prior): rank every available candidate by predicted
        # quality; argmax routes; if the top-2 are within the escalation margin, run both and let the judge decide.
        _cands=[c for c in _TOOLN if c!='jellyfish' or _jelly() is not None]
        _ordv=sorted(_cands,key=lambda c:-pl[c])
        _r1='run_'+_ordv[0]
        if _CASCADE:                                          # zero-offline-labels: cheapest-first, judge-gated (prior unused for routing)
            _ordv=[c for c in ['equi','dtt','sbert','mnn','jellyfish'] if c in _cands]
            _r1='run_equi'
        if _DR.get((BM,name)) in ('equi','sbert','mnn','dtt','jellyfish'):
            _r1='run_'+_DR[(BM,name)]                        # DISTILLED ROUTER: the local model (signals+samples) picks the first tool
        if _r1=='run_mnn' and len(sv)>MAX_N: _r1='run_sbert'   # cost guard, not a quality judgment
        if 'jellyfish' not in _TOOLN and _jelly() is not None and feats['exact_sig']<0.2 and pl['mnn']<0.6 and pl['sbert']<0.85:
            _r1='run_jellyfish'                              # legacy entity-gate: ONLY for the old 4-tool prior (no jelly labels)
        obs+=f'\nRECOMMENDED FIRST ACTION (start here unless a signal clearly says otherwise): {{"thought":"trust the prior","action":"{_r1}","args":{{}}}}'
        rec_holder[0]={'action':_r1,'args':{}}
        if _SIG_ONLY:
            # SIGNALS-ONLY routing (zero labels, ZERO model): the scout-legend rules pick the tool -- the
            # strongest no-LLM no-supervision router we can construct. Exists to MEASURE whether statistics
            # alone replace the model layers (the reviewer strawman); thresholds mirror the legend text.
            _p50=_ds.get('sim_p50',0.0); _mgn=_ds.get('sbert_margin',float('nan'))
            if _ds.get('exact_frac',0)>0.5: _r1='run_equi'
            elif (feats['avg_toks']>=8 or feats['avg_chars']>=60) and _jelly() is not None: _r1='run_jellyfish'
            elif _p50>=0.6 and not (_mgn==_mgn and _mgn<0.10): _r1='run_sbert'
            elif feats['digit_frac']>0.3 or feats['char_jac']>0.3: _r1='run_mnn'
            elif feats['multiword']>0.8 and feats['exact_sig']<0.05 and feats['char_jac']<0.15 and _jelly() is not None: _r1='run_jellyfish'
            else: _r1='run_mnn'
            if _r1=='run_mnn' and len(sv)>MAX_N: _r1='run_dtt'
            rec_holder[0]={'action':_r1,'args':{}}; _esc.update(armed=False,second=None,stash=None)
            obs+=f"\nSIGNALS-ONLY routing (legend rules, no model): -> {_r1}"
            print(f"    [sig-only] {name}: exact={_ds.get('exact_frac')} p50={_p50} mgn={_mgn} digit={feats['digit_frac']} cj={round(feats['char_jac'],2)} toks={feats['avg_toks']} -> {_r1}",flush=True)
        if _ESC>0 and len(_ordv)>=2:                         # near-tie escalation over the SAME ranking (any tool, incl. jellyfish)
            _sec='run_'+_ordv[1]
            if pl[_ordv[0]]-pl[_ordv[1]]<_ESC and _sec!=_r1 and not (_sec=='run_mnn' and len(sv)>MAX_N):
                _esc.update(armed=True,second=_sec)
        _exp_used=False
        _is_rec=feats.get('avg_toks',0)>=8 or feats.get('avg_chars',0)>=60   # RECORD-NESS policy (bimodal, 6x margin): multi-field records -> entity class; the 8B probe conflates semantic/entity exactly here, the statistic does not
        if _EXP_ROUTE:
            # EXPERIENCE ROUTING (user design, zero labels): kNN over the agent's OWN judge-verified tool quality
            # on past tasks (prequential -- only earlier tasks' records exist). Cost breaks ties. Near-ties get the
            # judged compare. Where experience is thin, fall through to catalog tool-select (and the judged outcome
            # of THIS task is recorded, so the next similar task routes by experience: picking improves over time).
            _mq=mem_lookup(feats,K=12)
            _cand2=[(t,v[0],v[1]) for t,v in _mq.items() if t in _TOOLCOST and v[1]>=2 and (t!='jellyfish' or _jelly() is not None) and not (t=='mnn' and len(sv)>MAX_N)]
            if len(_cand2)>=2 and _mq.get('_d2',9e9)>float(os.environ.get('AGENT_EXP_MAXD2','1.0')):
                _cand2=[]   # experience exists but is FAR in signal space (different domain) -> don't trust it; fall through to cold path
            if len(_cand2)>=2:
                _cand2.sort(key=lambda x:(-round(x[1],2),_TOOLCOST.get(x[0],9)))
                _r1='run_'+_cand2[0][0]
                _eff=_mq.get('_effort',(1.0,0))[0]           # memory's PLAN hint: how many tools similar tasks needed
                obs+=f"\nEXPERIENCE routing (judge-verified quality on similar past tasks): { {t:round(q,2) for t,q,_ in _cand2} } -> {_r1} (expected effort {_eff:.1f} tools)"
                print(f"    [exp-route] {name}: { {t:(round(q,2),n) for t,q,n in _cand2} } -> {_r1} eff={_eff:.1f}",flush=True)
                rec_holder[0]={'action':_r1,'args':{}}
                _esc.update(armed=False,second=None,stash=None)
                if _ESC>0 and (_cand2[0][1]-_cand2[1][1]<_ESC or _eff>=1.7):   # near-tie OR memory says tasks like this usually need >1 tool -> plan the compare upfront
                    _esc.update(armed=True,second='run_'+_cand2[1][0])
                _exp_used=True
        if (not _exp_used) and _is_rec and _jelly() is not None and (_TOOL_SELECT or _EXP_ROUTE or _PROBE_ROUTE):
            # RECORD POLICY (user design: "judge noise on records is figured out by the signals policy"):
            # values are unmistakably multi-field records -> entity class; route jellyfish directly, arm NO
            # semantic compare (the judge is noisy on EM pairs). Experience, once warm, takes precedence above.
            _r1='run_jellyfish'; rec_holder[0]={'action':_r1,'args':{}}
            _esc.update(armed=False,second=None,stash=None)
            obs+=f"\nRECORD policy: avg_toks={feats['avg_toks']} avg_chars={feats['avg_chars']} (multi-field records) -> run_jellyfish"
            print(f"    [record-policy] {name}: toks={feats['avg_toks']} chars={feats['avg_chars']} -> run_jellyfish",flush=True)
            _exp_used=True
        if _TOOL_SELECT and not _exp_used:
            # AGENTIC SELECTION (user design, zero labels): the model judges from the TOOL CATALOG + raw samples
            # which kind of tool the task needs (cost-constrained), naming PRIMARY + BACKUP; the judged compare
            # arbitrates primary vs backup on this task's actual pairs.
            _pri,_bak,_tsu=tool_select(sv,tv,feats=feats,scout=_ds)
            if _tsu: tin+=_tsu.prompt_tokens; tout+=_tsu.completion_tokens; calls+=1
            if _pri:
                if _pri=='jellyfish' and _jelly() is None: _pri,_bak='sbert',(_bak if _bak!='sbert' else 'jellyfish')
                if _pri=='mnn' and len(sv)>MAX_N: _pri='dtt'
                _r1='run_'+_pri
                obs+=f"\nAGENT tool-selection (catalog + raw samples): PRIMARY {_pri} BACKUP {_bak}"
                print(f"    [tool-select] {name}: primary={_pri} backup={_bak}",flush=True)
                rec_holder[0]={'action':_r1,'args':{}}
                _esc.update(armed=False,second=None,stash=None)
                if (_ESC>0 and _bak and _bak!=_pri and (_bak!='jellyfish' or _jelly() is not None)
                        and not (_bak=='mnn' and len(sv)>MAX_N)):
                    _esc.update(armed=True,second='run_'+_bak)
        if (_TYPE_PROBE or _PROBE_ROUTE) and not _is_rec:    # records: the policy already decided; the probe's semantic/entity confusion must not arm judge-flips
            # LLM CONTENT PROBE: model reads raw column samples and names the join type.
            #   TYPE_PROBE mode: disagreement with the prior's pick arms a judged comparison.
            #   PROBE_ROUTE mode (zero-label flagship): the probe PICKS the first tool; the judged compare against
            #   the type's twin (transform: mnn<->dtt) or sbert (entity fallback) guards against probe error.
            _tp,_tpu=type_probe(sbert_scored[0],sv,tv)
            if _tpu: tin+=_tpu.prompt_tokens; tout+=_tpu.completion_tokens; calls+=1
            if _tp and _PROBE_ROUTE:
                _tt=_TP_MAP[_tp]
                if _tt=='jellyfish' and _jelly() is None: _tt='sbert'
                if _tt=='mnn' and len(sv)>MAX_N: _tt='dtt'
                _r1='run_'+_tt
                _twin={'mnn':'dtt','dtt':'mnn','jellyfish':'sbert','sbert':None,'equi':None}.get(_tt)
                obs+=f"\nLLM type-probe ROUTES (reads raw column samples): {_tp} -> {_r1}"
                print(f"    [probe-route] {name}: {_tp} -> {_r1}",flush=True)
                rec_holder[0]={'action':_r1,'args':{}}
                _esc.update(armed=False,second=None,stash=None)
                if _ESC>0 and _twin and not (_twin=='mnn' and len(sv)>MAX_N):
                    _esc.update(armed=True,second='run_'+_twin)   # judged compare vs the class twin guards probe error
            elif _tp:
                _tt=_TP_MAP[_tp]
                if _tp=='transform' and _r1=='run_mnn': _tt='dtt'   # transform covers BOTH transform tools: challenge with the one not already picked
                elif _tp=='transform' and _r1=='run_dtt': _tt='mnn'
                obs+=f"\nLLM type-probe (reads raw column samples): {_tp} -> suggests run_{_tt}"
                print(f"    [type-probe] {name}: {_tp} -> run_{_tt} (first={_r1})",flush=True)
                _ok=(_tt!='jellyfish' or _jelly() is not None) and not (_tt=='mnn' and len(sv)>MAX_N)
                if _ok and 'run_'+_tt!=_r1 and _ESC>0 and _r1!='run_jellyfish':
                    # never challenge a jellyfish routing with a judge-compare: pair-judging is measurably
                    # unreliable on EM data (Amphibian/Artwork flips) -- the probe's semantic/entity confusion
                    # must not undo correct EM routing. (Challenges of transform/semantic routings stay.)
                    _esc.update(armed=True,second='run_'+_tt)
    if SELECTIVE and _esc['armed'] and _r1=='run_jellyfish':
        # FINAL GUARD (asymmetric, applies to ALL arming sources -- probe, catalog backup, exp near-tie):
        # never arm a judge-compare AGAINST a jellyfish primary; pair-judging is measurably unreliable on EM
        # data and flips correct EM routings (Amphibian/Artwork). Jellyfish-as-CHALLENGER stays allowed.
        _esc.update(armed=False,second=None,stash=None)
    prev_acc=0; stall=0
    for step in range(MAXSTEP):
        state=obs+f"\nSpent: tools_run={sorted(ran)} llm_calls={calls} budget_left={'inf' if budget<0 else budget-calls}. Next action JSON:"
        if _NO_LLM and SELECTIVE:                            # pure signal-rules: take the RECOMMENDED action, no controller LLM call
            action,args,thought,usage=rec_holder[0].get('action','finish'),rec_holder[0].get('args',{}) or {},'(signal-rules)',None
        else:
            action,args,thought,usage=llm_act(state)
        if usage: tin+=usage.prompt_tokens; tout+=usage.completion_tokens
        if action=='PARSEFAIL':                             # never forfeit a task on a bad parse: fall back to policy
            rec='run_'+(pl['rec'] if pl['rec'] in ('equi','sbert','mnn','dtt') else 'sbert')
            if rec not in ran: action,thought=rec,'(fallback -> prior-recommended tool)'
            elif last_pairs[0] and not accepted: action,thought='accept','(fallback -> accept last tool)'
            else: action,thought='finish','(fallback -> finish)'
        info=""
        if action=='probe': info=f"signals={ {k:round(v,3) for k,v in feats.items()} }"
        elif action=='prior':
            pl=prior_lookup(BM,name,feats); info=f"prior: expected_recall={ {k:pl[k] for k in ['equi','mnn','sbert','dtt']} } recommend={pl['rec']} mnn_poor={pl['mnn_poor']}"
        elif action=='run_mnn' and 'run_mnn' not in ran and len(sv)>MAX_N:
            ran.add('run_mnn')   # cost guard: MNN Join's transform search is too slow on big tables
            info=f"run_mnn REFUSED: {len(sv)} rows exceed the slow-MNN Join limit ({MAX_N}); use run_sbert/run_dtt instead"
        elif action in ('run_equi','run_sbert','run_mnn','run_dtt','run_jellyfish') and action not in ran:
            ran.add(action); last_tool[0]=action; extra=''
            if action=='run_equi': P=tool_equi(s,t,sc,tc,sv,tv)
            elif action=='run_sbert':
                if sbert_scored[0]:                          # reuse scout's SBERT pass (SELECTIVE) -> no double compute
                    scored=sbert_scored[0]; P={(a,b) for a,b,_ in scored}; sim=float(np.mean([x[2] for x in scored])) if scored else 0.0
                else:
                    P,sim,scored=tool_sbert(sv,tv); sbert_scored[0]=scored
                last_pairs[0]=set(P)
                extra=f" sbert_confidence={sim:.2f} | {sim_summary(scored)}"
            elif action=='run_mnn': P=tool_mnn(sv,tv)
            elif action=='run_jellyfish': P=tool_jellyfish(sv,tv); extra=" (off-the-shelf EM matcher: accepted only Yes-judged pairs)"
            else: P=tool_dtt(name)
            last_pairs[0]=set(P)
            for p in P: votes[p]+=1
            if action=='run_equi': accepted.update(P)        # exact matches are reliable -> auto-accept
            add_consensus()
            _cov=coverage(P,sv); _bij=bijectivity(P)
            if action=='run_jellyfish':                       # EM matcher: pairs are pre-verified Yes -> ACCEPT them; sparse-EM low coverage is correct
                _hint=(f"-> EM matcher: these pairs were each LLM-judged 'same entity' (high precision). coverage={_cov:.2f}. "
                       f"call accept to commit them and finish. Sparse entity joins have LOW coverage (most sources have no match) — that is CORRECT, do NOT escalate to sbert.")
            elif action in ('run_mnn','run_dtt'):           # TRANSFORM tools: matches are EXACT transformations -> accept on coverage (bijectivity irrelevant; repeats legit)
                _hint=(f"-> TRANSFORM tool: its matches are EXACT transformations. coverage={_cov:.2f}. If coverage is high, "
                       f"call accept to commit ALL pairs. Do NOT threshold (bijectivity {_bij:.2f} may be low for legitimate repeated targets).")
            elif action=='run_sbert':                          # SBERT: discriminate by the SIM DISTRIBUTION, not bijectivity
                _p10=float(np.percentile([sm for _,_,sm in scored],10)) if scored else 0.0   # weakest-match similarity
                if _cov>=0.9 and _p10>=0.5:                    # sims uniformly high -> every source has a real match (clean OR repeat-transform)
                    _hint=(f"-> SBERT all-confident (coverage={_cov:.2f}, sim_p10={_p10:.2f}: even the WEAKEST match is strong; "
                           f"low bijectivity {_bij:.2f} just means repeated targets, NOT non-matches): call accept to commit ALL pairs. Do NOT threshold.")
                else:                                          # low-sim tail -> sparse, forced non-matches present -> threshold
                    _hint=(f"-> SBERT has a low-sim tail (sim_p10={_p10:.2f} = likely forced non-matches): call threshold{{\"tau\":X}} "
                           f"at a cut ABOVE the low tail to keep only high-sim true matches.")
            else:
                _hint="-> EQUI exact matches are auto-accepted."
            _ca=coverage(accepted,sv); _nt=len([t for t in ran if t in ('run_equi','run_sbert','run_mnn','run_dtt')])
            # explicit copyable RECOMMENDED NEXT action (the weaker base controller ratifies a signal-grounded decision)
            if action=='run_jellyfish': _rec='{"action":"accept","args":{}}'   # EM matcher: its Yes-judged pairs are precise -> ACCEPT & finish; sparse-EM LOW coverage is CORRECT, do NOT escalate/fuse to sbert (that forces wrong matches)
            elif _ca>=0.95: _rec='{"action":"finish","args":{}}'
            elif len(P)==0:                                   # BUGFIX: tool proposed NOTHING -> escalate to the next unused tool (never fuse empties)
                _nxo=(['equi','dtt','sbert','mnn','jellyfish'] if _CASCADE else ['mnn','sbert','dtt','equi','jellyfish'])
                _nxa=[x for x in _nxo if 'run_'+x not in ran and 'run_'+x not in bad_tools and not (x=='mnn' and len(sv)>MAX_N) and not (x=='jellyfish' and _jelly() is None)]
                _rec=(f'{{"action":"run_{_nxa[0]}","args":{{}}}}' if _nxa else ('{"action":"finish","args":{}}' if not accepted else '{"action":"finish","args":{}}'))
            elif _nt>=2: _rec='{"action":"fuse","args":{}}'
            elif action in ('run_mnn','run_dtt'): _rec=('{"action":"accept","args":{}}' if _cov>=0.5 else '{"action":"run_sbert","args":{}}')
            elif action=='run_sbert' and _cov>=0.9 and _p10>=0.5: _rec='{"action":"accept","args":{}}'  # HIGH sim -> trust sbert (handles repeat-transforms: low bij but high sim)
            elif action=='run_sbert' and 'run_mnn' not in ran and 'run_mnn' not in bad_tools and len(sv)<=MAX_N: _rec='{"action":"run_mnn","args":{}}'  # LOW sim (even if bijective) -> sbert unreliable on this transform; escalate to mnn, then fuse
            elif action=='run_sbert': _rec=f'{{"action":"threshold","args":{{"tau":{round(max(0.4,_p10+0.05),2)}}}}}'  # mnn already tried OR large/entity table -> threshold
            else: _rec='{"action":"accept","args":{}}'
            if _ESC>0 and _esc['armed'] and _esc['stash'] is not None and action==_esc['second']:
                _rec='{"action":"accept","args":{}}'          # runner-up just ran -> go to the escalation compare-accept, not fuse
            elif _ESC>0 and _esc['armed'] and _esc['stash'] is None and len(P)>0 and action!=_esc['second']:
                _rec='{"action":"accept","args":{}}'          # challenge ARMED: funnel through accept->defer so the runner-up actually runs (don't let low-coverage re-routing pre-empt the judged compare)
            _exp='; '.join(f"{a[:28]!r}->{b[:28]!r}" for a,b in list(P)[:3])   # show ACTUAL matched pairs so the judge can sanity-check semantically
            info=(f"{action}: proposed {len(P)} pairs (e.g. {_exp}). coverage={_cov:.2f} bijectivity={_bij:.2f}{extra}. "
                  f"accepted_so_far={len(accepted)} coverage_acc={_ca:.2f}. {_hint}")
            if SELECTIVE: info+=f" || RECOMMENDED NEXT ACTION: {_rec}"; rec_holder[0]=json.loads(_rec)
        elif action=='threshold':
            tau=float(args.get('tau',0.7)); chosen_tau[0]=tau; keep={(s_,t_) for s_,t_,sm in sbert_scored[0] if sm>=tau}
            accepted.update(keep); last_pairs[0]=keep
            info=f"threshold tau={tau}: kept {len(keep)} sbert pairs with sim>=tau (of {len(sbert_scored[0])}). accepted={len(accepted)} coverage_acc={coverage(accepted,sv):.2f}"
            rec_holder[0]={'action':'finish','args':{}}
        elif action=='accept':
            if SELECTIVE:                                     # selective: trust the chosen tool's pairs, BUT verify a sample first (quality gate)
                _cand=list(last_pairs[0])
                if _ESC>0 and _esc['armed']:
                    # CLASS-AWARE EVIDENCE (v3, user design): a TRANSFORM tool's verification is its own output --
                    # every pair is an EXACT match under one explicit program, so its evidence = exact-overlap
                    # coverage (free, no judge). Pair-judging is reserved for semantic/entity pairs, where
                    # plausibility IS the question. (The pair-judge mislabels extraction-style transforms.)
                    _XFORM=_XFORM_TOOLS
                    def _evidence(tool,P):
                        nonlocal calls,tin,tout
                        if not P: return 0.0,'no pairs'
                        if tool in _XFORM: return coverage(P,sv),'exact-overlap under program'
                        Pl=list(P); samp=[Pl[i] for i in range(0,len(Pl),max(1,len(Pl)//_VK))][:_VK] if len(Pl)>_VK else Pl
                        ok=0
                        for a,b in samp:
                            if budget>=0 and calls>=budget: break
                            v,pin,po=A.verify(a,b); calls+=1; tin+=pin; tout+=po
                            if v: ok+=1
                        return ok/max(1,len(samp)),f'judged {len(samp)} pairs'
                    _rate,_evhow=_evidence(last_tool[0],set(_cand))
                    if _esc['stash'] is None and _esc['second'] not in ran:
                        _esc['stash']=(last_tool[0],set(_cand),_rate)
                        info=(f"accept DEFERRED (challenge armed): {last_tool[0]} evidence {_rate:.2f} ({_evhow}); "
                              f"run the challenger before committing. || RECOMMENDED NEXT ACTION: {{\"action\":\"{_esc['second']}\",\"args\":{{}}}}")
                        rec_holder[0]={'action':_esc['second'],'args':{}}
                    else:
                        _t1,_P1,_r1v=_esc['stash'] if _esc['stash'] else (last_tool[0],set(_cand),_rate)
                        _P2=set(_cand); _agree=_P1&_P2
                        if _t1 not in _XFORM and last_tool[0] not in _XFORM:
                            # both sides pair-judgeable -> DISAGREEMENT-FIRST judging (the informative pairs)
                            _m1={}; _m2={}
                            for a,b in _P1: _m1.setdefault(a,set()).add(b)
                            for a,b in _P2: _m2.setdefault(a,set()).add(b)
                            _conf=[a for a in set(_m1)&set(_m2) if _m1[a]!=_m2[a]]
                            if len(_conf)>=3:
                                _d1=[(a,sorted(_m1[a])[0]) for a in _conf[:8]]; _d2=[(a,sorted(_m2[a])[0]) for a in _conf[:8]]
                                _o1=_o2=0
                                for (a1,b1),(a2,b2) in zip(_d1,_d2):
                                    if budget>=0 and calls>=budget: break
                                    v1,p1,q1=A.verify(a1,b1); calls+=1; tin+=p1; tout+=q1
                                    v2,p2,q2=A.verify(a2,b2); calls+=1; tin+=p2; tout+=q2
                                    _o1+=bool(v1); _o2+=bool(v2)
                                _ra,_rb=_o1/max(1,len(_d1)),_o2/max(1,len(_d2)); _how=f"conflict-judged {len(_d1)} pairs"
                            else:
                                _ra,_rb=_r1v,_rate; _how="few conflicts; sampled judge rates"
                        else:
                            _ra,_rb=_r1v,_rate; _how="class evidence (exact-overlap for transform, judged pairs otherwise)"
                        if (_rb,coverage(_P2,sv),bijectivity(_P2))>=(_ra,coverage(_P1,sv),bijectivity(_P1)): _win,_wp=last_tool[0],_P2
                        else: _win,_wp=_t1,_P1
                        accepted.update(_wp|_agree); _esc['armed']=False
                        tool_q[_t1]=_ra; tool_q[last_tool[0]]=_rb   # record BOTH tools' evidence (experience store)
                        info=(f"accept ESCALATED-COMPARE ({_how}): {_t1} evidence={_ra:.2f} vs {last_tool[0]} evidence={_rb:.2f} -> committed {_win} "
                              f"({len(_wp)} pairs + {len(_agree)} agreement). accepted={len(accepted)} coverage_acc={coverage(accepted,sv):.2f} || RECOMMENDED NEXT ACTION: {{\"action\":\"finish\",\"args\":{{}}}}")
                        rec_holder[0]={'action':'finish','args':{}}
                elif _VERIFY_ACCEPT and _cand:                  # QUALITY GATE -- CLASS-APPROPRIATE EVIDENCE (v3):
                    if last_tool[0]=='run_jellyfish':
                        # the EM specialist ALREADY judged every pair 'same entity' -- re-judging with the weaker
                        # generic verifier and DISCARDING the join is the measured EM flip at gate level (it zeroed
                        # 24/50 autofj + 7/7 magellan in the first EM slate). Its own Yes-rate IS the evidence.
                        accepted.update(_cand); add_consensus(); tool_q[last_tool[0]]=coverage(set(_cand),sv)
                        info=f"accept: EM matcher pre-verified {len(_cand)} pairs -> committed. coverage_acc={coverage(accepted,sv):.2f}"
                        rec_holder[0]={'action':'finish','args':{}}
                        _gate_done=True
                    elif last_tool[0] in _XFORM_TOOLS:
                        _rate=coverage(set(_cand),sv); tool_q[last_tool[0]]=_rate
                        if _rate>=_VACC_TAU:                  # transform evidence = exact-overlap under the program (free)
                            accepted.update(_cand); add_consensus()
                            info=f"accept: exact-overlap evidence {_rate:.2f}>=tau -> committed {len(_cand)} pairs. coverage_acc={coverage(accepted,sv):.2f}"
                            rec_holder[0]={'action':'finish','args':{}}
                            _gate_done=True
                        else: _gate_done=False; _ok='-'; _samp=[]
                    elif _SIG_ONLY:
                        # signals-only: no judge exists -- rule-accept (the tool's own threshold/top-1 is the only
                        # evidence). This is precisely the strawman's missing correction layer; do NOT "fix" it.
                        accepted.update(_cand); add_consensus(); tool_q[last_tool[0]]=coverage(set(_cand),sv)
                        info=f"accept: SIGNALS-ONLY rule-accept {len(_cand)} pairs (no judge). coverage_acc={coverage(accepted,sv):.2f}"
                        rec_holder[0]={'action':'finish','args':{}}
                        _gate_done=True
                    else:
                        _samp=[_cand[i] for i in range(0,len(_cand),max(1,len(_cand)//_VK))][:_VK] if len(_cand)>_VK else _cand
                        _ok=0
                        for a,b in _samp:
                            if budget>=0 and calls>=budget: break
                            v,pin,po=A.verify(a,b); calls+=1; tin+=pin; tout+=po
                            if v: _ok+=1
                        _rate=_ok/max(1,len(_samp)); tool_q[last_tool[0]]=_rate
                        if _rate>=_VACC_TAU:                  # quality OK -> commit
                            accepted.update(_cand); add_consensus()
                            info=f"accept: VERIFIED {_ok}/{len(_samp)} sampled pairs correct (rate {_rate:.2f}>=tau {_VACC_TAU}) -> committed {len(_cand)} pairs. accepted={len(accepted)} coverage_acc={coverage(accepted,sv):.2f}"
                            rec_holder[0]={'action':'finish','args':{}}
                            _gate_done=True
                        else: _gate_done=False
                    if not _gate_done:                        # LOW quality -> REJECT this tool, escalate to a different one
                        _rej_best[0]=max(_rej_best[0],(coverage(set(_cand),sv),frozenset(_cand)),key=lambda x:x[0]) if _rej_best[0] else (coverage(set(_cand),sv),frozenset(_cand))
                        bad_tools.add(last_tool[0]); last_pairs[0]=set()
                        # v3 CLASS-AWARE switching: try the failed tool's CLASS SIBLING first (another transform tool),
                        # then SWITCH class (semantic -> entity ...). Cascade keeps cheapest-first.
                        _failed=last_tool[0].replace('run_','')
                        _esc_order=(['dtt','sbert','mnn','jellyfish'] if _CASCADE else _CLASS_NEXT.get(_failed,['mnn','sbert','dtt','equi']))
                        _av=[t for t in _esc_order if 'run_'+t not in ran and 'run_'+t not in bad_tools and not (t=='mnn' and len(sv)>MAX_N) and not (t=='jellyfish' and _jelly() is None)]
                        _nx=('run_'+_av[0]) if _av else 'finish'
                        info=(f"accept REJECTED: only {_ok}/{len(_samp)} sampled pairs verified correct (rate {_rate:.2f} < tau {_VACC_TAU}) -> "
                              f"{last_tool[0]} produced a LOW-QUALITY join. || RECOMMENDED NEXT ACTION: {{\"action\":\"{_nx}\",\"args\":{{}}}}")
                        rec_holder[0]={'action':_nx,'args':{}}
                else:
                    accepted.update(_cand); add_consensus()
                    if _CASCADE and not accepted:            # cascade: tool proposed nothing -> continue down the cost ladder, don't finish empty
                        _av2=[t for t in ['dtt','sbert','mnn','jellyfish'] if 'run_'+t not in ran and 'run_'+t not in bad_tools
                              and not (t=='mnn' and len(sv)>MAX_N) and not (t=='jellyfish' and _jelly() is None)]
                        _nx2=('run_'+_av2[0]) if _av2 else 'finish'
                        info=f"accept: nothing to commit -> cascade to next-cheapest. || RECOMMENDED NEXT ACTION: {{\"action\":\"{_nx2}\",\"args\":{{}}}}"
                        rec_holder[0]={'action':_nx2,'args':{}}
                    else:
                        info=f"accept: committed {len(_cand)} pairs from last tool. accepted={len(accepted)} coverage_acc={coverage(accepted,sv):.2f}"
                        rec_holder[0]={'action':'finish','args':{}}
                if last_tool[0]=='run_jellyfish' and not _esc['armed']:   # EM done -> finish (unless a near-tie compare is pending)
                    info+=' || RECOMMENDED NEXT ACTION: {"action":"finish","args":{}}'
                    rec_holder[0]={'action':'finish','args':{}}
            else:
                add_consensus()   # run-all consensus-only: wholesale accept of ONE tool's top-1 caused the 283063 precision loss
                info=f"accept: committed >=2-consensus. accepted={len(accepted)} coverage_acc={coverage(accepted,sv):.2f}"
        elif action=='verify':
            k=min(int(args.get('k',10)),25); resolved={a for a,_ in accepted}   # cap to bound cost on large tables
            cand=[p for p,c in votes.items() if c==1 and p[0] not in resolved][:k]
            kept=0
            for a,b in cand:
                if budget>=0 and calls>=budget: break
                v,pin,po=A.verify(a,b); calls+=1; tin+=pin; tout+=po
                if v: accepted.add((a,b)); kept+=1
            info=f"verify: checked {len(cand)} singletons, kept {kept}. accepted={len(accepted)} coverage_acc={coverage(accepted,sv):.2f}"
            rec_holder[0]={'action':'finish','args':{}}
        elif action=='fuse':                                  # COMBINE the tools run so far (= mnn+lotus / §14, ON DEMAND): pairs >=2 tools agree on are free-accepted; disagreements -> LLM/lotus decides
            before=len(accepted); add_consensus(); cons=len(accepted)-before
            resolved={a for a,_ in accepted}
            k=min(int(args.get('k',25)),50)
            cand=[p for p,c in votes.items() if c==1 and p[0] not in resolved][:k]   # disagreement singletons
            kept=0
            for a,b in cand:
                if budget>=0 and calls>=budget: break
                v,pin,po=A.verify(a,b); calls+=1; tin+=pin; tout+=po
                if v: accepted.add((a,b)); kept+=1
            _nt=len([t for t in ran if t in ('run_equi','run_sbert','run_mnn','run_dtt')])
            info=(f"fuse: combined {_nt} tools -> consensus(>=2 agree) auto-accepted {cons}, LLM/lotus-verified {len(cand)} disagreements (kept {kept}). "
                  f"accepted={len(accepted)} coverage_acc={coverage(accepted,sv):.2f}")
            rec_holder[0]={'action':'finish','args':{}}
        elif action=='finish': trace.append((thought,action,'done')); break
        else: info=f"(noop: {action} already run or unknown)"
        trace.append((thought,action,info)); obs=obs+f"\nStep{step+1} [{action}] -> {info}"
        if SELECTIVE and coverage(accepted,sv)>=0.98 and action in ('accept','threshold','fuse','verify'):
            trace.append(('(coverage complete)','finish','auto')); break   # full coverage committed -> stop early (kills base accept/fuse repeats)
        if len(accepted)==prev_acc: stall+=1
        else: stall=0
        prev_acc=len(accepted)
        if stall>=2 and accepted: trace.append(('(no further progress)','finish','auto')); break   # stop re-accept loops
    # --- REFINE (post-283063): deterministic PRECISION mechanism = §14 (consensus + verify/threshold), per class.
    #     The free-form loop wholesale-accepted single tools (precision loss) and under-verified; this guarantees the
    #     right mechanism regardless of the controller's path. FUSE within a class; route to entity-tools only if semantic.
    is_sem=pl['mnn_poor']
    if not SELECTIVE:
        # RUN-ALL backstop (default agent): force the free tools so fusion always has a pool.
        if 'run_equi' not in ran:
            Pe=tool_equi(s,t,sc,tc,sv,tv); ran.add('run_equi')
            for p in Pe: votes[p]+=1
            accepted.update(Pe)                                   # exact matches reliable
        if 'run_sbert' not in ran:
            Ps,_si,_sc=tool_sbert(sv,tv); sbert_scored[0]=_sc; ran.add('run_sbert')
            for p in Ps: votes[p]+=1
        if ('run_mnn' not in ran) and len(sv)<=MAX_N:
            Pq=tool_mnn(sv,tv); ran.add('run_mnn')       # MNN Join is CPU-cheap -> FUSE on every small table; skip ONLY when too large
            for p in Pq: votes[p]+=1
        # also fuse DTT (cached, free) when available
        Pd0=tool_dtt(name)
        if Pd0 and 'run_dtt' not in ran:
            ran.add('run_dtt')
            for p in Pd0: votes[p]+=1
    else:
        # SELECTIVE: do NOT force tools. Minimal fallback only: if the controller ran no real tool, run the prior-rec one.
        if not any(t in ran for t in ('run_equi','run_sbert','run_mnn','run_dtt','run_jellyfish')):
            _rec=pl['rec'] if pl['rec'] in ('equi','sbert','mnn','dtt') else 'sbert'
            if _rec=='mnn' and len(sv)>MAX_N: _rec='sbert'
            if _rec=='equi': Pf=tool_equi(s,t,sc,tc,sv,tv); accepted.update(Pf)
            elif _rec=='sbert':
                if sbert_scored[0]: Pf={(a,b) for a,b,_ in sbert_scored[0]}
                else: Pf,_si,_sc=tool_sbert(sv,tv); sbert_scored[0]=_sc
            elif _rec=='mnn': Pf=tool_mnn(sv,tv)
            else: Pf=tool_dtt(name)
            ran.add('run_'+_rec); last_pairs[0]=set(Pf)
            for p in Pf: votes[p]+=1
    if (not SELECTIVE) and _FUSE_UNION:
        accepted.update(votes.keys())                         # UNION fusion (truth-free, no verify): accept EVERY tool's pairs -> precision-dilution baseline
    else:
        add_consensus()                                       # >=2-tool agreement, free-accept (= §14)
    resolved={a for a,_ in accepted}
    if len(sv)>MAX_N and 'run_jellyfish' not in ran:    # LARGE table (MNN Join too slow): entity path = LLM-set threshold — but NOT if the EM matcher (jellyfish) already answered (its pairs are the join; sbert-threshold would only dilute)
        if chosen_tau[0]!=chosen_tau[0]:                     # nan -> pick tau from the sim distribution
            if _NO_LLM:                                      # rules: heuristic tau (no LLM) = p10 of the sim distribution + margin
                _scc=sbert_scored[0]; _p10v=float(np.percentile([sm for _,_,sm in _scc],10)) if _scc else 0.5
                chosen_tau[0]=max(0.4,round(_p10v+0.05,2))
            else:
                tau,u=llm_threshold(name,sv,sbert_scored[0]); chosen_tau[0]=tau
                if u: tin+=u.prompt_tokens; tout+=u.completion_tokens; calls+=1
        accepted |= {(a,b) for a,b,sm in sbert_scored[0] if sm>=chosen_tau[0] and a not in resolved}
    elif not SELECTIVE:                                       # TRANSFORM (run-all): precision via verifying disagreement singletons
        for (a,b) in [p for p,c in votes.items() if c==1 and p[0] not in resolved]:
            if budget>=0 and calls>=budget: break
            v,pin,po=A.verify(a,b); calls+=1; tin+=pin; tout+=po
            if v: accepted.add((a,b))
    elif not accepted and (last_pairs[0] or _rej_best[0]):    # SELECTIVE safety net: NEVER end empty
        if last_pairs[0]: accepted.update(last_pairs[0])
        else: accepted.update(_rej_best[0][1])                # all proposals gate-rejected -> commit the best-coverage one anyway (a weak join beats no join)
    if _BACKFILL and SELECTIVE:
        # v3 BACKFILL -- runs BEFORE scoring (v2 scored first: its backfill never reached the reported F1).
        # Commit judged matches for the UNCOVERED rows only. Free candidates from already-run tools first; else
        # RE-PROBE THE RESIDUAL: the leftover rows may be a DIFFERENT class than the committed join (transform
        # handled most rows, the rest are semantic/entity) -> catalog-select on the residual picks ITS tool.
        _have={a for a,_ in accepted}; _unc={v for v in sv if v not in _have}
        if len(_unc)>=max(3,0.05*len(sv)) and 'run_jellyfish' not in ran:
            _bf={p for p in votes if p[0] in _unc and p not in accepted}; _src='ran-tool proposals'
            if not _bf:
                _uncl=[v for v in sv if v in _unc]; _rt=None
                if _TOOL_SELECT or _EXP_ROUTE or _PROBE_ROUTE:
                    _rt,_rb,_ru=tool_select(_uncl,tv)
                    if _ru: tin+=_ru.prompt_tokens; tout+=_ru.completion_tokens; calls+=1
                    if _rt: print(f"    [residual-probe] {name}: {len(_uncl)} uncovered rows -> {_rt}",flush=True)
                _order=([_rt] if _rt else [])+[x for x in sorted(['equi','mnn','sbert','dtt'],key=lambda x:-pl[x])]
                for _tn in _order:
                    if _tn is None or 'run_'+_tn in ran or 'run_'+_tn in bad_tools: continue
                    if _tn=='mnn' and len(sv)>MAX_N: continue
                    if _tn=='jellyfish' and _jelly() is None: continue
                    if _tn=='equi': P=tool_equi(s,t,sc,tc,sv,tv)
                    elif _tn=='sbert':
                        if sbert_scored[0]: P={(a,b) for a,b,_ in sbert_scored[0]}
                        else: P,_si,_sc2=tool_sbert(sv,tv); sbert_scored[0]=_sc2
                    elif _tn=='mnn': P=tool_mnn(sv,tv)
                    elif _tn=='jellyfish': P=tool_jellyfish(_uncl,tv)
                    else: P=tool_dtt(name)
                    ran.add('run_'+_tn)
                    for p in P: votes[p]+=1
                    _bf={p for p in P if p[0] in _unc}; _src=f'residual run_{_tn}'
                    if _bf: break
            if _bf:
                _bl=list(_bf); _samp=_bl[::max(1,len(_bl)//_VK)][:_VK]
                _ok=0
                for a,b in _samp:
                    if budget>=0 and calls>=budget: break
                    v,pin,po=A.verify(a,b); calls+=1; tin+=pin; tout+=po
                    if v: _ok+=1
                if _ok>=max(1,(len(_samp)+1)//2):            # majority judged correct -> commit the residual pairs
                    accepted.update(_bf)
                    trace.append(('(backfill uncovered rows)','backfill',f'+{len(_bf)} pairs from {_src}, judged {_ok}/{len(_samp)}; coverage_acc={coverage(accepted,sv):.2f}'))
                else:
                    trace.append(('(backfill rejected by judge)','backfill',f'{_src}: {_ok}/{len(_samp)} -> not committed'))
    f1=A.score_rows(accepted,name,s,t,sc,tc,truth)
    # precision/recall (for failure analysis): map accepted value-pairs -> row-pairs vs truth
    s2r={}; [s2r.setdefault(_nrm(v),[]).append(i) for i,v in enumerate(s[sc])]
    t2r={}; [t2r.setdefault(_nrm(v),[]).append(j) for j,v in enumerate(t[tc])]
    pred=set()
    for a,b in accepted:
        for i in s2r.get(_nrm(a),[]):
            for j in t2r.get(_nrm(b),[]): pred.add((i,j))
    tp=len(pred&truth); prec=tp/len(pred) if pred else 0.0; rec=tp/len(truth) if truth else 0.0
    if _USE_MEM and SELECTIVE:                              # MEMORY: record each tool's verified-quality on this signal-profile (truth-free) for future picks
        _nt_run=len([t for t in ran if t.startswith('run_')])
        for _t,_q in tool_q.items(): mem_record(BM,name,feats,_t.replace('run_',''),_q,coverage(accepted,sv),ntools=_nt_run)
    _vq=max(tool_q.values()) if tool_q else float('nan')   # verified quality of the committed join
    return dict(name=name,f1=f1,prec=round(prec,3),rec=round(rec,3),tau=chosen_tau[0],steps=len(trace),
                tools='+'.join(sorted(ran)) or 'none',vq=round(_vq,3) if _vq==_vq else _vq,bad=','.join(sorted(bad_tools)),
                calls=calls,in_tok=tin,out_tok=tout,kept=len(accepted),trace=trace)

def main():
    am={}
    for f in glob.glob(f'{R.ARTIFACTS[BM]}/{BM}__*.json'):
        try: d=json.load(open(f)); am[d['dataset']]=d
        except Exception: pass
    names=sorted(am); lim=int(os.environ.get('AGENT_LIMIT','0'))
    sel=os.environ.get('NAMES','')
    if sel: names=[n for n in sel.split('||') if n in am]
    elif lim>0: names=names[:lim]
    _shuf=os.environ.get('AGENT_SHUFFLE','')
    if _shuf:                                                # order-randomization ablation: prequential order sensitivity
        import random as _rnd; _rnd.Random(int(_shuf)).shuffle(names)
        print(f"(dataset order shuffled, seed {_shuf})",flush=True)
    print(f"ReAct JOIN AGENT on {BM}: {len(names)} tasks, ctrl={CTRL} budget={BUDGET} maxstep={MAXSTEP}",flush=True)
    rows=[]
    for nm in names:
        _t0=time.time()
        try: r=react(nm,am[nm],BUDGET)
        except Exception as e:
            import traceback; print(f"  {nm:22s} ERR {str(e)[:60]}"); traceback.print_exc(); continue
        r['wall']=round(time.time()-_t0,3)
        rows.append({k:v for k,v in r.items() if k!='trace'})
        print(f"  {nm:22s} f1={r['f1']:.3f} tools=[{r['tools']:<28}] steps={r['steps']} calls={r['calls']} tok_in={r['in_tok']}",flush=True)
        if os.environ.get('SHOW_TRACE') and rows[-1]:  # print the ReAct transcript for the first few
            for th,ac,ob in r['trace']: print(f"        T:{th[:80]}\n        A:{ac}  O:{ob[:110]}")
    _mt='_'+_mnn_engine._METRIC           # tag by the fixed MNN Join metric
    _sh=('_shuf'+os.environ['AGENT_SHUFFLE']) if os.environ.get('AGENT_SHUFFLE') else ''      # tag by shuffle order (reorder exp)
    _sfx=_sh+_mt   # shuffle + metric tag
    D=pd.DataFrame(rows); D.to_csv(f'out_put_csv/react_v3_{BM}_b{BUDGET}{_sfx}.csv',index=False)
    print(f"\n=== ReAct {BM} (n={len(D)}) ===  row_f1={D.f1.mean():.3f}  LLM_calls={int(D.calls.sum())} (incl controller) tok_in={int(D.in_tok.sum())}")
    print(f"  TOOL-SET DISTRIBUTION:");
    for ts,c in D.tools.value_counts().items(): print(f"    {ts:<30} {c:>4} ({100*c/len(D):.0f}%)")
    print(f"  avg steps/task={D.steps.mean():.1f}  mnn skipped on {int((~D.tools.str.contains('mnn')).sum())} tasks")
    print("DONE react_join")

if __name__=='__main__': main()
