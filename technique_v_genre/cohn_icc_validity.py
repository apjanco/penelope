"""Does the per-author ICC reflect Cohn's scale, or is it a measure unto itself?

The chapter's headline statistic is a one-way ICC per author: the share of
variance in passage mediation level that lies between that author's titles
(Woolf 0.561, recompute_chapter_stats.py). Three rival explanations for a high
value:

  (a) Cohn — the novels differ in the grammatical features Cohn uses to define
      the levels (person, tense, reporting frames, free-indirect markers);
  (b) general heterogeneity — the novels differ in *everything*, and the
      readers' "mediation" picks that up;
  (c) instrument — the readers respond to something else (book identity,
      reputation, their own habits).

Three automated tests, no new API calls:

  Test 1  convergent: ICC of Cohn-grounded grammatical features, one by one and
          as a theory-signed composite built WITHOUT any model labels.
  Test 2  discriminant: ICC of features with no place in Cohn's definitions
          (sentence and word length, vocabulary richness, comma density,
          function-word style components). If Woolf tops these too, (b) is live.
  Test 3  decomposition: predict each passage's reader level from the Cohn
          features alone (leave-one-author-out, so author identity cannot be
          learned), then compute the ICC of the predicted part and of the
          residual. Between-title variance in the predicted part is grounded in
          Cohn's grammar; what remains is what the readers add beyond it.

Passages and levels exactly as in recompute_chapter_stats.py: scored/ rows from
gpt-5.4, qwen3-max and qwen3.6-local, is_soc, no error, per-passage mean across
readers and runs, vol. 1/2 folded into one title.

Features come from spaCy (en_core_web_sm) tags for person, tense and sentence
boundaries, plus small word lists: crude by design and blind to what the
readers see. Narrated monologue in particular has no reliable surface marker,
which is why the study uses LLM readers; the tests look for agreement in
*ordering and pattern*, not equal values.

Usage (della-vis2: activate the env by its /scratch path or python hangs):
  conda activate /scratch/gpfs/MM4/.conda-aj7878/envs/penelope
  python cohn_icc_validity.py
Writes cohn_icc_validity_report.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

TV = Path(__file__).resolve().parent
OUT = TV / "cohn_icc_validity_report.md"
SKIP = {"time_passes.jsonl", "qwen3_reasoning_smoketest.jsonl"}
READERS = {"gpt-5.4", "qwen3-max", "qwen3.6-local"}
AUTHORS = ["Woolf", "Joyce", "Faulkner", "Sinclair", "Richardson", "Ford", "James"]
RNG = np.random.default_rng(20261024)


# ---- ICC: same estimator as recompute_chapter_stats.py (copied, since importing
# that module runs its whole analysis)
def icc_one_way(groups: list[np.ndarray]) -> float:
    groups = [g for g in groups if len(g)]
    k = len(groups)
    if k < 2:
        return float("nan")
    ns = np.array([len(g) for g in groups])
    N = ns.sum()
    if N <= k:
        return float("nan")
    grand = np.concatenate(groups).mean()
    msb = sum(n * (g.mean() - grand) ** 2 for n, g in zip(ns, groups)) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    n0 = (N - (ns ** 2).sum() / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return float("nan") if denom == 0 else max(0.0, (msb - msw) / denom)


def bootstrap_icc_ci(groups: list[np.ndarray], n_boot: int = 1000) -> tuple[float, float]:
    boots = [icc_one_way([RNG.choice(g, size=len(g), replace=True) for g in groups]) for _ in range(n_boot)]
    boots = np.array([b for b in boots if not np.isnan(b)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---- Cohn-grounded features (sign = expected direction toward LESS mediation)
# psycho-narration: the narrator names a mental state from outside (lemmas, past tense)
PSYCHO_LEMMAS = {
    "feel", "know", "realize", "realise", "understand", "believe", "suppose",
    "resent", "fear", "hope", "wish", "desire", "long", "sense", "suspect",
    "remember", "recall", "notice", "perceive", "consider", "imagine", "regret",
    "doubt", "decide", "determine",
}
# quoted monologue: an explicit thought tag ("..., she thought")
THOUGHT_TAG_RE = re.compile(
    r"\b(?:she|he|i|they|we)\s+(?:thought|wondered|mused|reflected|said\s+to\s+(?:herself|himself|myself)|"
    r"told\s+(?:herself|himself|myself)|asked\s+(?:herself|himself|myself))\b|"
    r"\b(?:thought|wondered|mused|reflected)\s+(?:she|he|i)\b", re.IGNORECASE)
DEICTIC = {"now", "here", "today", "tonight", "tomorrow", "yesterday"}
PAST_MODALS = {"would", "could", "should", "might"}

COHN = [  # (name, sign)
    ("first_person_ratio", +1),
    ("present_aux_ratio", +1),
    ("psycho_verb_density", -1),
    ("thought_tag_density", +1),
    ("fid_question_exclaim", +1),
    ("deictic_in_past_sentence", +1),
]
NON_COHN = ["mean_sentence_length", "mean_word_length", "mattr50", "comma_density",
            "noun_ratio", "adj_ratio", "style_pc1", "style_pc2", "style_pc3"]

# function words for style PCs, excluding pronouns/auxiliaries the Cohn features use
FUNCTION_WORDS = [
    "the", "and", "of", "to", "a", "in", "that", "as", "with", "for", "on", "at",
    "but", "by", "from", "or", "so", "all", "this", "which", "there", "not", "no",
    "an", "into", "out", "up", "then", "if", "when", "what", "one", "some", "more",
    "only", "very", "over", "like", "than", "after", "about", "upon", "before",
    "such", "these", "those", "any", "again", "too", "even", "still", "though",
    "yet", "where", "while", "through", "down", "off", "how", "just", "never",
]


def _person(tok) -> str | None:
    p = tok.morph.get("Person")
    return p[0] if p and tok.tag_ in ("PRP", "PRP$") else None


def features(doc) -> dict:
    toks = [t for t in doc if t.is_alpha]
    lower = [t.lower_ for t in toks]
    words = [t.text for t in toks]
    n = max(len(lower), 1)
    sents = list(doc.sents)
    ns = max(len(sents), 1)

    persons = [_person(t) for t in doc]
    fp, tp = persons.count("1"), persons.count("3")
    # finite verbs only: present VBP/VBZ vs past VBD (modals excluded from both)
    pres = sum(t.tag_ in ("VBP", "VBZ") for t in doc)
    past = sum(t.tag_ == "VBD" for t in doc)
    psycho = sum(t.tag_ == "VBD" and t.lemma_.lower() in PSYCHO_LEMMAS for t in doc)
    text = doc.text

    fid = deic = 0
    for s in sents:
        sp = {_person(t) for t in s}
        past_s = any(t.tag_ == "VBD" or t.lower_ in PAST_MODALS for t in s)
        third_past = "3" in sp and "1" not in sp and past_s
        if s.text.rstrip("\"'”’) \n").endswith(("?", "!")) and third_past:
            fid += 1
        if any(t.lower_ in DEICTIC for t in s) and third_past:
            deic += 1

    win = 50
    if n >= win:
        mattr = np.mean([len(set(lower[i:i + win])) / win for i in range(0, n - win + 1, 10)])
    else:
        mattr = len(set(lower)) / n

    out = {
        "first_person_ratio": fp / (fp + tp) if fp + tp else np.nan,
        "present_aux_ratio": pres / (pres + past) if pres + past else np.nan,
        "psycho_verb_density": 100 * psycho / n,
        "thought_tag_density": 100 * len(THOUGHT_TAG_RE.findall(text)) / n,
        "fid_question_exclaim": fid / ns,
        "deictic_in_past_sentence": deic / ns,
        "mean_sentence_length": n / ns,
        "mean_word_length": float(np.mean([len(w) for w in words])) if words else np.nan,
        "mattr50": mattr,
        "comma_density": 100 * text.count(",") / n,
        "noun_ratio": sum(t.pos_ in ("NOUN", "PROPN") for t in toks) / n,
        "adj_ratio": sum(t.pos_ == "ADJ" for t in toks) / n,
        "_n": n,
    }
    for fw in FUNCTION_WORDS:
        out[f"fw_{fw}"] = lower.count(fw) / n
    return out


def load() -> pd.DataFrame:
    rows = []
    for f in sorted((TV / "scored").glob("*.jsonl")):
        if f.name in SKIP:
            continue
        for line in f.open():
            r = json.loads(line)
            if r.get("reader_id") not in READERS or r.get("error"):
                continue
            if r.get("is_soc") is not True or r.get("mediation_level") is None:
                continue
            rows.append((r["passage_id"], r["author"], r["title"], r["passage_text"],
                         r["reader_id"], float(r["mediation_level"])))
    df = pd.DataFrame(rows, columns=["passage_id", "author", "title", "text", "reader", "level"])
    df["title"] = df["title"].str.replace(r"\s*\(vol\. \d\)", "", regex=True)
    pp = df.groupby(["passage_id", "author", "title"]).agg(
        level=("level", "mean"), text=("text", "first")).reset_index()
    for rd in ("gpt-5.4", "qwen3-max"):
        pp[rd] = pp.passage_id.map(df[df.reader == rd].groupby("passage_id")["level"].mean())
    return pp


def author_icc(pp: pd.DataFrame, col: str) -> dict[str, float]:
    out = {}
    for a, g in pp.groupby("author"):
        g = g[g[col].notna()]
        out[a] = icc_one_way([gg[col].to_numpy(float) for _, gg in g.groupby("title")])
    return out


def rank_of(d: dict, a: str = "Woolf") -> int:
    return 1 + sorted(d, key=lambda k: -d[k]).index(a)


def fmt_row(name: str, d: dict, ref: dict | None = None) -> str:
    cells = " | ".join(f"{d[a]:.3f}" for a in AUTHORS)
    rho = f"{stats.spearmanr([ref[a] for a in AUTHORS], [d[a] for a in AUTHORS]).statistic:+.2f}" if ref else ""
    return f"| {name} | {cells} | {rank_of(d)} | {rho} |"


def main() -> None:
    pp = load().reset_index(drop=True)
    print(f"loaded {len(pp):,} passages", flush=True)
    # spaCy tagging is the slow step; cache it. Single process: n_process>1
    # stalled on della-vis2 (no workers appeared, main process spun threads).
    cache = TV / "cohn_icc_features.pkl"
    if cache.exists() and pd.read_pickle(cache).passage_id.tolist() == pp.passage_id.tolist():
        feats = pd.read_pickle(cache).drop(columns="passage_id")
        print("features from cache", flush=True)
    else:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
        rows = []
        for i, d in enumerate(nlp.pipe(pp.text, batch_size=128)):
            rows.append(features(d))
            if i % 2000 == 0:
                print(f"  tagged {i:,}/{len(pp):,}", flush=True)
        feats = pd.DataFrame(rows)
        feats.assign(passage_id=pp.passage_id).to_pickle(cache)
    pp = pd.concat([pp, feats], axis=1)

    # style PCs from function-word profile (z-scored), top 3 components
    fw = pp[[c for c in pp.columns if c.startswith("fw_")]].to_numpy()
    fw = (fw - fw.mean(0)) / (fw.std(0) + 1e-12)
    u, s, vt = np.linalg.svd(fw, full_matrices=False)
    for i in range(3):
        pp[f"style_pc{i + 1}"] = u[:, i] * s[i]
    var_expl = (s[:3] ** 2) / (s ** 2).sum()

    # theory-signed Cohn composite: no model labels involved
    z = pd.DataFrame({c: (pp[c] - pp[c].mean()) / pp[c].std() for c, _ in COHN})
    pp["cohn_composite"] = sum(sign * z[c].fillna(0) for c, sign in COHN) / len(COHN)

    # Test 3: predicted level from Cohn features, leave-one-author-out
    X = pp[[c for c, _ in COHN]].fillna(pp[[c for c, _ in COHN]].median()).to_numpy()
    y = pp["level"].to_numpy()
    pred_lin = np.zeros_like(y)
    pred_gb = np.zeros_like(y)
    for a in AUTHORS:
        tr, te = (pp.author != a).to_numpy(), (pp.author == a).to_numpy()
        sc = StandardScaler().fit(X[tr])
        lin = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(X[tr]), y[tr])
        pred_lin[te] = lin.predict(sc.transform(X[te]))
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X[tr], y[tr])
        pred_gb[te] = gb.predict(X[te])
    pp["pred_linear"], pp["pred_gb"] = pred_lin, pred_gb
    pp["resid_linear"], pp["resid_gb"] = y - pred_lin, y - pred_gb

    def r2(p):
        return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    model = author_icc(pp, "level")
    L = []
    L.append("# Is the per-author ICC a measure of Cohn's scale?\n")
    L.append(f"Passages: {len(pp):,} SOC-positive passages, {pp.title.nunique()} titles, 7 authors "
             "(pooled gpt-5.4 + qwen3-max + qwen3.6-local, per-passage mean; the chapter's ICC basis). "
             "Each table cell is that author's one-way ICC across their titles. "
             "**Woolf rank** = her position among the 7 authors (1 = highest). "
             "**ρ vs readers** = Spearman correlation, across the 7 authors, between that row's ICCs and "
             "the readers' ICCs (n = 7: descriptive, not a significance test).\n")
    head = "| measure | " + " | ".join(AUTHORS) + " | Woolf rank | ρ vs readers |"
    sep = "|---" * (len(AUTHORS) + 3) + "|"

    L.append("## Reference: the readers' mediation level\n")
    L += [head, sep, fmt_row("**readers, pooled**", model),
          fmt_row("gpt-5.4 only", author_icc(pp, "gpt-5.4"), model),
          fmt_row("qwen3-max only", author_icc(pp, "qwen3-max"), model)]

    # how each feature relates to level, for interpreting both tests
    L.append("\n## Do the features track level at all? (passage-level Spearman ρ with pooled level)\n")
    L.append("| feature | expected | ρ with level | within-author ρ |")
    L.append("|---|---|---|---|")
    for c in [c for c, _ in COHN] + ["cohn_composite"] + NON_COHN:
        sign = dict(COHN).get(c)
        ok = pp[c].notna()
        rho = stats.spearmanr(pp.loc[ok, c], pp.loc[ok, "level"]).statistic
        dm = pp.loc[ok].copy()
        for col in (c, "level"):
            dm[col] = dm[col] - dm.groupby("author")[col].transform("mean")
        rw = stats.spearmanr(dm[c], dm["level"]).statistic
        exp = {1: "+", -1: "−"}.get(sign, "+" if c == "cohn_composite" else "none")
        L.append(f"| {c} | {exp} | {rho:+.3f} | {rw:+.3f} |")

    L.append("\n## Test 1 — convergent: ICC of Cohn-grounded features (no model labels)\n")
    L += [head, sep]
    cohn_iccs = {}
    for c, _ in COHN:
        cohn_iccs[c] = author_icc(pp, c)
        L.append(fmt_row(c, cohn_iccs[c], model))
    comp = author_icc(pp, "cohn_composite")
    L.append(fmt_row("**Cohn composite (theory-signed)**", comp, model))

    L.append("\n## Test 2 — discriminant: ICC of features outside Cohn's definitions\n")
    L.append(f"Style PCs are the first three principal components of {len(FUNCTION_WORDS)} function-word "
             f"frequencies (pronouns and auxiliaries excluded), explaining "
             f"{', '.join(f'{v:.1%}' for v in var_expl)} of their variance.\n")
    L += [head, sep]
    non = {}
    for c in NON_COHN:
        non[c] = author_icc(pp, c)
        L.append(fmt_row(c, non[c], model))
    mean_non = {a: float(np.mean([non[c][a] for c in NON_COHN])) for a in AUTHORS}
    L.append(fmt_row("**mean of non-Cohn ICCs**", mean_non, model))
    mean_cohn = {a: float(np.mean([cohn_iccs[c][a] for c, _ in COHN])) for a in AUTHORS}
    L.append(fmt_row("mean of Cohn-feature ICCs (Test 1)", mean_cohn, model))

    L.append("\n## Test 3 — decomposition: the part of reader level that Cohn's grammar predicts\n")
    L.append(f"Leave-one-author-out prediction of pooled level from the {len(COHN)} Cohn features. "
             f"Out-of-author R²: linear (ridge) {r2(pred_lin):.3f}, gradient boosting {r2(pred_gb):.3f}. "
             "ICC is recomputed on the predicted values and on the residuals.\n")
    L += [head, sep, fmt_row("readers (observed)", model)]
    dec = {}
    for c in ("pred_linear", "resid_linear", "pred_gb", "resid_gb"):
        dec[c] = author_icc(pp, c)
        L.append(fmt_row(c, dec[c], model))

    # title-level: how well do predicted title means track observed title means, within author
    L.append("\n### Title means: observed vs Cohn-predicted (gradient boosting), within author\n")
    tm = pp.groupby(["author", "title"]).agg(obs=("level", "mean"), pred=("pred_gb", "mean"),
                                              resid=("resid_gb", "mean"), n=("level", "size")).reset_index()
    L.append("| author | titles | Spearman(observed, predicted title means) |")
    L.append("|---|---|---|")
    for a in AUTHORS:
        g = tm[tm.author == a]
        L.append(f"| {a} | {len(g)} | {stats.spearmanr(g.obs, g.pred).statistic:+.2f} |")
    L.append("\nWoolf title means (observed level, Cohn-predicted, residual):\n")
    L.append("| title | n | observed | predicted | residual |")
    L.append("|---|---|---|---|---|")
    for _, r in tm[tm.author == "Woolf"].sort_values("obs").iterrows():
        L.append(f"| {r.title} | {r.n} | {r.obs:.2f} | {r.pred:.2f} | {r.resid:+.2f} |")

    # bootstrap CIs for Woolf on the key measures
    L.append("\n### Woolf, 95% cluster-bootstrap CIs (passages resampled within title, 1,000 draws)\n")
    w = pp[pp.author == "Woolf"]
    for c in ("level", "cohn_composite", "pred_gb", "resid_gb", "mean_sentence_length", "style_pc1"):
        groups = [g[c].dropna().to_numpy(float) for _, g in w.groupby("title")]
        lo, hi = bootstrap_icc_ci(groups, n_boot=1000)
        L.append(f"- {c}: {icc_one_way(groups):.3f} [{lo:.3f}, {hi:.3f}]")

    # leave-out: does each result survive dropping the author's most extreme title?
    L.append("\n## Without each author's most extreme title (highest reader-level mean)\n")
    L.append("Woolf's chapter claim has two parts: *The Waves* departs, and her other nine novels still vary "
             "more than other authors' do. This separates which part each measure supports.\n")
    cols = ["level", "cohn_composite", "first_person_ratio", "present_aux_ratio", "pred_gb", "resid_gb"]
    L.append("| author | titles | " + " | ".join(cols) + " | mean Cohn-feature ICC | mean non-Cohn ICC |")
    L.append("|---" * (len(cols) + 4) + "|")
    for a in AUTHORS:
        g = pp[pp.author == a]
        top = g.groupby("title")["level"].mean().idxmax()
        for lab, gg in (("all", g), (f"without *{top}*", g[g.title != top])):
            def icc(c, gg=gg):
                return icc_one_way([x[c].dropna().to_numpy(float) for _, x in gg.groupby("title")])
            vals = [icc(c) for c in cols]
            coh = np.mean([icc(c) for c, _ in COHN])
            non = np.mean([icc(c) for c in NON_COHN])
            L.append(f"| {a} | {lab} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {coh:.3f} | {non:.3f} |")

    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
