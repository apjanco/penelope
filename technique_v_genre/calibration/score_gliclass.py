"""Score the calibration set with GLiClass — a self-hostable System One model.

This is the "Jev-like, but runs on Della" candidate. GLiClass is a zero-shot
classifier that takes label *descriptions* rather than a trained label set,
so the taxonomy from prompts.py can be handed to it verbatim, and it returns
a score per label — the same probability-per-category shape that makes Jev
interesting for measuring ambiguity directly rather than inferring it from
cross-reader disagreement.

Scale worth noting against what it replaces: this is a ~1.3GB model running
on one GPU, where llama-3.3-70B was 132GB across two A100s for 12h47m. Both
local candidates run on della-vis2's A40s, so neither needs the SLURM queue.

`single-label` classification because soc_type and mediation_level are each
a forced choice — the generative readers pick exactly one, and the
comparison has to be like-for-like. The full score vector is kept anyway;
that is the part no generative reader gives us.

soc_type and mediation_level are asked as two independent classifications,
for the same reason as in score_jev.py: across gpt-5.4, qwen3-max and llama,
mediation_level is a near-deterministic function of soc_type, and asking
separately shows whether that coupling is in the texts or in the models.

Usage:  python calibration/score_gliclass.py [--k 1] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TV = HERE.parent
sys.path.insert(0, str(HERE))

from score_jev import MEDIATION, SOC_TYPES  # same criteria text, one source  # noqa: E402

MODEL_ID = "knowledgator/gliclass-instruct-large-v1.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "scored_gliclass.jsonl"))
    args = ap.parse_args()

    import torch
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    print(f"  loading {MODEL_ID} on {args.device} ...")
    t0 = time.time()
    model = GLiClassModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    pipe = ZeroShotClassificationPipeline(
        model, tok, classification_type="single-label",
        device=args.device, max_length=2048, progress_bar=False)
    print(f"  loaded in {time.time()-t0:.1f}s")

    rows = [json.loads(l) for l in
            (HERE / "calibration_set.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    # Label text = "name: description" so the description actually informs the
    # embedding; GLiClass matches against the label string it is given.
    soc_labels = {f"{k}: {v}": k for k, v in SOC_TYPES.items()}
    med_labels = {f"level {k}: {v}": int(k) for k, v in MEDIATION.items()}

    texts = [r["passage_text"] for r in rows]
    t0 = time.time()
    soc_res = pipe(texts, list(soc_labels), threshold=0.0, batch_size=args.batch_size)
    med_res = pipe(texts, list(med_labels), threshold=0.0, batch_size=args.batch_size)
    dt = time.time() - t0
    print(f"  classified {len(texts)} passages x2 questions in {dt:.1f}s "
          f"({dt/max(len(texts),1)*1000:.0f} ms/passage)")

    out = Path(args.out)
    with out.open("w") as fh:
        for r, sres, mres in zip(rows, soc_res, med_res):
            s_sorted = sorted(sres, key=lambda d: -d["score"])
            m_sorted = sorted(mres, key=lambda d: -d["score"])
            s_top, m_top = s_sorted[0], m_sorted[0]
            s_probs = {soc_labels[d["label"]]: round(d["score"], 4) for d in sres}
            m_probs = {str(med_labels[d["label"]]): round(d["score"], 4) for d in mres}
            fh.write(json.dumps({
                "passage_id": r["passage_id"], "source_file": r["source_file"],
                "author": r.get("author"), "title": r.get("title"), "year": r.get("year"),
                "role": "calibration", "reader_id": "gliclass", "run": 0,
                "passage_text": r["passage_text"],
                # GLiClass has no is_soc question; every calibration passage was
                # already judged SoC by both cloud readers, so this is not a
                # detection test and is_soc is asserted, not measured.
                "is_soc": True,
                "soc_type": soc_labels[s_top["label"]],
                "mediation_level": med_labels[m_top["label"]],
                "gliclass_soc_type_probs": s_probs,
                "gliclass_soc_type_top": round(s_top["score"], 4),
                "gliclass_mediation_probs": m_probs,
                "gliclass_mediation_top": round(m_top["score"], 4),
            }) + "\n")
    print(f"  wrote {len(rows)} rows -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
