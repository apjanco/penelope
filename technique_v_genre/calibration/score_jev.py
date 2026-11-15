"""Score the calibration set with TypeSafe's Jev (System One) model.

Jev is not chat-completions, so it cannot go through score.py or
models.yaml: it takes a `state` plus typed questions and returns typed
answers with a probability distribution per question. That difference is
the reason to try it here. Every reader so far returns a hard label, and
the chapter's subject is ambiguity — currently inferred indirectly from
cross-reader disagreement, which AMBIGUITY.md warns conflates
cline-adjacency, detection uncertainty and model noise. A per-passage
probability distribution measures that directly.

Three questions per passage, mirroring what the generative readers emit so
evaluate_reader.py can score Jev on the same footing:

  is_soc           noul   — does the passage render consciousness at all
  soc_type         choice — the Humphrey/Steinberg taxonomy from prompts.py
  mediation_level  choice — Cohn's 0-3 cline, asked INDEPENDENTLY

That last point is deliberate. Across gpt-5.4, qwen3-max and llama,
mediation_level turned out to be a near-deterministic function of soc_type
(indirect -> L1 at 99-100%, omniscient -> L0 at 100%), so the schema's
"three orthogonal outputs" are not orthogonal in practice. Jev evaluates
each question in isolation against the same state, so asking both lets us
see whether the coupling is a property of the texts or an artifact of one
model choosing a type and then rationalising a level to match.

Rows are written in score.py's shape, plus the probability distributions,
which are the point.

Usage:  python calibration/score_jev.py [--k 1] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TV = HERE.parent
sys.path.insert(0, str(TV.parent))

from dotenv import load_dotenv  # noqa: E402

API = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

# Criteria text lifted from prompts.py's taxonomy so Jev is judged against
# the same definitions the generative readers were given, not a paraphrase.
SOC_TYPES = {
    "direct_interior_monologue": "First-person present-tense unmediated thought; no narrator framing.",
    "indirect_interior_monologue": "Third-person syntax with interior focalization; free indirect discourse.",
    "omniscient_description": "Narrator articulates character psychology with authority, from outside.",
    "soliloquy": "Character speaks aloud alone; theatrical address; rhetorical.",
    "free_association": "Logic-defying leaps; sound or image chaining over sense.",
    "imagery": "Extended metaphor or sensory image that IS the thought, not decoration.",
    "simulation_state_of_mind": "Renders emotional or perceptual state rather than propositional thought.",
    "reverie_fantasy": "Daydream, memory or wish diverging from present action.",
    "hybrid": "Two or more primary types inseparably fused in one passage.",
    "other_soc": "Clearly interior but resists the listed categories.",
}
MEDIATION = {
    "0": "Psycho-narration: the narrator reports the mind from outside ('she felt uneasy').",
    "1": "Narrated monologue (free indirect discourse): the character's thought bends the narrator's sentence, no quotation marks.",
    "2": "Quoted monologue: framed direct interior speech ('I am uneasy, she thought').",
    "3": "Autonomous monologue: unframed raw interior voice, no narratorial frame at all.",
}


def ask(state: str, key: str, retries: int = 4) -> dict | None:
    body = {
        "state": state,
        "model": MODEL,
        "questions": {
            "is_soc": {
                "type": "noul",
                "instructions": "This passage renders a character's consciousness "
                                "(thought, perception or feeling from within), rather than "
                                "only external action, dialogue or scene description.",
            },
            "soc_type": {
                "type": "choice",
                "instructions": "Which stream-of-consciousness technique best fits this passage?",
                "criteria": SOC_TYPES,
            },
            "mediation_level": {
                "type": "choice",
                "instructions": "How mediated is the rendering of consciousness, on Cohn's cline? "
                                "Judge the degree of narratorial framing, independently of "
                                "which technique label fits.",
                "criteria": MEDIATION,
            },
        },
    }
    data = json.dumps(body).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(
                API, data=data,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:200]
            print(f"    HTTP {e.code}: {detail}")
            if e.code in (400, 401, 403, 404, 422):
                return {"_fatal": f"{e.code}: {detail}"}
            time.sleep(3 * (i + 1))
        except Exception as e:  # noqa: BLE001
            print(f"    attempt {i+1} failed: {e}")
            time.sleep(3 * (i + 1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=1,
                    help="default 1: Jev returns a probability distribution per "
                         "question, so uncertainty is measured directly rather than "
                         "inferred from resampling. Its k-run variance on the "
                         "calibration set was 5%, so k=3 would triple the cost to "
                         "re-derive what the probabilities already state.")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--input", default=str(HERE / "calibration_set.jsonl"),
                    help="passages JSONL; defaults to the calibration set")
    ap.add_argument("--out", default=str(HERE / "scored_jev.jsonl"))
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    load_dotenv(TV.parent / ".env")
    key = os.environ.get("TYPESAFE_JEV")
    if not key:
        print("TYPESAFE_JEV not set in .env"); return 1

    rows = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                done.add((d["passage_id"], d["run"]))
    tasks = [(r, run) for r in rows for run in range(args.k)
             if (r["passage_id"], run) not in done]
    print(f"  {len(rows):,} passages x k={args.k}; {len(tasks):,} calls remaining after resume-skip")
    fh = out.open("a")
    lock = threading.Lock()
    n = 0
    fatal = []

    def work(task):
        r, run = task
        return r, run, ask(r["passage_text"], key)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for r, run, resp in pool.map(work, tasks):
            if resp is None or "_fatal" in resp:
                fatal.append(resp)
                if len(fatal) >= 10:
                    print(f"  aborting after {len(fatal)} unrecoverable failures: {resp}")
                    break
                continue
            a = resp.get("answers", {})
            ml = a.get("mediation_level", {}).get("choice")
            rec = {
                "passage_id": r["passage_id"], "source_file": r["source_file"],
                "author": r.get("author"), "title": r.get("title"), "year": r.get("year"),
                "role": r.get("role", "core"), "reader_id": "jev", "run": run,
                "passage_text": r["passage_text"],
                "is_soc": (a.get("is_soc", {}).get("noul", 0) >= 0.5),
                "soc_type": a.get("soc_type", {}).get("choice"),
                "mediation_level": int(ml) if ml is not None else None,
                # what no other reader gives us:
                "jev_is_soc_noul": a.get("is_soc", {}).get("noul"),
                "jev_soc_type_probs": a.get("soc_type", {}).get("probabilities"),
                "jev_soc_type_confidence": a.get("soc_type", {}).get("confidence"),
                "jev_mediation_probs": a.get("mediation_level", {}).get("probabilities"),
                "jev_mediation_confidence": a.get("mediation_level", {}).get("confidence"),
                "jev_usage": resp.get("usage"),
                "jev_model": resp.get("model"),
            }
            with lock:
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                n += 1
                if n % 250 == 0:
                    print(f"  {n:,}/{len(tasks):,} calls...")
    fh.close()
    print(f"  wrote {n} new rows -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
