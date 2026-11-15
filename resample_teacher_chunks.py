"""Resample two specific chunks through teacher_gold.py's own chunk-level
teacher_relabel() k times each, at both the production temperature (0.2) and
an exploratory temperature (0.8), to test whether the narrator_position:
present + indirect_interior_monologue reading for the Bertaux/embrace
passages is a stable fixed point of the teacher's own methodology or a
single-pass artifact.

Must run on the GPU node serving the vLLM teacher endpoint (via
teacher_della.slurm's server, not from a login/vis node).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/gpfs/MM4/apjanco/penelope")

import httpx
from openai import OpenAI

from scripts.teacher_gold import (
    _TokenCounter,
    _discover_base_url,
    teacher_relabel,
)
from scripts.soc_chunker import parse_chunked_file

MODEL = "Qwen/Qwen3.6-27B"
TARGET_CHUNKS = {
    "pg02413-madame-bovary_003": "chunking_positives/pg02413_madame_bovary.txt",
    "pg03019-pointed-roofs_002": "chunking_positives/pg03019_pointed_roofs.txt",
}
K = 3
TEMPS = [0.2, 0.8]
MAX_MODEL_LEN = 32768
MAX_TOKENS = 16384

out_path = Path("/scratch/gpfs/MM4/apjanco/penelope/resample_teacher_results.jsonl")


def main():
    base_url = _discover_base_url(None, Path("/scratch/gpfs/MM4/apjanco/penelope/teacher_endpoint.txt"))
    print("base_url:", base_url)
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    count_fn = _TokenCounter(base_url, MODEL, httpx.Client())

    with out_path.open("w") as fh:
        for chunk_id, file_path in TARGET_CHUNKS.items():
            chunks = {c.chunk_id: c for c in parse_chunked_file(Path("/scratch/gpfs/MM4/apjanco/penelope") / file_path)}
            chunk = chunks[chunk_id]
            for temp in TEMPS:
                for run in range(K):
                    instances, raw = teacher_relabel(
                        client, MODEL, chunk,
                        temperature=temp,
                        max_tokens=MAX_TOKENS,
                        retries=2,
                        count_fn=count_fn,
                        max_model_len=MAX_MODEL_LEN,
                    )
                    # Find the instance matching the passage of interest (longest one)
                    matches = sorted(instances, key=lambda i: len(i.passage), reverse=True)
                    top = matches[0] if matches else None
                    record = {
                        "chunk_id": chunk_id,
                        "temperature": temp,
                        "run": run,
                        "n_instances": len(instances),
                        "top_soc_type": top.soc_type if top else None,
                        "top_narrator_position": top.narrator_position if top else None,
                        "top_is_soc": top.is_soc if top else None,
                        "top_passage_preview": (top.passage[:100] if top else None),
                    }
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()
                    print(chunk_id, "temp=", temp, "run=", run, "->",
                          record["top_soc_type"], record["top_narrator_position"],
                          f"({record['n_instances']} instances total)")
    print("Written to", out_path)


if __name__ == "__main__":
    main()
