"""Validate and normalise exported annotation files into annotations/.

Each annotator exports one JSON file from the page. Drop those files into
human_review/exports/ and run this. It checks them against the passage set
they were built from, assigns stable anonymous labels, and writes the files
analyze_agreement.py reads.

The build stamp check is the point of doing this in a script rather than by
hand: an annotations file carries the sha256 of the passage set and of the
machine readings it was produced against, so a file coded against an older
set cannot silently be pooled with a newer one.
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
exports = HERE / "exports"
out = HERE / "annotations"

if not exports.exists():
    sys.exit(f"no exports/ directory — put each annotator's exported .json in {exports}")

set_sha = hashlib.sha256((HERE / "annotation_set.json").read_bytes()).hexdigest()
codes = {p["code"] for p in json.loads((HERE / "annotation_set.json").read_text())}
out.mkdir(exist_ok=True)

reg_path = HERE / "annotators.json"
reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}

files = sorted(exports.glob("*.json"))
if not files:
    sys.exit(f"no .json files in {exports}")

problems = []
docs = []
for f in files:
    try:
        d = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        problems.append(f"{f.name}: not valid JSON ({e})"); continue
    if not isinstance(d.get("responses"), dict):
        problems.append(f"{f.name}: no responses object"); continue
    built = (d.get("built") or {}).get("set_sha")
    if built != set_sha:
        problems.append(f"{f.name}: built against passage set {str(built)[:12]}, "
                        f"current set is {set_sha[:12]} — recode or rebuild, do not pool")
        continue
    stray = set(d["responses"]) - codes
    if stray:
        problems.append(f"{f.name}: {len(stray)} unknown passage codes, e.g. {sorted(stray)[:3]}")
        continue
    docs.append((d.get("exported") or "", f, d))

if problems:
    print("REJECTED:")
    for p in problems:
        print("  " + p)
    if not docs:
        sys.exit(1)

docs.sort()
print(f"\n{len(docs)} annotation file(s) accepted:\n")
for _, f, d in docs:
    who = d.get("label") or f.stem
    if who not in reg:
        reg[who] = f"annotator_{chr(ord('a') + len(reg))}"
    lab = reg[who]
    resp = d["responses"]
    coded = sum(1 for r in resp.values()
                if r.get("is_soc") is False or isinstance(r.get("mediation_level"), int))
    full = sum(1 for r in resp.values()
               if r.get("is_soc") is False
               or (isinstance(r.get("mediation_level"), int) and r.get("soc_type") and r.get("confidence")))
    revealed = sum(1 for r in resp.values() if r.get("saw_machine"))
    modes = {}
    for r in resp.values():
        modes[r.get("order_mode", "?")] = modes.get(r.get("order_mode", "?"), 0) + 1
    (out / f"{lab}.json").write_text(json.dumps({
        "label": lab, "source_file": f.name, "built": d.get("built"),
        "exported": d.get("exported"), "mode_last_used": d.get("mode_last_used"),
        "responses": resp,
    }, indent=1))
    print(f"  {lab:12s} <- {f.name}")
    print(f"      {coded} coded, {full} fully coded, {revealed} after a machine reveal")
    print(f"      order modes: {modes}")

reg_path.write_text(json.dumps(reg, indent=1))
print(f"\nwrote annotations/*.json — now run: python analyze_agreement.py")
