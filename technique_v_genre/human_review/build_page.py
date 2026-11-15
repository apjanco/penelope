"""Generate annotate.html — a self-contained static page, no server, no network.

The passages and the machine readings are injected as JSON rather than fetched,
for two reasons: a page opened over file:// cannot fetch a sibling JSON file
(the browser treats it as a cross-origin request), and a single file is what
makes the instrument archivable alongside the chapter.

A build stamp goes in too: the sha256 of each input and the build time. Every
export carries it, so an annotations file can always be matched to the exact
passage set and machine readings it was produced against, and the page warns
if you load a file built against a different set.
"""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def inject(name):
    raw = (HERE / name).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    # Escape "<" so no passage can close the script tag early.
    return json.dumps(json.loads(raw), ensure_ascii=False).replace("<", "\\u003c"), sha


passages, set_sha = inject("annotation_set.json")
machine, machine_sha = inject("machine_readings.json")
build = json.dumps({
    "set_sha": set_sha,
    "machine_sha": machine_sha,
    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
})

html = (HERE / "annotate.template.html").read_text()
for marker, value in (("/*__PASSAGES__*/null", passages),
                      ("/*__MACHINE__*/null", machine),
                      ("/*__BUILD__*/null", build)):
    assert marker in html, f"marker missing from template: {marker}"
    html = html.replace(marker, value)

(HERE / "annotate.html").write_text(html)

# _site/ is what goes to a static host. It holds the page and nothing else.
# Everything else in this folder must stay out of it: passage_key.json maps
# the opaque codes back to novel and author and carries the machine-derived
# reference level used to stratify section B, so publishing it would undo the
# blinding the codes exist to provide.
site = HERE / "_site"
site.mkdir(exist_ok=True)
(site / "index.html").write_text(html)

# Keep the excerpts out of search indexes and out of archive crawls. The page
# is for two or three named readers, not for the open web.
(site / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
(site / "_headers").write_text(
    "/*\n"
    "  X-Robots-Tag: noindex, nofollow, noarchive\n"
    "  Referrer-Policy: no-referrer\n"
    "  X-Content-Type-Options: nosniff\n")

print(f"annotate.html  {len(html):,} bytes")
print(f"  passage set      sha256 {set_sha[:16]}…")
print(f"  machine readings sha256 {machine_sha[:16]}…")
print(f"\n_site/ ready for upload — {len(list(site.iterdir()))} files:")
for f in sorted(site.iterdir()):
    print(f"    {f.name:12s} {f.stat().st_size:>9,} bytes")
