"""Tests for verify_corpus.py — including proof that each layer can fail.

The corpus currently passes every check, which on its own is worth very
little: a verifier that has never rejected anything is indistinguishable
from a verifier that cannot reject anything. So most of what is below
deliberately breaks a manifest entry in memory and asserts the specific
failure is caught, with the wrong-book case — the actual failure this
project has hit, where a mistyped Gutenberg ID lands real text under a
plausible filename — tested against real cached Gutenberg metadata rather
than a stub.

stdlib unittest rather than pytest: pytest is not installed in the
penelope env and this needs to be runnable without adding a dependency.
pytest will collect these classes unchanged if it is ever added.

    python -m unittest test_corpus_integrity -v      # offline + cached
    python -m unittest test_corpus_integrity.TestRealCorpus -v
"""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import verify_corpus as vc


def load_manifest() -> list[dict]:
    return json.loads(vc.MANIFEST_PATH.read_text())["corpus"]


def entry_for(manifest: list[dict], source_file: str) -> dict:
    for e in manifest:
        if e["source_file"] == source_file:
            return e
    raise AssertionError(f"{source_file} not in manifest")


def failures_for(manifest: list[dict], source_file: str, *, online: bool) -> list[str]:
    """Run the verifier over a (possibly mutated) manifest, one file's failures."""
    res = vc.verify(manifest, online=online, use_cache=True)
    return [f for f in res.failures if f.startswith(f"{source_file}:")]


class TestRealCorpus(unittest.TestCase):
    """The corpus as it actually stands must pass."""

    def test_offline_layers_pass(self):
        res = vc.verify(load_manifest(), online=False)
        self.assertEqual(res.failures, [], "offline verification failed")
        self.assertEqual(res.warnings, [], "offline verification raised warnings")

    def test_gutenberg_provenance_passes(self):
        """Layer 2, against cached RDF — no network needed once populated."""
        res = vc.verify(load_manifest(), online=True, use_cache=True)
        self.assertEqual(res.failures, [], "Gutenberg provenance failed")

    def test_every_entry_is_pinned(self):
        for e in load_manifest():
            with self.subTest(work=e["source_file"]):
                self.assertIn("sha256", e)
                self.assertIn("opening", e)
                self.assertIn("title_on_titlepage", e)

    def test_entries_without_external_records_document_their_origin(self):
        """Every text must be traceable to SOMETHING: a Gutenberg id, a
        Standard Ebooks id, or — failing both — a written note conceding
        the origin is unrecoverable. Updated when the SE path was added:
        an SE id is stronger provenance than a freetext note, not weaker,
        so requiring the note of SE entries was testing the old rule."""
        for e in load_manifest():
            if e.get("gutenberg_id") or e.get("standard_ebooks_id"):
                continue
            with self.subTest(work=e["source_file"]):
                self.assertTrue(
                    e.get("provenance_note"),
                    "a file with no external record must say where it came from",
                )

    def test_every_entry_has_some_provenance(self):
        for e in load_manifest():
            with self.subTest(work=e["source_file"]):
                self.assertTrue(
                    e.get("gutenberg_id") or e.get("standard_ebooks_id")
                    or e.get("provenance_note"),
                    "entry has no provenance of any kind",
                )


class TestCatchesWrongBook(unittest.TestCase):
    """Layer 2: the failure mode that motivated all of this."""

    def test_wrong_gutenberg_id_is_caught(self):
        """Point Pointed Roofs at Ulysses' ID; the title check must reject it."""
        man = load_manifest()
        e = entry_for(man, "pg03019_pointed_roofs.txt")
        e["gutenberg_id"] = 4300  # Ulysses
        fails = failures_for(man, "pg03019_pointed_roofs.txt", online=True)
        self.assertTrue(any("WRONG BOOK" in f for f in fails), fails)

    def test_wrong_author_is_caught(self):
        man = load_manifest()
        entry_for(man, "pg04300_ulysses.txt")["author"] = "Woolf"
        fails = failures_for(man, "pg04300_ulysses.txt", online=True)
        self.assertTrue(any("is by" in f for f in fails), fails)

    def test_volume_swap_against_gutenberg_is_caught(self):
        """The clean case: a title that names no volume, so the only claim
        is the manifest's `volume` field and the only contradiction is
        Gutenberg's. The Tunnel is Pilgrimage vol.4; claim it is vol.5."""
        man = load_manifest()
        entry_for(man, "pg56447_the_tunnel.txt")["volume"] = 5
        fails = failures_for(man, "pg56447_the_tunnel.txt", online=True)
        self.assertTrue(any("volume mismatch" in f for f in fails), fails)
        self.assertFalse(any("contradicts itself" in f for f in fails), fails)

    def test_volume_swap_in_a_numbered_title_is_caught(self):
        """Wings of the Dove vol.1 vs vol.2 — passes a title check trivially,
        since both volumes carry the same title."""
        man = load_manifest()
        entry_for(man, "pg29452_the_wings_of_the_dove_volume_1.txt")["volume"] = 2
        fails = failures_for(man, "pg29452_the_wings_of_the_dove_volume_1.txt", online=True)
        self.assertTrue(any("volume mismatch" in f for f in fails), fails)

    def test_manifest_contradicting_itself_about_volume_is_caught(self):
        man = load_manifest()
        e = entry_for(man, "pg29452_the_wings_of_the_dove_volume_1.txt")
        e["volume"] = 2  # title still reads "(vol. 1)"
        fails = failures_for(man, "pg29452_the_wings_of_the_dove_volume_1.txt", online=True)
        self.assertTrue(any("contradicts itself" in f for f in fails), fails)

    def test_filename_id_must_agree_with_manifest_id(self):
        man = load_manifest()
        entry_for(man, "pg09366_mary_olivier_a_life.txt")["gutenberg_id"] = 9298
        fails = failures_for(man, "pg09366_mary_olivier_a_life.txt", online=False)
        self.assertTrue(any("contradicts manifest gutenberg_id" in f for f in fails), fails)

    def test_year_outside_authors_lifespan_is_caught(self):
        man = load_manifest()
        entry_for(man, "pg09366_mary_olivier_a_life.txt")["year"] = 1850  # Sinclair b. 1863
        fails = failures_for(man, "pg09366_mary_olivier_a_life.txt", online=True)
        self.assertTrue(any("birthday" in f for f in fails), fails)


class TestCatchesWrongBookStandardEbooks(unittest.TestCase):
    """Layer 2 for the Standard Ebooks path, which has its own metadata
    source (a GitHub content.opf) and its own author-name convention
    ("William Faulkner", not "Faulkner, William")."""

    def test_wrong_se_id_is_caught(self):
        man = load_manifest()
        entry_for(man, "se_william_faulkner_soldiers_pay.txt")["standard_ebooks_id"] = (
            "ford-madox-ford/the-last-post")
        fails = failures_for(man, "se_william_faulkner_soldiers_pay.txt", online=True)
        self.assertTrue(any("WRONG BOOK" in f for f in fails), fails)
        self.assertTrue(any("is by" in f for f in fails), fails)

    def test_se_entries_are_actually_checked(self):
        """Guards against the SE branch silently not running: every
        SE-sourced entry must resolve to a cached/fetchable content.opf."""
        man = load_manifest()
        se = [e for e in man if e.get("standard_ebooks_id")]
        self.assertTrue(se, "no Standard Ebooks entries in the manifest")
        for e in se:
            with self.subTest(work=e["source_file"]):
                opf = vc.fetch_se_opf(e["standard_ebooks_id"].replace("/", "_"))
                self.assertIn("dc:title", opf)

    def test_se_author_name_order(self):
        """SE writes natural order; Gutenberg writes catalogue order."""
        self.assertEqual(vc.surname("William Faulkner"), "faulkner")
        self.assertEqual(vc.surname("Ford Madox Ford"), "ford")
        self.assertEqual(vc.surname("Faulkner, William"), "faulkner")


class TestCatchesContentDrift(unittest.TestCase):
    """Layer 1: the file is no longer what it was when it was verified."""

    def test_changed_content_is_caught(self):
        man = load_manifest()
        entry_for(man, "pg13292_the_romantic.txt")["sha256"] = "0" * 64
        fails = failures_for(man, "pg13292_the_romantic.txt", online=False)
        self.assertTrue(any("file changed" in f for f in fails), fails)

    def test_changed_opening_is_caught(self):
        man = load_manifest()
        entry_for(man, "pg13292_the_romantic.txt")["opening"] = "chapter i it was a dark and stormy night"
        fails = failures_for(man, "pg13292_the_romantic.txt", online=False)
        self.assertTrue(any("opening changed" in f for f in fails), fails)

    def test_word_count_drift_is_caught(self):
        man = load_manifest()
        entry_for(man, "pg13292_the_romantic.txt")["word_count"] = 1
        fails = failures_for(man, "pg13292_the_romantic.txt", online=False)
        self.assertTrue(any("word_count" in f for f in fails), fails)

    def test_same_text_under_two_filenames_is_caught(self):
        """The actual pg05670_jacobs_room error: one text, two filenames,
        two manifest entries. Needs a real duplicate file on disk, so this
        runs against a temporary positives/ rather than the real one."""
        man = load_manifest()
        original = entry_for(man, "pg13292_the_romantic.txt")
        dup = copy.deepcopy(original)
        dup["source_file"] = "pg99001_a_different_book.txt"
        dup["title"] = "A Different Book"
        dup["gutenberg_id"] = 99001
        man.append(dup)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for e in man:
                src = vc.POSITIVES_DIR / e["source_file"]
                if src.exists():
                    shutil.copy2(src, tmp / e["source_file"])
            shutil.copy2(vc.POSITIVES_DIR / original["source_file"],
                         tmp / dup["source_file"])
            real = vc.POSITIVES_DIR
            try:
                vc.POSITIVES_DIR = tmp
                res = vc.verify(man, online=False)
            finally:
                vc.POSITIVES_DIR = real

        self.assertTrue(
            any("byte-identical duplicate" in f for f in res.failures),
            res.failures,
        )

    def test_repeated_manifest_entry_is_caught(self):
        man = load_manifest()
        man.append(copy.deepcopy(entry_for(man, "pg13292_the_romantic.txt")))
        res = vc.verify(man, online=False)
        self.assertTrue(any("listed twice" in f for f in res.failures), res.failures)

    def test_missing_file_is_caught(self):
        man = load_manifest()
        entry_for(man, "pg13292_the_romantic.txt")["source_file"] = "pg99999_does_not_exist.txt"
        fails = failures_for(man, "pg99999_does_not_exist.txt", online=False)
        self.assertTrue(any("missing" in f for f in fails), fails)


class TestCatchesUndocumentedProvenance(unittest.TestCase):
    def test_non_gutenberg_file_without_a_note_is_caught(self):
        man = load_manifest()
        entry_for(man, "the_waves_cleaned.txt").pop("provenance_note", None)
        fails = failures_for(man, "the_waves_cleaned.txt", online=False)
        self.assertTrue(any("undocumented" in f for f in fails), fails)

    def test_claimed_titlepage_that_is_absent_is_caught(self):
        """orlando_cleaned is marked false because its only hit is the
        character's name; flipping it to true must fail."""
        man = load_manifest()
        e = entry_for(man, "the_years.txt")
        e["title_on_titlepage"] = True
        e["title"] = "Between the Acts"  # not on the_years' first page
        fails = failures_for(man, "the_years.txt", online=False)
        self.assertTrue(any("absent from first" in f for f in fails), fails)


class TestTitleNormalisation(unittest.TestCase):
    """The parsing that stands between a real mismatch and a false alarm."""

    def test_subtitle_and_marc_markers_are_stripped(self):
        self.assertEqual(vc.title_core("Some do not...: A novel"), "some do not")
        self.assertEqual(vc.title_core("The tunnel : $b Pilgrimage, Volume 4"), "the tunnel")
        self.assertEqual(vc.title_core("Mary Olivier: a Life"), "mary olivier")

    def test_roman_and_arabic_volumes_compare_equal(self):
        """PG writes vol.1 of Wings of the Dove as 'Volume 1 of 2' and vol.2
        as 'Volume II'. String comparison would flag a correct pair."""
        self.assertEqual(vc.extract_volume("The Wings of the Dove, Volume 1 of 2"), 1)
        self.assertEqual(vc.extract_volume("The Wings of the Dove, Volume II"), 2)
        self.assertEqual(vc.extract_volume("The Wings of the Dove (vol. 1)"), 1)
        self.assertIsNone(vc.extract_volume("The Ambassadors"))

    def test_surname_extraction(self):
        self.assertEqual(vc.surname("Richardson, Dorothy M. (Dorothy Miller)"), "richardson")
        self.assertEqual(vc.surname("Ford, Ford Madox"), "ford")
        self.assertEqual(vc.surname("Sinclair, May"), "sinclair")


if __name__ == "__main__":
    unittest.main(verbosity=2)
