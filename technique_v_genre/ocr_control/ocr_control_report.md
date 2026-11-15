# OCR control: *Soldiers' Pay*, raw scan vs. proofread descendant

Run 2026-10-02. Paired design (see `run_ocr_control.py` docstring): 260
passages sampled from the corpus's proofread Standard Ebooks text, each
aligned to the span of the Internet Archive scan that transcribes the same
words, OCR span scored with the corpus protocol (gpt-5.4 + qwen3-max, k=3,
same prompt). Clean-side judgements are the existing corpus rows in
`scored/se_william_faulkner_soldiers_pay.jsonl`. Noise floor = each
reader's own run-to-run agreement on the clean text.

Alignment: 260 aligned, 4 skipped (no unique anchor). All 260 end on the
right word; ~3 start one word off where the scan garbled the first word.
One passage (`_0053`) failed qwen3-max on all three OCR runs with malformed
JSON output, so qwen3-max n = 259.

The OCR spans lost 6.2% of their quotation marks (2,838 → 2,661); 102 of
260 passages lost at least one.

## Results

| | gpt-5.4 | qwen3-max |
|---|---|---|
| SOC detected (per run), clean → OCR | 67.3% → 68.2% | 56.9% → 51.2% |
| passages losing / gaining SOC (majority of runs) | 9 / 13, McNemar p = 0.52 | **19 / 4, p = 0.003** |
| level agreement, clean vs clean (floor) | 89.7% | 90.7% |
| level agreement, clean vs OCR | 87.2% | 81.6% |
| per-passage gap, 95% bootstrap CI | 3.5pp [−0.2, 7.7] | **9.0pp [3.8, 14.5]** |
| modal level shift, OCR − clean | +0.006 (8 up, 9 down) | −0.037 (9 up, 13 down) |
| ...quote-damaged passages only | −0.014 (2 up, 4 down) | −0.106 (3 up, 7 down; sign test p = 0.34) |

Level distributions (per run, SOC-positive): every level within ±2.3pp for
gpt-5.4; qwen3-max moves +5.2pp into L1 from the other three levels.

## Reading

**The pre-registered mechanism is not supported.** The prediction was that
losing quotation marks would push framed interior speech (L2) down to L1/L0.
Quote-damaged passages show at most a small downward drift under qwen3-max,
not significant at this n, and none under gpt-5.4. L2 mass does not drain in
either reader.

**OCR still matters, but through a different route and for one reader.**
gpt-5.4 reads the scan essentially as it reads the proofread text: the drop
in agreement is within resampling noise. qwen3-max is measurably less stable
on OCR: it more often decides a passage renders no consciousness at all, and
where it does score, it agrees with its clean reading 9 points less often
than with itself. The damage is to detection and stability generally (running
heads, hyphenation, garbled characters), not specifically to the quote frame.

**Decision this supports.** Do not pool scanned texts with the e-text
corpus. The effect is reader-dependent, so it would distort exactly the
cross-reader comparisons the chapter relies on, and it would land on
Faulkner, already the most reader-sensitive author. *Mosquitoes* and
*Sartoris* stay out of the main corpus in scanned form. (*Mosquitoes* is
present via the coauthor-supplied clean text and is read by Jev only.)
