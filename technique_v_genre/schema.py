"""Pydantic schema for the technique-vs-genre scoring pass.

Extends scripts/models.py's SocInstance with the three orthogonal outputs
HANDOFF_della.md section 3 specifies, each grounded in a specific source
from soc_literature_review.md rather than invented ad hoc:

- mediation_level: Cohn's (1978) mediation cline, not Humphrey's four types
  directly — psycho-narration / narrated monologue (FID) / quoted monologue /
  autonomous monologue. This is the fix for the Woolf-Richardson artifact:
  ordinal distance separates near-agreement (one level apart) from real
  instability (0 vs 3) in a way flat categorical agreement can't.
- fit_category + fit_alternatives: the technique/boundary/genre three-way
  sort from TECHNIQUE_VS_GENRE.md, promoted to an explicit field so it's
  measurable at the corpus level rather than inferred after the fact from
  cross-model agreement stats (which TECHNIQUE_VS_GENRE.md and the lit
  review both flag as an unreliable proxy on its own).
- antimimetic + antimimetic_defense: the genre pole's operational criterion
  from unnatural narratology (Richardson 2006, Alber 2010) — true only when
  the passage resists naturalization into any mediation level *and* the
  reader can articulate why, not merely when it's unstable under resampling.
  This is what separates genuine antimimetic narration from "the model
  hasn't converged yet," per soc_literature_review.md's point 3.

Revision (2026-08): replaces the single short `explanation` field from the
first draft of this schema with `argument` + `alternative_reasoning`,
restoring the skepticism gate scripts/prompts.py's JUDGE_PROMPT already
specifies for GRPO reward scoring — grounding (verbatim citation),
specificity (named type), and type_coherence (argue for the assigned type
AND explicitly engage the nearest alternative) — as generation-time
requirements, not just a post-hoc RL reward. This is what makes the output
usable the way confounds_report.md and woolf_richardson_comparison.md
actually use model output: as a quotable, argued position, not a label.
Resampling (k>=3 per passage per reader) is the default unit of evidence
going forward, per AMBIGUITY.md's stability methodology — a single-pass
call is a preliminary read, not a finding.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class MediationLevel(int, Enum):
    """Cohn's (1978) mediation cline — ordinal, not categorical."""

    PSYCHO_NARRATION = 0        # narrator reports the mind ("she felt uneasy")
    NARRATED_MONOLOGUE = 1      # FID: character's thought bends the narrator's sentence, no quotes
    QUOTED_MONOLOGUE = 2        # framed direct interior speech ("I am uneasy, she thought")
    AUTONOMOUS_MONOLOGUE = 3    # unframed raw interior voice (Molly Bloom)


class FitCategory(str, Enum):
    TECHNIQUE_FIT = "technique_fit"          # clean convergence on one mediation level / soc_type
    BOUNDARY_TENSION = "boundary_tension"    # stable split between two named, adjacent types
    NO_TYPE_FITS = "no_type_fits"            # candidate genre evidence — needs qualitative confirmation


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TechniqueGenreInstance(BaseModel):
    """One passage's scored instance under the technique-vs-genre schema."""

    passage: str = Field(description="Exact quoted passage from the text")
    is_soc: bool = Field(description="Whether this passage renders consciousness at all")

    # Primary measurement — Cohn's cline, not a flat label.
    mediation_level: int = Field(
        ge=0, le=3, description="0=psycho-narration, 1=narrated monologue/FID, 2=quoted monologue, 3=autonomous monologue"
    )

    # Humphrey/Steinberg type, kept for continuity with the existing taxonomy
    # and because fit_alternatives references these slugs.
    soc_type: str = Field(description="Primary SOC type from the Humphrey/Steinberg taxonomy")
    secondary_devices: list[str] = Field(default_factory=list)

    # The three-way sort, explicit.
    fit_category: FitCategory
    fit_alternatives: list[str] = Field(
        default_factory=list,
        description="Named types in contention; required (>=2) when fit_category == boundary_tension",
    )

    # Genre pole, operationalized per unnatural narratology rather than
    # just "unstable under resampling."
    antimimetic: bool = Field(
        default=False,
        description="True only if the passage resists naturalization into any mediation level, with a stated reason",
    )
    antimimetic_defense: str = Field(
        default="",
        description="Required, non-empty, when antimimetic=true: the specific reason no level/type applies",
    )

    narrator_position: str = Field(default="absent", description="absent | minimal | present | dominant")
    character_pov: str = Field(default="")

    # The argued case, not a label with a caption. Two fields rather than
    # one "explanation" so the alternative can't be silently skipped.
    argument: str = Field(
        description="The case FOR the assigned soc_type/mediation_level, grounded in verbatim "
        "evidence from `evidence`. As long as it needs to be — typically 2-4 sentences, not "
        "a caption. This is the quotable unit: it should read the way a passage-level finding "
        "in confounds_report.md or woolf_richardson_comparison.md reads."
    )
    alternative_reasoning: str = Field(
        default="",
        description="Required whenever is_soc=true. Name the strongest alternative type/level "
        "seriously considered and explain the call: why it was rejected (technique_fit), why it "
        "remains a live, defensible second reading (boundary_tension — should match fit_alternatives), "
        "or why nothing in the taxonomy accounts for the passage (no_type_fits — should match "
        "antimimetic_defense when antimimetic=true). This is the skepticism gate; a verdict without "
        "it is not a finding.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="2-3 short spans quoted VERBATIM from `passage` — must appear word-for-word "
        "in the passage text, not paraphrased.",
    )
    confidence: Confidence

    @model_validator(mode="after")
    def _check_consistency(self) -> "TechniqueGenreInstance":
        if self.fit_category == FitCategory.BOUNDARY_TENSION and len(self.fit_alternatives) < 2:
            raise ValueError("boundary_tension requires >=2 fit_alternatives")
        if self.antimimetic and not self.antimimetic_defense.strip():
            raise ValueError("antimimetic=true requires a non-empty antimimetic_defense")
        if self.is_soc and not self.alternative_reasoning.strip():
            raise ValueError("is_soc=true requires non-empty alternative_reasoning (the skepticism gate)")
        return self


class TechniqueGenreResponse(BaseModel):
    instances: list[TechniqueGenreInstance] = Field(default_factory=list)


class ScoredPassage(BaseModel):
    """One (passage x reader) row for the master table (HANDOFF_della.md Step 2)."""

    passage_id: str
    source_file: str
    author: str
    title: str
    year: int
    role: str
    reader_id: str  # e.g. "gpt-5.4", "mercury-2", "qwen3-max", "penelope-soc-v2"
    run: int = 0     # resample index, for k-run stability checks
    passage_text: str
    instance: TechniqueGenreInstance | None  # None if the reader found no SoC in the passage
