"""Pydantic models for the SOC analysis pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# SOC type enum — mirrors the SKILL.md taxonomy
# ---------------------------------------------------------------------------

class SocType(str, Enum):
    """Stream of consciousness type identifiers."""

    DIRECT_INTERIOR_MONOLOGUE = "direct_interior_monologue"
    INDIRECT_INTERIOR_MONOLOGUE = "indirect_interior_monologue"
    OMNISCIENT_DESCRIPTION = "omniscient_description"
    SOLILOQUY = "soliloquy"
    FREE_ASSOCIATION = "free_association"
    SPACE_MONTAGE = "space_montage"
    ORTHOGRAPHIC_MARKER = "orthographic_marker"
    IMAGERY = "imagery"
    SIMULATION_STATE_OF_MIND = "simulation_state_of_mind"
    REVERIE_FANTASY = "reverie_fantasy"
    HYBRID = "hybrid"


class NarratorPosition(str, Enum):
    ABSENT = "absent"
    MINIMAL = "minimal"
    PRESENT = "present"
    DOMINANT = "dominant"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Chunk model — intermediate representation after text extraction + chunking
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A segment of a literary text ready for LLM analysis."""

    source_file: str = Field(description="Original filename")
    chunk_id: str = Field(description="Unique identifier, e.g. 'mrs-dalloway_ch03'")
    chunk_label: str = Field(default="", description="Human-readable label")
    chunk_text: str = Field(description="Full text of the chunk")
    chunk_index: int = Field(description="Ordinal position in the source work")
    context_before: str = Field(default="", description="Overlap text from previous chunk")
    context_after: str = Field(default="", description="Overlap text from next chunk")


# ---------------------------------------------------------------------------
# SOC instance — a single classified passage returned by the LLM
# ---------------------------------------------------------------------------

class SocInstance(BaseModel):
    """A single stream-of-consciousness passage identified by the LLM."""

    passage: str = Field(description="Exact quoted passage from the text")
    soc_type: str = Field(description="Primary SOC type from SKILL.md taxonomy")
    secondary_devices: list[str] = Field(
        default_factory=list,
        description="Secondary techniques layered on the primary type",
    )
    affective_register: str = Field(
        default="n/a",
        description="Emotional register if simulation of state of mind",
    )
    narrator_position: str = Field(
        default="absent",
        description="absent | minimal | present | dominant",
    )
    character_pov: str = Field(default="", description="Character whose consciousness is rendered")
    explanation: str = Field(description="Reasoning for the classification")
    evidence: list[str] = Field(
        default_factory=list,
        description="2-3 specific textual features supporting classification",
    )
    confidence: str = Field(description="high | medium | low")
    notes: str = Field(default="", description="Ambiguity, hybrid transitions, observations")


# ---------------------------------------------------------------------------
# LLM response wrapper
# ---------------------------------------------------------------------------

class LLMResponse(BaseModel):
    """The expected JSON structure returned by the LLM for a single chunk."""

    soc_instances: list[SocInstance] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Full result row — chunk metadata + SOC instance, used for CSV/JSON export
# ---------------------------------------------------------------------------

class ResultRow(BaseModel):
    """A flat record combining chunk metadata with a single SOC detection."""

    model_label: str  # which LLM produced this result
    source_file: str
    chunk_id: str
    chunk_label: str
    chunk_index: int
    passage: str
    soc_type: str
    secondary_devices: str  # comma-separated for CSV friendliness
    affective_register: str
    narrator_position: str
    character_pov: str
    explanation: str
    evidence: str  # comma-separated
    confidence: str
    notes: str

    @classmethod
    def from_chunk_and_instance(
        cls, chunk: Chunk, inst: SocInstance, model_label: str = ""
    ) -> ResultRow:
        return cls(
            model_label=model_label,
            source_file=chunk.source_file,
            chunk_id=chunk.chunk_id,
            chunk_label=chunk.chunk_label,
            chunk_index=chunk.chunk_index,
            passage=inst.passage,
            soc_type=inst.soc_type,
            secondary_devices=", ".join(inst.secondary_devices),
            affective_register=inst.affective_register,
            narrator_position=inst.narrator_position,
            character_pov=inst.character_pov,
            explanation=inst.explanation,
            evidence=", ".join(inst.evidence),
            confidence=inst.confidence,
            notes=inst.notes,
        )
