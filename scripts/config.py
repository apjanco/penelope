"""Configuration loader for the SOC analysis pipeline."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Model profile — one LLM endpoint
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    """Configuration for a single LLM backend."""

    label: str
    base_url: str
    model_name: str
    api_key: str
    temperature: float = 0.1

    def validate(self) -> None:
        if not self.api_key or self.api_key.startswith("${"):
            raise ValueError(
                f"API key for model '{self.label}' is not set. "
                f"Check models.yaml and ensure the referenced env var is exported."
            )


def _resolve_env_vars(value: str) -> str:
    """Replace ${ENV_VAR} placeholders with actual environment values."""
    def _replace(m: re.Match) -> str:
        var_name = m.group(1)
        return os.environ.get(var_name, m.group(0))
    return re.sub(r"\$\{(\w+)\}", _replace, value)


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Pipeline configuration, loaded from environment or config files."""

    # Single-model fallback (used when no models.yaml is provided)
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o"
    api_key: str = ""

    chunk_size: int = 20_000
    chunk_overlap: int = 1_000

    skill_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "SKILL.md")

    # Multi-model profiles (populated from models.yaml)
    models: list[ModelProfile] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        env_file: Path | str | None = None,
        config_file: Path | str | None = None,
    ) -> Config:
        """Load configuration from .env and/or models.yaml.

        Priority:
          1. If config_file (models.yaml) is provided, load model profiles from it.
          2. Otherwise, fall back to single-model config from env vars / .env.
        """
        # Always load .env for API key env vars
        if env_file:
            load_dotenv(env_file)
        else:
            candidate = Path(__file__).resolve().parent.parent / ".env"
            if candidate.exists():
                load_dotenv(candidate)

        cfg = cls(
            base_url=os.getenv("LLM_BASE_URL", cls.base_url),
            model_name=os.getenv("LLM_MODEL_NAME", cls.model_name),
            api_key=os.getenv("LLM_API_KEY", cls.api_key),
            chunk_size=int(os.getenv("CHUNK_SIZE", str(cls.chunk_size))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(cls.chunk_overlap))),
        )

        # Load multi-model config if provided
        if config_file:
            cfg._load_models_yaml(Path(config_file))
        else:
            # Auto-detect models.yaml in project root
            candidate_yaml = Path(__file__).resolve().parent.parent / "models.yaml"
            if candidate_yaml.exists():
                cfg._load_models_yaml(candidate_yaml)

        return cfg

    def _load_models_yaml(self, path: Path) -> None:
        """Parse models.yaml and populate self.models."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load models.yaml. "
                "Install with: pip install pyyaml"
            )

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not data or "models" not in data:
            return

        # Override chunk settings if present in YAML
        if "chunk_size" in data:
            self.chunk_size = int(data["chunk_size"])
        if "chunk_overlap" in data:
            self.chunk_overlap = int(data["chunk_overlap"])

        self.models = []
        for entry in data["models"]:
            self.models.append(
                ModelProfile(
                    label=entry["label"],
                    base_url=_resolve_env_vars(str(entry["base_url"])),
                    model_name=_resolve_env_vars(str(entry["model_name"])),
                    api_key=_resolve_env_vars(str(entry.get("api_key", ""))),
                    temperature=float(entry.get("temperature", 0.1)),
                )
            )

    def get_model_profiles(self) -> list[ModelProfile]:
        """Return model profiles to run.

        If models.yaml was loaded, returns those profiles.
        Otherwise, returns a single profile from the .env config.
        """
        if self.models:
            return self.models
        # Fallback: single model from env vars
        return [
            ModelProfile(
                label=self.model_name,
                base_url=self.base_url,
                model_name=self.model_name,
                api_key=self.api_key,
            )
        ]

    def validate(self) -> None:
        """Raise if no usable model profiles are configured."""
        profiles = self.get_model_profiles()
        if not profiles:
            raise ValueError("No model profiles configured.")
        for p in profiles:
            p.validate()
