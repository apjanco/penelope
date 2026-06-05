"""Penelope — inference CLI.

Runs the fine-tuned SoC model on a passage and returns structured output.

Usage
-----
    python scripts/infer.py --input-text "She thought of him again..."
    python scripts/infer.py --input-file chunking/mrs_dalloway.txt
    python scripts/infer.py --adapter adapter/ --base-model Qwen/Qwen3-4B \\
        --input-text "..." --format json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Annotated, Optional

# Resolve project root so scripts.* imports work when called from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _lazy_imports():
    """Defer heavy imports so --help is fast."""
    try:
        import typer  # noqa: F401
    except ImportError:
        print("typer is required: pip install typer>=0.12.0", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field


class SoCInstance(BaseModel):
    is_soc: bool
    passage: str
    soc_type: str = ""
    secondary_devices: list[str] = Field(default_factory=list)
    affective_register: str = "n/a"
    narrator_position: str = ""
    character_pov: str = ""
    explanation: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: str = "medium"
    notes: str = ""


class SoCResult(BaseModel):
    instances: list[SoCInstance] = Field(default_factory=list)
    think_trace: str = ""
    raw_output: str = ""


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------


def run(
    input_text: str,
    model_path: str,
    adapter_path: str | None,
    base_model: str,
    max_new_tokens: int,
    enable_thinking: bool,
) -> SoCResult:
    from scripts.evaluate import load_model_and_tokenizer, run_inference
    from scripts.prompts import INFERENCE_SYSTEM_PROMPT

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        adapter_path=adapter_path,
        base_model=base_model,
        device="auto",
    )

    messages = [
        {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]

    raw = run_inference(
        model, tokenizer, messages,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
    )

    # Extract think trace
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    think_trace = think_match.group(1).strip() if think_match else ""

    # Parse JSON output
    json_str = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", json_str, re.DOTALL)
    parsed: dict = {}
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    instances = [
        SoCInstance(**inst)
        for inst in parsed.get("instances", [])
    ]

    return SoCResult(instances=instances, think_trace=think_trace, raw_output=raw)


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _format_json(result: SoCResult) -> str:
    return result.model_dump_json(indent=2)


def _format_csv(result: SoCResult) -> str:
    import csv
    import io
    buf = io.StringIO()
    fields = list(SoCInstance.model_fields.keys())
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for inst in result.instances:
        row = inst.model_dump()
        row["secondary_devices"] = "|".join(row["secondary_devices"])
        row["evidence"] = "|".join(row["evidence"])
        writer.writerow(row)
    return buf.getvalue()


def _format_markdown(result: SoCResult) -> str:
    lines: list[str] = []
    if result.think_trace:
        lines += ["## Reasoning trace\n", result.think_trace, ""]
    if not result.instances:
        lines.append("*No stream-of-consciousness instances found.*")
        return "\n".join(lines)
    for i, inst in enumerate(result.instances, 1):
        lines += [
            f"## Instance {i} — `{inst.soc_type}` ({inst.confidence})",
            f"> {inst.passage}",
            "",
            f"**Explanation:** {inst.explanation}",
            f"**Evidence:** {', '.join(inst.evidence)}",
            f"**Narrator position:** {inst.narrator_position}  |  "
            f"**Character:** {inst.character_pov or '—'}",
        ]
        if inst.secondary_devices:
            lines.append(f"**Secondary devices:** {', '.join(inst.secondary_devices)}")
        if inst.notes:
            lines.append(f"**Notes:** {inst.notes}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Typer CLI
# ---------------------------------------------------------------------------

import typer  # noqa: E402 (after sys.path setup)

app = typer.Typer(
    name="infer",
    help="Run the Penelope SoC model on a passage and return structured output.",
    add_completion=False,
)


@app.command()
def main(
    input_text: Annotated[
        Optional[str],
        typer.Option("--input-text", "-t", help="Passage text to analyse."),
    ] = None,
    input_file: Annotated[
        Optional[Path],
        typer.Option("--input-file", "-f", help="File containing the passage text."),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="HF model ID or local path (merged model)."),
    ] = "apjanco/penelope-soc-v1",
    adapter: Annotated[
        Optional[Path],
        typer.Option("--adapter", "-a", help="LoRA adapter directory (use with --base-model)."),
    ] = None,
    base_model: Annotated[
        str,
        typer.Option("--base-model", help="Base model ID when loading via --adapter."),
    ] = "Qwen/Qwen3-4B",
    format: Annotated[
        str,
        typer.Option("--format", help="Output format: json | csv | markdown."),
    ] = "json",
    max_new_tokens: Annotated[
        int,
        typer.Option("--max-new-tokens", help="Maximum tokens to generate."),
    ] = 4096,
    no_thinking: Annotated[
        bool,
        typer.Option("--no-thinking", help="Disable <think> mode (faster, less deliberative)."),
    ] = False,
) -> None:
    if input_text is None and input_file is None:
        typer.echo("Error: provide --input-text or --input-file.", err=True)
        raise typer.Exit(1)

    if input_file is not None:
        if not input_file.exists():
            typer.echo(f"Error: file not found: {input_file}", err=True)
            raise typer.Exit(1)
        text = input_file.read_text(encoding="utf-8")
    else:
        text = input_text  # type: ignore[assignment]

    result = run(
        input_text=text,
        model_path=model,
        adapter_path=str(adapter) if adapter else None,
        base_model=base_model,
        max_new_tokens=max_new_tokens,
        enable_thinking=not no_thinking,
    )

    fmt = format.lower()
    if fmt == "json":
        typer.echo(_format_json(result))
    elif fmt == "csv":
        typer.echo(_format_csv(result))
    elif fmt == "markdown":
        typer.echo(_format_markdown(result))
    else:
        typer.echo(f"Unknown format '{format}'. Use: json | csv | markdown", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
