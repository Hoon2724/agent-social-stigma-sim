"""Core generation pipeline: config -> conditions -> assembled dataset items.

Dataset generation ONLY. No judge, no two-agent simulation loop. Each item pairs
a visible_dialogue (what the judge will later see) with hidden_ground_truth
(condition + per-session intended trajectory) that must be withheld from the judge.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

import prompts
from schema import (
    SCHEMA_VERSION,
    SESSION_OUTPUT_JSON_SCHEMA,
    find_leak_terms,
    find_ungrounded_quotes,
    validate_item,
)
from trajectories import build_session_targets, target_final_ismi

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_PERSONAS = MODULE_DIR / "personas.json"


def _load_env() -> None:
    """Load OPENAI_API_KEY from the repo's simulation/.env if present."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (MODULE_DIR.parent / ".env", MODULE_DIR / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
    load_dotenv()  # also honor a .env in cwd / already-set vars


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_personas(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {p["persona_id"]: p for p in data["personas"]}


def _condition_id(spec: dict) -> str:
    return (
        f"{spec['trajectory_type']}__i{spec['initial_ismi']}"
        f"__s{spec['session_count']}x{spec['turns_per_session']}"
    )


def expand_conditions(config: dict, personas: dict) -> list[dict]:
    """Flatten config conditions into one spec per (condition, persona, repeat)."""
    seed_base = int(config.get("seed_base", 0))
    specs: list[dict] = []
    idx = 0
    for cond in config["conditions"]:
        trajectory_type = cond["trajectory_type"]
        initial_ismi = float(cond["initial_ismi"])
        session_count = int(cond["session_count"])
        turns_per_session = int(cond["turns_per_session"])
        repeats = int(cond.get("repeats", 1))
        persona_ids = cond["personas"]
        final_ismi = cond.get(
            "target_final_ismi", target_final_ismi(trajectory_type, initial_ismi)
        )
        for persona_id in persona_ids:
            if persona_id not in personas:
                raise KeyError(f"persona '{persona_id}' not found in personas file")
            for r in range(repeats):
                base = {
                    "trajectory_type": trajectory_type,
                    "initial_ismi": initial_ismi,
                    "target_final_ismi": round(float(final_ismi), 2),
                    "session_count": session_count,
                    "turns_per_session": turns_per_session,
                    "persona_id": persona_id,
                }
                cid = _condition_id(base)
                specs.append(
                    {
                        **base,
                        "condition_id": cid,
                        "sample_id": f"{cid}__{persona_id}__r{r + 1}",
                        "repeat": r + 1,
                        "seed": seed_base + idx,
                    }
                )
                idx += 1
    return specs


def _condition_block(spec: dict, persona: dict) -> dict:
    return {
        "trajectory_type": spec["trajectory_type"],
        "initial_ismi": spec["initial_ismi"],
        "target_final_ismi": spec["target_final_ismi"],
        "session_count": spec["session_count"],
        "turns_per_session": spec["turns_per_session"],
        "persona_id": spec["persona_id"],
        "diagnosis_context": persona.get("diagnosis_context", ""),
        "peer_style": prompts.PEER_STYLE[spec["trajectory_type"]],
    }


def assemble_item(
    spec: dict,
    persona: dict,
    session_targets: list[dict],
    model_output: dict,
    model_name: str,
    dataset_id: str,
    generation_params: dict,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "schema_version": SCHEMA_VERSION,
        "condition_id": spec["condition_id"],
        "sample_id": spec["sample_id"],
        "prompt_version": prompts.PROMPT_VERSION,
        "model": model_name,
        "seed": spec["seed"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_params": generation_params,
        "condition": _condition_block(spec, persona),
        "hidden_ground_truth": {"session_targets": session_targets},
        "visible_dialogue": model_output["visible_dialogue"],
        "author_evidence": model_output["author_evidence"],
    }


def generate_sample_sequentially(
    client,
    condition_block: dict,
    persona: dict,
    session_targets: list[dict],
    base_seed: int,
    temperature: float,
    max_output_tokens: int,
) -> dict:
    """Generate one sample session-by-session. Each call gets the ACTUAL
    previous session's turns as verbatim context, so cross-session
    backreferences are grounded in real text instead of the model trying to
    remember its own earlier output across one very long generation."""
    visible_dialogue: list[dict] = []
    markers_by_session: list[dict] = []
    prior_turns: list[dict] | None = None

    for i, target in enumerate(session_targets):
        system = prompts.build_system_prompt()
        user = prompts.build_session_user_prompt(
            condition_block, persona, session_targets, i, prior_turns
        )
        session_output = client.complete_json(
            system, user, SESSION_OUTPUT_JSON_SCHEMA,
            temperature=temperature, max_output_tokens=max_output_tokens,
            seed=base_seed + i,
        )
        visible_dialogue.append({"session": target["session"], "turns": session_output["turns"]})
        markers_by_session.append({"session": target["session"], "evidence": session_output["evidence"]})
        prior_turns = session_output["turns"]

    summary = (
        f"Generated sequentially across {len(session_targets)} sessions for "
        f"persona '{persona.get('persona_id', '?')}' under the "
        f"'{condition_block['trajectory_type']}' trajectory; see "
        "markers_by_session for session-specific evidence."
    )
    return {
        "visible_dialogue": visible_dialogue,
        "author_evidence": {"summary": summary, "markers_by_session": markers_by_session},
    }


def _output_paths(config: dict, mock: bool) -> tuple[Path, Path | None]:
    out_dir = Path(config.get("output_dir", "data/synthetic"))
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = config.get("output_basename", config.get("dataset_id", "ismi_dataset"))
    if mock:
        basename += ".mock"
    jsonl_path = out_dir / f"{basename}.jsonl"
    pretty_path = (
        out_dir / f"{basename}.pretty.json"
        if config.get("write_pretty_json", True)
        else None
    )
    return jsonl_path, pretty_path


def run(
    config: dict,
    *,
    mock: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    model_override: str | None = None,
) -> dict:
    """Execute the pipeline. Returns a run summary dict."""
    personas_file = config.get("personas_file")
    personas_path = Path(personas_file) if personas_file else DEFAULT_PERSONAS
    if not personas_path.is_absolute() and not personas_path.exists():
        personas_path = MODULE_DIR / personas_path
    personas = load_personas(personas_path)

    specs = expand_conditions(config, personas)
    if limit is not None:
        specs = specs[:limit]

    dataset_id = config.get("dataset_id", "ismi_dataset")
    model_name = model_override or config.get("model", "gpt-5.5")
    gen_cfg = config.get("generation", {})
    temperature = float(gen_cfg.get("temperature", 0.7))
    max_output_tokens = int(gen_cfg.get("max_output_tokens", 8000))

    # Pre-build (spec, persona, session_targets) for every sample. Prompts are
    # NOT pre-built here anymore: each session's prompt depends on the actual
    # previous session's generated output, so prompt construction happens
    # inside the per-session generation loop below.
    planned = []
    for spec in specs:
        persona = personas[spec["persona_id"]]
        session_targets = build_session_targets(
            spec["trajectory_type"], spec["initial_ismi"], spec["session_count"]
        )
        planned.append((spec, persona, session_targets))

    summary = {
        "dataset_id": dataset_id,
        "model": "mock" if mock else model_name,
        "prompt_version": prompts.PROMPT_VERSION,
        "planned_samples": len(planned),
        "conditions": sorted({s["condition_id"] for s in specs}),
        "mock": mock,
        "dry_run": dry_run,
    }

    if dry_run:
        # Validate the *shape* of each planned item using the ground truth alone.
        for spec, persona, session_targets in planned:
            stub = assemble_item(
                spec,
                persona,
                session_targets,
                model_output={
                    "visible_dialogue": [
                        {"session": t["session"], "turns": [
                            {"turn": 1, "speaker": "patient", "kind": "post",
                             "reply_to": None, "text": "placeholder"}
                        ]}
                        for t in session_targets
                    ],
                    "author_evidence": {"summary": "", "markers_by_session": []},
                },
                model_name="dry-run",
                dataset_id=dataset_id,
                generation_params={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                },
            )
            problems = validate_item(stub)
            if problems:
                summary.setdefault("dry_run_problems", []).append(
                    {"sample_id": spec["sample_id"], "problems": problems}
                )
        return summary

    # Real work: mock or API generation.
    client = None
    if not mock:
        _load_env()
        if not os.environ.get("OPENAI_API_KEY") and model_name.startswith("gpt-"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it or add it to simulation/.env, "
                "or run with --mock for an offline sample."
            )
        from llm_client import StructuredClient

        client = StructuredClient(model_name)

    jsonl_path, pretty_path = _output_paths(config, mock)
    items: list[dict] = []
    problems_log: list[dict] = []

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for spec, persona, session_targets in planned:
            condition_block = _condition_block(spec, persona)
            if mock:
                from mock import build_mock_output

                model_output = build_mock_output(condition_block, persona, session_targets)
                used_model = "mock"
                gen_params = {"temperature": None, "max_output_tokens": None}
            else:
                model_output = generate_sample_sequentially(
                    client, condition_block, persona, session_targets,
                    base_seed=spec["seed"], temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                used_model = model_name
                gen_params = {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }

            item = assemble_item(
                spec,
                persona,
                session_targets,
                model_output,
                used_model,
                dataset_id,
                gen_params,
            )
            struct_problems = validate_item(item)
            leaks = find_leak_terms(item)
            ungrounded = find_ungrounded_quotes(item)
            if struct_problems or leaks or ungrounded:
                problems_log.append(
                    {
                        "sample_id": spec["sample_id"],
                        "schema_problems": struct_problems,
                        "leak_hits": leaks,
                        "ungrounded_quotes": ungrounded,
                    }
                )
            jf.write(json.dumps(item, ensure_ascii=False) + "\n")
            items.append(item)

    if pretty_path is not None:
        with open(pretty_path, "w", encoding="utf-8") as pf:
            json.dump(items, pf, ensure_ascii=False, indent=2)

    summary.update(
        {
            "written": len(items),
            "jsonl_path": str(jsonl_path),
            "pretty_path": str(pretty_path) if pretty_path else None,
            "problems": problems_log,
        }
    )
    return summary
