"""Create the Minibook agents and project required by the main-experiment release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def request(method: str, url: str, *, api_key: str | None = None,
            payload: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.request(method, url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:3456/api/v1")
    parser.add_argument("--project-name", default="main-experiment-self-stigma-release")
    parser.add_argument("--out", type=Path, default=Path("config/minibook.local.json"))
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    agents = {}
    for role in ("patient", "peer", "general"):
        name = f"main-experiment-{role}"
        result = request("POST", f"{base_url}/agents", payload={"name": name})
        agents[role] = {"name": name, "api_key": result["api_key"]}

    project = request(
        "POST",
        f"{base_url}/projects",
        api_key=agents["patient"]["api_key"],
        payload={
            "name": args.project_name,
            "description": "NeurIPS workshop main-experiment reproduction",
        },
    )
    for role in ("peer", "general"):
        request(
            "POST",
            f"{base_url}/projects/{project['id']}/join",
            api_key=agents[role]["api_key"],
            payload={"role": "member"},
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "base_url": base_url,
        "project_id": project["id"],
        "agents": agents,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Minibook credentials to {args.out}")


if __name__ == "__main__":
    main()
