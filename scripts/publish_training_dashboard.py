"""Publish a separate three-panel W&B view for a 32-update training run."""

from __future__ import annotations

import argparse
import json
import netrc
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--proxy", default=os.getenv("HTTPS_PROXY"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((args.run_dir / "config.json").read_text())
    state = json.loads((args.run_dir / "state.json").read_text())
    if config["max_steps"] not in (32, 48, 80):
        raise ValueError("This dashboard template expects 32, 48, or 80 updates")
    base = os.getenv("WANDB_BASE_URL", "https://api.forge.coreweave.com").rstrip("/")
    credentials = netrc.netrc().authenticators(urlsplit(base).hostname)
    key = os.getenv("WANDB_API_KEY") or (credentials[2] if credentials else None)
    if not key:
        raise RuntimeError("Log in to the configured W&B host first")
    run_id = state["wandb_id"]
    template = (root / "configs/wandb_core_dashboard.json").read_text()
    payload = json.loads(template.replace("__RUN_ID__", run_id))
    destination = args.run_dir / "wandb-dashboard.json"
    saved = json.loads(destination.read_text()) if destination.exists() else {}
    mutation = """
    mutation SaveView($id: ID, $entity: String!, $project: String!,
        $name: String!, $displayName: String!, $spec: String!) {
      upsertView(input: {id: $id, entityName: $entity, projectName: $project,
        name: $name, displayName: $displayName, type: "project-view",
        spec: $spec, createdUsing: WANDB_SDK}) {
        view { id name } inserted
      }
    }
    """
    with httpx.Client(proxy=args.proxy, trust_env=False, timeout=30) as client:
        viewer = client.post(
            base + "/graphql",
            json={"query": "query { viewer { entity } }"},
            auth=("api", key),
        )
        viewer.raise_for_status()
        entity = viewer.json()["data"]["viewer"]["entity"]
        variables = {
            "entity": entity,
            "project": config["wandb_project"],
            "name": f"nw-{run_id}coremetrics-v",
            "displayName": payload["displayName"],
            "spec": json.dumps(payload["spec"]),
        }
        if saved.get("id"):
            variables["id"] = saved["id"]
        response = client.post(
            base + "/graphql",
            json={"query": mutation, "variables": variables},
            auth=("api", key),
        )
        response.raise_for_status()
        body = response.json()
        if body.get("errors"):
            raise RuntimeError(body["errors"])
        view = body["data"]["upsertView"]["view"]
    result = {
        **view,
        "run_id": run_id,
        "entity": entity,
        "project": config["wandb_project"],
        "panels": 3,
        "url": (
            f"https://forge.coreweave.com/wandb/{entity}/"
            f"{config['wandb_project']}/workspace?nw={run_id}coremetrics"
        ),
        "spec": payload["spec"],
    }
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "spec"}, indent=2))


if __name__ == "__main__":
    main()
