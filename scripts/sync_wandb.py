"""Probe authenticated W&B access and sync a finished experiment's native logs."""

from __future__ import annotations

import argparse
import json
import netrc
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--proxy", default=os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Upload finished local runs after a successful probe",
    )
    args = parser.parse_args()
    key = os.getenv("WANDB_API_KEY")
    if not key:
        credentials = netrc.netrc().authenticators("api.wandb.ai")
        key = credentials[2] if credentials else None
    if not key:
        raise RuntimeError("Configure WANDB_API_KEY or W&B login credentials first")
    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "project": "ARIN7101-Project",
    }
    try:
        with httpx.Client(proxy=args.proxy, trust_env=False, timeout=15) as client:
            response = client.post(
                "https://api.wandb.ai/graphql",
                json={"query": "query { viewer { username entity } }"},
                auth=("api", key),
            )
        report["http_status"] = response.status_code
        if response.status_code == 200:
            report["viewer"] = response.json().get("data", {}).get("viewer")
        report["authenticated"] = bool(report.get("viewer"))
    except httpx.HTTPError as exc:
        # Record error types only: proxy URLs or credentials must not enter logs.
        report.update(authenticated=False, error_type=type(exc).__name__)
    destination = args.run_dir / "wandb-connectivity.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not args.sync:
        return
    if not report["authenticated"]:
        raise RuntimeError(
            "W&B connectivity probe failed; offline data remains available"
        )
    state = json.loads((args.run_dir / "state.json").read_text())
    if state.get("status") != "completed":
        raise RuntimeError("This final-sync command expects a completed experiment")
    environment = {
        **os.environ,
        "WANDB_API_KEY": key,
        "WANDB_PROJECT": "ARIN7101-Project",
    }
    if args.proxy:
        environment.update(
            {
                name: args.proxy
                for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
            }
        )
    executable = Path(sys.executable).with_name("wandb")
    offline = sorted((args.run_dir / "wandb").glob("offline-run-*"))
    offline.extend(sorted((args.run_dir / "analysis/wandb").glob("offline-run-*")))
    if not offline:
        raise FileNotFoundError("No W&B offline run files found")
    for directory in offline:
        subprocess.run(
            [str(executable), "sync", "--project", "ARIN7101-Project", str(directory)],
            check=True,
            env=environment,
        )
    report["sync_command_succeeded"] = True
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Official W&B sync finished. Verify cloud run names, chart files and final metrics before marking the task complete."
    )


if __name__ == "__main__":
    main()
