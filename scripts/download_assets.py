"""Download exact upstream SEC inputs and the requested model with atomic writes."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DATASET_REVISION = "595c9e83e376412f14fa3d3243a6cd97f2ac8386"


def download(url: str, destination: Path, size: int | None = None) -> None:
    if destination.exists() and (size is None or destination.stat().st_size == size):
        print(f"cached {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(5):
        try:
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=180, trust_env=False
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes(4 * 1024 * 1024):
                        handle.write(chunk)
            if size is not None and temporary.stat().st_size != size:
                raise ValueError(f"Size mismatch for {destination}")
            temporary.replace(destination)
            print(
                f"downloaded {destination} ({destination.stat().st_size:,} bytes)",
                flush=True,
            )
            return
        except (httpx.HTTPError, ValueError) as exc:
            print(
                f"retry {attempt + 1}: {destination}: {type(exc).__name__}", flush=True
            )
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 10))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--model", action="store_true")
    args = parser.parse_args()
    jobs = []
    if args.dataset:
        root = f"https://huggingface.co/datasets/pat-jj/harness-1-train-data/resolve/{DATASET_REVISION}"
        jobs.append(
            (
                f"{root}/data/train-00000-of-00001.parquet",
                Path("data/raw/queries.parquet"),
                None,
            )
        )
        for i in range(43):
            name = f"train-{i:05d}.parquet"
            jobs.append(
                (
                    f"{root}/corpora/sec/train/{name}",
                    Path("data/raw/corpus") / name,
                    None,
                )
            )
    if args.model:
        with httpx.Client(trust_env=False, timeout=60) as client:
            files = client.get(
                f"https://huggingface.co/api/models/Qwen/Qwen3.5-9B/tree/{MODEL_REVISION}"
            )
            files.raise_for_status()
        for row in files.json():
            if row["type"] == "file" and row["path"] not in (
                "README.md",
                ".gitattributes",
            ):
                jobs.append(
                    (
                        f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{MODEL_REVISION}/{row['path']}",
                        Path("models/Qwen3.5-9B") / row["path"],
                        row["size"],
                    )
                )
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download, *job) for job in jobs]
        for future in as_completed(futures):
            future.result()
    if args.dataset:
        from sec_rl.sec import prepare_sec

        for count, name in [(256, "sec_subset"), (1024, "sec_1024")]:
            summary = prepare_sec(
                Path("data") / name,
                raw_dir=Path("data/raw"),
                train_size=count,
                dev_size=64,
                seed=42,
            )
            print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
