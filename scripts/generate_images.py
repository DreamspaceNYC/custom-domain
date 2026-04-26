#!/usr/bin/env python3
"""Generate storyboard images from STORYBOARD_PLAN.md via OpenAI Images API.

Usage:
  OPENAI_API_KEY=... python scripts/generate_images.py
  OPENAI_API_KEY=... python scripts/generate_images.py --start 1 --end 10
  python scripts/generate_images.py --dry-run
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORYBOARD_FILE = ROOT / "STORYBOARD_PLAN.md"
OUT_DIR = ROOT / "generated_images"

ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|.*?\|\s*`(.+?)`\s*\|", re.UNICODE)


def parse_prompts(path: Path) -> list[dict]:
    prompts: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROW_RE.match(line)
        if not match:
            continue
        idx = int(match.group(1))
        prompt = match.group(2).strip()
        prompts.append({"image_number": idx, "prompt": prompt})

    prompts.sort(key=lambda x: x["image_number"])
    if not prompts:
        raise ValueError(f"No storyboard rows found in {path}")

    expected = list(range(1, len(prompts) + 1))
    found = [p["image_number"] for p in prompts]
    if found != expected:
        raise ValueError(f"Storyboard numbering is not contiguous: found {found[:5]}...{found[-5:]}")
    return prompts


def call_images_api(api_key: str, prompt: str, model: str, size: str, retries: int = 3) -> bytes:
    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "response_format": "b64_json",
    }

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            "https://api.openai.com/v1/images/generations",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            data = parsed.get("data") or []
            if not data or "b64_json" not in data[0]:
                raise RuntimeError(f"Unexpected Images API response: {parsed}")
            return base64.b64decode(data[0]["b64_json"])
        except urllib.error.HTTPError as err:
            details = err.read().decode("utf-8", errors="replace")
            is_retryable = err.code in {408, 409, 429, 500, 502, 503, 504}
            if attempt < retries and is_retryable:
                sleep_s = attempt * 2
                print(f"retryable HTTP {err.code}; retrying in {sleep_s}s")
                time.sleep(sleep_s)
                continue
            raise RuntimeError(f"Images API HTTP {err.code}: {details}") from err
        except urllib.error.URLError as err:
            reason = str(getattr(err, "reason", err))
            if attempt < retries:
                sleep_s = attempt * 2
                print(f"network error '{reason}'; retrying in {sleep_s}s")
                time.sleep(sleep_s)
                continue
            raise RuntimeError(f"Network error calling Images API: {reason}") from err

    raise RuntimeError("Failed to generate image after retries.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyboard", type=Path, default=STORYBOARD_FILE)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest path (default: <out>/manifest.json)")
    parser.add_argument("--model", default="gpt-image-1")
    parser.add_argument("--size", default="1536x1024", help="Use a 16:9-compatible size")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=85)
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between requests")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = args.out.resolve()
    manifest_path = (args.manifest.resolve() if args.manifest else (out_dir / "manifest.json"))

    prompts = parse_prompts(args.storyboard)
    selected = [p for p in prompts if args.start <= p["image_number"] <= args.end]
    if not selected:
        raise SystemExit("No prompts selected. Check --start/--end values.")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not args.dry_run and not api_key:
        raise SystemExit("OPENAI_API_KEY is required unless --dry-run is used.")

    for item in selected:
        n = item["image_number"]
        prompt = item["prompt"]
        filename = out_dir / f"image_{n:03d}.png"
        print(f"[{n:03d}] generating {filename.name}")

        if args.dry_run:
            rel_file = filename.resolve().relative_to(ROOT) if filename.resolve().is_relative_to(ROOT) else filename.resolve()
            manifest.append({"image_number": n, "file": str(rel_file), "prompt": prompt, "status": "dry-run"})
            continue

        try:
            png_bytes = call_images_api(api_key, prompt, args.model, args.size)
        except RuntimeError as err:
            raise SystemExit(str(err)) from err

        filename.write_bytes(png_bytes)
        rel_file = filename.resolve().relative_to(ROOT) if filename.resolve().is_relative_to(ROOT) else filename.resolve()
        manifest.append({"image_number": n, "file": str(rel_file), "prompt": prompt, "status": "generated"})
        time.sleep(args.sleep)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done: {len(selected)} entries written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
