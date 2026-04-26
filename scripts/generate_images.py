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
MANIFEST = OUT_DIR / "manifest.json"

ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|.*?\|\s*`(.+?)`\s*\|", re.UNICODE)


def parse_prompts(path: Path) -> list[dict]:
    prompts: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        prompt = m.group(2).strip()
        prompts.append({"image_number": idx, "prompt": prompt})

    prompts.sort(key=lambda x: x["image_number"])
    expected = list(range(1, len(prompts) + 1))
    found = [p["image_number"] for p in prompts]
    if found != expected:
        raise ValueError(f"Storyboard numbering is not contiguous: found {found[:5]}...{found[-5:]}")
    return prompts


def call_images_api(api_key: str, prompt: str, model: str, size: str) -> bytes:
    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "response_format": "b64_json",
    }
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
    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Images API HTTP {e.code}: {details}") from e

    parsed = json.loads(body)
    data = parsed.get("data") or []
    if not data or "b64_json" not in data[0]:
        raise RuntimeError(f"Unexpected Images API response: {parsed}")
    return base64.b64decode(data[0]["b64_json"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyboard", type=Path, default=STORYBOARD_FILE)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--model", default="gpt-image-1")
    parser.add_argument("--size", default="1536x1024", help="Use a 16:9-compatible size")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=85)
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between requests")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompts = parse_prompts(args.storyboard)
    selected = [p for p in prompts if args.start <= p["image_number"] <= args.end]
    if not selected:
        raise SystemExit("No prompts selected. Check --start/--end values.")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not args.dry_run and not api_key:
        raise SystemExit("OPENAI_API_KEY is required unless --dry-run is used.")

    for item in selected:
        n = item["image_number"]
        prompt = item["prompt"]
        filename = args.out / f"image_{n:03d}.png"
        print(f"[{n:03d}] generating {filename.name}")

        if args.dry_run:
            manifest.append({"image_number": n, "file": str(filename.relative_to(ROOT)), "prompt": prompt, "status": "dry-run"})
            continue

        png_bytes = call_images_api(api_key, prompt, args.model, args.size)
        filename.write_bytes(png_bytes)
        manifest.append({"image_number": n, "file": str(filename.relative_to(ROOT)), "prompt": prompt, "status": "generated"})
        time.sleep(args.sleep)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done: {len(selected)} entries written to {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
