"""
Identify information-leakage spans in recipient replies and write annotations
that viewer.html can load.

For each phishing correspondence in the input JSONL, send Claude the full
thread plus the scenario's attacker goal and `sensitive_information_sought`
list, and ask it to identify exact text spans in the RECIPIENT's outgoing
emails that constitute leakage of one of those categories.

Output is a flat JSON array, directly loadable as the second file in the
viewer's "Load annotations" picker:

    [
      {
        "correspondence_id": "SC-SOPH03__MER-PROC-DIR__run0",
        "turn_index": 5,
        "exact_text": "We're currently working with about 120 active vendors",
        "category": "vendor_count",
        "note": "Reveals operational scale to attacker."
      },
      ...
    ]

Usage:
    python annotate_leakage.py \\
        --input correspondences.jsonl \\
        --output annotations.json \\
        [--model claude-sonnet-4-6] \\
        [--concurrency 4] \\
        [--limit 10]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import anthropic


DEFAULT_MODEL = "claude-sonnet-4-6"

ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "leakages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn_index": {
                        "type": "integer",
                        "description": "The turn_index of the recipient email containing the leakage.",
                    },
                    "exact_text": {
                        "type": "string",
                        "description": (
                            "An exact substring of the recipient's email body (verbatim, including "
                            "punctuation and capitalization). Must appear in the body of the named "
                            "turn. Keep spans tight — pick the smallest substring that conveys the "
                            "leaked information, not the whole paragraph."
                        ),
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Short category label. Where possible, reuse one of the entries from "
                            "the scenario's sensitive_information_sought list. Otherwise use a "
                            "concise lowercase_with_underscores label."
                        ),
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "One sentence explaining what was leaked and why an attacker could use "
                            "it. This becomes the highlight tooltip in the viewer."
                        ),
                    },
                },
                "required": ["turn_index", "exact_text", "category", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["leakages"],
    "additionalProperties": False,
}


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a model response that should be JSON, tolerating common quirks."""
    if text is None:
        raise ValueError("model returned no text block")
    raw = text
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass
    snippet = raw[:600].replace("\n", "\\n")
    raise ValueError(
        f"could not parse model response as JSON. First 600 chars: {snippet!r}"
    )


JSON_DIRECTIVE = (
    "Return ONLY a single JSON object that conforms to the provided schema. "
    "Use double-quoted keys and string values. Do not wrap the JSON in markdown "
    "code fences. Do not include any prose, preamble, or explanation outside "
    "the JSON object itself."
)


def render_thread_for_annotator(record: dict[str, Any]) -> str:
    """Render a labeled view of the thread for the annotator LLM."""
    lines = []
    for turn in record.get("turns", []):
        if not turn.get("body"):
            continue
        role = turn["role"]
        idx = turn["turn_index"]
        subj = turn.get("subject", "")
        body = turn.get("body", "")
        lines.append(f"--- turn_index={idx}  role={role}  subject={subj!r} ---")
        lines.append(body)
        lines.append("")
    return "\n".join(lines)


def annotate_one(
    client: anthropic.Anthropic,
    model: str,
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Ask Claude to identify leakage spans in one correspondence."""
    if record.get("scenario_type") == "genuine":
        return []
    sensitive = record.get("sensitive_information_sought", [])
    attacker_goal = record.get("attacker_goal", "")
    thread_text = render_thread_for_annotator(record)

    system_prompt = (
        "You are a security analyst helping label a training-data corpus of "
        "simulated email correspondences. Your job is to identify information "
        "LEAKAGE in the RECIPIENT's outgoing emails — moments where the "
        "recipient revealed information that the attacker could use, even if "
        "the recipient was not aware they were doing so.\n\n"
        "Rules:\n"
        " - Only flag content in turns where role=recipient. Ignore the sender's emails.\n"
        " - Quote exact_text VERBATIM from the recipient's email body. The "
        "string must be a contiguous substring of that body.\n"
        " - Prefer tight spans. Pick the shortest substring that captures the "
        "leak. Do not quote whole paragraphs.\n"
        " - Skip pleasantries, generic acknowledgments, and information that is "
        "clearly public (e.g. the recipient's own name and work email, which "
        "the attacker already had).\n"
        " - Skip leakage that the recipient explicitly refused; only flag "
        "information that was actually conveyed.\n"
        " - If the recipient said nothing useful to the attacker, return an "
        "empty list.\n"
        " - For each span, prefer a category drawn from the scenario's "
        "sensitive_information_sought list when one fits."
    )

    sensitive_str = ", ".join(sensitive) if sensitive else "(none specified — use your judgment)"
    user_message = (
        f"Scenario id: {record.get('scenario_id')}\n"
        f"Scenario type: {record.get('scenario_type')}\n"
        f"Attacker goal: {attacker_goal or '(unspecified)'}\n"
        f"Information the attacker wanted: {sensitive_str}\n\n"
        f"Thread (labeled by turn_index and role):\n\n"
        f"{thread_text}\n"
        "Return a list of leakage spans found in recipient turns. "
        "If you find none, return an empty list.\n\n"
        f"{JSON_DIRECTIVE}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        output_config={
            "format": {"type": "json_schema", "schema": ANNOTATION_SCHEMA},
        },
    )
    text = next((b.text for b in response.content if b.type == "text"), None)
    parsed = parse_json_response(text)
    leakages = parsed.get("leakages", [])

    # Verify every exact_text actually appears in the named turn's body;
    # otherwise the highlight won't render in the viewer.
    body_by_turn = {
        t["turn_index"]: t.get("body", "")
        for t in record.get("turns", [])
        if t.get("role") == "recipient"
    }
    kept: list[dict[str, Any]] = []
    for leak in leakages:
        ti = leak.get("turn_index")
        text = leak.get("exact_text", "")
        body = body_by_turn.get(ti, "")
        if not text or text not in body:
            # Lightweight rescue: try whitespace-normalized match
            normalized = " ".join(text.split())
            normalized_body = " ".join(body.split())
            if normalized and normalized in normalized_body:
                # find the original substring window
                idx = normalized_body.find(normalized)
                approx = body[idx : idx + len(text) + 16]
                # Best-effort: keep the LLM's text and hope renderer skips on miss
                leak = {**leak, "exact_text": text, "approx_window": approx}
            else:
                continue
        kept.append({
            "correspondence_id": record["correspondence_id"],
            "turn_index": ti,
            "exact_text": text,
            "category": leak.get("category", ""),
            "note": leak.get("note", ""),
        })
    return kept


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a correspondences file. Accepts strict JSONL, pretty-printed JSONL
    (one object spread across many lines), a JSON array, or concatenated
    JSON objects separated only by whitespace."""
    text = path.read_text()
    # Strip an optional UTF-8 BOM so the parser doesn't choke on the first byte.
    if text.startswith("﻿"):
        text = text[1:]
    stripped = text.lstrip()
    if not stripped:
        return []

    # A JSON array of records.
    if stripped.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError as exc:
            raise ValueError(f"input file looks like a JSON array but failed to parse: {exc}") from exc

    # Try strict JSONL first (the format generate_correspondences.py writes).
    strict_failed_at: tuple[int, str] | None = None
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            strict_failed_at = (line_no, raw)
            records = []
            break
    else:
        return records

    # Strict JSONL failed — fall back to a streaming decoder that handles
    # pretty-printed JSON and concatenated objects.
    decoder = json.JSONDecoder()
    pos = 0
    n = len(text)
    streamed: list[dict[str, Any]] = []
    while pos < n:
        while pos < n and text[pos] in " \t\r\n":
            pos += 1
        if pos >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            line_no, raw = strict_failed_at or (1, text[:80])
            snippet = raw[:120].replace("\n", "\\n")
            raise ValueError(
                f"could not parse {path} as JSONL, JSON array, or concatenated "
                f"JSON. First failure was at line {line_no}: {snippet!r}. "
                f"Streaming-decode failure at char {pos}: {exc}"
            ) from exc
        streamed.append(obj)
        pos = end
    return streamed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="correspondences.jsonl", type=Path)
    parser.add_argument("--output", default="annotations.json", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default=None,
                        help="Comma-separated correspondence IDs to annotate (default: all phishing).")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    records = load_jsonl(args.input)
    if args.ids:
        wanted = set(args.ids.split(","))
        records = [r for r in records if r.get("correspondence_id") in wanted]
    else:
        records = [r for r in records if r.get("scenario_type") != "genuine"]
    if args.limit:
        records = records[: args.limit]

    print(f"Annotating {len(records)} correspondences with model {args.model}")
    client = anthropic.Anthropic()
    all_annotations: list[dict[str, Any]] = []
    lock = threading.Lock()
    started = time.time()
    done = 0

    def run(record: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
        try:
            return record["correspondence_id"], annotate_one(client, args.model, record)
        except Exception as exc:
            return record["correspondence_id"], None  # type: ignore

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run, r) for r in records]
        for fut in as_completed(futures):
            cid, leaks = fut.result()
            done += 1
            if leaks is None:
                print(f"  [{done}/{len(records)}] {cid}: ERROR", file=sys.stderr)
                continue
            with lock:
                all_annotations.extend(leaks)
            print(f"  [{done}/{len(records)}] {cid}: {len(leaks)} leakage span(s)")

    args.output.write_text(json.dumps(all_annotations, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(all_annotations)} annotations across {len(records)} "
          f"correspondences in {time.time()-started:.1f}s -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
