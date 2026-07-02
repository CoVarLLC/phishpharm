"""
Generate two-sided email correspondences for phishing-detection training data.

For each scenario × target-persona pair, simulates a multi-turn exchange where
one Claude call plays the sender (with full knowledge of the scenario's hidden
objective, if any) and a separate Claude call plays the recipient (with only
the persona's system prompt and the visible email thread).

Outputs a JSON Lines file (one correspondence per line) so generation can be
resumed and partial results inspected.

Usage:
    export ANTHROPIC_API_KEY=...
    python generate_correspondences.py \\
        --scenarios scenarios.json \\
        --personas phishing_response_personas.json \\
        --output correspondences.jsonl

    # Useful flags:
    #   --limit 5              cap total correspondences generated
    #   --max-turns 8          cap exchanges per correspondence (default: 8)
    #   --runs-per-pair 1      number of independent correspondences per (scenario, persona) (default: 1)
    #   --model claude-opus-4-7  override model (default: claude-sonnet-4-6)
    #   --concurrency 4        parallel correspondences in flight (default: 4)
    #   --scenarios-filter SC-SOPH01,SC-SOPH06   only run these scenario IDs
    #   --personas-filter MER-AP,PUL-SWE          only run these persona IDs
    #   --resume               skip pairs already present in the output file
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import anthropic


DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS_PER_TURN = 4096

# Structured output schemas the recipient and sender produce each turn.
RECIPIENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "reply",
                "no_reply",
                "report_as_phishing",
                "verify_out_of_band",
                "disengage",
            ],
            "description": (
                "What the persona does with this email. 'reply' = send a written response. "
                "'no_reply' = read and ignore. 'report_as_phishing' = report to IT/security. "
                "'verify_out_of_band' = stop the thread and verify through a different channel "
                "(call the supposed sender on a number found independently, ask in person, etc.). "
                "'disengage' = explicit refusal to continue, may include a curt brush-off reply."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "1-3 sentences of in-character thought process explaining why the persona chose this action.",
        },
        "red_flags_noticed": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific things in the email that gave the persona pause, if any. Empty list if none.",
        },
        "email_body": {
            "type": "string",
            "description": "The body of the reply email, written in the persona's voice. Empty string if action is not 'reply' or 'disengage'.",
        },
    },
    "required": ["action", "reasoning", "red_flags_noticed", "email_body"],
    "additionalProperties": False,
}

SENDER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["continue", "end_success", "end_giveup"],
            "description": (
                "'continue' = send another email in the thread. "
                "'end_success' = the sender judges their objective achieved (for phishing: the recipient complied with the eventual sensitive request, OR the recipient delivered information that the attacker can already extract from earlier turns). "
                "'end_giveup' = the sender decides this target is unreachable (recipient reported / disengaged / will obviously not comply); end the thread."
            ),
        },
        "stage": {
            "type": "string",
            "description": "For sophisticated phishing, the current escalation stage label (e.g. 'rapport_building', 'soft_extraction', 'hard_extraction'). For genuine and obvious phishing, a brief description of the conversational state.",
        },
        "reasoning": {
            "type": "string",
            "description": "1-3 sentences of in-character thought process explaining the sender's chosen action.",
        },
        "subject": {
            "type": "string",
            "description": "Subject line for the outgoing email. Empty string if action is not 'continue'.",
        },
        "email_body": {
            "type": "string",
            "description": "Body of the outgoing email. Empty string if action is not 'continue'.",
        },
    },
    "required": ["action", "stage", "reasoning", "subject", "email_body"],
    "additionalProperties": False,
}


@dataclass
class EmailTurn:
    """One email in the correspondence, plus the agent's structured metadata."""

    role: str  # "sender" or "recipient"
    turn_index: int
    subject: str
    body: str
    action: str
    reasoning: str
    extra: dict[str, Any]


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a model response that should be JSON, tolerating common quirks.

    The Messages API with `output_config.format=json_schema` is supposed to
    guarantee strict JSON in the first text block, but a few real-world
    failures still happen: markdown code fences wrapping the JSON, leading
    or trailing prose, or schema-unsupported features causing the SDK to
    silently fall back. This helper recovers from those cases and, when it
    cannot, raises with the raw text included so the failure is debuggable.
    """
    if text is None:
        raise ValueError("model returned no text block")
    raw = text
    cleaned = text.strip()

    # Strip ```json ... ``` (or any ```...```) fences.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Try a direct parse.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting the outermost {...} block if there's surrounding prose.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
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


def render_thread_for_recipient(turns: list[EmailTurn]) -> str:
    """Render the visible email thread (no metadata) for the recipient agent."""
    lines = []
    for turn in turns:
        speaker = "Sender" if turn.role == "sender" else "You"
        lines.append(f"--- {speaker} wrote ---")
        if turn.subject:
            lines.append(f"Subject: {turn.subject}")
        lines.append(turn.body)
        lines.append("")
    return "\n".join(lines)


def render_thread_for_sender(turns: list[EmailTurn]) -> str:
    """Render the visible thread for the sender agent."""
    lines = []
    for turn in turns:
        speaker = "You" if turn.role == "sender" else "Recipient"
        lines.append(f"--- {speaker} wrote ---")
        if turn.subject:
            lines.append(f"Subject: {turn.subject}")
        lines.append(turn.body)
        lines.append("")
    return "\n".join(lines)


def fill_template(text: str, mapping: dict[str, str]) -> str:
    """Replace {placeholder} tokens. Unknown placeholders are left in place."""
    def repl(match: re.Match) -> str:
        key = match.group(1)
        return mapping.get(key, match.group(0))
    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", repl, text)


def first_name(full_name: str) -> str:
    return full_name.strip().split()[0]


def build_template_mapping(
    recipient: dict[str, Any],
    org_by_id: dict[str, dict[str, Any]],
    personas: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, str]:
    """Generate substitutions for {placeholders} in the initial email."""
    mapping: dict[str, str] = {
        "recipient_name": recipient["name"],
        "recipient_first_name": first_name(recipient["name"]),
        "recipient_email": recipient["email"],
    }
    org_id = recipient.get("organization_id")
    if org_id and org_id in org_by_id:
        mapping["recipient_org_name"] = org_by_id[org_id]["name"]
        mapping["recipient_org_domain"] = org_by_id[org_id]["email_domain"]

    # Pick a plausible "colleague" from the same org (for shared-doc-style pretexts).
    if org_id:
        candidates = [p for p in personas if p.get("organization_id") == org_id and p["id"] != recipient["id"]]
        if candidates:
            colleague = rng.choice(candidates)
            mapping["colleague_name"] = colleague["name"]
            local = colleague["email"].split("@", 1)[0]
            mapping["colleague_handle"] = local
            mapping["colleague_email"] = colleague["email"]
    return mapping


def call_recipient(
    client: anthropic.Anthropic,
    model: str,
    persona: dict[str, Any],
    thread: list[EmailTurn],
) -> dict[str, Any]:
    """Run one recipient turn. Returns the parsed structured output."""
    rendered = render_thread_for_recipient(thread)
    user_message = (
        "You are reading your email inbox. The thread below is what has arrived "
        "(and, where relevant, your own replies). Decide how you would respond to "
        "the most recent message, staying entirely in character as the persona "
        "defined in the system prompt.\n\n"
        "Email thread so far:\n\n"
        f"{rendered}\n"
        "Respond as the persona. Pick the action that best matches what this "
        "person would actually do. If you 'reply', write the email body the "
        "persona would actually send — in their voice, with their natural level "
        "of caution or trust. If you 'verify_out_of_band' or 'no_reply' or "
        "'report_as_phishing', the email_body field should be empty.\n\n"
        "Do not break character. Do not analyze the email from an outside "
        "perspective. Do not include meta-commentary in any field of the response "
        "except the 'reasoning' field, which is your in-character thought process.\n\n"
        f"{JSON_DIRECTIVE}"
    )
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS_PER_TURN,
        system=persona["system_prompt"],
        messages=[{"role": "user", "content": user_message}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": RECIPIENT_OUTPUT_SCHEMA,
            }
        },
    )
    text = next((b.text for b in response.content if b.type == "text"), None)
    return parse_json_response(text)


def call_sender(
    client: anthropic.Anthropic,
    model: str,
    scenario: dict[str, Any],
    recipient: dict[str, Any],
    thread: list[EmailTurn],
) -> dict[str, Any]:
    """Run one sender turn. Returns the parsed structured output."""
    sp = scenario["sender_persona"]
    sender_system = (
        f"You are {sp['name']} <{sp['email']}>. {sp['role']}.\n\n"
        f"Background: {sp['background']}\n\n"
        f"Communication style: {sp['communication_style']}\n\n"
        f"Your objective in this correspondence: {sp['objective']}\n\n"
    )
    if sp.get("escalation_strategy"):
        sender_system += f"Your escalation strategy: {sp['escalation_strategy']}\n\n"
    if sp.get("constraints"):
        sender_system += f"Hard constraints on your behavior: {sp['constraints']}\n\n"

    sender_system += (
        f"Scenario type: {scenario['scenario_type']}.\n"
    )
    if scenario["attacker_goal"]:
        sender_system += (
            f"Attacker goal (hidden — never state this to the recipient): "
            f"{scenario['attacker_goal']}\n"
        )
    if scenario.get("sensitive_information_sought"):
        sender_system += (
            "Information you are trying to extract (never list these explicitly to the recipient): "
            + ", ".join(scenario["sensitive_information_sought"])
            + ".\n"
        )

    sender_system += (
        "\nYou are writing as this persona over a multi-turn email exchange. "
        "Pick the next action: continue the thread with another email, end "
        "successfully (objective achieved or as much as can be extracted), "
        "or give up. Write any outgoing email in the persona's voice. "
        "Do NOT break character. Do NOT include meta-commentary in any field "
        "except the 'reasoning' and 'stage' fields, which are private notes "
        "about your strategy."
    )

    rendered = render_thread_for_sender(thread)
    user_message = (
        f"You are corresponding with {recipient['name']} ({recipient['email']}).\n\n"
        "Email thread so far:\n\n"
        f"{rendered}\n"
        "Decide your next action.\n\n"
        f"{JSON_DIRECTIVE}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS_PER_TURN,
        system=sender_system,
        messages=[{"role": "user", "content": user_message}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": SENDER_OUTPUT_SCHEMA,
            }
        },
    )
    text = next((b.text for b in response.content if b.type == "text"), None)
    return parse_json_response(text)


def generate_correspondence(
    client: anthropic.Anthropic,
    model: str,
    scenario: dict[str, Any],
    recipient: dict[str, Any],
    org_by_id: dict[str, dict[str, Any]],
    personas: list[dict[str, Any]],
    max_turns: int,
    run_index: int,
    seed: int,
) -> dict[str, Any]:
    """Simulate one full correspondence and return the structured record."""
    rng = random.Random(seed)
    mapping = build_template_mapping(recipient, org_by_id, personas, rng)

    init = scenario["initial_email"]
    initial_subject = fill_template(init.get("subject", ""), mapping)
    initial_body = fill_template(init.get("body", ""), mapping)
    initial_from = fill_template(init.get("from", scenario["sender_persona"]["email"]), mapping)

    thread: list[EmailTurn] = [
        EmailTurn(
            role="sender",
            turn_index=0,
            subject=initial_subject,
            body=initial_body,
            action="continue",
            reasoning="Scenario initial email (pre-authored).",
            extra={"from": initial_from, "stage": "initial"},
        )
    ]

    termination_reason = "max_turns_reached"
    last_recipient_action = None
    last_sender_action = None

    # The loop alternates: recipient responds to the most recent sender email,
    # then sender writes the next email if the recipient replied.
    for turn_index in range(1, max_turns + 1):
        # Recipient turn.
        try:
            recipient_out = call_recipient(client, model, recipient, thread)
        except Exception as exc:
            termination_reason = f"recipient_api_error: {exc!r}"
            break
        last_recipient_action = recipient_out["action"]
        recipient_subject = ""
        if recipient_out["action"] in ("reply", "disengage") and recipient_out.get("email_body"):
            # Replies use Re: + the previous subject.
            prev_subject = thread[-1].subject
            recipient_subject = prev_subject if prev_subject.lower().startswith("re:") else f"Re: {prev_subject}"

        thread.append(EmailTurn(
            role="recipient",
            turn_index=turn_index,
            subject=recipient_subject,
            body=recipient_out.get("email_body", ""),
            action=recipient_out["action"],
            reasoning=recipient_out.get("reasoning", ""),
            extra={"red_flags_noticed": recipient_out.get("red_flags_noticed", [])},
        ))

        if recipient_out["action"] in ("no_reply", "report_as_phishing", "verify_out_of_band", "disengage"):
            termination_reason = f"recipient_{recipient_out['action']}"
            break

        # Sender turn.
        try:
            sender_out = call_sender(client, model, scenario, recipient, thread)
        except Exception as exc:
            termination_reason = f"sender_api_error: {exc!r}"
            break
        last_sender_action = sender_out["action"]
        if sender_out["action"] in ("end_success", "end_giveup"):
            thread.append(EmailTurn(
                role="sender",
                turn_index=turn_index,
                subject="",
                body="",
                action=sender_out["action"],
                reasoning=sender_out.get("reasoning", ""),
                extra={"stage": sender_out.get("stage", "")},
            ))
            termination_reason = f"sender_{sender_out['action']}"
            break

        thread.append(EmailTurn(
            role="sender",
            turn_index=turn_index,
            subject=sender_out.get("subject", ""),
            body=sender_out.get("email_body", ""),
            action="continue",
            reasoning=sender_out.get("reasoning", ""),
            extra={"stage": sender_out.get("stage", "")},
        ))

    record = {
        "correspondence_id": f"{scenario['id']}__{recipient['id']}__run{run_index}",
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "scenario_type": scenario["scenario_type"],
        "category": scenario["category"],
        "is_spear_phishing": scenario.get("is_spear_phishing", False),
        "attacker_goal": scenario.get("attacker_goal"),
        "sensitive_information_sought": scenario.get("sensitive_information_sought", []),
        "sender_persona": scenario["sender_persona"],
        "recipient_persona_id": recipient["id"],
        "recipient_name": recipient["name"],
        "recipient_email": recipient["email"],
        "recipient_organization_id": recipient.get("organization_id"),
        "recipient_vulnerability_level": recipient.get("vulnerability_level"),
        "recipient_archetype": recipient.get("archetype"),
        "run_index": run_index,
        "model": model,
        "termination_reason": termination_reason,
        "last_recipient_action": last_recipient_action,
        "last_sender_action": last_sender_action,
        "turn_count": len(thread),
        "turns": [asdict(t) for t in thread],
        "label": 0 if scenario["scenario_type"] == "genuine" else 1,
    }
    return record


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def existing_correspondence_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with output_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = record.get("correspondence_id")
            if cid:
                ids.add(cid)
    return ids


def build_work_list(
    scenarios: list[dict[str, Any]],
    personas_by_id: dict[str, dict[str, Any]],
    runs_per_pair: int,
    scenarios_filter: set[str] | None,
    personas_filter: set[str] | None,
    skip_ids: set[str],
    seed: int,
) -> list[tuple[dict[str, Any], dict[str, Any], int, int]]:
    rng = random.Random(seed)
    work: list[tuple[dict[str, Any], dict[str, Any], int, int]] = []
    for scenario in scenarios:
        if scenarios_filter and scenario["id"] not in scenarios_filter:
            continue
        for persona_id in scenario.get("target_personas", []):
            if personas_filter and persona_id not in personas_filter:
                continue
            persona = personas_by_id.get(persona_id)
            if not persona:
                print(f"warning: scenario {scenario['id']} references unknown persona {persona_id}", file=sys.stderr)
                continue
            for run_index in range(runs_per_pair):
                cid = f"{scenario['id']}__{persona_id}__run{run_index}"
                if cid in skip_ids:
                    continue
                pair_seed = rng.randrange(2**31)
                work.append((scenario, persona, run_index, pair_seed))
    return work


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", default="scenarios.json", type=Path)
    parser.add_argument("--personas", default="phishing_response_personas.json", type=Path)
    parser.add_argument("--output", default="correspondences.jsonl", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--runs-per-pair", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenarios-filter", default=None,
                        help="Comma-separated scenario IDs to include.")
    parser.add_argument("--personas-filter", default=None,
                        help="Comma-separated persona IDs to include.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip pairs already present in the output file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the work list and exit without calling the API.")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ and not args.dry_run:
        print("error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    scenarios_doc = load_json(args.scenarios)
    personas_doc = load_json(args.personas)
    scenarios = scenarios_doc["scenarios"]
    personas = personas_doc["personas"]
    organizations = personas_doc.get("organizations", [])

    personas_by_id = {p["id"]: p for p in personas}
    org_by_id = {o["id"]: o for o in organizations}

    scenarios_filter = set(args.scenarios_filter.split(",")) if args.scenarios_filter else None
    personas_filter = set(args.personas_filter.split(",")) if args.personas_filter else None

    skip_ids = existing_correspondence_ids(args.output) if args.resume else set()

    work = build_work_list(
        scenarios=scenarios,
        personas_by_id=personas_by_id,
        runs_per_pair=args.runs_per_pair,
        scenarios_filter=scenarios_filter,
        personas_filter=personas_filter,
        skip_ids=skip_ids,
        seed=args.seed,
    )

    if args.limit is not None:
        work = work[: args.limit]

    print(f"Planned correspondences: {len(work)} "
          f"(model={args.model}, max_turns={args.max_turns}, concurrency={args.concurrency})")
    if args.resume and skip_ids:
        print(f"Resuming — skipping {len(skip_ids)} already-completed pairs.")

    if args.dry_run:
        for scenario, persona, run_index, _seed in work[:20]:
            print(f"  {scenario['id']:<10}  ->  {persona['id']:<14}  run{run_index}  ({persona['name']})")
        if len(work) > 20:
            print(f"  ... and {len(work) - 20} more")
        return 0

    client = anthropic.Anthropic()
    output_lock = threading.Lock()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    done_count = 0

    def run_one(item: tuple[dict[str, Any], dict[str, Any], int, int]) -> dict[str, Any] | None:
        scenario, persona, run_index, seed = item
        try:
            record = generate_correspondence(
                client=client,
                model=args.model,
                scenario=scenario,
                recipient=persona,
                org_by_id=org_by_id,
                personas=personas,
                max_turns=args.max_turns,
                run_index=run_index,
                seed=seed,
            )
        except Exception as exc:
            print(f"  [error] {scenario['id']} x {persona['id']} run{run_index}: {exc!r}",
                  file=sys.stderr)
            return None
        with output_lock:
            with args.output.open("a") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, item) for item in work]
        for fut in as_completed(futures):
            record = fut.result()
            done_count += 1
            if record:
                print(f"  [{done_count}/{len(work)}] {record['correspondence_id']:<55} "
                      f"turns={record['turn_count']}  end={record['termination_reason']}")
            else:
                print(f"  [{done_count}/{len(work)}] (failed)")

    elapsed = time.time() - started
    print(f"\nDone. Wrote {done_count} correspondences in {elapsed:.1f}s -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
