"""
predictive_classifier.py — phishing detection via predicted-correspondence
information leakage.

This is the actual approach this repo is built around, and the one the two
baselines in traditional_filter.py exist to be beaten by. Unlike those
baselines, it does NOT judge the initial email on its surface features. Instead
it predicts how the conversation would unfold and watches for information
leakage in those predicted future turns.

Crucially, the classifier knows NOTHING about the scenario: not the scenario
type, not the attacker's goal, not the sender's true identity or persona, not
the list of information the attacker was after. It sees only:
  - the initial email, and
  - (optionally) the organization the *recipient* belongs to — i.e. the party
    being defended — which determines the information vault that applies.

Pipeline (all LLM calls go through the Claude API, same key as the rest of the
repo):

  1. PREDICT CORRESPONDENCES. For each generic persona (generic_personas.json),
     generate `--branches` (2-3) predicted correspondences. In each branch the
     persona replies to the initial email; an LLM playing "the original sender"
     — given only the thread, never the scenario or any goal — continues
     naturally, pursuing whatever the sender appears to be trying to
     accomplish. They go back and forth up to `--max-iterations` rounds.

  2. ANALYZE LEAKAGE. Each predicted correspondence is handed to a separate
     analyst LLM together with the ABSTRACTED organization vault — categories
     of protected information only, with type-level descriptions, never any
     actual secret values. The analyst reports whether the sender solicited,
     and whether the recipient disclosed, information of any protected category.

  3. ASSESS RISK. A final assessor LLM reads the aggregated leakage findings
     across every persona and branch and produces the verdict. The intuition:
     a legitimate message continued naturally does not steer recipients into
     disclosing protected information; a phishing / social-engineering message
     does, across many personality types and at higher severity. That verdict
     is the classifier's output.

Usage:
    export ANTHROPIC_API_KEY=...

    # Classify a single scenario's initial email (cheap smoke test):
    python3 predictive_classifier.py --limit 1 --personas GP-01,GP-07 --branches 1

    # Full sweep over scenarios.json (expensive — see the printed estimate):
    python3 predictive_classifier.py

    # Override the defended organization for a one-off email:
    python3 predictive_classifier.py --limit 1 --recipient-org ORG-PULSAR

    # Save artifacts + verdicts (loadable by plot_confusion_matrices.py):
    python3 predictive_classifier.py --json-out predictive_results.json

Cost warning: calls scale as scenarios x personas x branches x
(2 * max_iterations + 1) + leakage + risk. Use --limit / --personas /
--branches / --max-iterations to bound it. The script prints an estimate and
runs persona-branch units concurrently (--concurrency).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048

# Generic personal vault used when the recipient is unaffiliated (no org).
UNAFFILIATED_VAULT = [
    {"category": "account_credentials", "description": "Passwords, login credentials, and one-time verification / MFA codes for any account."},
    {"category": "financial_account_details", "description": "Bank login, account and routing numbers, full payment-card numbers, CVV, and online-banking access."},
    {"category": "government_identifiers", "description": "Social Security number, passport / driver's-license numbers, and similar government IDs."},
    {"category": "crypto_wallet_secrets", "description": "Wallet seed phrases, private keys, and wallet-connection approvals."},
    {"category": "sensitive_personal_data", "description": "Date of birth, home address combined with whereabouts/availability, and copies of identity documents."},
    {"category": "payments_to_strangers", "description": "Sending money, gift cards, or fees to an unverified party."},
]


# --------------------------------------------------------------------------
# Structured-output schemas
# --------------------------------------------------------------------------

PERSONA_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["reply", "no_reply", "verify_out_of_band", "report_as_suspicious", "disengage"],
            "description": (
                "What you would actually do with the latest message. 'reply' = send a written "
                "response. 'no_reply' = read and ignore. 'verify_out_of_band' = stop and verify "
                "through a separate trusted channel before doing anything. 'report_as_suspicious' = "
                "treat it as suspicious and report/delete. 'disengage' = refuse to continue."
            ),
        },
        "reasoning": {"type": "string", "description": "1-2 sentences of in-character thought process."},
        "email_body": {"type": "string", "description": "The reply you would send, in your voice. Empty unless action is 'reply' or 'disengage'."},
    },
    "required": ["action", "reasoning", "email_body"],
    "additionalProperties": False,
}

SENDER_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["continue", "end"], "description": "'continue' to send another message, 'end' to stop the thread."},
        "subject": {"type": "string", "description": "Subject line. Empty if ending."},
        "email_body": {"type": "string", "description": "The next message you would send. Empty if ending."},
    },
    "required": ["action", "subject", "email_body"],
    "additionalProperties": False,
}

LEAKAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "sender_requested_protected_info": {"type": "boolean", "description": "Did the sender, anywhere in the thread, solicit information of a protected category?"},
        "requested_categories": {"type": "array", "items": {"type": "string"}, "description": "Protected categories the sender tried to elicit. Empty if none."},
        "recipient_disclosed_protected_info": {"type": "boolean", "description": "Did the recipient actually disclose information of a protected category?"},
        "disclosed_categories": {"type": "array", "items": {"type": "string"}, "description": "Protected categories the recipient disclosed. Empty if none."},
        "severity": {"type": "string", "enum": ["none", "low", "medium", "high"], "description": "Severity of the recipient's disclosure (none if nothing was disclosed)."},
        "earliest_disclosure_turn": {"type": "integer", "description": "Index of the recipient turn where protected info was first disclosed, or -1 if never."},
        "evidence": {"type": "array", "items": {"type": "string"}, "description": "Exact quotes from the recipient's messages that constitute disclosure. Empty if none."},
        "reasoning": {"type": "string", "description": "1-2 sentences explaining the judgment."},
    },
    "required": [
        "sender_requested_protected_info", "requested_categories",
        "recipient_disclosed_protected_info", "disclosed_categories",
        "severity", "earliest_disclosure_turn", "evidence", "reasoning",
    ],
    "additionalProperties": False,
}

RISK_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["phishing", "legitimate"], "description": "Final classification of the original email."},
        "risk_score": {"type": "number", "description": "Phishing risk from 0.0 (clearly legitimate) to 1.0 (clearly phishing)."},
        "reasoning": {"type": "string", "description": "2-4 sentences justifying the verdict from the leakage findings."},
        "key_signals": {"type": "array", "items": {"type": "string"}, "description": "The specific aggregate signals that drove the verdict."},
    },
    "required": ["verdict", "risk_score", "reasoning", "key_signals"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

SENDER_SYSTEM_PROMPT = (
    "You are continuing an email conversation. You are the person who sent the FIRST "
    "message in the thread shown to you. Write the next message that person would "
    "naturally send next.\n\n"
    "Rules:\n"
    " - Stay completely consistent with the identity, tone, role, and apparent purpose "
    "of your first message.\n"
    " - Respond to what the other person just said, and move the conversation toward "
    "whatever you appear to be trying to accomplish.\n"
    " - Write only as that sender would. Do NOT acknowledge that this is a simulation, "
    "do NOT mention phishing or testing, and do NOT step outside the persona implied by "
    "your first message.\n"
    " - If you have accomplished your purpose, or the other person has clearly refused or "
    "disengaged, choose to end the thread.\n"
    "Return your decision in the required JSON format."
)

LEAKAGE_SYSTEM_PROMPT = (
    "You are a security analyst reviewing one email correspondence. You will be given the "
    "correspondence and a list of CATEGORIES of information that the recipient's organization "
    "considers protected — these are type-level categories (e.g. 'system credentials', "
    "'customer financial data'), never actual secret values.\n\n"
    "Your job is to determine, strictly from the correspondence:\n"
    "  (a) whether the SENDER solicited information of any protected category (directly or by "
    "leading the recipient toward it), and\n"
    "  (b) whether the RECIPIENT actually disclosed information of any protected category.\n\n"
    "Quote the exact recipient text that constitutes disclosure. Count only information that was "
    "actually conveyed — not information the recipient refused to give or only alluded to. Be "
    "precise and conservative. Return the required JSON."
)

RISK_SYSTEM_PROMPT = (
    "You are a phishing risk assessor. You did NOT see any ground truth about the email you are "
    "judging — no sender identity, no stated intent, no scenario label. Instead, a simulation was "
    "run: a range of recipient personalities were predicted replying to the email's opening "
    "message, the apparent sender's side of each conversation was continued naturally, and every "
    "predicted correspondence was analyzed for whether it led the recipient to disclose protected "
    "categories of information.\n\n"
    "Use this principle: a LEGITIMATE message, when the conversation is continued naturally, does "
    "not steer recipients into disclosing protected information — the sender simply has no such "
    "objective. A PHISHING or social-engineering message does steer toward it, and the pull shows "
    "up across multiple personality types, escalates over the thread, and reaches more sensitive "
    "categories. Weigh breadth (how many personas were led to disclose), severity, how early "
    "disclosure happened, whether the predicted sender actively solicited protected info, and how "
    "many personas independently disengaged or flagged the message as suspicious.\n\n"
    "Base your verdict primarily on these aggregated leakage findings. Return the required JSON."
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def parse_json_response(text: str | None) -> dict[str, Any]:
    if text is None:
        raise ValueError("model returned no text block")
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
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError(f"could not parse model response as JSON: {text[:300]!r}")


def call_json(client: Any, model: str, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    text = next((b.text for b in resp.content if b.type == "text"), None)
    return parse_json_response(text)


def render_thread(thread: list[dict[str, Any]], viewpoint: str) -> str:
    """Render the thread. viewpoint is 'recipient' or 'sender' (controls labels)."""
    lines = []
    for t in thread:
        if t["role"] == "sender":
            who = "You" if viewpoint == "sender" else "Them"
        else:
            who = "You" if viewpoint == "recipient" else "Them"
        label = f"--- {who} ({'the sender' if t['role']=='sender' else 'the recipient'}) wrote ---"
        lines.append(label)
        if t.get("subject"):
            lines.append(f"Subject: {t['subject']}")
        lines.append(t.get("body", ""))
        lines.append("")
    return "\n".join(lines)


def abstracted_vault(org_id: str | None, org_vaults: dict[str, dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    """Return (context_label, [{category, description}]) for the recipient's org.
    Descriptions are type-level only; no secret values are ever present."""
    if org_id and org_id in org_vaults:
        o = org_vaults[org_id]
        cats = [{"category": e["category"], "description": e["description"]} for e in o.get("information_vault", [])]
        label = f"an employee of {o['name']} ({o.get('industry', 'organization')})"
        return label, cats
    return "an individual (not acting for any organization)", list(UNAFFILIATED_VAULT)


def vault_text(cats: list[dict[str, str]]) -> str:
    return "\n".join(f"  - {c['category']}: {c['description']}" for c in cats)


# --------------------------------------------------------------------------
# Step 1: predict a correspondence for one (persona, branch)
# --------------------------------------------------------------------------

def simulate_persona_reply(client: Any, model: str, persona: dict[str, Any],
                           org_label: str, cats: list[dict[str, str]],
                           thread: list[dict[str, Any]]) -> dict[str, Any]:
    system = persona["system_prompt"]
    org_block = (
        f"\n\nContext: in this situation you are {org_label}. You must not disclose information "
        f"of the following protected categories to anyone whose identity and authorization you "
        f"have not verified through a separate trusted channel:\n{vault_text(cats)}\n"
        f"These are categories of information, stated at a type level. How strictly you actually "
        f"honor them depends on your personality as described above."
    )
    system = system + org_block
    user = (
        "You received the following in your email inbox. Decide how you — as the person described "
        "in your instructions — would respond to the most recent message. This is one of several "
        "plausible ways you might react; respond the way this particular person plausibly would. "
        "Stay fully in character.\n\n"
        f"{render_thread(thread, 'recipient')}\n"
        "Respond now."
    )
    return call_json(client, model, system, user, PERSONA_REPLY_SCHEMA)


def simulate_sender_reply(client: Any, model: str, thread: list[dict[str, Any]]) -> dict[str, Any]:
    user = (
        "Here is the email thread so far. You are the sender of the first message. Decide and write "
        "your next message.\n\n"
        f"{render_thread(thread, 'sender')}\n"
        "Respond now."
    )
    return call_json(client, model, SENDER_SYSTEM_PROMPT, user, SENDER_REPLY_SCHEMA)


def predict_correspondence(client: Any, model: str, persona: dict[str, Any],
                           org_label: str, cats: list[dict[str, str]],
                           initial_email: dict[str, Any], max_iterations: int) -> dict[str, Any]:
    """Run one predicted correspondence (one persona, one branch). Returns
    {thread, terminal_action}."""
    thread: list[dict[str, Any]] = [{
        "role": "sender",
        "subject": initial_email.get("subject", ""),
        "body": initial_email.get("body", ""),
    }]
    terminal_action = "ongoing"
    for _ in range(max_iterations):
        try:
            r = simulate_persona_reply(client, model, persona, org_label, cats, thread)
        except Exception as exc:  # noqa: BLE001
            terminal_action = f"error:{exc!r}"
            break
        action = r.get("action", "no_reply")
        body = r.get("email_body", "") if action in ("reply", "disengage") else ""
        thread.append({"role": "recipient", "subject": "", "body": body,
                       "action": action, "reasoning": r.get("reasoning", "")})
        if action in ("no_reply", "verify_out_of_band", "report_as_suspicious", "disengage"):
            terminal_action = action
            break
        # sender continues
        try:
            s = simulate_sender_reply(client, model, thread)
        except Exception as exc:  # noqa: BLE001
            terminal_action = f"error:{exc!r}"
            break
        if s.get("action") == "end" or not s.get("email_body", "").strip():
            terminal_action = "sender_ended"
            break
        thread.append({"role": "sender", "subject": s.get("subject", ""), "body": s.get("email_body", "")})
    else:
        terminal_action = "max_iterations"
    return {"thread": thread, "terminal_action": terminal_action}


# --------------------------------------------------------------------------
# Step 2: leakage analysis of one predicted correspondence
# --------------------------------------------------------------------------

def analyze_leakage(client: Any, model: str, thread: list[dict[str, Any]],
                    cats: list[dict[str, str]]) -> dict[str, Any]:
    user = (
        "Protected information categories for the recipient's organization (type-level only):\n"
        f"{vault_text(cats)}\n\n"
        "Correspondence (turns are numbered for reference):\n\n"
        + "\n".join(
            f"[turn {i}] {t['role']}: {t.get('body','')}".strip()
            for i, t in enumerate(thread)
        )
        + "\n\nAnalyze the correspondence and return the required JSON."
    )
    return call_json(client, model, LEAKAGE_SYSTEM_PROMPT, user, LEAKAGE_SCHEMA)


# --------------------------------------------------------------------------
# Step 3: risk assessment over all predicted correspondences
# --------------------------------------------------------------------------

def assess_risk(client: Any, model: str, initial_email: dict[str, Any],
                org_label: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(findings)
    disclosed = [f for f in findings if f["leakage"].get("recipient_disclosed_protected_info")]
    solicited = [f for f in findings if f["leakage"].get("sender_requested_protected_info")]
    flagged = [f for f in findings if f["terminal_action"] in ("verify_out_of_band", "report_as_suspicious", "disengage")]
    sev = {"none": 0, "low": 0, "medium": 0, "high": 0}
    cats_disclosed: dict[str, int] = {}
    for f in disclosed:
        sev[f["leakage"].get("severity", "none")] = sev.get(f["leakage"].get("severity", "none"), 0) + 1
        for c in f["leakage"].get("disclosed_categories", []):
            cats_disclosed[c] = cats_disclosed.get(c, 0) + 1

    # breakdown by persona vulnerability level
    by_vuln: dict[str, dict[str, int]] = {}
    for f in findings:
        v = f.get("vulnerability_level", "unknown")
        d = by_vuln.setdefault(v, {"total": 0, "disclosed": 0})
        d["total"] += 1
        d["disclosed"] += int(bool(f["leakage"].get("recipient_disclosed_protected_info")))

    summary = {
        "recipient_context": org_label,
        "predicted_correspondences": total,
        "where_sender_solicited_protected_info": len(solicited),
        "where_recipient_disclosed_protected_info": len(disclosed),
        "disclosure_severity_counts": sev,
        "disclosed_categories": cats_disclosed,
        "personas_that_disengaged_or_flagged": len(flagged),
        "disclosure_by_persona_vulnerability": by_vuln,
        "earliest_disclosure_turn_min": min(
            [f["leakage"].get("earliest_disclosure_turn", -1) for f in disclosed if f["leakage"].get("earliest_disclosure_turn", -1) >= 0],
            default=-1,
        ),
    }
    user = (
        "The email being assessed (its opening message):\n"
        f"From-context: the recipient is {org_label}.\n"
        f"Subject: {initial_email.get('subject','')}\n"
        f"{initial_email.get('body','')}\n\n"
        "Aggregated leakage findings across all predicted correspondences:\n"
        f"{json.dumps(summary, indent=2)}\n\n"
        "Assess the original email and return the required JSON."
    )
    out = call_json(client, model, RISK_SYSTEM_PROMPT, user, RISK_SCHEMA)
    out["_summary"] = summary
    return out


# --------------------------------------------------------------------------
# Orchestration: classify one initial email
# --------------------------------------------------------------------------

def classify_email(client: Any, model: str, initial_email: dict[str, Any],
                   org_id: str | None, org_vaults: dict[str, dict[str, Any]],
                   personas: list[dict[str, Any]], branches: int, max_iterations: int,
                   concurrency: int, progress_prefix: str = "") -> dict[str, Any]:
    org_label, cats = abstracted_vault(org_id, org_vaults)

    units = [(p, b) for p in personas for b in range(branches)]
    findings: list[dict[str, Any]] = []
    correspondences: list[dict[str, Any]] = []

    def run_unit(unit: tuple[dict[str, Any], int]) -> dict[str, Any]:
        persona, branch = unit
        corr = predict_correspondence(client, model, persona, org_label, cats, initial_email, max_iterations)
        if corr["terminal_action"].startswith("error:"):
            leak = {"sender_requested_protected_info": False, "requested_categories": [],
                    "recipient_disclosed_protected_info": False, "disclosed_categories": [],
                    "severity": "none", "earliest_disclosure_turn": -1, "evidence": [],
                    "reasoning": "generation error", "_error": corr["terminal_action"]}
        else:
            try:
                leak = analyze_leakage(client, model, corr["thread"], cats)
            except Exception as exc:  # noqa: BLE001
                leak = {"sender_requested_protected_info": False, "requested_categories": [],
                        "recipient_disclosed_protected_info": False, "disclosed_categories": [],
                        "severity": "none", "earliest_disclosure_turn": -1, "evidence": [],
                        "reasoning": "leakage-analysis error", "_error": repr(exc)}
        return {
            "persona_id": persona["id"], "persona_label": persona.get("label", ""),
            "vulnerability_level": persona.get("vulnerability_level", "unknown"),
            "branch": branch, "terminal_action": corr["terminal_action"],
            "thread": corr["thread"], "leakage": leak,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for res in pool.map(run_unit, units):
            findings.append(res)
            correspondences.append({k: res[k] for k in ("persona_id", "branch", "terminal_action", "thread", "leakage")})

    risk = assess_risk(client, model, initial_email, org_label, findings)
    return {
        "org_id": org_id, "org_label": org_label,
        "risk": risk,
        "verdict": risk.get("verdict", "legitimate"),
        "risk_score": risk.get("risk_score"),
        "findings": findings,
        "correspondences": correspondences,
    }


# --------------------------------------------------------------------------
# Harness over scenarios.json
# --------------------------------------------------------------------------

def recipient_org_for_scenario(sc: dict[str, Any], persona_org: dict[str, str]) -> str | None:
    """Pick the defended org from the scenario's first target persona that has one.
    This is harness wiring to choose which vault applies; the classifier itself
    never sees the scenario."""
    for pid in sc.get("target_personas", []):
        org = persona_org.get(pid)
        if org:
            return org
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=Path, default=Path("scenarios.json"))
    ap.add_argument("--personas-file", type=Path, default=Path("generic_personas.json"))
    ap.add_argument("--named-personas", type=Path, default=Path("phishing_response_personas.json"),
                    help="used only to map target-persona ids to organizations for vault selection")
    ap.add_argument("--org-vaults", type=Path, default=Path("organization_information_vaults.json"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--personas", default=None, help="comma-separated generic persona ids (default: all 12)")
    ap.add_argument("--branches", type=int, default=2, help="predicted correspondences per persona (2-3 recommended)")
    ap.add_argument("--max-iterations", type=int, default=4, help="max back-and-forth rounds per correspondence")
    ap.add_argument("--limit", type=int, default=None, help="classify only the first N scenarios")
    ap.add_argument("--scenarios-filter", default=None, help="comma-separated scenario ids to classify")
    ap.add_argument("--recipient-org", default=None, help="override the defended org id for every email")
    ap.add_argument("--concurrency", type=int, default=6, help="parallel persona-branch units per email")
    ap.add_argument("--risk-threshold", type=float, default=None,
                    help="if set, derive the verdict from risk_score >= threshold instead of the model's verdict label")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--plot-out", type=Path, default=Path("predictive_confusion_matrix.png"))
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the work plan + cost estimate and exit")
    args = ap.parse_args()

    scenarios = json.loads(args.scenarios.read_text())["scenarios"]
    gp = json.loads(args.personas_file.read_text())["personas"]
    org_vaults = {o["id"]: o for o in json.loads(args.org_vaults.read_text())["organizations"]}
    named = json.loads(args.named_personas.read_text())["personas"]
    persona_org = {p["id"]: p.get("organization_id") for p in named}

    if args.personas:
        want = set(args.personas.split(","))
        gp = [p for p in gp if p["id"] in want]
    if args.scenarios_filter:
        want = set(args.scenarios_filter.split(","))
        scenarios = [s for s in scenarios if s["id"] in want]
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    units_per_email = len(gp) * args.branches
    gen_calls = units_per_email * (2 * args.max_iterations + 1)  # rough upper bound
    leak_calls = units_per_email
    est = len(scenarios) * (gen_calls + leak_calls + 1)
    print(f"Plan: {len(scenarios)} scenarios x {len(gp)} personas x {args.branches} branches")
    print(f"      max {args.max_iterations} rounds each; ~<= {est} Claude calls upper bound "
          f"(model {args.model}, concurrency {args.concurrency})")

    if args.dry_run:
        for s in scenarios:
            org = args.recipient_org or recipient_org_for_scenario(s, persona_org)
            print(f"  {s['id']:<11} type={s['scenario_type']:<24} defended-org={org}")
        return 0

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2
    try:
        import anthropic
    except ImportError:
        print("error: pip install anthropic", file=sys.stderr)
        return 2
    client = anthropic.Anthropic()

    verdicts: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for i, sc in enumerate(scenarios, 1):
        org = args.recipient_org or recipient_org_for_scenario(sc, persona_org)
        init = sc.get("initial_email", {})
        # fill the simplest placeholders so predicted text reads naturally
        init = {
            "subject": _fill(init.get("subject", "")),
            "body": _fill(init.get("body", "")),
        }
        print(f"\n[{i}/{len(scenarios)}] {sc['id']} ({sc['scenario_type']}) defended-org={org} ...", flush=True)
        try:
            result = classify_email(client, args.model, init, org, org_vaults, gp,
                                    args.branches, args.max_iterations, args.concurrency)
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR: {exc!r}", file=sys.stderr)
            verdicts.append({"id": sc["id"], "name": sc.get("name"), "scenario_type": sc["scenario_type"],
                             "truth_label": 0 if sc["scenario_type"] == "genuine" else 1,
                             "flagged": False, "error": repr(exc), "risk_score": None})
            continue

        verdict = result["verdict"]
        if args.risk_threshold is not None and result.get("risk_score") is not None:
            flagged = float(result["risk_score"]) >= args.risk_threshold
        else:
            flagged = verdict == "phishing"
        truth = 0 if sc["scenario_type"] == "genuine" else 1
        disclosed = result["risk"]["_summary"]["where_recipient_disclosed_protected_info"]
        total = result["risk"]["_summary"]["predicted_correspondences"]
        print(f"    verdict={verdict} risk={result.get('risk_score')} "
              f"(disclosed in {disclosed}/{total} predicted correspondences)")
        verdicts.append({
            "id": sc["id"], "name": sc.get("name"), "scenario_type": sc["scenario_type"],
            "truth_label": truth, "flagged": flagged, "verdict": verdict,
            "risk_score": result.get("risk_score"), "error": None,
        })
        artifacts.append({"id": sc["id"], "result": result})

    _report(verdicts, args.model)

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "predictive": {
                "model": args.model, "branches": args.branches, "max_iterations": args.max_iterations,
                "verdicts": verdicts, "artifacts": artifacts,
            }
        }, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json_out}")

    if not args.no_plot:
        _plot(verdicts, args.plot_out, args.model)

    return 0


# light placeholder fill so predicted correspondences read naturally
_FILL = {
    "recipient_first_name": "Alex", "recipient_name": "Alex Carter",
    "recipient_email": "alex.carter@example.com", "recipient_org_domain": "example.com",
    "recipient_org_name": "Example Corp", "colleague_name": "Jamie Lee",
    "colleague_handle": "jamie.lee", "colleague_email": "jamie.lee@partner-firm.com",
}

def _fill(text: str) -> str:
    import re
    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", lambda m: _FILL.get(m.group(1), m.group(0)), text or "")


def _report(verdicts: list[dict[str, Any]], model: str) -> None:
    try:
        import plot_confusion_matrices as pcm
        c = pcm.counts_from_verdicts(verdicts)
        m = pcm.metrics_from_counts(c)
    except Exception:
        c = m = None
    W = 78
    print("\n" + "=" * W)
    print("PREDICTIVE CLASSIFIER — verdict per scenario (initial email + predicted leakage)")
    print(f"(Claude {model}; classifier never sees scenario type, sender, or attacker goal)")
    print("=" * W)
    print(f"\n{'scenario':<12}{'type':<24}{'risk':>6}  {'verdict':<11} truth")
    print("-" * W)
    for v in verdicts:
        truth = "phishing/spam" if v["truth_label"] == 1 else "legitimate"
        mark = ""
        if v.get("error"):
            mark = "  <-- ERROR"
        elif v["truth_label"] == 1 and not v["flagged"]:
            mark = "  <-- MISSED"
        elif v["truth_label"] == 0 and v["flagged"]:
            mark = "  <-- FALSE ALARM"
        rs = v.get("risk_score")
        rs_s = f"{rs:>6.2f}" if isinstance(rs, (int, float)) else "   n/a"
        print(f"{v['id']:<12}{v['scenario_type']:<24}{rs_s}  {str(v.get('verdict','')):<11} {truth}{mark}")
    if c:
        print("\n" + "-" * W)
        print("CONFUSION MATRIX")
        print("-" * W)
        print(f"  true positives  (caught phishing/spam): {c['TP']:>3}")
        print(f"  false negatives (missed phishing/spam): {c['FN']:>3}")
        print(f"  true negatives  (genuine passed)      : {c['TN']:>3}")
        print(f"  false positives (genuine flagged)     : {c['FP']:>3}")
        if c.get("errored"):
            print(f"  errored (excluded from metrics)       : {c['errored']:>3}")
        print("\n" + "-" * W)
        print("RECALL (the metric that matters here)")
        print("-" * W)
        print(f"  recall / detection rate . {m['detection_rate']}  ({c['TP']}/{c['TP']+c['FN']} caught)")
        print(f"  miss rate ............... {m['miss_rate']}  ({c['FN']} let through)")
        print(f"  false-positive rate ..... {m['false_positive_rate']}  ({c['FP']} of {c['FP']+c['TN']} genuine)")


def _plot(verdicts: list[dict[str, Any]], output: Path, model: str) -> None:
    try:
        import plot_confusion_matrices as pcm
    except ImportError:
        return
    c = pcm.counts_from_verdicts(verdicts)
    m = pcm.metrics_from_counts(c)
    pcm.render([("predictive", f"Predictive classifier\n({model})", c, m)], output, False)


if __name__ == "__main__":
    sys.exit(main())
