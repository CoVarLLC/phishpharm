"""
export_to_viewer.py -- convert predictive_cache.json into files the viewer can load.

Reads:
  predictive_cache.json    (written by predictive_classifier.py)
  scenarios.json           (for scenario metadata)
  generic_personas.json    (for persona labels / vulnerability levels)

Writes:
  correspondences_predictive.jsonl  -- drop-in for correspondences.jsonl in the viewer
  annotations_predictive.json       -- leakage highlights, drop-in for annotations.json

Usage:
  python export_to_viewer.py

Then open viewer.html, click Load JSONL -> correspondences_predictive.jsonl,
and Load annotations -> annotations_predictive.json.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache",           type=Path, default=Path("predictive_cache.json"))
    ap.add_argument("--results",         type=Path, default=None,
                    help="predictive_results*.json written by --json-out; used to pull intent projections")
    ap.add_argument("--scenarios",       type=Path, default=Path("scenarios.json"))
    ap.add_argument("--personas-file",   type=Path, default=Path("generic_personas.json"))
    ap.add_argument("--out-jsonl",       type=Path, default=Path("correspondences_predictive.jsonl"))
    ap.add_argument("--out-annotations", type=Path, default=Path("annotations_predictive.json"))
    args = ap.parse_args()

    # load inputs
    try:
        import ftfy
        _fix = ftfy.fix_text
    except ImportError:
        _fix = None

    def read_json(path: Path) -> str:
        """Read a JSON file with smart mixed-encoding handling.

        Two failure modes handled:
        1. File has raw cp1252 bytes (e.g. 0xE1 = á, 0x97 = em dash) mixed with
           valid UTF-8 sequences -- UTF-8 decode fails.  Decoded as cp1252 then
           ftfy repairs any resulting mojibake from the UTF-8 sequences.
        2. File decodes as UTF-8 but already contains pre-stored mojibake strings
           (e.g. â€" stored as literal UTF-8 characters) -- ftfy fixes these too.

        Requires: pip install ftfy
        """
        raw = path.read_bytes()
        for enc in ("utf-8-sig", "utf-8"):
            try:
                text = raw.decode(enc)
                return _fix(text) if _fix else text
            except UnicodeDecodeError:
                pass
        # Bytes aren't valid UTF-8: decode as cp1252 (never fails for single bytes),
        # then let ftfy reverse the mojibake that results from UTF-8 sequences being
        # read as cp1252 (e.g. E2 80 94 -> â€" -> --).
        text = raw.decode("cp1252", errors="replace")
        return _fix(text) if _fix else text

    cache           = json.loads(read_json(args.cache))
    scenarios_by_id = {s["id"]: s for s in json.loads(read_json(args.scenarios))["scenarios"]}
    personas_by_id  = {p["id"]: p for p in json.loads(read_json(args.personas_file))["personas"]}

    # build scenario -> intent_projections index from results file if provided,
    # falling back to the cache's scenarios key for older runs
    intent_by_scenario: dict[str, list] = {}
    if args.results and args.results.exists():
        results_data = json.loads(read_json(args.results))
        for artifact in results_data.get("predictive", {}).get("artifacts", []):
            sc_id = artifact.get("id")
            ip = artifact.get("result", {}).get("risk", {}).get("_summary", {}).get("intent_projections")
            if sc_id and ip:
                intent_by_scenario[sc_id] = ip
        print(f"Loaded intent projections for {len(intent_by_scenario)} scenario(s) from {args.results}")
    else:
        for sc_id, sc_data in cache.get("scenarios", {}).items():
            ip = sc_data.get("intent_projections")
            if ip:
                intent_by_scenario[sc_id] = ip

    correspondences = []
    annotations = []

    for key, unit in cache.get("units", {}).items():
        # key format: SC-SOPH01__GP-01__0
        parts = key.split("__")
        if len(parts) != 3:
            print(f"Skipping unexpected key format: {key}")
            continue
        scenario_id, persona_id, branch_str = parts
        branch = int(branch_str)

        thread   = unit.get("thread", [])
        leakage  = unit.get("leakage", {})
        terminal = unit.get("terminal_action", "")

        sc      = scenarios_by_id.get(scenario_id, {})
        persona = personas_by_id.get(persona_id, {})

        corr_id = f"{scenario_id}__{persona_id}__branch{branch}"

        # build turns list -- cache format is already compatible with viewer
        turns = []
        last_recipient_action = None
        for t in thread:
            turns.append({
                "role":      t.get("role", ""),
                "subject":   t.get("subject", ""),
                "body":      t.get("body", ""),
                "action":    t.get("action", ""),
                "reasoning": t.get("reasoning", ""),
            })
            if t.get("role") == "recipient" and t.get("action"):
                last_recipient_action = t["action"]

        scenario_intent = intent_by_scenario.get(scenario_id, [])[:1]

        corr = {
            "correspondence_id":            corr_id,
            "scenario_id":                  scenario_id,
            "scenario_name":                sc.get("name", scenario_id),
            "scenario_type":                sc.get("scenario_type", ""),
            "category":                     sc.get("category", ""),
            "is_spear_phishing":            sc.get("is_spear_phishing", False),
            "attacker_goal":                sc.get("attacker_goal"),
            "sensitive_information_sought": sc.get("sensitive_information_sought", []),
            "sender_persona":               sc.get("sender_persona", {}),
            "recipient_persona_id":         persona_id,
            "recipient_name":               persona.get("label", persona_id),
            "recipient_email":              f"{persona_id.lower()}@example.com",
            "recipient_organization_id":    sc.get("target_org_id", ""),
            "recipient_vulnerability_level": persona.get("vulnerability_level", ""),
            "recipient_archetype":          persona.get("archetype", ""),
            "run_index":                    branch,
            "model":                        "claude-sonnet-4-6",
            "termination_reason":           terminal,
            "last_recipient_action":        last_recipient_action,
            "turn_count":                   len(turns),
            "turns":                        turns,
            "label":                        sc.get("scenario_type", ""),
            "intent_projections":           scenario_intent,
            # leakage summary stored for reference
            "_leakage_summary": {
                "sender_solicited":    leakage.get("sender_requested_protected_info", False),
                "recipient_disclosed": leakage.get("recipient_disclosed_protected_info", False),
                "severity":            leakage.get("severity", "none"),
                "categories":          leakage.get("disclosed_categories", []),
            },
        }
        correspondences.append(corr)

        # build annotation records from leakage evidence spans
        evidence_texts  = leakage.get("evidence", [])
        disclosed_cats  = leakage.get("disclosed_categories", [])
        leakage_note    = leakage.get("reasoning", "")

        for ev_idx, evidence in enumerate(evidence_texts):
            if not evidence.strip():
                continue
            # find the recipient turn that contains this exact text
            turn_index = -1
            for ti, t in enumerate(turns):
                if t.get("role") == "recipient" and evidence in t.get("body", ""):
                    turn_index = ti
                    break
            cat = (disclosed_cats[ev_idx] if ev_idx < len(disclosed_cats)
                   else (disclosed_cats[0] if disclosed_cats else "information_leakage"))
            annotations.append({
                "correspondence_id": corr_id,
                "turn_index":        turn_index,
                "exact_text":        evidence,
                "category":          cat,
                "note":              leakage_note,
            })

    # write outputs
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for c in correspondences:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    args.out_annotations.write_text(
        json.dumps(annotations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # summary
    by_type: dict[str, int] = {}
    for c in correspondences:
        t = c["scenario_type"]
        by_type[t] = by_type.get(t, 0) + 1

    disclosed_count = sum(
        1 for c in correspondences if c["_leakage_summary"]["recipient_disclosed"]
    )
    with_intent = sum(1 for c in correspondences if c.get("intent_projections"))

    print(f"Wrote {len(correspondences)} correspondence(s) to {args.out_jsonl}")
    print(f"  by type: { {k: v for k, v in sorted(by_type.items())} }")
    print(f"  with leakage annotations: {disclosed_count} / {len(correspondences)}")
    print(f"  with intent projections:  {with_intent} / {len(correspondences)}")
    print(f"Wrote {len(annotations)} annotation(s) to {args.out_annotations}")
    print()
    print("To view: open viewer.html")
    print("  Load JSONL        -> " + str(args.out_jsonl))
    print("  Load annotations  -> " + str(args.out_annotations))


if __name__ == "__main__":
    main()
