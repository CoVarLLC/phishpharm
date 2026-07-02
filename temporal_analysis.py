"""
temporal_analysis.py -- track how risk score evolves turn-by-turn through a correspondence.

For each scenario, takes the predicted threads already in predictive_cache.json,
truncates them at depth 1, 2, 3, ... and re-runs leakage analysis + risk assessment
at each depth. Plots both trajectories on one chart so you can see how the classifier's
confidence on a legitimate vs. phishing thread diverges as the conversation unfolds.

Defaults: SC-G04 (genuine conference follow-up) and SC-SOPH01 (BEC vendor change).
Both have threads up to 9 turns, giving a meaningful evolution curve.

Usage:
    python temporal_analysis.py
    python temporal_analysis.py --genuine SC-G03 --phishing SC-SOPH09
    python temporal_analysis.py --max-turns 6 --output temporal.png
    python temporal_analysis.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Reuse analysis functions from predictive_classifier
sys.path.insert(0, str(Path(__file__).parent))
from predictive_classifier import analyze_leakage, assess_risk, abstracted_vault


DEFAULT_GENUINE  = "SC-G04"
DEFAULT_PHISHING = "SC-SOPH01"
DEFAULT_MODEL    = "claude-sonnet-4-6"

TYPE_COLOR = {
    "genuine":                "#2e7d32",
    "obvious_phishing":       "#e65100",
    "sophisticated_phishing": "#c62828",
}
TYPE_LABEL = {
    "genuine":                "Legitimate",
    "obvious_phishing":       "Obvious phishing",
    "sophisticated_phishing": "Sophisticated phishing",
}


# --------------------------------------------------------------------------
# Per-depth leakage analysis
# --------------------------------------------------------------------------

def leakage_at_depth(
    client: Any,
    model: str,
    depth: int,
    units: list[dict[str, Any]],
    cats: list[dict[str, str]],
    temporal_cache: dict[str, Any],
    cache_lock: threading.Lock,
    cache_save_fn: Any,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Run leakage analysis on all units truncated to `depth` turns. Returns findings list."""

    def run_unit(unit: dict[str, Any]) -> dict[str, Any] | None:
        thread_full = unit["thread"]
        thread = thread_full[:depth]
        if not thread:
            return None

        # terminal_action: real outcome if thread already ended at or before depth,
        # otherwise "ongoing" (we haven't seen the end yet)
        terminal = unit["terminal_action"] if len(thread_full) <= depth else "ongoing"

        cache_key = f"{unit['unit_key']}__depth{depth}"
        with cache_lock:
            cached_leak = temporal_cache.get("leakage", {}).get(cache_key)
        if cached_leak:
            return _finding(unit, thread, terminal, cached_leak)

        # If there are no recipient turns yet there can be no leakage
        has_recipient_body = any(
            t.get("role") == "recipient" and t.get("body", "").strip()
            for t in thread
        )
        if not has_recipient_body:
            leak: dict[str, Any] = {
                "sender_requested_protected_info": False, "requested_categories": [],
                "recipient_disclosed_protected_info": False, "disclosed_categories": [],
                "severity": "none", "earliest_disclosure_turn": -1, "evidence": [],
                "reasoning": "no recipient reply yet",
            }
        else:
            try:
                leak = analyze_leakage(client, model, thread, cats)
            except Exception as exc:  # noqa: BLE001
                leak = {
                    "sender_requested_protected_info": False, "requested_categories": [],
                    "recipient_disclosed_protected_info": False, "disclosed_categories": [],
                    "severity": "none", "earliest_disclosure_turn": -1, "evidence": [],
                    "reasoning": f"error: {exc!r}",
                }

        with cache_lock:
            temporal_cache.setdefault("leakage", {})[cache_key] = leak
        cache_save_fn()

        return _finding(unit, thread, terminal, leak)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(run_unit, units))
    return [r for r in results if r is not None]


def _finding(unit: dict[str, Any], thread: list, terminal: str, leak: dict) -> dict[str, Any]:
    return {
        "persona_id":        unit["persona_id"],
        "vulnerability_level": unit["vulnerability_level"],
        "terminal_action":   terminal,
        "thread":            thread,
        "leakage":           leak,
    }


# --------------------------------------------------------------------------
# Full trajectory for one scenario
# --------------------------------------------------------------------------

def scenario_trajectory(
    client: Any,
    model: str,
    scenario: dict[str, Any],
    org_id: str | None,
    org_vaults: dict[str, Any],
    units: list[dict[str, Any]],
    max_turns: int,
    temporal_cache: dict[str, Any],
    cache_lock: threading.Lock,
    cache_save_fn: Any,
    concurrency: int,
) -> list[tuple[int, float]]:
    """Return [(depth, risk_score), ...] for this scenario."""
    org_label, cats = abstracted_vault(org_id, org_vaults)
    actual_max = min(max_turns, max((len(u["thread"]) for u in units), default=1))
    initial_email = {
        "subject": units[0]["thread"][0].get("subject", ""),
        "body":    units[0]["thread"][0].get("body", ""),
    }

    trajectory: list[tuple[int, float]] = []
    for depth in range(1, actual_max + 1):
        print(f"    depth {depth}/{actual_max} ...", flush=True)
        findings = leakage_at_depth(
            client, model, depth, units, cats,
            temporal_cache, cache_lock, cache_save_fn, concurrency,
        )
        if not findings:
            continue
        try:
            risk = assess_risk(client, model, initial_email, org_label, findings,
                               intent_projections=None)
            score = float(risk.get("risk_score", 0.0))
        except Exception as exc:  # noqa: BLE001
            print(f"      risk assessment error at depth {depth}: {exc!r}", flush=True)
            score = trajectory[-1][1] if trajectory else 0.0
        trajectory.append((depth, score))
        print(f"      risk_score = {score:.3f}", flush=True)

    return trajectory


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def plot_trajectories(
    trajectories: list[dict[str, Any]],
    output: Path,
    threshold: float = 0.5,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed -- skipping plot (pip install matplotlib)")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for t in trajectories:
        depths = [d for d, _ in t["trajectory"]]
        scores = [s for _, s in t["trajectory"]]
        color  = TYPE_COLOR.get(t["scenario_type"], "#555")
        label  = f"{TYPE_LABEL.get(t['scenario_type'], t['scenario_type'])}: {t['scenario_name']}"

        ax.plot(depths, scores, color=color, linewidth=2.5, marker="o",
                markersize=7, label=label, zorder=3)

        # Shade under the line
        ax.fill_between(depths, scores, alpha=0.08, color=color)

    # Threshold line
    xlim = ax.get_xlim()
    ax.axhline(threshold, color="#888", linewidth=1.2, linestyle="--", zorder=2,
               label=f"Decision threshold ({threshold})")

    # Shade risk zones
    ax.axhspan(threshold, 1.0, alpha=0.04, color="#c62828")
    ax.axhspan(0.0, threshold, alpha=0.04, color="#2e7d32")

    ax.set_xlim(left=1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Messages in thread (turn depth)", fontsize=11)
    ax.set_ylabel("Risk score", fontsize=11)
    ax.set_title("Risk score evolution over email thread", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(True, alpha=0.3, zorder=1)
    ax.set_xticks(range(1, max(len(t["trajectory"]) for t in trajectories) + 2))

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genuine",        default=DEFAULT_GENUINE,
                    help=f"Genuine scenario ID (default: {DEFAULT_GENUINE})")
    ap.add_argument("--phishing",       default=DEFAULT_PHISHING,
                    help=f"Phishing scenario ID (default: {DEFAULT_PHISHING})")
    ap.add_argument("--cache",          type=Path, default=Path("predictive_cache.json"))
    ap.add_argument("--temporal-cache", type=Path, default=Path("temporal_cache.json"))
    ap.add_argument("--scenarios",      type=Path, default=Path("scenarios.json"))
    ap.add_argument("--named-personas", type=Path, default=Path("phishing_response_personas.json"))
    ap.add_argument("--org-vaults",     type=Path, default=Path("organization_information_vaults.json"))
    ap.add_argument("--generic-personas", type=Path, default=Path("generic_personas.json"))
    ap.add_argument("--model",          default=DEFAULT_MODEL)
    ap.add_argument("--max-turns",      type=int, default=9)
    ap.add_argument("--concurrency",    type=int, default=6)
    ap.add_argument("--output",         type=Path, default=Path("temporal_analysis.png"))
    ap.add_argument("--json-out",       type=Path, default=None)
    ap.add_argument("--dry-run",        action="store_true")
    args = ap.parse_args()

    scenario_ids = [args.genuine, args.phishing]

    # Load data
    def read_json(p: Path) -> Any:
        raw = p.read_bytes()
        for enc in ("utf-8-sig", "utf-8", "cp1252"):
            try:
                return json.loads(raw.decode(enc))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        raise ValueError(f"Could not decode {p}")

    cache           = read_json(args.cache)
    scenarios_by_id = {s["id"]: s for s in read_json(args.scenarios)["scenarios"]}
    org_vaults      = {o["id"]: o for o in read_json(args.org_vaults)["organizations"]}
    named_personas  = read_json(args.named_personas)["personas"]
    generic_personas = {p["id"]: p for p in read_json(args.generic_personas)["personas"]}
    persona_org     = {p["id"]: p.get("organization_id") for p in named_personas}

    # Validate scenario IDs
    for sc_id in scenario_ids:
        if sc_id not in scenarios_by_id:
            print(f"error: scenario {sc_id!r} not found in scenarios.json", file=sys.stderr)
            return 1
        sc_units = [k for k in cache.get("units", {}) if k.startswith(sc_id + "__")]
        if not sc_units:
            print(f"error: no cached units found for {sc_id}. Run predictive_classifier.py first.",
                  file=sys.stderr)
            return 1

    # Build units list per scenario
    def get_units(sc_id: str) -> list[dict[str, Any]]:
        result = []
        for key, unit in cache.get("units", {}).items():
            if not key.startswith(sc_id + "__"):
                continue
            parts = key.split("__")
            persona_id = parts[1] if len(parts) >= 2 else "unknown"
            gp = generic_personas.get(persona_id, {})
            result.append({
                "unit_key":          key,
                "thread":            unit.get("thread", []),
                "terminal_action":   unit.get("terminal_action", "unknown"),
                "persona_id":        persona_id,
                "vulnerability_level": gp.get("vulnerability_level", "unknown"),
            })
        return result

    # Dry run: show plan + cost estimate
    if args.dry_run:
        print("Temporal analysis plan:")
        for sc_id in scenario_ids:
            sc   = scenarios_by_id[sc_id]
            units = get_units(sc_id)
            actual_max = min(args.max_turns, max((len(u["thread"]) for u in units), default=1))
            leakage_calls = len(units) * actual_max
            risk_calls    = actual_max
            print(f"  {sc_id} ({sc['scenario_type']}): {sc.get('name','')}")
            print(f"    {len(units)} units x {actual_max} depths = "
                  f"{leakage_calls} leakage calls + {risk_calls} risk calls")
        total_units = sum(
            min(args.max_turns, max((len(u["thread"]) for u in get_units(sc_id)), default=1))
            * len(get_units(sc_id))
            for sc_id in scenario_ids
        )
        print(f"\n  Total API calls (upper bound): ~{total_units + sum(min(args.max_turns, max((len(u['thread']) for u in get_units(sc_id)), default=1)) for sc_id in scenario_ids)}")
        print("  (already-cached depths will be skipped)")
        return 0

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("error: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1
    try:
        import anthropic
    except ImportError:
        print("error: pip install anthropic", file=sys.stderr)
        return 1
    client = anthropic.Anthropic()

    # Load or create temporal cache
    temporal_cache: dict[str, Any] = {}
    if args.temporal_cache.exists():
        try:
            temporal_cache = read_json(args.temporal_cache)
            print(f"Loaded temporal cache: {len(temporal_cache.get('leakage', {}))} entries")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not load temporal cache: {exc}", file=sys.stderr)

    cache_lock = threading.Lock()

    def save_temporal_cache() -> None:
        args.temporal_cache.write_text(
            json.dumps(temporal_cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Run trajectory for each scenario
    all_trajectories = []
    for sc_id in scenario_ids:
        sc    = scenarios_by_id[sc_id]
        units = get_units(sc_id)
        org_id = persona_org.get(units[0]["persona_id"]) if units else None
        # Fall back to first target persona's org
        for pid in sc.get("target_personas", []):
            if persona_org.get(pid):
                org_id = persona_org[pid]
                break

        print(f"\n[{sc_id}] {sc.get('name','')} ({sc['scenario_type']}) org={org_id}")
        trajectory = scenario_trajectory(
            client, args.model, sc, org_id, org_vaults, units,
            args.max_turns, temporal_cache, cache_lock, save_temporal_cache,
            args.concurrency,
        )
        all_trajectories.append({
            "scenario_id":   sc_id,
            "scenario_name": sc.get("name", sc_id),
            "scenario_type": sc["scenario_type"],
            "trajectory":    trajectory,
        })

    # Print results as text
    print("\n" + "=" * 60)
    print("TEMPORAL RISK SCORE TRAJECTORIES")
    print("=" * 60)
    for t in all_trajectories:
        print(f"\n{t['scenario_id']} -- {t['scenario_name']}")
        print(f"  {'depth':<8} {'risk_score'}")
        for depth, score in t["trajectory"]:
            bar = "#" * int(score * 20)
            flag = " <-- FLAGGED" if score >= 0.5 else ""
            print(f"  {depth:<8} {score:.3f}  {bar}{flag}")

    # Save JSON
    if args.json_out:
        args.json_out.write_text(
            json.dumps({"trajectories": all_trajectories}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")

    # Plot
    plot_trajectories(all_trajectories, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
