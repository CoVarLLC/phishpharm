"""
plot_risk_breakdown.py -- generate 4-panel risk breakdown figures for each scenario.

Reads predictive_results_v2.json and produces one PNG per scenario in risk_breakdowns/.

Panels:
  1. Disclosure unit chart  (squares: disclosed vs clean)
  2. Terminal action distribution (horizontal bar chart)
  3. Intent projection status (triggered or not, with reason)
  4. Risk gauge + verdict badge

Usage:
    python plot_risk_breakdown.py
    python plot_risk_breakdown.py --results predictive_results_v2.json --out-dir risk_breakdowns
    python plot_risk_breakdown.py --scenario SC-SOPH09   # single scenario
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette (matches slide deck)
# ---------------------------------------------------------------------------
C = dict(
    bg        = "#F4F7FB",
    card      = "#FFFFFF",
    navy      = "#0F2B4C",
    teal      = "#0D7B8A",
    mid       = "#4A90C4",
    light     = "#A8C8E8",
    disclosed = "#C0392B",
    amber     = "#C97C0A",
    green     = "#1B6B3A",
    muted     = "#7F8C9A",
    text      = "#1A2535",
    divider   = "#D0D8E4",
)

TERMINAL_LABELS = {
    "reply":               "Replied",
    "no_reply":            "No reply",
    "report_as_phishing":  "Reported phishing",
    "verify_out_of_band":  "Verified OOB",
    "disengage":           "Disengaged",
    "end_success":         "Sender succeeded",
    "end_giveup":          "Sender gave up",
    "sender_ended":        "Sender ended",
    "max_iterations":      "Max iterations",
    "continue":            "Continued",
}


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _panel_unit_chart(ax, total: int, disclosed: int, scenario_id: str) -> None:
    """Panel 1: grid of squares, red = disclosed, light blue = clean."""
    ax.set_facecolor(C["card"])
    ax.set_aspect("equal")

    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)

    sq = 0.75
    gap = 0.18
    step = sq + gap

    for idx in range(total):
        r = idx // cols
        c = idx % cols
        x = c * step
        y = (rows - 1 - r) * step
        color = C["disclosed"] if idx < disclosed else C["light"]
        rect = mpatches.FancyBboxPatch(
            (x, y), sq, sq,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="white", linewidth=1.2,
        )
        ax.add_patch(rect)

    ax.set_xlim(-0.1, cols * step)
    ax.set_ylim(-0.1, rows * step)
    ax.axis("off")

    pct = disclosed / total * 100 if total else 0
    ax.set_title(
        f"{disclosed}/{total} correspondences disclosed\n{pct:.0f}% disclosure rate",
        fontsize=9, color=C["text"], pad=6, fontweight="bold",
    )


def _panel_terminal_actions(ax, dist: dict[str, int]) -> None:
    """Panel 2: horizontal bar chart of terminal action distribution."""
    ax.set_facecolor(C["card"])

    # Separate real actions from API error keys
    error_keys = {k: v for k, v in dist.items() if k.startswith("error:")}
    clean_dist  = {k: v for k, v in dist.items() if not k.startswith("error:")}
    n_errors    = sum(error_keys.values())

    if not clean_dist:
        ax.axis("off")
        ax.text(0.5, 0.58, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color=C["muted"])
        if n_errors:
            ax.text(0.5, 0.38,
                    f"{n_errors} simulation(s) errored\n(API overload during run)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color=C["disclosed"], linespacing=1.5)
        ax.set_title("Terminal actions", fontsize=9, color=C["text"], pad=6, fontweight="bold")
        return

    labels = [TERMINAL_LABELS.get(k, k) for k in clean_dist]
    values = list(clean_dist.values())
    total = sum(values)

    # Sort by count descending
    paired = sorted(zip(values, labels), reverse=True)
    values, labels = zip(*paired)

    y = np.arange(len(labels))
    # Light-to-dark blues: highest bar gets darkest shade
    n = len(values)
    blues = plt.cm.Blues(np.linspace(0.35, 0.85, n))
    bar_colors = blues[::-1]  # sorted descending, so first bar is darkest
    bars = ax.barh(y, values, color=bar_colors, height=0.55, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, max(values) * 1.35)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.set_facecolor(C["card"])
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=C["divider"], linewidth=0.5, zorder=0)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.03,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center", ha="left", fontsize=8, color=C["text"], fontweight="bold",
        )

    if n_errors:
        ax.set_xlabel(f"count  ({n_errors} errored, excluded)", fontsize=7, color=C["disclosed"])
    else:
        ax.set_xlabel("count", fontsize=7, color=C["muted"])

    ax.set_title("Terminal actions", fontsize=9, color=C["text"], pad=6, fontweight="bold")
    ax.invert_yaxis()


def _panel_intent(ax, intent_projections, disclosure_pct: float, threshold_pct: float = 10.0) -> None:
    """Panel 3: intent projection status."""
    ax.set_facecolor(C["card"])
    ax.axis("off")

    triggered = bool(intent_projections)

    if triggered:
        ip = intent_projections[0]
        goal_text = ip.get("apparent_sender_goal", "")
        # Truncate to ~180 chars
        if len(goal_text) > 180:
            goal_text = goal_text[:177] + "…"

        ax.text(0.5, 0.92, "INTENT PROJECTION", ha="center", va="top",
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                color=C["amber"])
        ax.text(0.5, 0.80, "TRIGGERED", ha="center", va="top",
                transform=ax.transAxes, fontsize=13, fontweight="bold",
                color=C["amber"])

        # Wrap goal text
        wrapped = textwrap.fill(goal_text, width=42)
        ax.text(0.5, 0.66, wrapped, ha="center", va="top",
                transform=ax.transAxes, fontsize=7.5, color=C["text"],
                linespacing=1.4)

        ax.text(0.5, 0.10,
                f"{disclosure_pct:.0f}% disclosure < {threshold_pct:.0f}% threshold → IP triggered",
                ha="center", va="bottom", transform=ax.transAxes,
                fontsize=7, color=C["muted"], style="italic")
    else:
        ax.text(0.5, 0.92, "INTENT PROJECTION", ha="center", va="top",
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                color=C["muted"])
        ax.text(0.5, 0.72, "NOT TRIGGERED", ha="center", va="top",
                transform=ax.transAxes, fontsize=13, fontweight="bold",
                color=C["muted"])
        ax.text(0.5, 0.56,
                f"{disclosure_pct:.0f}% > {threshold_pct:.0f}% threshold",
                ha="center", va="top", transform=ax.transAxes,
                fontsize=9, color=C["muted"])
        ax.text(0.5, 0.42,
                "Leakage signal was sufficient\nIP step skipped",
                ha="center", va="top", transform=ax.transAxes,
                fontsize=8, color=C["muted"], linespacing=1.5)

    ax.set_title("Intent projection", fontsize=9, color=C["text"], pad=6, fontweight="bold")


def _panel_risk_gauge(ax, risk_score: float, verdict: str, reasoning: str) -> None:
    """Panel 4: semicircular gauge + verdict badge + reasoning snippet."""
    ax.set_facecolor(C["card"])
    ax.set_aspect("equal")
    ax.axis("off")

    # Gauge arc background (grey)
    theta = np.linspace(np.pi, 0, 200)
    r_out, r_in = 1.0, 0.6
    ax.fill_between(
        np.concatenate([r_out * np.cos(theta), r_in * np.cos(theta[::-1])]),
        np.concatenate([r_out * np.sin(theta), r_in * np.sin(theta[::-1])]),
        color=C["divider"], zorder=1,
    )

    # Filled portion (teal → amber → red based on score)
    if risk_score < 0.4:
        fill_color = C["teal"]
    elif risk_score < 0.65:
        fill_color = C["amber"]
    else:
        fill_color = C["disclosed"]

    fill_end = np.pi - risk_score * np.pi
    theta_fill = np.linspace(np.pi, fill_end, 200)
    ax.fill_between(
        np.concatenate([r_out * np.cos(theta_fill), r_in * np.cos(theta_fill[::-1])]),
        np.concatenate([r_out * np.sin(theta_fill), r_in * np.sin(theta_fill[::-1])]),
        color=fill_color, zorder=2,
    )

    # Needle
    needle_angle = np.pi - risk_score * np.pi
    ax.annotate("",
        xy=(0.75 * np.cos(needle_angle), 0.75 * np.sin(needle_angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=C["navy"], lw=2.0),
        zorder=3,
    )
    ax.add_patch(mpatches.Circle((0, 0), 0.08, color=C["navy"], zorder=4))

    # Score label
    ax.text(0, -0.12, f"{risk_score:.2f}", ha="center", va="top",
            fontsize=18, fontweight="bold", color=C["navy"], zorder=5)

    # Verdict badge
    v_color = C["disclosed"] if verdict.lower() == "phishing" else C["green"]
    badge = mpatches.FancyBboxPatch(
        (-0.45, -0.42), 0.90, 0.22,
        boxstyle="round,pad=0.04",
        facecolor=v_color, edgecolor="none", zorder=5,
    )
    ax.add_patch(badge)
    ax.text(0, -0.31, verdict.upper(), ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=6)

    # Reasoning snippet (truncated)
    snippet = reasoning[:160].replace("\n", " ")
    if len(reasoning) > 160:
        snippet += "…"
    wrapped = textwrap.fill(snippet, width=38)
    ax.text(0, -0.56, wrapped, ha="center", va="top",
            fontsize=6.5, color=C["muted"], linespacing=1.35, zorder=5)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.85, 1.15)
    ax.set_title("Risk score", fontsize=9, color=C["text"], pad=6, fontweight="bold")


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_scenario(scenario_id: str, artifact: dict, out_dir: Path) -> Path:
    result   = artifact["result"]
    risk     = result["risk"]
    summary  = risk["_summary"]

    total      = summary.get("predicted_correspondences", 0)
    disclosed  = summary.get("where_recipient_disclosed_protected_info", 0)
    disc_pct   = disclosed / total * 100 if total else 0
    term_dist  = summary.get("terminal_action_distribution", {})
    intent_proj = summary.get("intent_projections") or []
    risk_score = risk.get("risk_score", 0.0)
    verdict    = risk.get("verdict", "unknown")
    reasoning  = risk.get("reasoning", "")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    fig.patch.set_facecolor(C["bg"])
    for ax in axes:
        ax.set_facecolor(C["card"])

    _panel_unit_chart(axes[0], total, disclosed, scenario_id)
    _panel_terminal_actions(axes[1], term_dist)
    _panel_intent(axes[2], intent_proj, disc_pct)
    _panel_risk_gauge(axes[3], risk_score, verdict, reasoning)

    # Card backgrounds
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"{scenario_id}  |  Risk breakdown",
        fontsize=12, fontweight="bold", color=C["navy"], y=1.01,
    )
    fig.tight_layout(pad=1.2)

    out_path = out_dir / f"{scenario_id}_risk_breakdown.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not decode {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 4-panel risk breakdown PNGs for each scenario.")
    ap.add_argument("--results",  "-r", type=Path, default=Path("predictive_results_v2.json"))
    ap.add_argument("--out-dir",  "-o", type=Path, default=Path("risk_breakdowns"))
    ap.add_argument("--scenario", "-s", type=str,  default=None,
                    help="Render a single scenario ID only")
    args = ap.parse_args()

    data      = _read_json(args.results)
    artifacts = data.get("predictive", {}).get("artifacts", [])

    if not artifacts:
        print("No artifacts found in", args.results)
        return

    args.out_dir.mkdir(exist_ok=True)

    targets = artifacts
    if args.scenario:
        targets = [a for a in artifacts if a["id"] == args.scenario]
        if not targets:
            print(f"Scenario {args.scenario!r} not found")
            return

    for artifact in targets:
        sc_id = artifact["id"]
        out   = render_scenario(sc_id, artifact, args.out_dir)
        print(f"  {sc_id:15s} -> {out}")

    print(f"\nDone. {len(targets)} figure(s) written to {args.out_dir}/")


if __name__ == "__main__":
    main()
