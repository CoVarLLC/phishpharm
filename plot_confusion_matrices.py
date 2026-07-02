"""
plot_confusion_matrices.py -- render confusion matrices from classifier result files.

Can be used standalone (CLI) or imported as a module by traditional_filter.py and
predictive_classifier.py.

Standalone usage:
    # From a traditional_filter.py --json-out file:
    python plot_confusion_matrices.py --input filter_results.json

    # From a predictive_classifier.py --json-out file:
    python plot_confusion_matrices.py --input predictive_results.json

    # Both side by side:
    python plot_confusion_matrices.py --input filter_results.json predictive_results.json

    # Row-normalized rates:
    python plot_confusion_matrices.py --input filter_results.json --normalize

    # Custom output path:
    python plot_confusion_matrices.py --input predictive_results.json --output cm.png

Module interface (used by traditional_filter.py and predictive_classifier.py):
    import plot_confusion_matrices as pcm

    counts  = pcm.counts_from_verdicts(list_of_verdict_dicts)
    metrics = pcm.metrics_from_counts(counts)
    pcm.render([("rule_based", "Rule-based filter", counts, metrics)], Path("out.png"), normalize=False)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Human-readable labels for each classifier key found in result files.
CLASSIFIER_KEYS: list[tuple[str, str]] = [
    ("rule_based",  "Rule-based filter\n(PILFER + SpamAssassin-style)"),
    ("llm",         "Single-message LLM\n(initial email only)"),
    ("predictive",  "Predictive classifier\n(information leakage)"),
]


# --------------------------------------------------------------------------
# Core functions (importable)
# --------------------------------------------------------------------------

def counts_from_verdicts(verdicts: list[dict[str, Any]]) -> dict[str, int]:
    """Count TP / FN / TN / FP / errored from a list of verdict dicts.

    Each dict must have:
        truth_label : int   -- 1 = phishing/spam, 0 = legitimate
        flagged     : bool  -- True = classifier said phishing
        error       : any   -- non-None/non-False means the email errored out
    """
    tp = fn = tn = fp = errored = 0
    for v in verdicts:
        if v.get("error"):
            errored += 1
            continue
        truth   = int(v.get("truth_label", 0))
        flagged = bool(v.get("flagged", False))
        if truth == 1 and flagged:
            tp += 1
        elif truth == 1 and not flagged:
            fn += 1
        elif truth == 0 and not flagged:
            tn += 1
        else:
            fp += 1
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "errored": errored}


def metrics_from_counts(c: dict[str, int]) -> dict[str, str]:
    """Derive recall, miss rate, and false-positive rate as formatted strings."""
    tp, fn, tn, fp = c["TP"], c["FN"], c["TN"], c["FP"]
    pos = tp + fn
    neg = tn + fp

    def pct(num: int, den: int) -> str:
        if den == 0:
            return "n/a  "
        return f"{num / den * 100:5.1f}%"

    return {
        "detection_rate":     pct(tp, pos),
        "miss_rate":          pct(fn, pos),
        "false_positive_rate": pct(fp, neg),
    }


def render(
    classifiers: list[tuple[str, str, dict[str, int], dict[str, str]]],
    output: Path,
    normalize: bool,
) -> None:
    """Render one confusion-matrix panel per classifier and save as a PNG.

    Args:
        classifiers : list of (key, label, counts_dict, metrics_dict)
        output      : PNG file path
        normalize   : if True, shade cells by row-normalized rate instead of raw count
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("matplotlib is not installed -- skipping plot (pip install matplotlib)")
        return

    n = len(classifiers)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8))
    if n == 1:
        axes = [axes]

    # Colour scheme: light-to-dark blues throughout.
    cmap_good = plt.cm.Blues   # type: ignore[attr-defined]
    cmap_bad  = plt.cm.Blues   # type: ignore[attr-defined]

    for ax, (key, label, c, m) in zip(axes, classifiers):
        tp, fn, tn, fp = c["TP"], c["FN"], c["TN"], c["FP"]

        # 2x2 matrix: rows = actual, cols = predicted
        # [[TN, FP], [FN, TP]]
        matrix = np.array([[tn, fp], [fn, tp]], dtype=float)
        row_totals = matrix.sum(axis=1, keepdims=True)

        if normalize:
            display = np.where(row_totals > 0, matrix / row_totals, 0.0)
            fmt = lambda v: f"{v:.1%}"
        else:
            display = matrix
            fmt = lambda v: str(int(v))

        # Draw cells manually so we can colour diagonal vs off-diagonal differently.
        for row in range(2):
            for col in range(2):
                val    = display[row, col]
                raw    = matrix[row, col]
                on_diag = (row == col)
                norm_val = float(val / display.max()) if display.max() > 0 else 0.0
                colour = cmap_good(0.25 + norm_val * 0.6) if on_diag else cmap_bad(0.15 + norm_val * 0.65)
                ax.add_patch(mpatches.Rectangle((col, 1 - row), 1, 1, color=colour))
                text = fmt(val)
                if normalize and raw != val:
                    text += f"\n({int(raw)})"
                ax.text(col + 0.5, 1 - row + 0.5, text,
                        ha="center", va="center", fontsize=13, fontweight="bold",
                        color="white" if norm_val > 0.55 else "#222")

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(["predicted\nlegitimate", "predicted\nphishing"], fontsize=9)
        ax.set_yticklabels(["actual\nphishing", "actual\nlegitimate"], fontsize=9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Grid lines between cells
        ax.axhline(1, color="white", linewidth=2)
        ax.axvline(1, color="white", linewidth=2)

        # Title and metrics subtitle
        recall = m["detection_rate"].strip()
        fpr    = m["false_positive_rate"].strip()
        ax.set_title(label, fontsize=10, fontweight="bold", pad=10)
        ax.set_xlabel(
            f"recall {recall}  |  false-positive rate {fpr}",
            fontsize=8, labelpad=8, color="#444",
        )
        if c.get("errored"):
            ax.annotate(
                f"{c['errored']} errored (excluded)",
                xy=(0.5, -0.12), xycoords="axes fraction",
                ha="center", fontsize=7, color="#888",
            )

    fig.suptitle("Phishing classifier confusion matrices", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


# --------------------------------------------------------------------------
# Text printing (standalone)
# --------------------------------------------------------------------------

def print_matrix(label: str, c: dict[str, int], m: dict[str, str]) -> None:
    W = 60
    print(f"\n{label}")
    print("-" * W)
    print(f"  true positives  (caught phishing/spam): {c['TP']:>3}")
    print(f"  false negatives (missed phishing/spam): {c['FN']:>3}")
    print(f"  true negatives  (genuine passed)      : {c['TN']:>3}")
    print(f"  false positives (genuine flagged)     : {c['FP']:>3}")
    if c.get("errored"):
        print(f"  errored (excluded from metrics)       : {c['errored']:>3}")
    print()
    print(f"  recall / detection rate . {m['detection_rate']}  ({c['TP']}/{c['TP']+c['FN']} caught)")
    print(f"  miss rate ............... {m['miss_rate']}  ({c['FN']} missed)")
    print(f"  false-positive rate ..... {m['false_positive_rate']}  ({c['FP']} of {c['FP']+c['TN']} genuine flagged)")


# --------------------------------------------------------------------------
# JSON loading helpers
# --------------------------------------------------------------------------

def _read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not decode {path}")


def _load_classifiers_from_file(path: Path, normalize: bool) -> list[tuple[str, str, dict, dict]]:
    """Extract (key, label, counts, metrics) entries from a result file."""
    data   = _read_json(path)
    labels = dict(CLASSIFIER_KEYS)
    result = []

    # traditional_filter.py format: top-level keys are "rule_based" and/or "llm"
    for key in ("rule_based", "llm"):
        if key in data and "verdicts" in data[key]:
            c = counts_from_verdicts(data[key]["verdicts"])
            m = metrics_from_counts(c)
            model = data[key].get("model", "")
            lbl   = labels.get(key, key)
            if model:
                lbl += f"\n({model})"
            result.append((key, lbl, c, m))

    # predictive_classifier.py format: top-level key is "predictive"
    if "predictive" in data and "verdicts" in data["predictive"]:
        c   = counts_from_verdicts(data["predictive"]["verdicts"])
        m   = metrics_from_counts(c)
        model = data["predictive"].get("model", "")
        lbl   = labels.get("predictive", "Predictive classifier")
        if model:
            lbl += f"\n({model})"
        result.append(("predictive", lbl, c, m))

    return result


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  "-i", type=Path, nargs="+", required=True,
                    help="one or more result JSON files (filter_results.json, predictive_results.json)")
    ap.add_argument("--output", "-o", type=Path, default=Path("confusion_matrices.png"),
                    help="output PNG path (default: confusion_matrices.png)")
    ap.add_argument("--normalize", action="store_true",
                    help="shade cells by row-normalized rate rather than raw count")
    ap.add_argument("--no-plot", action="store_true",
                    help="print matrices as text only, skip PNG")
    args = ap.parse_args()

    classifiers: list[tuple] = []
    for path in args.input:
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 1
        loaded = _load_classifiers_from_file(path, args.normalize)
        if not loaded:
            print(f"warning: no recognisable classifier results found in {path}", file=sys.stderr)
        classifiers.extend(loaded)

    if not classifiers:
        print("error: nothing to plot", file=sys.stderr)
        return 1

    # Print text matrices
    for key, label, c, m in classifiers:
        print_matrix(label.replace("\n", " -- "), c, m)

    # Render PNG
    if not args.no_plot:
        render(classifiers, args.output, args.normalize)

    return 0


if __name__ == "__main__":
    sys.exit(main())
