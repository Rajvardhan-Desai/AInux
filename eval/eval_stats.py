"""
AInux Statistical Analysis
Consumes results.json from eval_harness.py and produces:

  1. Per-system accuracy by category (Table I in paper)
  2. Paired Wilcoxon signed-rank tests: AInux vs CLI, AInux vs NaSh
  3. Cohen's d effect sizes
  4. 95% confidence intervals
  5. Four publication-quality figures (saved as PNG)
  6. LaTeX table snippets for direct paste into paper

Usage:
    python eval_stats.py --results results.json --outdir figures/
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

matplotlib = importlib.import_module("matplotlib")
matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")
import numpy as np
from scipy import stats


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Load and structure results
# ---------------------------------------------------------------------------

def load_results(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def by_system(results: List[Dict]) -> Dict[str, List[Dict]]:
    out = defaultdict(list)
    for r in results:
        out[r["system"]].append(r)
    return dict(out)


def by_category(results: List[Dict]) -> Dict[str, List[Dict]]:
    out = defaultdict(list)
    for r in results:
        out[r["category"]].append(r)
    return dict(out)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def accuracy(records: List[Dict]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["correct"]) / len(records)


def false_positive_rate(records: List[Dict]) -> float:
    """Rate at which safe commands are incorrectly blocked."""
    total = sum(1 for r in records if not r.get("blocked", False) or r.get("false_positive"))
    fp = sum(1 for r in records if r.get("false_positive", False))
    return fp / len(records) if records else 0.0


def mean_latency(records: List[Dict]) -> float:
    lats = [r["latency"] for r in records if r["latency"] is not None]
    return float(np.mean(lats)) if lats else 0.0


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_test(
    a_correct: List[int],   # 1=correct, 0=incorrect — same task order
    b_correct: List[int],
    label_a: str,
    label_b: str,
) -> Dict:
    """
    Wilcoxon signed-rank test (non-parametric, appropriate for ordinal data).
    Falls back to paired t-test for symmetric reporting.
    """
    a = np.array(a_correct, dtype=float)
    b = np.array(b_correct, dtype=float)
    diff = a - b

    if np.all(diff == 0):
        return {"stat": 0.0, "p": 1.0, "significant": False,
                "test": "wilcoxon", "label_a": label_a, "label_b": label_b,
                "cohen_d": 0.0, "ci_95": (0.0, 0.0)}

    try:
        stat, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        test_name = "Wilcoxon signed-rank"
    except Exception:
        stat, p = stats.ttest_rel(a, b)
        test_name = "paired t-test"

    # Cohen's d
    d = float(np.mean(diff) / (np.std(diff) + 1e-9))

    # 95% CI on the mean difference via bootstrap
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(5000):
        sample = rng.choice(diff, size=len(diff), replace=True)
        boot_means.append(np.mean(sample))
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    return {
        "test": test_name,
        "stat": float(stat),
        "p": float(p),
        "significant": bool(p < 0.05),
        "cohen_d": d,
        "ci_95": (ci_lo, ci_hi),
        "label_a": label_a,
        "label_b": label_b,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
    }


def align_by_task(
    results_a: List[Dict],
    results_b: List[Dict],
) -> Tuple[List[int], List[int]]:
    """Return parallel correct arrays ordered by task_id."""
    a_map = {r["task_id"]: int(r["correct"]) for r in results_a}
    b_map = {r["task_id"]: int(r["correct"]) for r in results_b}
    common = sorted(set(a_map) & set(b_map))
    return [a_map[t] for t in common], [b_map[t] for t in common]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COLORS = {
    "TraditionalCLI": "#4C72B0",
    "NaSh":           "#DD8452",
    "AInux":          "#55A868",
}
SYSTEMS = ["TraditionalCLI", "NaSh", "AInux"]
LABELS  = {"TraditionalCLI": "Traditional CLI", "NaSh": "NaSh (baseline)", "AInux": "AInux (ours)"}


def available_systems(sys_results: Dict[str, List[Dict]]) -> List[str]:
    return [s for s in SYSTEMS if sys_results.get(s)]


def fig_accuracy_by_category(sys_results: Dict[str, List[Dict]], outdir: str, systems_to_plot: List[str]) -> None:
    categories = ["T1", "T2", "T3", "T4", "T5"]
    cat_labels  = ["File Ops", "Packages", "Diagnostics", "Web/SSL", "Dev Setup"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(categories))
    width = 0.25

    if not systems_to_plot:
        return

    width = 0.8 / len(systems_to_plot)
    for i, sys in enumerate(systems_to_plot):
        records = sys_results.get(sys, [])
        cat_acc = []
        for cat in categories:
            cat_recs = [r for r in records if r["category"] == cat]
            cat_acc.append(accuracy(cat_recs) * 100)
        offset = (i - (len(systems_to_plot) - 1) / 2) * width
        bars = ax.bar(x + offset, cat_acc, width, label=LABELS[sys],
                      color=COLORS[sys], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, cat_acc):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xlabel("Task Category")
    ax.set_ylabel("Command Accuracy (%)")
    ax.set_title("Command Generation Accuracy by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_accuracy_by_category.png"), dpi=150)
    plt.close()
    print("[Stats] Saved fig_accuracy_by_category.png")


def fig_latency_comparison(sys_results: Dict[str, List[Dict]], outdir: str, systems_to_plot: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    data = []
    labels = []
    plotted_systems = []
    for sys in systems_to_plot:
        lats = [r["latency"] for r in sys_results.get(sys, [])
                if r["latency"] is not None and r["latency"] > 0]
        if lats:
            data.append(lats)
            labels.append(LABELS[sys])
            plotted_systems.append(sys)

    if not data:
        plt.close()
        return

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, sys in zip(bp["boxes"], plotted_systems):
        patch.set_facecolor(COLORS[sys])
        patch.set_alpha(0.8)

    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Command Generation Latency")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_latency.png"), dpi=150)
    plt.close()
    print("[Stats] Saved fig_latency.png")


def fig_safety_classification(sys_results: Dict[str, List[Dict]], outdir: str, systems_to_plot: List[str]) -> None:
    """
    For AInux and NaSh: show true-positive block rate vs false-positive block rate.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    categories_safety = ["True Positive\n(dangerous blocked)",
                         "False Positive\n(safe blocked)",
                         "Miss\n(dangerous passed)"]

    # We count blocked records
    safety_systems = [s for s in ["NaSh", "AInux"] if s in systems_to_plot]
    if not safety_systems:
        plt.close()
        return

    for idx, sys in enumerate(safety_systems):
        records = sys_results.get(sys, [])
        tp = sum(1 for r in records if r.get("blocked") and not r.get("false_positive"))
        fp = sum(1 for r in records if r.get("false_positive"))
        total = len(records) or 1
        vals = [tp / total * 100, fp / total * 100,
                (total - tp - fp) / total * 100]
        x = np.arange(len(categories_safety))
        offset = (idx - (len(safety_systems) - 1) / 2) * 0.35
        ax.bar(x + offset, vals, 0.35,
               label=LABELS[sys], color=COLORS[sys], alpha=0.85, edgecolor="white")

    ax.set_xticks(np.arange(len(categories_safety)))
    ax.set_xticklabels(categories_safety, fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Safety Classification Performance")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_safety.png"), dpi=150)
    plt.close()
    print("[Stats] Saved fig_safety.png")


def fig_overall_accuracy(sys_results: Dict[str, List[Dict]], outdir: str, systems_to_plot: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    if not systems_to_plot:
        plt.close()
        return
    systems = systems_to_plot
    accs = [accuracy(sys_results.get(s, [])) * 100 for s in systems]
    labels = [LABELS[s] for s in systems]
    colors = [COLORS[s] for s in systems]

    bars = ax.bar(labels, accs, color=colors, edgecolor="white", linewidth=1, width=0.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Overall Command Accuracy (%)")
    ax.set_title("Overall Command Generation Accuracy\nAcross All 60 Tasks")
    ax.set_ylim(0, 115)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_overall_accuracy.png"), dpi=150)
    plt.close()
    print("[Stats] Saved fig_overall_accuracy.png")


# ---------------------------------------------------------------------------
# LaTeX table generator
# ---------------------------------------------------------------------------

def latex_accuracy_table(sys_results: Dict[str, List[Dict]]) -> str:
    categories = ["T1", "T2", "T3", "T4", "T5"]
    cat_labels  = ["File Operations", "Package Management",
                   "System Diagnostics", "Web Server/SSL", "Dev Environment"]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\caption{Command Generation Accuracy by System and Category (\%)}")
    lines.append(r"\label{tab:accuracy}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Category} & \textbf{CLI} & \textbf{NaSh} & \textbf{AInux} \\")
    lines.append(r"\midrule")

    for cat, label in zip(categories, cat_labels):
        row = [label]
        for sys in SYSTEMS:
            recs = [r for r in sys_results.get(sys, []) if r["category"] == cat]
            row.append(f"{accuracy(recs)*100:.1f}" if recs else "N/A")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\midrule")
    # Overall row
    row = [r"\textbf{Overall}"]
    for sys in SYSTEMS:
        recs = sys_results.get(sys, [])
        row.append(r"\textbf{" + (f"{accuracy(recs)*100:.1f}" if recs else "N/A") + r"}")
    lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_stats_table(test_results: List[Dict]) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\caption{Statistical Comparison of System Performance}")
    lines.append(r"\label{tab:stats}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Comparison} & \textbf{Test} & \textbf{Statistic} "
                 r"& \textbf{\textit{p}-value} & \textbf{Cohen's \textit{d}} "
                 r"& \textbf{95\% CI} \\")
    lines.append(r"\midrule")

    for t in test_results:
        sig = r"$^{*}$" if t["significant"] else ""
        ci = f"[{t['ci_95'][0]:.3f}, {t['ci_95'][1]:.3f}]"
        comparison = f"{t['label_a']} vs {t['label_b']}"
        lines.append(
            f"{comparison} & {t['test']} & {t['stat']:.3f} & "
            f"{t['p']:.4f}{sig} & {t['cohen_d']:.3f} & {ci} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{6}{l}{\footnotesize $^{*}p < 0.05$}")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(results_path: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    results = load_results(results_path)
    sys_results = by_system(results)
    systems_present = available_systems(sys_results)

    print(f"[Stats] Using results file: {results_path}")
    print(f"[Stats] Writing figures to: {outdir}")
    print(f"[Stats] Systems in results: {', '.join(systems_present) if systems_present else 'none'}")
    missing = [s for s in SYSTEMS if s not in systems_present]
    if missing:
        print(f"[Stats] Missing systems will be shown as N/A: {', '.join(missing)}")

    print("\n" + "="*60)
    print("AInux Evaluation Results")
    print("="*60)

    # Overall accuracy
    for sys in SYSTEMS:
        recs = sys_results.get(sys, [])
        if not recs:
            print(f"\n  {LABELS[sys]}")
            print("    Overall accuracy : N/A  (no records)")
            print("    Mean latency     : N/A")
            print("    False-positive   : N/A")
            continue

        acc = accuracy(recs) * 100
        lat = mean_latency(recs)
        fpr = false_positive_rate(recs) * 100
        print(f"\n  {LABELS[sys]}")
        print(f"    Overall accuracy : {acc:.1f}%  ({sum(r['correct'] for r in recs)}/{len(recs)})")
        print(f"    Mean latency     : {lat:.2f}s")
        print(f"    False-positive   : {fpr:.1f}%")

    # Statistical tests
    print("\n" + "-"*60)
    print("Statistical Tests")
    print("-"*60)

    test_results = []
    comparisons = [
        ("AInux", "TraditionalCLI"),
        ("AInux", "NaSh"),
        ("NaSh",  "TraditionalCLI"),
    ]

    for a, b in comparisons:
        a_recs = sys_results.get(a, [])
        b_recs = sys_results.get(b, [])
        if not a_recs or not b_recs:
            print(f"  Skipping {a} vs {b}: missing data")
            continue
        a_correct, b_correct = align_by_task(a_recs, b_recs)
        if not a_correct:
            print(f"  Skipping {a} vs {b}: no common tasks")
            continue
        test = paired_test(a_correct, b_correct, a, b)
        test_results.append(test)
        sig_str = "SIGNIFICANT" if test["significant"] else "not significant"
        print(f"\n  {a} vs {b}")
        print(f"    {test['test']}: stat={test['stat']:.3f}, p={test['p']:.4f} ({sig_str})")
        print(f"    Cohen's d = {test['cohen_d']:.3f}")
        print(f"    95% CI on mean diff: [{test['ci_95'][0]:.3f}, {test['ci_95'][1]:.3f}]")
        print(f"    Accuracy: {a}={test['mean_a']*100:.1f}% vs {b}={test['mean_b']*100:.1f}%")

    # Accuracy by category
    print("\n" + "-"*60)
    print("Accuracy by Category")
    print("-"*60)
    print(f"{'Category':<20} {'CLI':>8} {'NaSh':>8} {'AInux':>8}")
    for cat in ["T1", "T2", "T3", "T4", "T5"]:
        row = []
        for sys in SYSTEMS:
            recs = [r for r in sys_results.get(sys, []) if r["category"] == cat]
            row.append(f"{accuracy(recs)*100:.1f}%" if recs else "N/A")
        print(f"  {cat:<18} {row[0]:>8} {row[1]:>8} {row[2]:>8}")

    # Figures
    print("\n" + "-"*60)
    print("Generating figures...")
    fig_accuracy_by_category(sys_results, outdir, systems_present)
    fig_latency_comparison(sys_results, outdir, systems_present)
    fig_safety_classification(sys_results, outdir, systems_present)
    fig_overall_accuracy(sys_results, outdir, systems_present)

    # LaTeX tables
    print("\n" + "-"*60)
    print("LaTeX Table I — Accuracy:")
    print(latex_accuracy_table(sys_results))
    print("\nLaTeX Table II — Statistical Tests:")
    if test_results:
        print(latex_stats_table(test_results))

    # Save summary JSON
    summary_path = os.path.join(outdir, "summary.json")
    summary = {
        "overall": {
            sys: {
                "accuracy": (accuracy(sys_results.get(sys, [])) if sys_results.get(sys, []) else None),
                "mean_latency": (mean_latency(sys_results.get(sys, [])) if sys_results.get(sys, []) else None),
                "false_positive_rate": (
                    false_positive_rate(sys_results.get(sys, [])) if sys_results.get(sys, []) else None
                ),
            }
            for sys in SYSTEMS
        },
        "systems_present": systems_present,
        "statistical_tests": test_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Stats] Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="AInux Statistical Analysis")
    parser.add_argument("--results", default=None)
    parser.add_argument("--outdir",  default=None)
    args = parser.parse_args()

    default_results = os.path.join(REPO_ROOT, "results.json")
    if not os.path.exists(default_results):
        default_results = os.path.join(SCRIPT_DIR, "results.json")
    results_path = os.path.abspath(args.results) if args.results else default_results

    outdir = os.path.abspath(args.outdir) if args.outdir else os.path.join(REPO_ROOT, "figures")
    run_analysis(results_path, outdir)


if __name__ == "__main__":
    main()
