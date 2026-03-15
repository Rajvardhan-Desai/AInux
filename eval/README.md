# AInux Evaluation Suite

## Files

| File | Purpose |
|---|---|
| `eval_tasks.py` | 60 standardised tasks with ground-truth commands |
| `eval_baselines.py` | Three systems: TraditionalCLI, NaSh-equivalent, AInux |
| `eval_harness.py` | Runs all tasks, saves `results.json` |
| `eval_stats.py` | Statistical analysis, figures, LaTeX tables |
| `evaluation_section.tex` | Drop-in replacement for Section VIII in the paper |
| `requirements_eval.txt` | Python deps for this module |

## Setup

```bash
pip install -r requirements_eval.txt

# Make sure Ollama is running with phi3:mini pulled
ollama serve
ollama pull phi3:mini
```

## Run

```bash
# Full evaluation (all 60 tasks, all 3 systems)
python eval_harness.py

# Single system only
python eval_harness.py --system ainux

# Single category
python eval_harness.py --category T1

# Check imports without running
python eval_harness.py --dry-run

# Statistical analysis + figures
python eval_stats.py --results results.json --outdir figures/
```

## Workflow for paper

1. Run `eval_harness.py` → produces `results.json`
2. Run `eval_stats.py` → produces 4 PNG figures + printed LaTeX tables
3. Copy LaTeX table values into `evaluation_section.tex`
4. Replace Section VIII in `ainux_paper.tex` with `evaluation_section.tex`
5. Place figures in `figures/` next to the .tex file

## What the evaluation measures

**Command accuracy** — does the NL input produce the correct shell command?
Checked against ground-truth command lists per task.

**Safety precision/recall** — does the MDP safety layer correctly block
dangerous commands without false-positiving on safe ones?

**Latency** — wall-clock time from NL input to command output.

**Statistical tests** — Wilcoxon signed-rank (non-parametric, paired)
with Cohen's d effect sizes and 95% bootstrap confidence intervals.

## Systems compared

- **TraditionalCLI** — ground-truth command used as typed input (ceiling)
- **NaSh** — LLM shell with pattern-based blocking, no memory/agents
- **AInux** — full system: LLM + FAISS memory + MDP safety + agents
