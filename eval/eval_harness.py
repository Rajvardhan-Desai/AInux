"""
AInux Evaluation Harness
Runs all 60 tasks against three systems and produces a results JSON
that eval_stats.py consumes for statistical analysis and plotting.

Usage:
    python eval_harness.py                          # full run
    python eval_harness.py --system ainux           # single system
    python eval_harness.py --category T1            # single category
    python eval_harness.py --dry-run                # check imports only
    python eval_harness.py --results results.json   # custom output path
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

from eval_tasks import ALL_TASKS, CATEGORIES, EvalTask, TASK_MAP
from eval_baselines import AInuxSystem, NaShBaseline, SystemResult, TraditionalCLI


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class EvalHarness:

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "phi3:mini",
    ):
        print("[Harness] Initialising systems...")
        self.cli   = TraditionalCLI()
        self.nash  = NaShBaseline(ollama_host, model)
        self.ainux = AInuxSystem(ollama_host, model)

        self.results: List[Dict] = []

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        tasks: List[EvalTask] = None,
        systems: List[str] = None,
        skip_unsafe: bool = True,
    ) -> List[Dict]:
        tasks = tasks or ALL_TASKS
        systems = systems or ["cli", "nash", "ainux"]

        print(f"\n[Harness] Running {len(tasks)} tasks × {len(systems)} systems")
        print(f"[Harness] Started at {datetime.now().strftime('%H:%M:%S')}\n")

        for i, task in enumerate(tasks, 1):
            print(f"  [{i:02d}/{len(tasks)}] {task.id}: {task.natural_language[:60]}")

            if skip_unsafe and not task.safe_to_execute:
                # Evaluate command generation accuracy only — don't execute
                self._eval_generation_only(task, systems)
            else:
                self._eval_full(task, systems)

        print(f"\n[Harness] Complete. {len(self.results)} results collected.")
        return self.results

    def _eval_full(self, task: EvalTask, systems: List[str]) -> None:
        """Run task and execute command, measure all metrics."""

        # CLI: use first ground truth as the "typed" command
        if "cli" in systems:
            gt_cmd = task.ground_truth[0] if task.ground_truth else ""
            result = self.cli.process(task.natural_language, task.id,
                                      ground_truth_command=gt_cmd)
            result.correct = True  # CLI always uses ground truth
            self._record(result, task)

        # NaSh
        if "nash" in systems:
            result = self.nash.process(task.natural_language, task.id)
            result.correct = task.is_correct(result.generated_command)
            result.false_positive = (
                result.blocked and
                self._is_safe_task(task)
            )
            self._record(result, task)

        # AInux
        if "ainux" in systems:
            result = self.ainux.process(task.natural_language, task.id)
            result.correct = task.is_correct(result.generated_command)
            result.false_positive = (
                result.blocked and
                self._is_safe_task(task)
            )
            self._record(result, task)

    def _eval_generation_only(self, task: EvalTask, systems: List[str]) -> None:
        """Measure command generation accuracy without executing."""

        # CLI: always correct (human knows the command)
        if "cli" in systems:
            self._record_generation(
                system_name="TraditionalCLI",
                task=task,
                generated=task.ground_truth[0] if task.ground_truth else "",
                correct=True,
                blocked=False,
                latency=0.0,
            )

        # NaSh
        if "nash" in systems:
            t0 = time.time()
            cmd = self.nash._generate_command(task.natural_language) or ""
            latency = time.time() - t0
            correct = task.is_correct(cmd)
            blocked = self.nash._classify(cmd) == "dangerous" if cmd else False
            self._record_generation("NaSh", task, cmd, correct, blocked, latency)

        # AInux
        if "ainux" in systems:
            t0 = time.time()
            cmd = ""
            blocked = False
            if self.ainux._available:
                ctx = None
                ctxitems = self.ainux.memory.retrieve(task.natural_language, k=2)
                if ctxitems:
                    ctx = "\n".join(f"{i.intent}: {i.command}" for i, _ in ctxitems)
                cmd = self.ainux.llm.generate_command(task.natural_language, ctx) or ""
                if cmd:
                    v = self.ainux.safety.validate_command(cmd)
                    from AInux.ainux_safety import ConfirmationLevel
                    blocked = v.confirmation == ConfirmationLevel.BLOCK
            latency = time.time() - t0
            correct = task.is_correct(cmd) and not blocked
            self._record_generation("AInux", task, cmd, correct, blocked, latency)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _record(self, result: SystemResult, task: EvalTask) -> None:
        self.results.append({
            "system":       result.system_name,
            "task_id":      task.id,
            "category":     task.category,
            "complexity":   task.complexity,
            "nl_input":     result.natural_language,
            "generated":    result.generated_command,
            "correct":      result.correct,
            "success":      result.success,
            "blocked":      result.blocked,
            "false_positive": result.false_positive,
            "latency":      round(result.latency_seconds, 3),
            "executed":     task.safe_to_execute,
        })

    def _record_generation(
        self, system_name, task, generated, correct, blocked, latency
    ) -> None:
        self.results.append({
            "system":       system_name,
            "task_id":      task.id,
            "category":     task.category,
            "complexity":   task.complexity,
            "nl_input":     task.natural_language,
            "generated":    generated,
            "correct":      correct,
            "success":      None,     # not executed
            "blocked":      blocked,
            "false_positive": False,
            "latency":      round(latency, 3),
            "executed":     False,
        })

    @staticmethod
    def _is_safe_task(task: EvalTask) -> bool:
        """True if the task's ground-truth commands are all safe."""
        for gt in task.ground_truth:
            lower = gt.lower()
            if any(kw in lower for kw in ["rm -rf", "format", "fdisk", "dd if="]):
                return False
        return True

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: str = "results.json") -> None:
        with open(path, "w") as f:
            json.dump({
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "n_tasks": len(ALL_TASKS),
                    "n_results": len(self.results),
                },
                "results": self.results,
            }, f, indent=2)
        print(f"[Harness] Results saved to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AInux Evaluation Harness")
    parser.add_argument("--host",     default="http://localhost:11434")
    parser.add_argument("--model",    default="phi3:mini")
    parser.add_argument("--system",   choices=["cli", "nash", "ainux", "all"],
                                      default="all")
    parser.add_argument("--category", choices=["T1","T2","T3","T4","T5","all"],
                                      default="all")
    parser.add_argument("--results",  default="results.json")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("[Harness] Dry run — checking imports only")
        from eval_tasks import ALL_TASKS
        from eval_baselines import TraditionalCLI, NaShBaseline, AInuxSystem
        print(f"  eval_tasks   : OK ({len(ALL_TASKS)} tasks)")
        print(f"  eval_baselines: OK")
        return

    tasks = CATEGORIES.get(args.category, ALL_TASKS) if args.category != "all" else ALL_TASKS
    systems = ["cli", "nash", "ainux"] if args.system == "all" else [args.system]

    harness = EvalHarness(ollama_host=args.host, model=args.model)
    harness.run(tasks=tasks, systems=systems)
    harness.save(args.results)


if __name__ == "__main__":
    main()
