"""
AInux - AI-Native Linux Environment
Main orchestrator integrating:
  - Universal LLM Runtime (Chat Completion API standard)
  - FAISS Memory Layer with three-tier persistence
  - Four-tier Risk Scoring Safety Framework
  - Autonomous Agents (package, file, diagnostics)

Usage:
    python -m AInux.ainux_core
    python -m AInux.ainux_core --voice
    python -m AInux.ainux_core --model phi3:mini
    python -m AInux.ainux_core --host http://192.168.1.10:8000
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from typing import Optional

from .ainux_agents import AgentDispatcher
from .ainux_llm_runtime import (
    DEFAULT_LLM_HOST,
    LocalLLMRuntime,
    LLMRuntimeConfig,
    normalize_llm_host,
)
from .ainux_memory import AInuxMemory
from .ainux_safety import ConfirmationLevel, RiskScoringChecker


# ---------------------------------------------------------------------------
# AI Shell
# ---------------------------------------------------------------------------

class AIShell:
    """
    Primary interface: natural language → intent → plan → safe execution.
    Implements the AiShell described in paper Sections III.A and IV.

    Routing:
      Simple requests (single-command intent) → _handle_simple_command
      Complex/multi-step requests              → _handle_agent_task
    """

    INTENT_CATEGORIES = [
        "information_retrieval",
        "system_modification",
        "process_management",
        "file_operations",
        "troubleshooting",
        "package_management",
        "learning_explanation",
    ]

    def __init__(
        self,
        model: str = "gpt-oss-20b-MXFP4",
        llm_host: str = DEFAULT_LLM_HOST,
        persist_memory: bool = True,
    ):
        print("[AInux] Initialising components...")

        # LLM runtime
        cfg = LLMRuntimeConfig(host=normalize_llm_host(llm_host), model=model)
        self.llm = LocalLLMRuntime(cfg)

        # Memory
        self.memory = AInuxMemory(persist=persist_memory)
        stats = self.memory.stats()
        print(
            f"[AInux] Memory: {stats['total']} items "
            f"(short={stats['short']}, mid={stats['mid']}, long={stats['long']})"
        )

        # Four-tier risk scoring safety framework
        self.safety = RiskScoringChecker()

        # Agent dispatcher
        self.dispatcher = AgentDispatcher(self.llm, self.safety)

        self._platform = platform.system().lower()
        print("[AInux] Ready.\n")

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def _classify_intent(self, user_input: str) -> str:
        """
        Classify user input into one of the six intent categories.
        Uses regex pattern matching; falls back to 'information_retrieval'.
        """
        import re
        lower = user_input.lower()

        intent_patterns = {
            "file_operations": [
                r"\bfile\b", r"\bfolder\b", r"\bdirector\b",
                r"\bcreate\b", r"\bdelete\b", r"\brename\b",
                r"\bcopy\b", r"\bmove\b", r"\blist files\b",
            ],
            "package_management": [
                r"\binstall\b", r"\bupgrade\b", r"\bpip\b",
                r"\bapt\b", r"\bnpm\b",
            ],
            "process_management": [
                r"\bprocess\b", r"\bps\b", r"\bkill\b",
                r"\bstart\b", r"\bstop\b", r"\brestart\b",
            ],
            "troubleshooting": [
                r"\bdiagnos\b", r"\blog\b", r"\berror\b",
                r"\bfix\b", r"\bdebug\b", r"\bmonitor\b",
            ],
            "information_retrieval": [
                r"\bshow\b", r"\blist\b", r"\bdisplay\b",
                r"\bwhat\b", r"\bwhere\b", r"\bhow much\b",
            ],
        }

        scores = {}
        for intent, patterns in intent_patterns.items():
            scores[intent] = sum(1 for p in patterns if re.search(p, lower))

        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "information_retrieval"

    # ------------------------------------------------------------------
    # Memory-augmented context retrieval
    # ------------------------------------------------------------------

    def _get_context(self, user_input: str) -> Optional[str]:
        """Retrieve relevant past interactions from memory."""
        results = self.memory.retrieve(user_input, k=3)
        if not results:
            return None
        lines = []
        for item, score in results:
            lines.append(
                f"[Past: {item.intent}] {item.text} → {item.command} ({item.outcome})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Single command mode
    # ------------------------------------------------------------------

    def _handle_simple_command(self, user_input: str, context: Optional[str]) -> None:
        """Handle short, single-command requests."""
        command = self.llm.generate_command(user_input, context, self._platform)

        if not command:
            print("[AInux] I couldn't generate a command for that request.")
            print("        Try rephrasing, or type 'help' for examples.")
            return

        # Validate with four-tier risk scoring framework
        validation = self.safety.validate_command(command)
        print(f"\n  Generated command : {command}")
        print(f"  Safety            : {validation.action_class.value} (score={validation.score})")
        if validation.affected_components:
            print(f"  Affects           : {', '.join(validation.affected_components)}")

        if validation.confirmation == ConfirmationLevel.BLOCK:
            print(f"\n  BLOCKED: {validation.reason}")
            return

        if validation.confirmation in (ConfirmationLevel.CONFIRM, ConfirmationLevel.WARN):
            explanation = self.llm.explain_command(command)
            print(f"\n  What this does: {explanation}")
            if validation.confirmation == ConfirmationLevel.CONFIRM:
                ans = input("  Proceed? (YES to confirm): ").strip()
                if ans.upper() != "YES":
                    print("  Cancelled.")
                    return

        # Execute
        import subprocess
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=60, cwd=os.getcwd()
        )

        outcome = "success" if result.returncode == 0 else "failure"
        print(f"\n{'─' * 50}")
        print(f"  Status : {'SUCCESS' if outcome == 'success' else 'FAILED'}")
        if result.stdout.strip():
            print(f"  Output :\n{result.stdout.strip()}")
        if result.stderr.strip() and outcome == "failure":
            print(f"  Error  : {result.stderr.strip()}")
        print(f"{'─' * 50}")

        # Store in memory
        intent = self._classify_intent(user_input)
        self.memory.store(
            text=user_input,
            command=command,
            intent=intent,
            outcome=outcome,
            layer="short",
        )

    # ------------------------------------------------------------------
    # Agent mode (complex / multi-step requests)
    # ------------------------------------------------------------------

    def _handle_agent_task(self, user_input: str, context: Optional[str]) -> None:
        """Delegate multi-step tasks to the appropriate autonomous agent."""

        def confirm_cb(cmd: str, reason: str) -> bool:
            print(f"\n  [Agent] WARNING: {reason}")
            print(f"  Command: {cmd}")
            ans = input("  Proceed? (YES to confirm): ").strip()
            return ans.upper() == "YES"

        print("\n  [Agent] Planning task...")
        agent_name, result = self.dispatcher.dispatch(
            user_input, context=context, confirm_callback=confirm_cb
        )

        print(f"\n{'─' * 50}")
        print(f"  Agent   : {agent_name}")
        print(f"  Status  : {'SUCCESS' if result.success else 'FAILED'}")
        print(f"  Steps   : {len(result.commands_executed)}")
        print(f"  Time    : {result.duration_seconds:.1f} s")

        if result.commands_executed:
            print("  Commands executed:")
            for cmd in result.commands_executed:
                print(f"    • {cmd}")

        if result.output:
            out = result.output
            if len(out) > 1000:
                out = out[:1000] + "\n  [... truncated]"
            print(f"  Output  :\n{out}")

        if result.error:
            print(f"  Error   : {result.error}")

        if result.rolled_back:
            print("  [Agent] System rolled back to previous state.")

        if result.summary:
            print(f"  Summary : {result.summary}")

        print(f"{'─' * 50}")

        # Store each executed command to memory
        intent = self._classify_intent(user_input)
        for cmd in result.commands_executed:
            self.memory.store(
                text=user_input,
                command=cmd,
                intent=intent,
                outcome="success" if result.success else "failure",
                layer="mid",
            )

    # ------------------------------------------------------------------
    # Routing: simple vs agent
    # ------------------------------------------------------------------

    def _is_multi_step(self, user_input: str) -> bool:
        """
        Heuristic: route to an agent if the request implies multiple operations
        or falls in a domain with a specialised agent.
        """
        import re
        lower = user_input.lower()
        multi_step_indicators = [
            r"\bset\s+up\b", r"\bconfigure\b", r"\binstall\s+and\b",
            r"\bthen\b", r"\bafter\s+that\b", r"\balso\b",
            r"\bcreate.*and\b", r"\bbackup\b", r"\bmigrate\b",
            r"\bdeploy\b", r"\boptimise\b", r"\bdiagnos\b",
        ]
        return any(re.search(p, lower) for p in multi_step_indicators)

    def process(self, user_input: str) -> None:
        """Main dispatch method for a single user utterance."""
        context = self._get_context(user_input)

        if self._is_multi_step(user_input):
            self._handle_agent_task(user_input, context)
        else:
            self._handle_simple_command(user_input, context)

        # Periodically consolidate memory
        if len(self.memory._items) % 20 == 0 and len(self.memory._items) > 0:
            result = self.memory.consolidate()
            if result["promoted"] > 0:
                print(
                    f"  [Memory] Promoted {result['promoted']} item(s) to long-term memory."
                )

    # ------------------------------------------------------------------
    # Interactive REPL
    # ------------------------------------------------------------------

    def run_interactive(self, voice: bool = False) -> None:
        """Main interactive read-eval-print loop."""
        print("AInux — AI-Native Linux Shell")
        print("Type 'help' for examples, 'memory' for memory stats, 'exit' to quit.\n")

        while True:
            try:
                if voice:
                    user_input = self._get_voice_input()
                    if user_input:
                        print(f"AInux [voice]> {user_input}")
                    else:
                        continue
                else:
                    user_input = input("AInux> ").strip()

                if not user_input:
                    continue

                lower = user_input.lower()

                if lower in ("exit", "quit", "q", "bye"):
                    self.memory.consolidate()
                    print("Goodbye.")
                    break

                elif lower in ("help", "h", "?"):
                    self._show_help()

                elif lower in ("memory", "mem"):
                    stats = self.memory.stats()
                    print(
                        f"\n  Memory: {stats['total']} items "
                        f"(short={stats['short']}, mid={stats['mid']}, "
                        f"long={stats['long']}, index={stats['index_size']})\n"
                    )

                elif lower in ("status", "mode", "info"):
                    self._show_status()

                else:
                    self.process(user_input)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit or press Ctrl+C again.")
            except Exception as e:
                print(f"\n  Unexpected error: {e}")
                import traceback
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Voice input
    # ------------------------------------------------------------------

    def _get_voice_input(self) -> Optional[str]:
        try:
            import importlib
            voice_input = importlib.import_module("voice_input")
            print("Listening...")
            return voice_input.listen_for_command(timeout=6, phrase_time_limit=8)
        except Exception as e:
            print(f"Voice error: {e}")
            return input("AInux> ").strip()

    # ------------------------------------------------------------------
    # Help and status
    # ------------------------------------------------------------------

    def _show_help(self) -> None:
        print("""
  AInux Help — Natural Language Examples
  ─────────────────────────────────────────
  File operations:
    • "List all Python files in this directory"
    • "Create folder logs/archive/2025"
    • "Find files modified in the last 24 hours"

  Package management (agent):
    • "Install nginx and configure it"
    • "Set up a Python environment for data analysis"
    • "Update all installed packages"

  System diagnostics (agent):
    • "Diagnose high CPU usage"
    • "Show memory and disk health"
    • "Check which services are running"

  Context-aware:
    • "Do what I did last time for the backup"
    • "Show me the logs again"

  Shell commands:
    • memory   — show memory statistics
    • status   — show system status
    • exit     — quit AInux
""")

    def _show_status(self) -> None:
        llm_status = "connected" if self.llm.is_available() else "offline (regex fallback active)"
        stats = self.memory.stats()
        mem_str = (
            f"{stats['total']} items "
            f"(short={stats['short']}, mid={stats['mid']}, long={stats['long']})"
        )
        print(f"""
  AInux Status
  ─────────────────────────────────────────
  LLM       : {self.llm.config.model} — {llm_status}
  Memory    : {mem_str}
  Safety    : four-tier risk scoring framework active
  Agents    : PackageManagement, FileOperations, SystemDiagnostics
  Platform  : {platform.system()} {platform.release()}
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AInux — AI-Native Linux Shell")
    parser.add_argument("--model", default="gpt-oss-20b-MXFP4")
    parser.add_argument(
        "--host", default=DEFAULT_LLM_HOST,
        help="Chat Completion API endpoint URL (local or cloud)"
    )
    parser.add_argument("--voice", action="store_true", help="Enable voice input")
    parser.add_argument(
        "--no-persist", action="store_true",
        help="Disable memory persistence (no disk writes)"
    )
    args = parser.parse_args()

    shell = AIShell(
        model=args.model,
        llm_host=normalize_llm_host(args.host),
        persist_memory=not args.no_persist,
    )
    shell.run_interactive(voice=args.voice)