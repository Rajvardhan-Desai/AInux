from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""
AInux Evaluation Baselines
Three systems under comparison:

  1. TraditionalCLI   — direct subprocess execution, no NL parsing
  2. NaShBaseline     — LLM shell with confirmation prompts only
                        (no memory, no agents, no MDP safety scoring)
                        models the guardrailed LLM shell from NaSh [1]
  3. AInuxSystem      — full AInux with memory, MDP safety, agents

[1] Gyawali et al., "NaSh: Guardrails for an LLM-Powered Natural
    Language Shell," arXiv:2506.13028, 2025.
"""

import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class SystemResult:
    system_name: str
    task_id: str
    natural_language: str
    generated_command: str
    success: bool               # execution success (return code 0)
    correct: bool               # command matches ground truth
    blocked: bool               # safety block triggered
    false_positive: bool        # safe command incorrectly blocked
    latency_seconds: float
    output: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseSystem(ABC):
    @abstractmethod
    def process(self, natural_language: str, task_id: str) -> SystemResult:
        ...

    def _run_command(self, command: str, timeout: int = 30) -> tuple[bool, str, str]:
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True,
                text=True, timeout=timeout, cwd="/tmp"
            )
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", "Timeout"
        except Exception as e:
            return False, "", str(e)


# ---------------------------------------------------------------------------
# Baseline 1: Traditional CLI
# Represents a user typing commands without any NL assistance.
# For automated evaluation, we pass the ground-truth command directly
# (simulating a competent CLI user) to establish the execution baseline.
# ---------------------------------------------------------------------------

class TraditionalCLI(BaseSystem):
    """
    Simulates a traditional CLI user.
    In the automated harness, we use the ground-truth command as the
    "user-typed" command. In the human study, participants type freely.
    """

    def process(self, natural_language: str, task_id: str,
                ground_truth_command: Optional[str] = None) -> SystemResult:
        t0 = time.time()

        # In automated mode, use ground truth to measure pure execution success
        command = ground_truth_command or ""
        if not command:
            return SystemResult(
                system_name="TraditionalCLI",
                task_id=task_id,
                natural_language=natural_language,
                generated_command="",
                success=False,
                correct=False,
                blocked=False,
                false_positive=False,
                latency_seconds=time.time() - t0,
                error="No command provided",
            )

        success, output, error = self._run_command(command)
        return SystemResult(
            system_name="TraditionalCLI",
            task_id=task_id,
            natural_language=natural_language,
            generated_command=command,
            success=success,
            correct=True,   # ground truth by definition
            blocked=False,
            false_positive=False,
            latency_seconds=time.time() - t0,
            output=output,
            error=error,
        )


# ---------------------------------------------------------------------------
# Baseline 2: NaSh-equivalent
# LLM-powered shell with confirmation prompts but NO:
#   - persistent memory
#   - MDP safety scoring
#   - autonomous agents
# This directly mirrors the NaSh design from Gyawali et al.
# ---------------------------------------------------------------------------

DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"\bformat\b",
    r"\bfdisk\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
    r":\(\)\s*\{",         # fork bomb
    r"wget\s+.*\|\s*(bash|sh)",
    r"curl\s+.*\|\s*(bash|sh)",
]

RISKY_PATTERNS = [
    r"rm\s+",
    r"\bdel\s+",
    r"\brmdir\b",
    r"\bchmod\b",
    r"\bchown\b",
    r"apt(-get)?\s+remove",
    r"\bkill\b",
]


class NaShBaseline(BaseSystem):
    """
    NaSh-equivalent: LLM generates a command, simple pattern-based safety
    check blocks/warns, no memory, no planning, no agents.
    Confirmation is simulated as 'auto-confirm' for non-dangerous commands
    in the automated harness (mimics a user who proceeds through warnings).
    """

    def __init__(self, ollama_host: str = "http://127.0.0.1:12345",
                 model: str = "phi3:mini"):
        self.ollama_host = ollama_host
        self.model = model
        self._llm_available = self._check_ollama()

    def _check_ollama(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _generate_command(self, natural_language: str) -> Optional[str]:
        if not self._llm_available:
            return self._regex_fallback(natural_language)
        try:
            import requests
            prompt = (
                f"Convert this to a Linux shell command. "
                f"Output ONLY the command, nothing else.\n"
                f"Request: {natural_language}\nCommand:"
            )
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={"model": self.model, "prompt": prompt,
                      "stream": False, "options": {"temperature": 0.1}},
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            # Clean up
            raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()
            return raw.split("\n")[0].strip() or None
        except Exception:
            return self._regex_fallback(natural_language)

    def _regex_fallback(self, text: str) -> Optional[str]:
        """Minimal regex fallback when Ollama is unavailable."""
        lower = text.lower()
        if re.search(r"list.*files|show.*files|ls", lower):
            return "ls -la"
        if re.search(r"current.*dir|where.*am|pwd", lower):
            return "pwd"
        if re.search(r"disk.*space|disk.*usage", lower):
            return "df -h"
        if re.search(r"memory", lower):
            return "free -h"
        if re.search(r"processes", lower):
            return "ps aux"
        if re.search(r"uptime", lower):
            return "uptime"
        return None

    def _classify(self, command: str) -> str:
        """Returns 'dangerous', 'risky', or 'safe'."""
        cmd = command.lower()
        for p in DANGEROUS_PATTERNS:
            if re.search(p, cmd):
                return "dangerous"
        for p in RISKY_PATTERNS:
            if re.search(p, cmd):
                return "risky"
        return "safe"

    def process(self, natural_language: str, task_id: str) -> SystemResult:
        t0 = time.time()

        command = self._generate_command(natural_language)
        if not command:
            return SystemResult(
                system_name="NaSh",
                task_id=task_id,
                natural_language=natural_language,
                generated_command="",
                success=False, correct=False, blocked=False,
                false_positive=False,
                latency_seconds=time.time() - t0,
                error="LLM could not generate command",
            )

        classification = self._classify(command)

        if classification == "dangerous":
            return SystemResult(
                system_name="NaSh",
                task_id=task_id,
                natural_language=natural_language,
                generated_command=command,
                success=False, correct=False, blocked=True,
                false_positive=False,
                latency_seconds=time.time() - t0,
                error="Blocked: dangerous command",
            )

        # Risky: NaSh shows confirmation prompt (auto-confirmed in harness)
        # Safe: execute directly
        success, output, error = self._run_command(command)
        return SystemResult(
            system_name="NaSh",
            task_id=task_id,
            natural_language=natural_language,
            generated_command=command,
            success=success, correct=False,  # correctness set by harness
            blocked=False, false_positive=False,
            latency_seconds=time.time() - t0,
            output=output, error=error,
        )


# ---------------------------------------------------------------------------
# Baseline 3: AInux Full System
# ---------------------------------------------------------------------------

class AInuxSystem(BaseSystem):
    """
    Full AInux: local LLM + FAISS memory + MDP safety + agents.
    """

    def __init__(self, ollama_host: str = "http://127.0.0.1:12345",
                 model: str = "phi3:mini"):
        import sys, os
        # Allow importing from parent directory
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from ainux_llm_runtime import LocalLLMRuntime, OllamaConfig
            from ainux_memory import AInuxMemory
            from ainux_safety import MDPSafetyChecker, ConfirmationLevel

            self._confirmation_level_block = ConfirmationLevel.BLOCK

            cfg = OllamaConfig(host=ollama_host, model=model)
            self.llm = LocalLLMRuntime(cfg)
            self.memory = AInuxMemory(persist=False)   # no disk I/O during eval
            self.safety = MDPSafetyChecker()
            self._available = True
        except ImportError as e:
            print(f"[AInuxSystem] Import error: {e}", flush=True); import traceback; traceback.print_exc()
            self._available = False

    def process(self, natural_language: str, task_id: str) -> SystemResult:
        t0 = time.time()

        if not self._available:
            return SystemResult(
                system_name="AInux",
                task_id=task_id,
                natural_language=natural_language,
                generated_command="",
                success=False, correct=False, blocked=False,
                false_positive=False,
                latency_seconds=time.time() - t0,
                error="AInux modules not available",
            )

        # Retrieve memory context
        context_items = self.memory.retrieve(natural_language, k=3)
        context = "\n".join(
            f"{i.intent}: {i.command}" for i, _ in context_items
        ) or None

        # Generate command
        command = self.llm.generate_command(natural_language, context)

        if not command:
            return SystemResult(
                system_name="AInux",
                task_id=task_id,
                natural_language=natural_language,
                generated_command="",
                success=False, correct=False, blocked=False,
                false_positive=False,
                latency_seconds=time.time() - t0,
                error="LLM could not generate command",
            )

        # MDP safety validation
        validation = self.safety.validate_command(command)
        if validation.confirmation == self._confirmation_level_block:
            return SystemResult(
                system_name="AInux",
                task_id=task_id,
                natural_language=natural_language,
                generated_command=command,
                success=False, correct=False, blocked=True,
                false_positive=False,
                latency_seconds=time.time() - t0,
                error=f"Blocked by MDP safety: {validation.reason}",
            )

        # Execute
        success, output, error = self._run_command(command)

        # Store to memory
        self.memory.store(
            text=natural_language,
            command=command,
            intent=task_id.split("_")[0],
            outcome="success" if success else "failure",
            layer="short",
        )

        return SystemResult(
            system_name="AInux",
            task_id=task_id,
            natural_language=natural_language,
            generated_command=command,
            success=success, correct=False,  # set by harness
            blocked=False, false_positive=False,
            latency_seconds=time.time() - t0,
            output=output, error=error,
        )

