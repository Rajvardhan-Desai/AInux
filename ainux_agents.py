"""
AInux Autonomous Agents
Implements the agent execution framework described in paper Sections IV and V.

Each agent follows a plan-execute-verify lifecycle:
  1. Receive intent and memory context
  2. Generate a full plan via the LLM
  3. Validate each step with the four-tier risk scoring safety layer
  4. Execute with file-level checkpointing; restore on failure
  5. Verify outcomes
  6. Return a natural language summary

Three agents are implemented:
  - PackageManagementAgent  (apt, pip, npm)
  - FileOperationsAgent      (files, directories, permissions)
  - SystemDiagnosticsAgent   (read-only diagnostics; SAFE commands only)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ainux_safety import ConfirmationLevel, RiskScoringChecker, ValidationResult
from .ainux_llm_runtime import LocalLLMRuntime

logger = logging.getLogger("ainux.agents")


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    success: bool
    commands_executed: List[str]
    output: str
    error: str = ""
    rolled_back: bool = False
    summary: str = ""
    duration_seconds: float = 0.0


@dataclass
class Checkpoint:
    """Snapshot created before a risky command to enable rollback."""
    timestamp: float
    command: str
    backup_paths: List[str] = field(default_factory=list)
    snapshot_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    Abstract base for all AInux agents.
    Implements the planner-verifier loop with file-level checkpointing.

    Safety validation uses the four-tier risk scoring framework
    (paper Section III.E, Eq. 2–3) before each command is executed.
    """

    def __init__(self, llm: LocalLLMRuntime, safety: RiskScoringChecker):
        self.llm = llm
        self.safety = safety
        self._checkpoints: List[Checkpoint] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        intent: str,
        context: Optional[str] = None,
        confirm_callback=None,
    ) -> AgentResult:
        """
        Main agent lifecycle.
        confirm_callback(command, reason) -> bool  (True = proceed)
        """
        t0 = time.time()
        executed: List[str] = []
        outputs: List[str] = []

        # Steps 1–2: Generate plan
        plan = self._generate_plan(intent, context)
        if not plan:
            return AgentResult(
                success=False,
                commands_executed=[],
                output="",
                error="Could not generate a plan for this intent.",
                duration_seconds=time.time() - t0,
            )

        # Step 3: Validate plan (Eq. 2 — product of individual scores)
        plan_score, validations = self.safety.validate_plan(plan)
        logger.info(f"Plan safety score: {plan_score:.3f}")

        # Step 4: Execute with checkpointing
        for cmd, validation in zip(plan, validations):

            # Hard block (DANGEROUS tier, score=0.0)
            if validation.confirmation == ConfirmationLevel.BLOCK:
                return AgentResult(
                    success=False,
                    commands_executed=executed,
                    output="\n".join(outputs),
                    error=f"Command blocked: '{cmd}'. {validation.reason}",
                    duration_seconds=time.time() - t0,
                )

            # Needs user confirmation (RISKY tier, score=0.3, or root escalation)
            if validation.confirmation == ConfirmationLevel.CONFIRM:
                if confirm_callback:
                    proceed = confirm_callback(cmd, validation.reason)
                else:
                    proceed = self._default_confirm(cmd, validation.reason)
                if not proceed:
                    return AgentResult(
                        success=False,
                        commands_executed=executed,
                        output="\n".join(outputs),
                        error=f"User declined to execute: '{cmd}'",
                        duration_seconds=time.time() - t0,
                    )

            # Take checkpoint before risky or reversible operations
            if validation.action_class.value in ("risky", "reversible"):
                self._take_checkpoint(cmd)

            # Execute
            result = self._execute_command(cmd)
            executed.append(cmd)

            if result["success"]:
                if result["output"]:
                    outputs.append(result["output"])
            else:
                # Attempt rollback to most recent checkpoint
                rolled = self._rollback()
                return AgentResult(
                    success=False,
                    commands_executed=executed,
                    output="\n".join(outputs),
                    error=f"Command failed: '{cmd}'\n{result['error']}",
                    rolled_back=rolled,
                    duration_seconds=time.time() - t0,
                )

        # Steps 5–6: Verify and summarise
        summary = self._summarize(intent, executed, "\n".join(outputs))

        return AgentResult(
            success=True,
            commands_executed=executed,
            output="\n".join(outputs),
            summary=summary,
            duration_seconds=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _generate_plan(self, intent: str, context: Optional[str]) -> List[str]:
        """Generate an ordered list of shell commands for the intent."""
        ...

    @abstractmethod
    def domain(self) -> str:
        """Human-readable agent domain name."""
        ...

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_command(self, command: str) -> Dict:
        """
        Execute a shell command and return a result dict.

        Uses a list-form invocation where possible; shell=True is required
        here because commands may include pipes and redirections that are
        generated by the LLM for diagnostic tasks. The safety layer
        validates commands before this method is called.
        """
        try:
            result = subprocess.run(
                command, shell=True,
                capture_output=True, text=True,
                timeout=60, cwd=os.getcwd(),
            )
            return {
                "success":     result.returncode == 0,
                "output":      result.stdout.strip(),
                "error":       result.stderr.strip(),
                "return_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "", "error": "Command timed out after 60 s"}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    # ------------------------------------------------------------------
    # Checkpointing and rollback
    # ------------------------------------------------------------------

    def _take_checkpoint(self, command: str) -> None:
        """
        Create a lightweight checkpoint before executing a risky/reversible
        command by backing up any file paths referenced in the command.

        Uses tempfile.mkstemp() to create backup files atomically and
        securely (avoids the TOCTOU race condition of mktemp()).
        """
        checkpoint = Checkpoint(timestamp=time.time(), command=command)

        file_args = re.findall(
            r"(?:^|\s)(/[^\s]+|\./[^\s]+|[A-Za-z0-9_./]+\.[a-z]+)",
            command,
        )
        for path in file_args:
            p = Path(path)
            if p.is_file():
                # mkstemp returns (fd, name); close the fd before copying into it
                fd, tmp = tempfile.mkstemp(suffix=f"_{p.name}.ainux_bak")
                try:
                    os.close(fd)
                    shutil.copy2(str(p), tmp)
                    checkpoint.backup_paths.append(f"{path}:{tmp}")
                except Exception as exc:
                    # Clean up the temp file if the copy failed
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    logger.warning(f"Checkpoint failed for {path}: {exc}")

        self._checkpoints.append(checkpoint)
        logger.debug(f"Checkpoint created for: {command}")

    def _rollback(self) -> bool:
        """Restore files from the most recent checkpoint."""
        if not self._checkpoints:
            return False

        cp = self._checkpoints.pop()
        logger.info(f"Rolling back checkpoint for: {cp.command}")
        restored = 0

        for mapping in cp.backup_paths:
            original, backup = mapping.split(":", 1)
            if Path(backup).exists():
                try:
                    shutil.copy2(backup, original)
                    os.unlink(backup)
                    restored += 1
                except Exception as e:
                    logger.warning(f"Rollback failed for {original}: {e}")

        return restored > 0

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def _summarize(self, intent: str, commands: List[str], output: str) -> str:
        """Ask the LLM to summarise what was accomplished."""
        if not self.llm.is_available():
            return f"Completed {len(commands)} step(s) for: {intent}"
        prompt = (
            f"Summarise in one sentence what was accomplished:\n"
            f"Goal: {intent}\n"
            f"Commands run: {'; '.join(commands)}\n"
            f"Output snippet: {output[:300]}"
        )
        try:
            response = self.llm._call_llm(prompt)
            return response.strip() if response else f"Completed: {intent}"
        except Exception:
            return f"Completed {len(commands)} step(s) for: {intent}"

    def _default_confirm(self, command: str, reason: str) -> bool:
        print(f"\n[{self.domain()}] WARNING: {reason}")
        print(f"  Command: {command}")
        ans = input("Proceed? (YES to confirm): ").strip()
        return ans.upper() == "YES"


# ---------------------------------------------------------------------------
# Agent 1: Package Management
# ---------------------------------------------------------------------------

class PackageManagementAgent(BaseAgent):
    """
    Handles installation, updates, and dependency resolution.
    Supports apt/apt-get, pip, and npm.
    """

    PLAN_PROMPT_TEMPLATE = (
        "You are a Linux package management expert.\n"
        "Task: {intent}\n"
        "Context: {context}\n\n"
        "Generate a safe sequence of package management commands (apt, pip, or npm as appropriate).\n"
        "Always run 'apt-get update' before apt installs.\n"
        "Always use --yes / -y flags to avoid interactive prompts.\n"
        "Output one command per line, no explanations.\n"
        "Commands:"
    )

    def domain(self) -> str:
        return "PackageManagement"

    def _generate_plan(self, intent: str, context: Optional[str]) -> List[str]:
        if self.llm.is_available():
            prompt = self.PLAN_PROMPT_TEMPLATE.format(
                intent=intent,
                context=context or "none",
            )
            try:
                response = self.llm._call_llm(prompt)
                if response:
                    plan = self.llm._extract_plan(response, max_steps=10)
                    if plan:
                        return plan
            except Exception as e:
                logger.warning(f"LLM plan generation failed: {e}")

        return self._regex_plan(intent)

    def _regex_plan(self, intent: str) -> List[str]:
        lower = intent.lower()

        m = re.search(r"install\s+(\S+)", lower)
        if m:
            pkg = m.group(1)
            if re.search(r"\.txt$", pkg):
                return [f"pip install -r {pkg}"]
            if re.search(r"\bpip\b", lower):
                return [f"pip install {pkg}"]
            if re.search(r"\bnpm\b", lower):
                return [f"npm install {pkg}"]
            return ["apt-get update -y", f"apt-get install -y {pkg}"]

        if re.search(r"update|upgrade", lower):
            return ["apt-get update -y", "apt-get upgrade -y"]

        return []


# ---------------------------------------------------------------------------
# Agent 2: File Operations
# ---------------------------------------------------------------------------

class FileOperationsAgent(BaseAgent):
    """
    Manages files, directories, permissions, and batch operations.
    Includes hierarchical path construction (paper Section IV.B.1).
    """

    def domain(self) -> str:
        return "FileOperations"

    def _generate_plan(self, intent: str, context: Optional[str]) -> List[str]:
        if self.llm.is_available():
            prompt = (
                f"You are a Linux file system expert.\n"
                f"Task: {intent}\n"
                f"Context: {context or 'none'}\n\n"
                f"Generate safe file operation commands.\n"
                f"For creating nested directories, use mkdir -p.\n"
                f"Output one command per line, no explanations.\n"
                f"Commands:"
            )
            try:
                response = self.llm._call_llm(prompt)
                if response:
                    plan = self.llm._extract_plan(response, max_steps=12)
                    if plan:
                        return plan
            except Exception as e:
                logger.warning(f"LLM plan generation failed: {e}")

        return self._regex_plan(intent)

    def _regex_plan(self, intent: str) -> List[str]:
        lower = intent.lower()

        # Hierarchical path construction (paper Section IV.B.1)
        m = re.search(r"create\s+(?:director(?:y|ies)|folder)\s+(.+)", lower)
        if m:
            path = m.group(1).strip().replace(" ", "/")
            return [f"mkdir -p {path}"]

        if re.search(r"list|show|ls", lower):
            return ["ls -la"]

        if re.search(r"find.*python|python.*files?", lower):
            return ["find . -name '*.py' -type f"]

        m = re.search(r"rename\s+(\S+)\s+to\s+(\S+)", lower)
        if m:
            return [f"mv {m.group(1)} {m.group(2)}"]

        m = re.search(r"copy\s+(\S+)\s+to\s+(\S+)", lower)
        if m:
            return [f"cp -r {m.group(1)} {m.group(2)}"]

        return []


# ---------------------------------------------------------------------------
# Agent 3: System Diagnostics
# ---------------------------------------------------------------------------

class SystemDiagnosticsAgent(BaseAgent):
    """
    Monitors system health, analyses logs, and identifies performance bottlenecks.
    Read-only by design — all generated commands are validated as SAFE (score=1.0)
    before execution; any non-SAFE command from the LLM is dropped.
    """

    DIAGNOSTIC_COMMANDS: Dict[str, List[str]] = {
        "cpu":       ["top -bn1 | head -20", "mpstat 1 1"],
        "memory":    ["free -h", "vmstat -s | head -10"],
        "disk":      ["df -h", "du -sh /* 2>/dev/null | sort -rh | head -10"],
        "network":   ["ss -tuln", "netstat -s 2>/dev/null | head -20"],
        "logs":      ["journalctl -n 50 --no-pager", "dmesg | tail -20"],
        "processes": ["ps aux --sort=-%cpu | head -15"],
        "services":  ["systemctl --type=service --state=running --no-pager"],
        "general":   ["uname -a", "uptime", "free -h", "df -h"],
    }

    def domain(self) -> str:
        return "SystemDiagnostics"

    def _generate_plan(self, intent: str, context: Optional[str]) -> List[str]:
        lower = intent.lower()

        # Keyword shortcut → predefined safe diagnostic set
        for category, commands in self.DIAGNOSTIC_COMMANDS.items():
            if re.search(category, lower):
                return commands

        # LLM fallback for complex or composite diagnostics
        if self.llm.is_available():
            prompt = (
                f"You are a Linux diagnostics expert.\n"
                f"Task: {intent}\n\n"
                f"Generate READ-ONLY diagnostic commands (no system modifications).\n"
                f"Output one command per line, no explanations.\n"
                f"Commands:"
            )
            try:
                response = self.llm._call_llm(prompt)
                if response:
                    plan = self.llm._extract_plan(response, max_steps=8)
                    if plan:
                        # Safety gate: keep only commands classified as SAFE
                        from .ainux_safety import classify_action, ActionClass
                        safe_plan = [
                            c for c in plan
                            if classify_action(c) == ActionClass.SAFE
                        ]
                        if safe_plan:
                            return safe_plan
            except Exception as e:
                logger.warning(f"LLM plan generation failed: {e}")

        return self.DIAGNOSTIC_COMMANDS["general"]


# ---------------------------------------------------------------------------
# Agent registry and dispatcher
# ---------------------------------------------------------------------------

class AgentDispatcher:
    """Routes user intents to the appropriate specialised agent."""

    INTENT_PATTERNS: Dict[str, List[str]] = {
        "package": [
            r"\binstall\b", r"\bupgrade\b", r"\bupdate\b",
            r"\bapt\b", r"\bpip\b", r"\bnpm\b", r"\bremove package\b",
        ],
        "file": [
            r"\bfile\b", r"\bfolder\b", r"\bdirector\b",
            r"\bcreate\b", r"\brename\b", r"\bcopy\b",
            r"\bmove file\b", r"\blist files\b", r"\bfind files\b",
        ],
        "diagnostics": [
            r"\bdiagnos\b", r"\bmonitor\b", r"\bhealth\b",
            r"\bperformance\b", r"\bcpu\b", r"\bmemory\b",
            r"\bdisk\b", r"\blogs?\b", r"\bprocesses\b",
            r"\bnetwork stat\b", r"\bservices\b",
        ],
    }

    def __init__(self, llm: LocalLLMRuntime, safety: RiskScoringChecker):
        self.agents = {
            "package":     PackageManagementAgent(llm, safety),
            "file":        FileOperationsAgent(llm, safety),
            "diagnostics": SystemDiagnosticsAgent(llm, safety),
        }

    def dispatch(
        self,
        intent: str,
        context: Optional[str] = None,
        confirm_callback=None,
    ) -> Tuple[str, AgentResult]:
        """
        Route to the best-matching agent. Returns (agent_name, AgentResult).
        Defaults to 'diagnostics' for unrecognised intents (safer than 'file').
        """
        lower = intent.lower()

        scores: Dict[str, int] = {k: 0 for k in self.INTENT_PATTERNS}
        for agent_name, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, lower):
                    scores[agent_name] += 1

        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0:
            best = "diagnostics"   # safer default for unrecognised intents

        agent = self.agents[best]
        result = agent.run(intent, context=context, confirm_callback=confirm_callback)
        return best, result