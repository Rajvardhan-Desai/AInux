"""
AInux Safety Verification Layer
Models command safety as a Markov Decision Process.

State space S: system state (files, processes, network, permissions)
Action space A: classified into safe / reversible / risky / dangerous
Transition T: S x A -> S  (deterministic approximation)
Reward R: S x A -> R      (negative for unsafe actions)

Safety score for an action sequence A in state S:
    Safety(A, S) = prod_{i=1}^{|A|}  SafetyScore(a_i, S_{i-1})   (Eq. 10)

Individual action safety (Eq. 11):
    1.0  if a in A_safe
    0.7  if a in A_reversible
    0.3  if a in A_risky
    0.0  if a in A_dangerous

Actions with Safety(A, S) < 0.5 require explicit user confirmation.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ActionClass(Enum):
    SAFE        = "safe"
    REVERSIBLE  = "reversible"
    RISKY       = "risky"
    DANGEROUS   = "dangerous"

class ConfirmationLevel(Enum):
    NONE    = "none"
    WARN    = "warn"        # show warning, proceed
    CONFIRM = "confirm"     # require YES
    BLOCK   = "block"       # hard block, no override

SAFETY_SCORES: Dict[ActionClass, float] = {
    ActionClass.SAFE:       1.0,
    ActionClass.REVERSIBLE: 0.7,
    ActionClass.RISKY:      0.3,
    ActionClass.DANGEROUS:  0.0,
}

CONFIRMATION_THRESHOLD = 0.5   # Safety(A, S) < 0.5 → require confirmation


# ---------------------------------------------------------------------------
# System State
# ---------------------------------------------------------------------------

@dataclass
class SystemState:
    """
    Partial observation of the system state used by the MDP.
    Populated lazily via subprocess calls.
    """
    cwd: str = ""
    user: str = ""
    is_root: bool = False
    has_sudo: bool = False
    mounted_filesystems: List[str] = field(default_factory=list)

    @classmethod
    def observe(cls) -> "SystemState":
        """Collect current system state."""
        import getpass, os, platform
        state = cls()
        state.cwd = os.getcwd()
        try:
            state.user = getpass.getuser()
            state.is_root = (state.user == "root")
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["sudo", "-n", "true"],
                capture_output=True, timeout=2
            )
            state.has_sudo = (result.returncode == 0)
        except Exception:
            pass
        return state


# ---------------------------------------------------------------------------
# Classifier rules  (pattern -> ActionClass)
# ---------------------------------------------------------------------------

# Tuples of (regex_pattern, ActionClass)
# Evaluated top-to-bottom; first match wins.
_CLASSIFICATION_RULES: List[Tuple[str, ActionClass]] = [

    # --- Dangerous (score 0.0, always block) ---
    (r"rm\s+-rf\s+/(?:\s|$)",               ActionClass.DANGEROUS),  # rm -rf /
    (r"rm\s+-rf\s+\*",                       ActionClass.DANGEROUS),  # rm -rf *
    (r"^\s*format(?:\.com)?\b",              ActionClass.DANGEROUS),
    (r"\bfdisk\b",                           ActionClass.DANGEROUS),
    (r"\bmkfs\b",                            ActionClass.DANGEROUS),
    (r"\bdd\s+if=",                          ActionClass.DANGEROUS),
    (r":\(\)\s*\{.*:\|:.*\}",               ActionClass.DANGEROUS),  # fork bomb
    (r"\bshutdown\b",                        ActionClass.DANGEROUS),
    (r"\breboot\b",                          ActionClass.DANGEROUS),
    (r"\bpoweroff\b",                        ActionClass.DANGEROUS),
    (r"\bhalt\b",                            ActionClass.DANGEROUS),
    (r"\binit\s+[06]\b",                     ActionClass.DANGEROUS),
    (r"\bkillall\b",                         ActionClass.DANGEROUS),
    (r"wget\s+.*\|\s*(bash|sh)",             ActionClass.DANGEROUS),  # remote exec
    (r"curl\s+.*\|\s*(bash|sh)",             ActionClass.DANGEROUS),

    # --- Risky (score 0.3) ---
    (r"rm\s+-rf\s+\S+",                      ActionClass.RISKY),      # rm -rf <path>
    (r"rm\s+(-f\s+)?\S+",                    ActionClass.RISKY),      # rm file
    (r"\bdel\s+\S+",                         ActionClass.RISKY),      # Windows del
    (r"\brmdir\s+",                          ActionClass.RISKY),
    (r"\bchmod\s+[0-7]{3,4}",               ActionClass.RISKY),
    (r"\bchown\b",                           ActionClass.RISKY),
    (r"\bapt(-get)?\s+remove\b",             ActionClass.RISKY),
    (r"\byum\s+remove\b",                    ActionClass.RISKY),
    (r"\bpip\s+uninstall\b",                 ActionClass.RISKY),
    (r"\bsystemctl\s+(stop|disable)\b",      ActionClass.RISKY),
    (r"\bkill\s+-9\b",                       ActionClass.RISKY),

    # --- Reversible (score 0.7) ---
    (r"\bmv\s+\S+\s+\S+",                   ActionClass.REVERSIBLE),  # mv (can undo)
    (r"\bcp\s+\S+\s+\S+",                   ActionClass.REVERSIBLE),  # cp
    (r"\bapt(-get)?\s+install\b",            ActionClass.REVERSIBLE),
    (r"\byum\s+install\b",                   ActionClass.REVERSIBLE),
    (r"\bpip\s+install\b",                   ActionClass.REVERSIBLE),
    (r"\bsystemctl\s+start\b",              ActionClass.REVERSIBLE),
    (r"\bsystemctl\s+enable\b",             ActionClass.REVERSIBLE),
    (r"\bnano\b|\bvim\b|\bvi\b|\bemacs\b",  ActionClass.REVERSIBLE),  # editors
    (r"\bcron\b|\bcrontab\b",               ActionClass.REVERSIBLE),

    # --- Safe (score 1.0) — everything else defaults here ---
]


def classify_action(command: str) -> ActionClass:
    """Classify a single shell command into an ActionClass."""
    cmd = command.strip().lower()
    for pattern, cls in _CLASSIFICATION_RULES:
        if re.search(pattern, cmd):
            return cls
    return ActionClass.SAFE


# ---------------------------------------------------------------------------
# MDP Safety Checker
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    action_class: ActionClass
    score: float
    confirmation: ConfirmationLevel
    reason: str
    affected_components: List[str] = field(default_factory=list)
    reversible: bool = True


class MDPSafetyChecker:
    """
    Implements the MDP safety framework described in the paper.

    For a single command:
        safety_score = SafetyScore(a, S)

    For a sequence of commands (a plan):
        Safety(A, S) = prod_{i=1}^{|A|} SafetyScore(a_i, S_{i-1})
    """

    def __init__(self):
        self._state: Optional[SystemState] = None

    @property
    def state(self) -> SystemState:
        if self._state is None:
            self._state = SystemState.observe()
        return self._state

    def refresh_state(self) -> None:
        self._state = SystemState.observe()

    # ------------------------------------------------------------------
    # Single command validation
    # ------------------------------------------------------------------

    def validate_command(self, command: str) -> ValidationResult:
        """
        Validate a single command. Returns a ValidationResult with score,
        confirmation level, and human-readable reason.
        """
        cls = classify_action(command)
        score = SAFETY_SCORES[cls]
        affected = self._identify_affected_components(command)
        reversible = cls in (ActionClass.SAFE, ActionClass.REVERSIBLE)

        if cls == ActionClass.DANGEROUS:
            confirmation = ConfirmationLevel.BLOCK
            reason = f"Command classified as DANGEROUS (score=0.0). Execution blocked."
        elif cls == ActionClass.RISKY:
            confirmation = ConfirmationLevel.CONFIRM
            reason = f"Command classified as RISKY (score=0.3). Requires explicit confirmation."
        elif cls == ActionClass.REVERSIBLE:
            confirmation = ConfirmationLevel.WARN
            reason = f"Command classified as REVERSIBLE (score=0.7). Potentially impactful."
        else:
            confirmation = ConfirmationLevel.NONE
            reason = f"Command classified as SAFE (score=1.0)."

        # Escalate if running as root
        if self.state.is_root and cls == ActionClass.REVERSIBLE:
            confirmation = ConfirmationLevel.CONFIRM
            reason += " Escalated to CONFIRM because running as root."

        return ValidationResult(
            action_class=cls,
            score=score,
            confirmation=confirmation,
            reason=reason,
            affected_components=affected,
            reversible=reversible,
        )

    # ------------------------------------------------------------------
    # Action sequence (plan) validation   — Equation 10
    # ------------------------------------------------------------------

    def validate_plan(self, commands: List[str]) -> Tuple[float, List[ValidationResult]]:
        """
        Compute Safety(A, S) = product of individual safety scores.
        Returns (plan_safety_score, list_of_per_command_results).
        """
        results = [self.validate_command(cmd) for cmd in commands]
        # Product of individual scores (Eq. 10)
        plan_score = 1.0
        for r in results:
            plan_score *= r.score
        return plan_score, results

    def requires_confirmation(self, plan_score: float) -> bool:
        return plan_score < CONFIRMATION_THRESHOLD

    # ------------------------------------------------------------------
    # Plan scoring for the planner  — Equation 14
    # ------------------------------------------------------------------

    def score_plan(
        self,
        commands: List[str],
        estimated_success_prob: float = 0.9,
        estimated_cost: float = 0.1,
        alpha: float = 0.4,
        beta: float = 0.5,
        gamma: float = 0.1,
    ) -> float:
        """
        Score(plan) = α * P(success) + β * Safety - γ * Cost   (Eq. 14)
        Default weights: α=0.4, β=0.5, γ=0.1 (as specified in paper).
        """
        safety, _ = self.validate_plan(commands)
        return alpha * estimated_success_prob + beta * safety - gamma * estimated_cost

    # ------------------------------------------------------------------
    # Component dependency analysis
    # ------------------------------------------------------------------

    def _identify_affected_components(self, command: str) -> List[str]:
        """Heuristic: identify which system components a command touches."""
        affected = []
        cmd = command.lower()

        component_patterns = {
            "filesystem":    [r"\brm\b", r"\bmv\b", r"\bcp\b", r"\bmkdir\b", r"\btouch\b"],
            "packages":      [r"\bapt\b", r"\byum\b", r"\bpip\b", r"\bnpm\b", r"\bpacman\b"],
            "processes":     [r"\bkill\b", r"\bkillall\b", r"\bsystemctl\b", r"\bservice\b"],
            "network":       [r"\bifconfig\b", r"\bip\b", r"\biptables\b", r"\bfirewall\b"],
            "permissions":   [r"\bchmod\b", r"\bchown\b", r"\bsudo\b"],
            "users":         [r"\buseradd\b", r"\buserdel\b", r"\bpasswd\b", r"\bsu\b"],
            "disk":          [r"\bmkfs\b", r"\bfdisk\b", r"\bparted\b", r"\bdd\b"],
        }

        for component, patterns in component_patterns.items():
            for pattern in patterns:
                if re.search(pattern, cmd):
                    affected.append(component)
                    break

        return affected
