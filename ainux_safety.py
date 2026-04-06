"""
AInux Safety Verification Layer
Implements the four-tier risk scoring framework described in paper Section III.E.

Each command is classified into one of four tiers and assigned a safety score:
    SAFE        1.0  — read-only queries, no side effects
    REVERSIBLE  0.7  — operations that can be undone (installs, copies)
    RISKY       0.3  — operations with lasting effects (permissions, deletes)
    DANGEROUS   0.0  — hard-blocked; catastrophic or irreversible system damage

Plan-level safety score (Eq. 2 in paper):
    Safety(A, S) = prod_{i=1}^{|A|}  SafetyScore(a_i, S_{i-1})

Individual action safety score (Eq. 3 in paper):
    1.0  if a in A_safe
    0.7  if a in A_reversible
    0.3  if a in A_risky
    0.0  if a in A_dangerous

Plans with Safety(A, S) < 0.5 require explicit user confirmation before execution.
Commands classified as DANGEROUS (score = 0.0) are hard-blocked and cannot be
executed without an explicit out-of-band operator override (direct CLI invocation).

Pattern counts (verified against paper Section VII):
    Dangerous   : 22 patterns
    Risky       :  9 patterns
    Reversible  :  8 patterns
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
    Partial observation of the system state used by the safety checker.
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
        import getpass
        import os
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
# Counts: 22 DANGEROUS, 9 RISKY, 8 REVERSIBLE  (matches paper Section VII)
# Evaluated top-to-bottom; first match wins.
# ---------------------------------------------------------------------------

_CLASSIFICATION_RULES: List[Tuple[str, ActionClass]] = [

    # -----------------------------------------------------------------------
    # DANGEROUS (score 0.0) — 22 patterns
    # Hard-blocked; represent catastrophic or irreversible operations.
    # -----------------------------------------------------------------------

    # 1  Recursive root delete
    (r"rm\s+-rf\s+/(?:\s|$)",               ActionClass.DANGEROUS),
    # 2  Recursive wildcard delete
    (r"rm\s+-rf\s+\*",                       ActionClass.DANGEROUS),
    # 3  DOS format command
    (r"^\s*format(?:\.com)?\b",              ActionClass.DANGEROUS),
    # 4  Partition editor (interactive, destructive)
    (r"\bfdisk\b",                           ActionClass.DANGEROUS),
    # 5  Filesystem creation (overwrites partition data)
    (r"\bmkfs\b",                            ActionClass.DANGEROUS),
    # 6  Raw disk read/write
    (r"\bdd\s+if=",                          ActionClass.DANGEROUS),
    # 7  Fork bomb
    (r":\(\)\s*\{.*:\|:.*\}",               ActionClass.DANGEROUS),
    # 8  System shutdown
    (r"\bshutdown\b",                        ActionClass.DANGEROUS),
    # 9  Reboot
    (r"\breboot\b",                          ActionClass.DANGEROUS),
    # 10 Power off
    (r"\bpoweroff\b",                        ActionClass.DANGEROUS),
    # 11 Halt
    (r"\bhalt\b",                            ActionClass.DANGEROUS),
    # 12 SysV runlevel 0 or 6 (halt/reboot)
    (r"\binit\s+[06]\b",                     ActionClass.DANGEROUS),
    # 13 Kill all processes
    (r"\bkillall\b",                         ActionClass.DANGEROUS),
    # 14 Remote code execution via wget pipe
    (r"wget\s+.*\|\s*(bash|sh)\b",          ActionClass.DANGEROUS),
    # 15 Remote code execution via curl pipe
    (r"curl\s+.*\|\s*(bash|sh)\b",          ActionClass.DANGEROUS),
    # 16 Secure file wipe (unrecoverable)
    (r"\bshred\b",                           ActionClass.DANGEROUS),
    # 17 Interactive partition editor
    (r"\bparted\b",                          ActionClass.DANGEROUS),
    # 18 Direct raw disk write (e.g. echo > /dev/sda)
    (r">\s*/dev/sd[a-z]\b",                 ActionClass.DANGEROUS),
    # 19 Base64-encoded payload execution (obfuscated code injection)
    (r"eval\s+.*base64",                     ActionClass.DANGEROUS),
    # 20 Netcat reverse shell
    (r"nc\s+.*-e\s+/bin/(bash|sh)\b",       ActionClass.DANGEROUS),
    # 21 Setuid bit (privilege escalation)
    (r"chmod\s+[+]s\b",                     ActionClass.DANGEROUS),
    # 22 Process substitution execution (bash <(curl ...) attack vector)
    (r"bash\s+<\(",                          ActionClass.DANGEROUS),

    # -----------------------------------------------------------------------
    # RISKY (score 0.3) — 9 patterns
    # Require explicit YES confirmation; lasting but non-catastrophic effects.
    # -----------------------------------------------------------------------

    # 1  Recursive forced delete of a named path
    (r"rm\s+-rf\s+\S+",                      ActionClass.RISKY),
    # 2  Delete a file (with or without -f)
    (r"rm\s+(-f\s+)?\S+",                    ActionClass.RISKY),
    # 3  Remove directory
    (r"\brmdir\s+",                          ActionClass.RISKY),
    # 4  Change file permissions
    (r"\bchmod\s+[0-7]{3,4}",               ActionClass.RISKY),
    # 5  Change file ownership
    (r"\bchown\b",                           ActionClass.RISKY),
    # 6  Remove installed package
    (r"\bapt(-get)?\s+remove\b",             ActionClass.RISKY),
    # 7  Uninstall Python package
    (r"\bpip\s+uninstall\b",                 ActionClass.RISKY),
    # 8  Stop or disable a system service
    (r"\bsystemctl\s+(stop|disable)\b",      ActionClass.RISKY),
    # 9  Force-kill a process
    (r"\bkill\s+-9\b",                       ActionClass.RISKY),

    # -----------------------------------------------------------------------
    # REVERSIBLE (score 0.7) — 8 patterns
    # Show warning and proceed; operations that can be undone.
    # -----------------------------------------------------------------------

    # 1  Move file or directory
    (r"\bmv\s+\S+\s+\S+",                   ActionClass.REVERSIBLE),
    # 2  Copy file or directory
    (r"\bcp\s+\S+\s+\S+",                   ActionClass.REVERSIBLE),
    # 3  Install apt/apt-get package
    (r"\bapt(-get)?\s+install\b",            ActionClass.REVERSIBLE),
    # 4  Install yum package
    (r"\byum\s+install\b",                   ActionClass.REVERSIBLE),
    # 5  Install Python package
    (r"\bpip\s+install\b",                   ActionClass.REVERSIBLE),
    # 6  Start a system service
    (r"\bsystemctl\s+start\b",              ActionClass.REVERSIBLE),
    # 7  Enable a system service at boot
    (r"\bsystemctl\s+enable\b",             ActionClass.REVERSIBLE),
    # 8  Schedule or modify cron jobs
    (r"\bcron\b|\bcrontab\b",               ActionClass.REVERSIBLE),

    # -----------------------------------------------------------------------
    # SAFE (score 1.0) — default
    # All commands not matched above are classified SAFE.
    # -----------------------------------------------------------------------
]


def classify_action(command: str) -> ActionClass:
    """Classify a single shell command into an ActionClass.

    Rules are evaluated top-to-bottom; first match wins.
    Commands not matching any rule default to ActionClass.SAFE.
    """
    cmd = command.strip().lower()
    for pattern, cls in _CLASSIFICATION_RULES:
        if re.search(pattern, cmd):
            return cls
    return ActionClass.SAFE


# ---------------------------------------------------------------------------
# Four-tier Risk Scoring Checker
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    action_class: ActionClass
    score: float
    confirmation: ConfirmationLevel
    reason: str
    affected_components: List[str] = field(default_factory=list)
    reversible: bool = True


class RiskScoringChecker:
    """
    Implements the four-tier risk scoring framework described in paper Section III.E.

    For a single command, returns a ValidationResult with:
        - tier classification (SAFE / REVERSIBLE / RISKY / DANGEROUS)
        - numeric risk score (1.0 / 0.7 / 0.3 / 0.0)
        - required confirmation level (NONE / WARN / CONFIRM / BLOCK)

    For a sequence of commands (a plan), computes the plan-level safety score
    as the product of individual scores (Eq. 2):
        Safety(A, S) = prod_{i=1}^{|A|} SafetyScore(a_i, S_{i-1})

    Plans with Safety(A, S) < 0.5 require explicit user confirmation.
    DANGEROUS commands (score = 0.0) are always hard-blocked.
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
        Validate a single command and return a ValidationResult.

        Implements the individual action safety score (Eq. 3):
            1.0  SAFE       — execute immediately
            0.7  REVERSIBLE — show warning, proceed
            0.3  RISKY      — require explicit YES confirmation
            0.0  DANGEROUS  — hard block, no override
        """
        cls = classify_action(command)
        score = SAFETY_SCORES[cls]
        affected = self._identify_affected_components(command)
        reversible = cls in (ActionClass.SAFE, ActionClass.REVERSIBLE)

        if cls == ActionClass.DANGEROUS:
            confirmation = ConfirmationLevel.BLOCK
            reason = (
                "Command classified as DANGEROUS (score=0.0). "
                "Execution permanently blocked. Use the terminal directly if this is intentional."
            )
        elif cls == ActionClass.RISKY:
            confirmation = ConfirmationLevel.CONFIRM
            reason = "Command classified as RISKY (score=0.3). Requires explicit YES confirmation."
        elif cls == ActionClass.REVERSIBLE:
            confirmation = ConfirmationLevel.WARN
            reason = "Command classified as REVERSIBLE (score=0.7). Review before proceeding."
        else:
            confirmation = ConfirmationLevel.NONE
            reason = "Command classified as SAFE (score=1.0)."

        # Escalate REVERSIBLE to CONFIRM when running as root
        if self.state.is_root and cls == ActionClass.REVERSIBLE:
            confirmation = ConfirmationLevel.CONFIRM
            reason += " Escalated to CONFIRM: running as root."

        return ValidationResult(
            action_class=cls,
            score=score,
            confirmation=confirmation,
            reason=reason,
            affected_components=affected,
            reversible=reversible,
        )

    # ------------------------------------------------------------------
    # Plan validation — Eq. 2
    # ------------------------------------------------------------------

    def validate_plan(self, commands: List[str]) -> Tuple[float, List[ValidationResult]]:
        """
        Compute the plan-level safety score (Eq. 2):
            Safety(A, S) = product of individual SafetyScore(a_i, S_{i-1})

        Returns (plan_safety_score, per_command_results).
        A plan score of 0.0 means at least one command is DANGEROUS.
        """
        results = [self.validate_command(cmd) for cmd in commands]
        plan_score = 1.0
        for r in results:
            plan_score *= r.score
        return plan_score, results

    def requires_confirmation(self, plan_score: float) -> bool:
        """Return True if the plan score falls below the confirmation threshold (0.5)."""
        return plan_score < CONFIRMATION_THRESHOLD

    # ------------------------------------------------------------------
    # Plan scoring for the planner — Eq. 4
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
        Composite plan score used during plan selection (Eq. 4):
            Score(plan) = α * P(success) + β * Safety − γ * Cost

        Default weights: α=0.4, β=0.5, γ=0.1 (as specified in paper Section IV.B.2).
        Safety term dominates (β=0.5) to prioritise safe plans over fast ones.
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
            "filesystem":  [r"\brm\b", r"\bmv\b", r"\bcp\b", r"\bmkdir\b", r"\btouch\b"],
            "packages":    [r"\bapt\b", r"\byum\b", r"\bpip\b", r"\bnpm\b", r"\bpacman\b"],
            "processes":   [r"\bkill\b", r"\bkillall\b", r"\bsystemctl\b", r"\bservice\b"],
            "network":     [r"\bifconfig\b", r"\bip\b", r"\biptables\b", r"\bfirewall\b"],
            "permissions": [r"\bchmod\b", r"\bchown\b", r"\bsudo\b"],
            "users":       [r"\buseradd\b", r"\buserdel\b", r"\bpasswd\b", r"\bsu\b"],
            "disk":        [r"\bmkfs\b", r"\bfdisk\b", r"\bparted\b", r"\bdd\b"],
        }

        for component, patterns in component_patterns.items():
            for pattern in patterns:
                if re.search(pattern, cmd):
                    affected.append(component)
                    break

        return affected


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# Earlier versions of the codebase referred to this class as MDPSafetyChecker.
# That name is retained as an alias so existing imports continue to work.
# The canonical name is RiskScoringChecker, matching the paper's terminology.
# ---------------------------------------------------------------------------
MDPSafetyChecker = RiskScoringChecker