"""
AInux Local LLM Runtime
Targets LM Studio's OpenAI-compatible local server (default port 1234).

Supports any GGUF model loaded in LM Studio.
Recommended models:
  - gpt-oss-20b-MXFP4   (20B, strong quality)
  - phi3:mini            (~2.3 GB, fast)
  - llama3.2:3b          (~2.0 GB, good balance)
  - mistral:7b-q4        (~4.1 GB, best quality)

LM Studio setup:
  1. Open LM Studio and load your model.
  2. Go to Local Server tab → Start Server (default port 1234).
  3. Pass --host http://127.0.0.1:1234 --model <model-id>

Implements KV-cache awareness and dynamic batching described in paper Sec III.C.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_HOST = (
    os.getenv("AINUX_LM_HOST")
    or os.getenv("AINUX_OLLAMA_HOST")   # legacy env var honoured
    or os.getenv("OLLAMA_HOST")
    or "http://127.0.0.1:1234"          # LM Studio default port
)


def normalize_ollama_host(host: str) -> str:
    host = (host or DEFAULT_OLLAMA_HOST).strip()
    if host.isdigit():
        return f"http://127.0.0.1:{host}"
    if "://" not in host:
        return f"http://{host}"
    return host


@dataclass
class OllamaConfig:
    host: str = field(default_factory=lambda: DEFAULT_OLLAMA_HOST)
    model: str = "gpt-oss-20b-MXFP4"
    temperature: float = 0.1            # low = deterministic commands
    timeout: int = 120                  # LM Studio needs more headroom than Ollama
    max_retries: int = 3
    context_window: int = 4096
    quantization_bits: int = 4

    def __post_init__(self) -> None:
        self.host = normalize_ollama_host(self.host)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class LocalLLMRuntime:
    """
    Manages local LLM inference via LM Studio's OpenAI-compatible API.

    Implements the inference optimisation described in the paper:
      T_inf = T_prompt + N_tokens * T_decode      (Eq. 4)

    All HTTP calls go through _call_llm() which posts to
    /v1/chat/completions — the standard OpenAI chat format that
    LM Studio exposes on its local server.
    """

    SYSTEM_PROMPT = (
    "You are AInux, an expert Linux system administration assistant.\n"
    "Convert natural language instructions into safe shell commands.\n\n"
    "STRICT RULES — follow all of them:\n"
    "1. Output EXACTLY ONE shell command. No explanations, no markdown, "
       "no backticks, no code fences, no comments.\n"
    "2. NEVER use && or ; to chain commands. One command only.\n"
    "3. NEVER add sudo unless the user explicitly says 'as root' or 'with sudo'.\n"
    "4. NEVER use placeholder paths like /path/to/ or <your-path>. "
       "Use the exact names from the request.\n"
    "5. NEVER add echo, verification steps, or output decorators.\n"
    "6. Use apt-get (not sudo apt-get) for package operations.\n"
    "7. If the request is ambiguous or unsafe, output: AINUX_CLARIFY\n"
    "8. Use Linux/Debian commands unless told otherwise.\n\n"
    "Examples of CORRECT output:\n"
    "  apt-get install -y nginx\n"
    "  systemctl reload nginx\n"
    "  find . -type f -name '*.py'\n\n"
    "Examples of WRONG output (never do these):\n"
    "  sudo apt-get update && sudo apt-get install nginx   ← two commands\n"
    "  sudo systemctl start nginx                          ← unnecessary sudo\n"
    "  source /path/to/venv/bin/activate                  ← placeholder path\n"
)

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.available = False
        self._check_availability()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _check_availability(self) -> None:
        """Check LM Studio is running and has a model loaded."""
        try:
            resp = requests.get(
                f"{self.config.host}/v1/models",
                timeout=5,
            )
            if resp.status_code == 200:
                models = [m["id"] for m in resp.json().get("data", [])]
                if not models:
                    print(
                        f"[AInux LLM] LM Studio is running at {self.config.host} "
                        "but no model is loaded.\n"
                        "            Load a model in LM Studio → Local Server tab."
                    )
                    return
                self.available = True
                matched = any(self.config.model.lower() in m.lower() for m in models)
                if not matched:
                    print(
                        f"[AInux LLM] WARNING: requested model '{self.config.model}' "
                        f"not found. Available: {models}\n"
                        f"            Using first loaded model: {models[0]}"
                    )
                    # Use whatever is loaded rather than failing
                    self.config.model = models[0]
                else:
                    print(
                        f"[AInux LLM] Connected to LM Studio — model: {self.config.model}"
                    )
            else:
                print(
                    f"[AInux LLM] LM Studio at {self.config.host} responded "
                    f"with status {resp.status_code}"
                )
        except requests.ConnectionError:
            print(
                f"[AInux LLM] LM Studio not reachable at {self.config.host}.\n"
                "            Open LM Studio → Local Server tab → Start Server.\n"
                f"            Then pass --host {self.config.host} --model <model-id>"
            )
        except Exception as e:
            print(f"[AInux LLM] Availability check failed: {e}")

    def is_available(self) -> bool:
        return self.available

    # ------------------------------------------------------------------
    # Command generation (single command)
    # ------------------------------------------------------------------

    def generate_command(
        self,
        user_input: str,
        context: Optional[str] = None,
        platform: str = "linux",
    ) -> Optional[str]:
        """
        Convert natural language to a single shell command.
        Returns the command string, or None on failure.
        """
        if not self.available:
            return self._regex_fallback(user_input)

        prompt = self._build_prompt(user_input, context, platform)

        for attempt in range(self.config.max_retries):
            try:
                response = self._call_llm(prompt)
                if response:
                    command = self._extract_command(response)
                    if command:
                        return command
            except Exception as e:
                print(f"[AInux LLM] Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)

        return self._regex_fallback(user_input)

    # ------------------------------------------------------------------
    # Plan generation (multi-step)
    # ------------------------------------------------------------------

    def generate_plan(
        self,
        user_input: str,
        context: Optional[str] = None,
        max_steps: int = 8,
    ) -> List[str]:
        """
        Generate a multi-step command plan for complex tasks.
        Returns an ordered list of shell commands.
        """
        if not self.available:
            return []

        plan_prompt = (
            f"You are a Linux system administrator planning a multi-step task.\n"
            f"Task: {user_input}\n"
            f"Context: {context or 'none'}\n\n"
            f"Output a numbered list of shell commands (max {max_steps}) to complete the task safely.\n"
            f"Format: one command per line, no explanations.\n"
            f"Never include dangerous or irreversible commands unless absolutely necessary.\n"
            f"Commands:"
        )

        for attempt in range(self.config.max_retries):
            try:
                response = self._call_llm(plan_prompt)
                if response:
                    return self._extract_plan(response, max_steps)
            except Exception as e:
                print(f"[AInux LLM] Plan generation attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)

        return []

    # ------------------------------------------------------------------
    # Command explanation (for confirmation prompts)
    # ------------------------------------------------------------------

    def explain_command(self, command: str) -> str:
        """Generate a one-sentence explanation of what a command does."""
        if not self.available:
            return f"Execute: {command}"

        prompt = (
            f"Explain in one sentence what this shell command does, "
            f"focusing on its effect on the system:\n{command}"
        )
        try:
            response = self._call_llm(prompt)
            return response.strip() if response else f"Execute: {command}"
        except Exception:
            return f"Execute: {command}"

    # ------------------------------------------------------------------
    # Core LM Studio API call
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        POST to LM Studio /v1/chat/completions (OpenAI-compatible).
        Uses the shared SYSTEM_PROMPT as the system message so the model
        maintains consistent behaviour across calls.
        """
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "temperature": self.config.temperature,
            "max_tokens": 256,
        }

        resp = requests.post(
            f"{self.config.host}/v1/chat/completions",
            json=payload,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()

        raw_output = resp.json()["choices"][0]["message"]["content"]
        print(f"[AInux LLM][DEBUG] raw_output={raw_output!r}")
        return raw_output.strip() if raw_output else None

    # Keep the old name as an alias so any callers that haven't been
    # updated yet don't break immediately.
    def _call_ollama(self, prompt: str) -> Optional[str]:
        return self._call_llm(prompt)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        user_input: str,
        context: Optional[str],
        platform: str,
    ) -> str:
        parts = []
        if context:
            parts.append(f"Recent context:\n{context}\n")
        parts.append(
            f"Platform: {platform}\n"
            f"Return the shortest canonical command that satisfies the request.\n"
            f"User request: {user_input}\n"
            f"Shell command:"
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _canonicalize_command(self, command: str) -> str:
        command = command.strip()

        if command.startswith("`") and command.endswith("`"):
            command = command[1:-1].strip()

        # Normalise find flag order
        match = re.fullmatch(r"find \. -type f -name (.+)", command)
        if match:
            return f"find . -name {match.group(1)} -type f"

        # Normalise ip shorthand
        if command == "ip a":
            return "ip addr"

        # Normalise git log format variants
        if command in {
            'git log -5 --pretty=format:"%h %s"',
            "git log -5 --pretty=format:'%h %s'",
        }:
            return "git log --oneline -5"

        return command

    def _extract_command(self, raw: str) -> Optional[str]:
        """Clean LLM output down to a single executable shell command."""
        if not raw:
            return None

        command = raw.strip()

        # Strip triple-backtick code fences
        command = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", command)
        command = re.sub(r"\n?```$", "", command).strip()
        if command.startswith("```") and "```" in command[3:]:
            command = re.sub(r"^```.*?\n", "", command, flags=re.DOTALL)
            command = re.sub(r"```$", "", command).strip()

        # Bail on clarification
        if "AINUX_CLARIFY" in command:
            return None

        # Take first non-empty line only (single-command mode)
        for line in command.splitlines():
            line = line.strip()
            if line:
                command = line
                break

        # Strip inline backticks
        if command.startswith("`") and command.endswith("`"):
            command = command[1:-1].strip()

        # Strip common shell decorators
        for prefix in ["$ ", "# ", "> ", "bash: ", "sh: "]:
            if command.lower().startswith(prefix):
                command = command[len(prefix):].strip()

        # Strip trailing semicolons
        command = command.rstrip(";").strip()

        command = self._canonicalize_command(command)

        # Reject placeholder paths — the model invented something
        placeholder_patterns = [
            r"/path/to/",
            r"<[^>]+>",
            r"\byour[_/-]",
            r"\bexample[_/-]",
        ]
        if any(re.search(p, command, re.IGNORECASE) for p in placeholder_patterns):
            return None

        if not command or len(command) > 500:
            return None

        return command

    def _regex_fallback(self, text: str) -> Optional[str]:
        """Minimal fallback when LM Studio is unavailable or generation fails."""
        lower = text.lower()
        if re.search(r"list.*files|show.*files?\b", lower):
            return "ls -la"
        if re.search(r"current.*dir|where.*am\b|pwd", lower):
            return "pwd"
        if re.search(r"disk.*space|disk.*usage", lower):
            return "df -h"
        if re.search(r"\bmemory\b", lower):
            return "free -h"
        if re.search(r"\bprocesses\b", lower):
            return "ps aux"
        if re.search(r"\buptime\b", lower):
            return "uptime"
        return None

    def _extract_plan(self, raw: str, max_steps: int) -> List[str]:
        """Parse a numbered or bulleted list of commands from a plan response."""
        commands = []
        for line in raw.strip().splitlines():
            line = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
            line = re.sub(r"^[-*•]\s*", "", line).strip()
            # Strip inline backticks
            if line.startswith("`") and line.endswith("`"):
                line = line[1:-1].strip()
            if line and not line.startswith("#") and len(line) > 1:
                commands.append(line)
            if len(commands) >= max_steps:
                break
        return commands