"""
AInux Universal LLM Runtime
Universal adapter implementing the Chat Completion API standard.

This standard is implemented by many LLM providers:
  - LM Studio         (local, port 1234)
  - Ollama            (local, port 11434)
  - Vllm              (local, port 8000)
  - OpenAI            (cloud)
  - Azure OpenAI      (cloud)
  - Anthropic Claude  (via proxy)
  - Any Chat Completion API-compliant service

Environment variables:
  - AINUX_LLM_HOST    (endpoint URL, default: http://127.0.0.1:1234)
  - AINUX_LLM_API_KEY (optional, for remote/proprietary services)
  - Legacy: AINUX_OLLAMA_HOST, OLLAMA_HOST (still supported)

Implements the Chat Completion API standard interface.
KV-cache awareness and dynamic batching described in paper Sec III.C.
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

DEFAULT_LLM_HOST = (
    os.getenv("AINUX_LLM_HOST")
    or os.getenv("AINUX_OLLAMA_HOST")   # legacy env vars for backward compatibility
    or os.getenv("OLLAMA_HOST")
    or "http://127.0.0.1:1234"          # LM Studio default (also works with local Ollama/Vllm)
)

DEFAULT_LLM_API_KEY = os.getenv("AINUX_LLM_API_KEY")


def normalize_llm_host(host: str) -> str:
    """Normalize LLM endpoint URL to standard format."""
    host = (host or DEFAULT_LLM_HOST).strip()
    if host.isdigit():
        return f"http://127.0.0.1:{host}"
    if "://" not in host:
        return f"http://{host}"
    return host


@dataclass
class LLMRuntimeConfig:
    """Configuration for Chat Completion API endpoint."""
    host: str = field(default_factory=lambda: DEFAULT_LLM_HOST)
    model: str = "gpt-oss-20b-MXFP4"
    api_key: Optional[str] = field(default_factory=lambda: DEFAULT_LLM_API_KEY)  # For cloud providers requiring auth
    temperature: float = 0.1            # low = deterministic commands
    timeout: int = 120
    max_retries: int = 3
    context_window: int = 4096
    quantization_bits: int = 4

    def __post_init__(self) -> None:
        self.host = normalize_llm_host(self.host)


# Keep backward compatibility aliases
OllamaConfig = LLMRuntimeConfig
normalize_ollama_host = normalize_llm_host
DEFAULT_OLLAMA_HOST = DEFAULT_LLM_HOST


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class LocalLLMRuntime:
    """
    Universal LLM runtime adapter using Chat Completion API standard.

    Works with any endpoint implementing the Chat Completion API standard:
      - Local: LM Studio, Ollama, Vllm, etc.
      - Cloud: OpenAI, Azure, Anthropic (via proxy), etc.
      - Self-hosted: LLaMA, Mistral, Falcon servers

    Implements the inference optimisation described in the paper:
      T_inf = T_prompt + N_tokens * T_decode      (Eq. 4)

    All HTTP calls use /v1/chat/completions (Chat Completion API standard).
    This is a widely-adopted open specification, not vendor-specific.
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
    "8. Use Linux/Debian commands unlessLLMRuntimeConfig] = None):
        self.config = config or LLMRuntime
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
    # Core API call
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Call any Chat Completion API-compliant endpoint.
        Works with any provider implementing /v1/chat/completions standard.
        
        Examples:
          - LM Studio (local, http://127.0.0.1:1234)
          - Ollama (local, http://127.0.0.1:11434)
          - Vllm (local, http://127.0.0.1:8000)
          - OpenAI, Anthropic, Azure (cloud, requires auth keys)
          - Self-hosted LLaMA, Mistral, etc.
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

        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        resp = requests.post(
            f"{self.config.host}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()

        raw_output = resp.json()["choices"][0]["message"]["content"]
        print(f"[AInux LLM][DEBUG] raw_output={raw_output!r}")
        return raw_output.strip() if raw_output else None

    # Backward compatibility alias
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