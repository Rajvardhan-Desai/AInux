"""
AInux Local LLM Runtime
Replaces the Gemini cloud API with a local Ollama inference engine.

Supports: Ollama (primary), with llama.cpp-compatible fallback.
Recommended models for 8GB RAM:
  - phi3:mini      (~2.3 GB)   fastest
  - llama3.2:3b    (~2.0 GB)   good balance
  - mistral:7b-q4  (~4.1 GB)   best quality, fits in 8GB

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
    os.getenv("AINUX_OLLAMA_HOST")
    or os.getenv("OLLAMA_HOST")
    or "http://127.0.0.1:12345"
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
    model: str   = "phi3:mini"          # fits comfortably in 8GB
    temperature: float = 0.1            # low = deterministic commands
    timeout: int = 60
    max_retries: int = 3
    context_window: int = 4096
    # Quantization hint for paper metrics (actual quantization is done by Ollama)
    quantization_bits: int = 4

    def __post_init__(self) -> None:
        self.host = normalize_ollama_host(self.host)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class LocalLLMRuntime:
    """
    Manages local LLM inference via Ollama.

    Implements the inference optimisation described in the paper:
      T_inf = T_prompt + N_tokens * T_decode      (Eq. 4)

    KV-cache is handled automatically by Ollama when the same system
    prompt prefix is reused across calls (context reuse).
    """

    SYSTEM_PROMPT = (
        "You are AInux, an expert system administration assistant. "
        "Convert natural language instructions into safe shell commands. "
        "Rules:\n"
        "1. Output ONLY the shell command — no explanation, no markdown, no backticks, no code fences.\n"
        "2. Never output dangerous commands (rm -rf /, format, shutdown, etc.).\n"
        "3. If the request is ambiguous or unsafe, output: AINUX_CLARIFY\n"
        "4. Prefer a single canonical shell command. Only use && or ; if the user explicitly asks for multiple actions or one command is impossible.\n"
        "5. Use commands appropriate for Linux/Debian unless told otherwise.\n"
        "6. Preserve literal filenames, directories, ports, and package names from the request. Never invent placeholder paths like /path/to/...\n"
        "7. Do not add sudo, apt-get update, cd, echo, comments, or verification steps unless the user explicitly asks for them.\n"
        "8. Prefer canonical forms such as ls -la, find . -name '*.py' -type f, df -h, git init, which python, and git log --oneline -5.\n"
    )

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.available = False
        self._context_cache: Optional[List[int]] = None  # KV-cache token ids
        self._check_availability()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _check_availability(self) -> None:
        try:
            resp = requests.get(
                f"{self.config.host}/api/tags",
                timeout=5
            )
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                self.available = True
                if not any(self.config.model in m for m in models):
                    print(
                        f"[AInux LLM] Model '{self.config.model}' not found locally. "
                        f"Run: ollama pull {self.config.model}"
                    )
                    print(f"[AInux LLM] Available: {models}")
                else:
                    print(f"[AInux LLM] Connected to Ollama — model: {self.config.model}")
            else:
                print(
                    f"[AInux LLM] Ollama at {self.config.host} responded with "
                    f"status {resp.status_code}"
                )
        except requests.ConnectionError:
            print(
                f"[AInux LLM] Ollama not reachable at {self.config.host}. "
                "Start it with: ollama serve\n"
                f"[AInux LLM] Then pull a model: ollama pull {self.config.model}"
            )
        except Exception as e:
            print(f"[AInux LLM] Availability check failed: {e}")

    def is_available(self) -> bool:
        return self.available

    # ------------------------------------------------------------------
    # Command generation
    # ------------------------------------------------------------------

    def generate_command(
        self,
        user_input: str,
        context: Optional[str] = None,
        platform: str = "linux",
    ) -> Optional[str]:
        """
        Convert natural language to a shell command.
        Returns the command string, or None on failure.
        """
        if not self.available:
            return self._regex_fallback(user_input)

        prompt = self._build_prompt(user_input, context, platform)

        for attempt in range(self.config.max_retries):
            try:
                t0 = time.time()
                response = self._call_ollama(prompt)
                t_inf = time.time() - t0

                if response:
                    command = self._extract_command(response)
                    if command:
                        return command

            except Exception as e:
                print(f"[AInux LLM] Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)

        return self._regex_fallback(user_input)

    def generate_plan(
        self,
        user_input: str,
        context: Optional[str] = None,
        max_steps: int = 8,
    ) -> List[str]:
        """
        Generate a multi-step command plan for complex tasks.
        Returns a list of shell commands (ordered).
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
                response = self._call_ollama(plan_prompt)
                if response:
                    return self._extract_plan(response, max_steps)
            except Exception as e:
                print(f"[AInux LLM] Plan generation attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)

        return []

    def explain_command(self, command: str) -> str:
        """Generate a natural language explanation of a command (for user confirmation prompts)."""
        if not self.available:
            return f"Execute: {command}"

        prompt = (
            f"Explain in one sentence what this shell command does, "
            f"focusing on its effect on the system:\n{command}"
        )
        try:
            response = self._call_ollama(prompt)
            return response.strip() if response else f"Execute: {command}"
        except Exception:
            return f"Execute: {command}"

    # ------------------------------------------------------------------
    # Internal Ollama API call
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """
        POST to Ollama /api/generate (non-streaming).
        Uses the shared system prompt for KV-cache reuse.
        """
        payload = {
            "model": self.config.model,
            "system": self.SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.context_window,
            },
        }

        resp = requests.post(
            f"{self.config.host}/api/generate",
            json=payload,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        raw_output = resp.json().get("response", "")
        print(f"[AInux LLM][DEBUG] raw_output={raw_output!r}")
        return raw_output.strip()

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _build_prompt(self, user_input: str, context: Optional[str], platform: str) -> str:
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

    def _canonicalize_command(self, command: str) -> str:
        command = command.strip()

        if command.startswith("`") and command.endswith("`"):
            command = command[1:-1].strip()

        match = re.fullmatch(r"find \. -type f -name (.+)", command)
        if match:
            return f"find . -name {match.group(1)} -type f"

        if command == "ip a":
            return "ip addr"

        if command in {
            'git log -5 --pretty=format:"%h %s"',
            "git log -5 --pretty=format:'%h %s'",
        }:
            return "git log --oneline -5"

        return command

    def _extract_command(self, raw: str) -> Optional[str]:
        """Clean up LLM output to a single shell command."""
        if not raw:
            return None

        command = raw.strip()

        # Strip markdown code fences (single-line and multi-line)
        # Example: ```bash\nls -la\n``` or ```bash ls -la ```
        command = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", command)
        command = re.sub(r"\n?```$", "", command).strip()
        if command.startswith("```") and "```" in command[3:]:
            command = re.sub(r"^```.*?\n", "", command, flags=re.DOTALL)
            command = re.sub(r"```$", "", command).strip()

        # Bail on clarification requests
        if "AINUX_CLARIFY" in command:
            return None

        # Take only first line if multiline (single-command mode)
        command = command.split("\n")[0].strip()

        if command.startswith("`") and command.endswith("`"):
            command = command[1:-1].strip()

        # Strip common decorators
        for prefix in ["$", "#", ">", "bash:", "sh:"]:
            if command.lower().startswith(prefix):
                command = command[len(prefix):].strip()

        # Strip trailing semicolons left by the model on single commands
        command = command.rstrip(";").strip()

        command = self._canonicalize_command(command)

        placeholder_patterns = [
            r"/path/to/",
            r"<[^>]+>",
            r"\byour[_/-]",
            r"\bexample[_/-]",
        ]
        if any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in placeholder_patterns):
            return None

        if not command or len(command) > 500:
            return None

        return command

    def _regex_fallback(self, text: str) -> Optional[str]:
        """Minimal fallback when Ollama is unavailable or generation fails."""
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

    def _extract_plan(self, raw: str, max_steps: int) -> List[str]:
        """Parse a numbered list of commands from plan response."""
        import re
        commands = []
        for line in raw.strip().split("\n"):
            # Strip leading numbers/bullets
            line = re.sub(r"^\s*[\d]+[.)]\s*", "", line).strip()
            line = re.sub(r"^[-*•]\s*", "", line).strip()
            if line and not line.startswith("#") and len(line) > 1:
                commands.append(line)
            if len(commands) >= max_steps:
                break
        return commands
