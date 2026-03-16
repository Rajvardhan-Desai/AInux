"""
AInux Evaluation Task Suite
60 standardised tasks across 5 categories with ground-truth commands
and automated correctness verification.

Categories:
  T1 - File Operations       (12 tasks)
  T2 - Package Management    (12 tasks)
  T3 - System Diagnostics    (12 tasks)
  T4 - Web Server / SSL      (12 tasks)
  T5 - Development Setup     (12 tasks)

Each task has:
  - natural_language : what the user would type
  - ground_truth     : list of acceptable correct commands (any match = correct)
  - category         : T1-T5
  - complexity       : "simple" | "multi_step"
  - safe_to_execute  : whether automated runner can actually run it
  - verify_fn        : optional callable(output) -> bool for output validation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import re
import os


@dataclass
class EvalTask:
    id: str
    natural_language: str
    ground_truth: List[str]          # any match counts as correct
    category: str                    # T1-T5
    complexity: str                  # simple | multi_step
    safe_to_execute: bool = True     # False = check generation only, don't run
    verify_fn: Optional[Callable] = None  # validate output beyond return code

    def is_correct(self, generated_command: str) -> bool:
        """
        Check if generated command matches any ground truth.
        Uses normalised comparison: strip whitespace, collapse spaces.
        """
        gen = _normalise(generated_command)
        for gt in self.ground_truth:
            if _normalise(gt) == gen:
                return True
            # Partial match for commands with variable args
            if self.verify_fn and self.verify_fn(generated_command):
                return True
        return False


def _normalise(cmd: str) -> str:
    cmd = cmd.strip()
    if cmd.startswith("```") and cmd.endswith("```"):
        cmd = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cmd)
        cmd = re.sub(r"\s*```$", "", cmd)
    if cmd.startswith("`") and cmd.endswith("`"):
        cmd = cmd[1:-1].strip()
    cmd = re.sub(r'\s+', ' ', cmd.lower())
    # Treat single and double quotes as equivalent so models that use
    # double-quotes aren't penalised when the ground truth uses single-quotes
    cmd = cmd.replace('"', "'")
    # Strip trailing semicolons the model may append
    cmd = cmd.rstrip(';').strip()
    return cmd


def _is_self_signed_ssl_command(generated_command: str) -> bool:
    cmd = _normalise(generated_command)
    required = [
        "openssl req",
        "-x509",
        "-newkey",
        "-keyout",
        "-out",
    ]
    return all(token in cmd for token in required) and ("-nodes" in cmd or "-noenc" in cmd)


def _is_nvm_install_command(generated_command: str) -> bool:
    cmd = _normalise(generated_command)
    return bool(re.fullmatch(
        r"(?:curl -o-|wget -qo-) https://raw\.githubusercontent\.com/nvm-sh/nvm/v0\.39\.\d+/install\.sh \| bash",
        cmd,
    ))


# ---------------------------------------------------------------------------
# T1 — File Operations
# ---------------------------------------------------------------------------

FILE_TASKS = [
    EvalTask(
        id="T1_01",
        natural_language="List all files in the current directory with details",
        ground_truth=["ls -la", "ls -l", "ls --all -l"],
        category="T1", complexity="simple",
    ),
    EvalTask(
        id="T1_02",
        natural_language="Show the current working directory",
        ground_truth=["pwd"],
        category="T1", complexity="simple",
    ),
    EvalTask(
        id="T1_03",
        natural_language="Create a directory called project_logs",
        ground_truth=["mkdir project_logs", "mkdir -p project_logs"],
        category="T1", complexity="simple",
    ),
    EvalTask(
        id="T1_04",
        natural_language="Create nested directory structure logs/archive/2025",
        ground_truth=["mkdir -p logs/archive/2025"],
        category="T1", complexity="simple",
    ),
    EvalTask(
        id="T1_05",
        natural_language="Find all Python files in the current directory",
        ground_truth=["find . -name '*.py'", "find . -name '*.py' -type f", "find . -type f -name '*.py'", "ls *.py"],
        category="T1", complexity="simple",
    ),
    EvalTask(
        id="T1_06",
        natural_language="Show disk usage of the current directory",
        ground_truth=["du -sh .", "du -sh"],
        category="T1", complexity="simple",
    ),
    EvalTask(
        id="T1_07",
        natural_language="Find files modified in the last 24 hours",
        ground_truth=["find . -mtime -1", "find . -mtime -1 -type f", "find . -mtime -1 -ls"],
        category="T1", complexity="simple",
    ),
    EvalTask(
        id="T1_08",
        natural_language="Count the number of files in the current directory",
        ground_truth=["ls -1 | wc -l", "ls | wc -l"],
        category="T1", complexity="simple",
    ),
    EvalTask(
        id="T1_09",
        natural_language="Show the 10 largest files in this directory",
        ground_truth=["du -sh * | sort -rh | head -10", "find . -type f -ls | sort -k7 -rn | head -10"],
        category="T1", complexity="simple",
    ),
    EvalTask(
        id="T1_10",
        natural_language="Rename file report.txt to report_final.txt",
        ground_truth=["mv report.txt report_final.txt"],
        category="T1", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T1_11",
        natural_language="Copy directory backup to backup_old recursively",
        ground_truth=["cp -r backup backup_old"],
        category="T1", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T1_12",
        natural_language="Show file permissions for all files in current directory",
        ground_truth=["ls -la", "ls -l", "stat *"],
        category="T1", complexity="simple",
    ),
]

# ---------------------------------------------------------------------------
# T2 — Package Management
# ---------------------------------------------------------------------------

PACKAGE_TASKS = [
    EvalTask(
        id="T2_01",
        natural_language="Update the package list",
        ground_truth=["apt-get update", "apt update", "apt-get update -y"],
        category="T2", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T2_02",
        natural_language="Install nginx",
        ground_truth=["apt-get install -y nginx", "apt install -y nginx", "apt-get update && apt-get install nginx"],
        category="T2", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T2_03",
        natural_language="Install Python package requests",
        ground_truth=["pip install requests", "pip3 install requests"],
        category="T2", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T2_04",
        natural_language="List all installed pip packages",
        ground_truth=["pip list", "pip3 list", "pip freeze"],
        category="T2", complexity="simple",
    ),
    EvalTask(
        id="T2_05",
        natural_language="Check if git is installed",
        ground_truth=["which git", "git --version", "dpkg -l git"],
        category="T2", complexity="simple",
    ),
    EvalTask(
        id="T2_06",
        natural_language="Upgrade all installed packages",
        ground_truth=["apt-get upgrade -y", "apt upgrade -y", "apt-get update && apt-get upgrade -y"],
        category="T2", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T2_07",
        natural_language="Install numpy and pandas for data analysis",
        ground_truth=["pip install numpy pandas", "pip3 install numpy pandas"],
        category="T2", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T2_08",
        natural_language="Show which version of Python is installed",
        ground_truth=["python --version", "python3 --version"],
        category="T2", complexity="simple",
    ),
    EvalTask(
        id="T2_09",
        natural_language="Install requirements from requirements.txt",
        ground_truth=["pip install -r requirements.txt", "pip3 install -r requirements.txt"],
        category="T2", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T2_10",
        natural_language="Remove the package vim",
        ground_truth=["apt-get remove -y vim", "apt remove -y vim", "apt-get remove vim"],
        category="T2", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T2_11",
        natural_language="Search for available packages matching curl",
        ground_truth=["apt-cache search curl", "apt search curl"],
        category="T2", complexity="simple",
    ),
    EvalTask(
        id="T2_12",
        natural_language="Show details about the nginx package",
        ground_truth=["apt-cache show nginx", "apt show nginx", "dpkg -l nginx"],
        category="T2", complexity="simple",
    ),
]

# ---------------------------------------------------------------------------
# T3 — System Diagnostics
# ---------------------------------------------------------------------------

DIAGNOSTICS_TASKS = [
    EvalTask(
        id="T3_01",
        natural_language="Show CPU usage",
        ground_truth=["top -bn1", "top", "mpstat 1 1", "top -bn1 | head -20"],
        category="T3", complexity="simple",
    ),
    EvalTask(
        id="T3_02",
        natural_language="Show memory usage",
        ground_truth=["free -h", "free -m"],
        category="T3", complexity="simple",
    ),
    EvalTask(
        id="T3_03",
        natural_language="Show disk space usage",
        ground_truth=["df -h", "df -h --total"],
        category="T3", complexity="simple",
    ),
    EvalTask(
        id="T3_04",
        natural_language="Show all running processes",
        ground_truth=["ps aux", "ps -ef"],
        category="T3", complexity="simple",
    ),
    EvalTask(
        id="T3_05",
        natural_language="Show the last 50 lines of system logs",
        ground_truth=["journalctl -n 50", "journalctl -n 50 --no-pager", "tail -50 /var/log/syslog", "tail -n 50 /var/log/syslog"],
        category="T3", complexity="simple",
    ),
    EvalTask(
        id="T3_06",
        natural_language="Show network interface configuration",
        ground_truth=["ifconfig", "ip a", "ip addr", "ip addr show"],
        category="T3", complexity="simple",
    ),
    EvalTask(
        id="T3_07",
        natural_language="Show system uptime",
        ground_truth=["uptime"],
        category="T3", complexity="simple",
    ),
    EvalTask(
        id="T3_08",
        natural_language="Show which services are currently running",
        ground_truth=[
            "systemctl --type=service --state=running",
            "systemctl list-units --type=service --state=running",
            "service --status-all",
        ],
        category="T3", complexity="simple",
    ),
    EvalTask(
        id="T3_09",
        natural_language="Show open network connections",
        ground_truth=["ss -tuln", "netstat -tuln", "netstat -an", "ss -tulnp"],
        category="T3", complexity="simple",
    ),
    EvalTask(
        id="T3_10",
        natural_language="Find which process is using the most CPU",
        ground_truth=["ps aux --sort=-%cpu | head -5", "top -bn1 | sort -k9 -rn | head -5"],
        category="T3", complexity="simple",
    ),
    EvalTask(
        id="T3_11",
        natural_language="Show system kernel version",
        ground_truth=["uname -r", "uname -a"],
        category="T3", complexity="simple",
    ),
    EvalTask(
        id="T3_12",
        natural_language="Check if port 80 is in use",
        ground_truth=["ss -tuln | grep :80", "netstat -tuln | grep :80", "netstat -an | grep :80", "lsof -i :80"],
        category="T3", complexity="simple",
    ),
]

# ---------------------------------------------------------------------------
# T4 — Web Server / SSL  (multi-step, agent territory)
# ---------------------------------------------------------------------------

WEBSERVER_TASKS = [
    EvalTask(
        id="T4_01",
        natural_language="Check if nginx is running",
        ground_truth=["systemctl status nginx", "service nginx status"],
        category="T4", complexity="simple",
    ),
    EvalTask(
        id="T4_02",
        natural_language="Start the nginx service",
        ground_truth=["systemctl start nginx", "service nginx start"],
        category="T4", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T4_03",
        natural_language="Enable nginx to start on boot",
        ground_truth=["systemctl enable nginx"],
        category="T4", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T4_04",
        natural_language="Reload nginx configuration without downtime",
        ground_truth=["systemctl reload nginx", "nginx -s reload"],
        category="T4", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T4_05",
        natural_language="Test nginx configuration for syntax errors",
        ground_truth=["nginx -t", "nginx -T"],
        category="T4", complexity="simple",
    ),
    EvalTask(
        id="T4_06",
        natural_language="Show the nginx error log",
        ground_truth=["tail -50 /var/log/nginx/error.log", "tail /var/log/nginx/error.log", "cat /var/log/nginx/error.log"],
        category="T4", complexity="simple",
    ),
    EvalTask(
        id="T4_07",
        natural_language="Open port 443 in the firewall",
        ground_truth=["ufw allow 443", "ufw allow https", "iptables -A INPUT -p tcp --dport 443 -j ACCEPT"],
        category="T4", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T4_08",
        natural_language="Show firewall status",
        ground_truth=["ufw status", "ufw status verbose", "iptables -L", "iptables --list-rules"],
        category="T4", complexity="simple",
    ),
    EvalTask(
        id="T4_09",
        natural_language="Check SSL certificate expiry for localhost",
        ground_truth=[
            "openssl s_client -connect localhost:443 2>/dev/null | openssl x509 -noout -dates",
            "echo | openssl s_client -connect localhost:443 2>/dev/null | openssl x509 -noout -enddate",
        ],
        category="T4", complexity="simple",
    ),
    EvalTask(
        id="T4_10",
        natural_language="Generate a self-signed SSL certificate",
        ground_truth=[
            "openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout server.key -out server.crt",
        ],
        category="T4", complexity="simple",
        verify_fn=_is_self_signed_ssl_command,
        safe_to_execute=False,
    ),
    EvalTask(
        id="T4_11",
        natural_language="Show the nginx access log",
        ground_truth=["tail -100 /var/log/nginx/access.log", "tail /var/log/nginx/access.log", "cat /var/log/nginx/access.log"],
        category="T4", complexity="simple",
    ),
    EvalTask(
        id="T4_12",
        natural_language="Restart nginx",
        ground_truth=["systemctl restart nginx", "service nginx restart"],
        category="T4", complexity="simple",
        safe_to_execute=False,
    ),
]

# ---------------------------------------------------------------------------
# T5 — Development Environment Setup  (multi-step, agent territory)
# ---------------------------------------------------------------------------

DEVSETUP_TASKS = [
    EvalTask(
        id="T5_01",
        natural_language="Create a Python virtual environment called venv",
        ground_truth=["python3 -m venv venv", "python -m venv venv", "virtualenv venv"],
        category="T5", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T5_02",
        natural_language="Activate the virtual environment",
        ground_truth=["source venv/bin/activate", ". venv/bin/activate"],
        category="T5", complexity="simple",
    ),
    EvalTask(
        id="T5_03",
        natural_language="Show the current Python path",
        ground_truth=["which python", "which python3"],
        category="T5", complexity="simple",
    ),
    EvalTask(
        id="T5_04",
        natural_language="Initialise a new git repository",
        ground_truth=["git init"],
        category="T5", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T5_05",
        natural_language="Show git status",
        ground_truth=["git status"],
        category="T5", complexity="simple",
    ),
    EvalTask(
        id="T5_06",
        natural_language="Show the last 5 git commits",
        ground_truth=["git log --oneline -5", "git log -5", "git log -5 --oneline", "git log -5 --pretty=format:'%h %s'"],
        category="T5", complexity="simple",
    ),
    EvalTask(
        id="T5_07",
        natural_language="Set git global username to John",
        ground_truth=['git config --global user.name "John"'],
        category="T5", complexity="simple",
        safe_to_execute=False,
    ),
    EvalTask(
        id="T5_08",
        natural_language="Show all environment variables",
        ground_truth=["env", "printenv"],
        category="T5", complexity="simple",
    ),
    EvalTask(
        id="T5_09",
        natural_language="Check if Docker is installed",
        ground_truth=["which docker", "docker --version"],
        category="T5", complexity="simple",
    ),
    EvalTask(
        id="T5_10",
        natural_language="Show running Docker containers",
        ground_truth=["docker ps", "docker ps -a"],
        category="T5", complexity="simple",
    ),
    EvalTask(
        id="T5_11",
        natural_language="Show all git branches",
        ground_truth=["git branch", "git branch -a"],
        category="T5", complexity="simple",
    ),
    EvalTask(
        id="T5_12",
        natural_language="Install node version manager nvm",
        ground_truth=[
            'curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash',
            'curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash',
            'wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash',
        ],
        category="T5", complexity="simple",
        verify_fn=_is_nvm_install_command,
        safe_to_execute=False,
    ),
]

# ---------------------------------------------------------------------------
# Full task suite
# ---------------------------------------------------------------------------

ALL_TASKS: List[EvalTask] = (
    FILE_TASKS + PACKAGE_TASKS + DIAGNOSTICS_TASKS + WEBSERVER_TASKS + DEVSETUP_TASKS
)

TASK_MAP = {t.id: t for t in ALL_TASKS}

CATEGORIES = {
    "T1": FILE_TASKS,
    "T2": PACKAGE_TASKS,
    "T3": DIAGNOSTICS_TASKS,
    "T4": WEBSERVER_TASKS,
    "T5": DEVSETUP_TASKS,
}
