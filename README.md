## AInux AI-Native Linux Environment

AInux is an LLM-augmented Linux shell that lets you interact with your system using natural language. It integrates a local LLM runtime, a persistent vector memory layer, an MDP-based safety framework, and autonomous agents capable of multi-step system administration tasks.

---

## Architecture

```
User Input (natural language)
        │
        ▼
   AI Shell (ainux_core.py)
   ├── Intent classification
   ├── Memory retrieval (FAISS)
   └── Route: simple command OR agent task
        │                        │
        ▼                        ▼
  LLM Runtime            Agent Dispatcher
  (Ollama local)         ├── PackageManagementAgent
        │                ├── FileOperationsAgent
        ▼                └── SystemDiagnosticsAgent
  MDP Safety Checker
  (validate before execution)
        │
        ▼
  Execute → Store to Memory
```

---

## Components

| Module | Description |
|---|---|
| `ainux_core.py` | Main shell and orchestrator |
| `ainux_llm_runtime.py` | Local LLM via Ollama (phi3:mini, llama3.2:3b, etc.) |
| `ainux_memory.py` | FAISS vector store with three-layer memory and decay scoring |
| `ainux_safety.py` | MDP-based safety verification (Equations 10–11 from paper) |
| `ainux_agents.py` | Autonomous agents with planner-verifier loops and rollback |

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally
- 8GB RAM minimum (recommended: phi3:mini at 2.3GB)

---

## Setup

### 1. Install Ollama and pull a model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve                    # run in a separate terminal
ollama pull phi3:mini           # 2.3GB — works on 8GB RAM
```

Other supported models (all fit in 8GB RAM):
```bash
ollama pull llama3.2:3b         # 2.0GB
ollama pull mistral:7b-q4       # 4.1GB — best quality
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~80MB) automatically.

### 3. Run AInux

```bash
# From the repo root:
python -m AInux.ainux_core

# With a specific model:
python -m AInux.ainux_core --model llama3.2:3b

# With voice input:
python -m AInux.ainux_core --voice

# Without persisting memory to disk:
python -m AInux.ainux_core --no-persist
```

---

## Example interactions

```
AInux> list all Python files here
  Generated command : find . -name '*.py' -type f
  Safety            : safe (score=1.0)

AInux> set up a Python environment for data analysis
  [Agent] Planning task...
  Agent   : package
  Steps   : 3
    • apt-get update -y
    • apt-get install -y python3-venv python3-pip
    • pip install numpy pandas matplotlib scikit-learn jupyter

AInux> diagnose high CPU usage
  [Agent] Planning task...
  Agent   : diagnostics
  Steps   : 2
    • top -bn1 | head -20
    • ps aux --sort=-%cpu | head -15
```

---

## Memory

Memories persist across sessions to `~/.ainux/memory/items.json`. The memory system uses three layers:

- **Short-term**: current session, not persisted
- **Mid-term**: session state transitions, persisted at end of session
- **Long-term**: stable preferences and repeated workflows, always persisted

Decay scoring prioritises recent and frequently accessed memories:

```
score(m, t) = sim(q, m) · exp(−λ(t − t_m)) · (1 + log(1 + f_m))
```

---

## Safety

Every command and plan is validated before execution using the MDP safety framework:

| Class | Score | Behaviour |
|---|---|---|
| SAFE | 1.0 | Execute immediately |
| REVERSIBLE | 0.7 | Show warning, proceed |
| RISKY | 0.3 | Require explicit YES confirmation |
| DANGEROUS | 0.0 | Hard block, no override |

Plan safety = product of individual command scores. Plans scoring below 0.5 require confirmation.

---
