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
  (Chat Completion      ├── PackageManagementAgent
   API standard)        ├── FileOperationsAgent
        │                └── SystemDiagnosticsAgent
        ▼
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
| `ainux_llm_runtime.py` | Universal Chat Completion API runtime (works with LM Studio, Ollama, Vllm, and any Chat Completion API-compliant service) |
| `ainux_memory.py` | FAISS vector store with three-layer memory and decay scoring |
| `ainux_safety.py` | MDP-based safety verification (Equations 10–11 from paper) |
| `ainux_agents.py` | Autonomous agents with planner-verifier loops and rollback |

---

## Requirements

- Python 3.9+
- Any Chat Completion API-compliant LLM endpoint (local or cloud)
- 8GB RAM minimum (for local models)

---

## Setup

### 1. Configure your Chat Completion API endpoint

**What is Chat Completion API?**

The Chat Completion API is an open standard implemented by many LLM providers. It's not proprietary to OpenAI—it's used by dozens of projects and services.

**Option A: Local (Recommended)**

All of the following implement the Chat Completion API standard on localhost:

- **LM Studio** (easiest for beginners)
  - Download: https://lmstudio.ai
  - Load a model → Local Server tab → Start Server (default port 1234)
  - Models: gpt-oss-20b-MXFP4 (20B, best quality), phi3:mini (2.3GB, fast)

- **Ollama**
  - Download: https://ollama.com
  - `ollama serve` (runs on port 11434)
  - `ollama pull llama3.2:3b`

- **Vllm**
  - `pip install vllm`
  - `python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf` (port 8000)

**Option B: Cloud & Remote Services**

Set `AINUX_LLM_API_KEY` for services that require authentication:

```bash
export AINUX_LLM_API_KEY=sk-proj-...
python -m AInux.ainux_core --host https://api.openai.com/v1 --model gpt-4
```

Other Chat Completion API providers:
- OpenAI (https://api.openai.com/v1)
- Azure OpenAI (your-resource.openai.azure.com/v1)
- Anthropic Claude (via APIrouter or proxy)
- Others: any service exposing /v1/chat/completions

**Environment variables:**

- `AINUX_LLM_HOST` — endpoint URL (default: `http://127.0.0.1:1234`)
- `AINUX_LLM_API_KEY` — auth key for remote/proprietary services
- Legacy: `AINUX_OLLAMA_HOST`, `OLLAMA_HOST` (still supported)

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~80MB) automatically.

### 3. Run AInux

```bash
# From the repo root (LLM endpoint must be running):
python -m AInux.ainux_core

# With a specific model:
python -m AInux.ainux_core --model "llama3.2:3b"

# With custom LLM endpoint:
python -m AInux.ainux_core --host http://192.168.1.10:8000

# Cloud provider example (with API key):
export AINUX_LLM_API_KEY=sk-...
python -m AInux.ainux_core --host https://api.openai.com/v1 --model "gpt-4"

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
