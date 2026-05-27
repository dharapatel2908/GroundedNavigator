# NavAgent (Open Source)

Fully local, open-source LLM-guided autonomous navigation agent.
No API keys. No cloud calls.

| Component | Model | Backend |
|-----------|-------|---------|
| Scene perception (VLM) | **LLaVA** | Ollama |
| Task planner (LLM) | **Mistral** | Ollama |
| Simulation | **AI2-THOR** | Local |

## Architecture

```
Natural language command
        │
        ▼
  ThorEnv (AI2-THOR)       ──  RGB frame + visible object list
        │
        ▼
  VLM Describer (LLaVA)    ──  Local Ollama → scene description
        │
        ▼
  NavPlanner (Mistral)     ──  Local Ollama → JSON action plan
        │
        ▼
  ThorEnv.step()           ──  Execute discrete actions
        │
        ▼
  EvalLogger               ──  success, steps, latency → results.csv
```

## Setup

### 1. Install Ollama
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download
```

### 2. Pull models
```bash
ollama pull llava      # VLM for scene description (~4 GB)
ollama pull mistral    # LLM for task planning    (~4 GB)
ollama serve           # Start Ollama server (default: localhost:11434)
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
# Single episode
python main.py --scene FloorPlan1 --goal "find the television"

# Full eval suite
python eval/run_eval.py --scenarios eval/scenarios/sample.json --output results.csv
```

## Swap models

Edit the model constants at the top of each file:

| File | Constant | Alternatives |
|------|----------|--------------|
| `perception/vlm_describer.py` | `VISION_MODEL = "llava"` | `llava:13b`, `bakllava` |
| `planner/planner.py` | `PLANNER_MODEL = "mistral"` | `llama3`, `gemma2`, `phi3` |

## Project structure

```
navagent/
├── main.py
├── env/thor_env.py                # AI2-THOR wrapper
├── perception/vlm_describer.py   # LLaVA via Ollama
├── planner/prompts.py
├── planner/planner.py             # Mistral via Ollama (LangChain)
├── eval/run_eval.py
├── eval/scenarios/sample.json
├── requirements.txt
└── README.md
```

## Eval metrics

| Metric | Description |
|--------|-------------|
| Success rate | % episodes where target object reached |
| Avg steps | Mean action steps per episode |
| Avg latency | Wall-clock time (local inference included) |
