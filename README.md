# GroundedNavigator

Fully local, open-source LLM-guided autonomous navigation agent for indoor environments.
No API keys. No cloud calls. Runs on your existing hardware.

| Component | Model | Backend |
|-----------|-------|---------|
| Simulation | **AI2-THOR** | Local |
| Scene Understanding (VLM) | **LLaVA** | Ollama |
| Task Planner (LLM) | **Mistral** | Ollama |
| Environment Wrapper | **ThorEnv** | Custom |

---

## Pipeline

```
Natural Language Goal
        │
        ▼
AI2-THOR Environment
        │
RGB Frame + Visible Objects
        │
        ▼
LLaVA
(Scene Understanding)
        │
Scene Description
        │
        ▼
Mistral
(Task Planning)
        │
JSON Action Plan
        │
        ▼
ThorEnv.step()
        │
Execute Navigation
        │
        ▼
Evaluation
(Success Rate · Avg Steps · Latency)
```

---

## How It Works

At each step, the agent:

1. **Perceives** — AI2-THOR renders a first-person RGB frame and returns a list of visible objects
2. **Understands** — LLaVA receives the frame and the goal, and produces a natural language scene description
3. **Plans** — Mistral receives the scene description, goal, and visible objects, and outputs a JSON action plan with a reasoning trace
4. **Executes** — ThorEnv executes up to 5 actions from the plan, then replans

The replanning loop repeats until the goal object is reached or the step budget is exhausted.

---

## Setup

### 1. Install Ollama
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: https://ollama.com/download
```

### 2. Pull models
```bash
ollama pull llava      # VLM for scene understanding (~4 GB)
ollama pull mistral    # LLM for task planning (~4 GB)
ollama serve           # Start Ollama server (default: localhost:11434)
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Run

**Single episode:**
```bash
python main.py --scene FloorPlan1 --goal "find the television"
```

**Full eval suite:**
```bash
python eval/run_eval.py --scenarios eval/scenarios/sample.json --output results.csv
```

---

## Project Structure

```
navagent/
├── main.py                          # Entry point — single episode runner
├── env/
│   └── thor_env.py                  # AI2-THOR environment wrapper
├── perception/
│   └── vlm_describer.py             # LLaVA scene understanding via Ollama
├── planner/
│   ├── planner.py                   # Mistral task planner via LangChain + Ollama
│   └── prompts.py                   # System and user prompt templates
├── eval/
│   ├── run_eval.py                  # Batch evaluation harness
│   └── scenarios/
│       └── sample.json              # Evaluation scenarios
├── requirements.txt
└── README.md
```

---

## Action Space

| Action | Description |
|--------|-------------|
| `MoveAhead` | Move forward one step |
| `RotateLeft` | Rotate left 45° |
| `RotateRight` | Rotate right 45° |
| `LookUp` | Tilt camera up |
| `LookDown` | Tilt camera down |
| `Stop` | Declare goal reached |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Success rate | % of episodes where the target object was reached |
| Avg steps | Mean action steps per episode |
| Avg latency | Wall-clock time including local inference |

Evaluated across 7 scenarios in 5 room types: living room, office, kitchen, bathroom, bedroom.

Results are written to CSV for reproducibility.

---

## Swap Models

Edit the model constants at the top of each file:

| File | Constant | Alternatives |
|------|----------|--------------|
| `perception/vlm_describer.py` | `VISION_MODEL = "llava"` | `llava:13b`, `bakllava` |
| `planner/planner.py` | `PLANNER_MODEL = "mistral"` | `llama3`, `gemma2`, `phi3` |

---

## Demo

[Watch the demo on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7466208647279308800/)

---

## Tech Stack

- [AI2-THOR](https://ai2thor.allenai.org/) — photorealistic indoor simulation
- [LLaVA](https://ollama.com/library/llava) — vision-language model for scene understanding
- [Mistral](https://ollama.com/library/mistral) — LLM for task planning
- [Ollama](https://ollama.com/) — local model serving
- [LangChain](https://www.langchain.com/) — LLM orchestration
