# GroundedNavigator

Fully local, open-source language-guided autonomous navigation agent.
No API keys. No cloud. No GPU required.

> A robot receives a plain English command like "find the refrigerator," looks through a first-person 3D camera, and navigates there — no pre-built map, no hardcoded waypoints.

| Component | Model | Backend |
|-----------|-------|---------|
| Scene perception (VLM) | **LLaVA** | Ollama |
| Task planner (LLM) | **Llama3** | Ollama |
| Simulation | **PyBullet** | Local |
| Fallback controller | **Geometric** | Coordinate-based |

## Results

Evaluated across 7 indoor navigation scenes (living room, kitchen, bathroom, bedroom):

| Metric | Value |
|--------|-------|
| Goals completed | 7 / 7 |
| Avg steps to goal | 12 |
| Avg latency | ~135s |
| Hardware | Local CPU, no GPU |

Scenes: find the television, navigate to the sofa, find the laptop, locate the refrigerator, navigate to the sink, find the toilet, find the bed.

## Architecture

```
Natural language command
        │
        ▼
  PyBullet 3D environment  ──  RGB frame (first-person camera)
        │                  ──  Object list + distances
        ▼
  LLaVA (VLM)             ──  Ollama → scene description
        │
        ▼
  Llama3 planner           ──  Ollama + LangChain → JSON action plan
        │
        ├── Valid plan → Execute actions
        │
        └── Parse error / only rotations → Geometric fallback controller
                                           (coordinate-based navigation)
        │
        ▼
  Success check            ──  Agent within 2m of target → done
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
ollama pull llava      # VLM for scene perception (~4.7 GB)
ollama pull llama3     # LLM for task planning   (~4.7 GB)
ollama serve           # Start Ollama server (localhost:11434)
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

## Project structure

```
GroundedNavigator/
├── main.py                          # Episode runner + geometric fallback
├── env/
│   └── thor_env.py                  # PyBullet 3D environment wrapper
├── perception/
│   └── vlm_describer.py             # LLaVA via Ollama
├── planner/
│   ├── prompts.py                   # System + user prompt templates
│   └── planner.py                   # Llama3 via Ollama (LangChain)
├── eval/
│   ├── run_eval.py                  # Batch eval harness → CSV
│   └── scenarios/
│       └── sample.json              # 7 benchmark scenarios
├── requirements.txt
└── README.md
```

## Swap models

| File | Constant | Alternatives |
|------|----------|--------------|
| `perception/vlm_describer.py` | `VISION_MODEL = "llava"` | `llava:13b`, `bakllava` |
| `planner/planner.py` | `PLANNER_MODEL = "llama3"` | `mistral`, `gemma2`, `phi3` |

## Key design decisions

**Coordinate-based success** — the agent must physically reach within 2m of the target object. Word-matching on VLM descriptions is not reliable; LLaVA hallucinates object locations frequently.

**Geometric fallback controller** — when Llama3 fails to produce valid JSON or outputs only rotations, a coordinate-based controller computes the correct direction and moves the agent toward the goal. Hybrid systems outperform pure LLM planners on local CPU hardware.

**VLM call throttling** — LLaVA is called every 5 steps rather than every step. Scene descriptions are cached between calls, cutting per-episode latency significantly with no meaningful drop in navigation quality.

## Eval metrics

| Metric | Description |
|--------|-------------|
| Goals completed | Number of targets physically reached |
| Avg steps | Mean action steps per episode |
| Avg latency | Wall-clock time including local inference |

## Results & Analysis

**Evaluated across 7 indoor navigation scenes:**

| Scene | Goal | Steps | Latency |
|-------|------|-------|---------|
| FloorPlan1 | find the television | 16 | 148s |
| FloorPlan1 | navigate to the sofa | 7 | 100s |
| FloorPlan2 | find the laptop | 11 | 130s |
| FloorPlan201 | find the refrigerator | 13 | 123s |
| FloorPlan201 | navigate to the sink | 7 | 97s |
| FloorPlan301 | find the toilet | 10 | 119s |
| FloorPlan401 | find the bed | 20 | 228s |

**What the numbers show:**

**Easiest goals (7 steps):** Sofa and sink were closest to the agent's spawn position, requiring minimal rotation before the geometric controller could navigate directly.

**Hardest goal (20 steps, 228s):** The bed in FloorPlan401 took the longest — the bedroom scene has the most furniture density, increasing the number of corrective rotations needed before the agent could move in a straight line toward the target.

**Latency scales with steps:** Each step costs ~10–12s on local CPU — dominated by LLaVA inference. The VLM is called every 5 steps, so a 20-step episode triggers 4 LLaVA calls vs 2 for a 7-step episode.

**Limitations to note:**

- The geometric fallback has direct coordinate access to object positions — something a real robot would not have. In a real deployment, the agent would rely entirely on the VLM + LLM pipeline, which is significantly less reliable on local hardware.
- PyBullet's simplified 3D environment (box geometry, flat lighting) is far from real-world visual complexity. LLaVA's hallucination rate would likely increase significantly on real camera feeds with dynamic lighting, occlusion, and visual clutter.
- Latency at ~135s/episode is impractical for real-time AMR deployment. A GPU-accelerated setup or a smaller, faster VLM (e.g. LLaVA-1.5 7B) would be needed.

**What this project establishes:**

The hybrid VLM + geometric controller architecture is a viable proof of concept for language-guided navigation in simulation. The key insight is that pure LLM planning is insufficient on local hardware — reliable navigation requires geometric grounding as a fallback. Closing the sim-to-real gap is the next meaningful research direction.