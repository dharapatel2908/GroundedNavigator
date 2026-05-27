"""
vlm_describer.py
Uses LLaVA (via Ollama) for open-source, fully local scene description.

Setup:
    1. Install Ollama: https://ollama.com/download
    2. Pull model:    ollama pull llava
    3. Start server:  ollama serve   (runs on http://localhost:11434)
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "llava"          # or "llava:13b" for better quality


def describe_scene(frame_b64: str, goal: str) -> str:
    """
    Given a base64 JPEG frame and a navigation goal, return a natural-language
    scene description using LLaVA running locally via Ollama.
    """
    prompt = (
        f"You are assisting a mobile robot. "
        f"The robot's current navigation goal is: '{goal}'.\n"
        f"Describe the scene from the robot's first-person perspective in 3-5 sentences. "
        f"List visible objects and their approximate positions (left/center/right, near/far). "
        f"Note any obstacles blocking the path toward the goal. Be concise."
    )

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [frame_b64],
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)  # 5 min
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Ollama. Make sure it's running: `ollama serve`"
        )
