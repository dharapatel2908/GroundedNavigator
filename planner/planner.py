"""
planner.py
LangChain-based task planner using Mistral (via Ollama) — fully open source.

Setup:
    ollama pull mistral     # or: ollama pull llama3
"""

import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE

PLANNER_MODEL = "mistral"       # swap to "llama3", "gemma2", etc.


class NavPlanner:
    def __init__(self, model: str = PLANNER_MODEL, temperature: float = 0.0):
        self.llm = ChatOllama(model=model, temperature=temperature)

    def plan(
        self,
        scene_description: str,
        goal: str,
        visible_objects: list[str],
    ) -> dict:
        """
        Returns {"reasoning": str, "actions": [str, ...]}
        Falls back to RotateLeft if JSON parsing fails.
        """
        user_content = USER_TEMPLATE.format(
            scene_description=scene_description,
            goal=goal,
            visible_objects=", ".join(visible_objects) if visible_objects else "none visible",
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
        response = self.llm.invoke(messages)
        text = response.content.strip()

        # Strip markdown fences if the model adds them
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"reasoning": "parse error — rotating to recover", "actions": ["RotateLeft"]}
