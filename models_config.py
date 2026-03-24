# models_config.py

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class LocalModel:
    name: str          # ollama model name, e.g. "llama3.2:latest"
    role: str          # "general", "reasoning", "code", "math"
    tier: str          # "fast", "balanced", "max"
    description: str


# NOTE: adjust these names to match models you have pulled with Ollama.
# Example pulls (run in terminal, not here):
#   ollama pull llama3.2:latest
#   ollama pull mistral:latest
#   ollama pull qwen2.5:7b-instruct
#   ollama pull deepseek-r1:7b
#   ollama pull qwen2.5-coder:7b
#   ollama pull qwen2.5:7b-math
# Make sure they exist: `ollama list`.

MODELS: List[LocalModel] = [
    # General chat & reasoning
    LocalModel(
        name="llama3.2:latest",
        role="general",
        tier="balanced",
        description="General chat & reasoning, good default model."
    ),
    LocalModel(
        name="mistral:latest",
        role="general",
        tier="fast",
        description="Fast, small general-purpose model."
    ),
    LocalModel(
        name="qwen2.5:7b-instruct",
        role="general",
        tier="max",
        description="Strong at knowledge & RAG-style tasks."
    ),

    # Reasoning / judge
    LocalModel(
        name="deepseek-r1:7b",
        role="reasoning",
        tier="max",
        description="Good reasoning model; can be used as aggregator/judge."
    ),

    # Code specialist (optional, if pulled)
    LocalModel(
        name="qwen2.5-coder:7b",
        role="code",
        tier="balanced",
        description="Code-specialized model for programming questions."
    ),

    # Math / logical specialist (optional, if pulled)
    LocalModel(
        name="qwen2.5:7b-math",
        role="math",
        tier="balanced",
        description="Math-focused model for numerical / formula reasoning."
    ),
]


def get_models_by_tier(tier: str) -> List[LocalModel]:
    return [m for m in MODELS if m.tier == tier]


def get_default_sources() -> List[str]:
    """
    Default models to query in parallel for general questions.
    """
    return [
        "llama3.2:latest",
        "mistral:latest",
        "qwen2.5:7b-instruct",
    ]


def get_aggregator_model() -> str:
    """
    Single model used to summarize & combine per-model answers.
    """
    return "deepseek-r1:7b"


# ----------------- Simple router helpers for MCA project ----------------- #

def get_task_models(task: str, mode: str = "balanced") -> List[str]:
    """
    Very simple routing:
    - task: "general", "code", "math"
    - mode: "fast", "balanced", "max"
    Returns a list of model names to use for this request.
    """
    tier = mode.lower()

    # base filter: tier or all if no models for that tier
    candidates: List[LocalModel] = [m for m in MODELS if m.tier == tier]
    if not candidates:
        candidates = MODELS[:]  # fallback

    if task == "code":
        # prefer code models + some general
        code_models = [m.name for m in candidates if m.role == "code"]
        general_models = [m.name for m in candidates if m.role == "general"]
        return (code_models or []) + general_models[:2]

    if task == "math":
        math_models = [m.name for m in candidates if m.role == "math"]
        general_models = [m.name for m in candidates if m.role == "general"]
        return (math_models or []) + general_models[:2]

    # default: general task
    general_models = [m.name for m in candidates if m.role == "general"]
    if not general_models:
        general_models = [m.name for m in MODELS if m.role == "general"]
    return general_models


def classify_task_from_question(question: str) -> str:
    """
    Very naive keyword-based classifier:
    returns "code", "math", or "general".
    You can mention this as a simple router in your MCA report.
    """
    q_lower = question.lower()

    code_keywords = ["code", "python", "java", "c++", "bug", "function", "api", "program"]
    math_keywords = ["solve", "equation", "integral", "derivative", "probability", "matrix"]

    if any(k in q_lower for k in code_keywords):
        return "code"
    if any(k in q_lower for k in math_keywords):
        return "math"
    return "general"
