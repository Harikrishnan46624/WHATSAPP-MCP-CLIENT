# Cost calculation module

MODEL_PRICING = {
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    }
}

def normalize_model_name(model: str) -> str:
    if not model:
        return ""
    if model.startswith("gpt-4o-mini"):
        return "gpt-4o-mini"
    return model

def calculate_llm_cost(model: str, prompt: int, completion: int) -> float:
    base = normalize_model_name(model)
    pricing = MODEL_PRICING.get(base)
    if not pricing:
        return 0.0
    return round(
        (prompt * pricing["input"]) +
        (completion * pricing["output"]),
        6
    )