"""VLM inference package for Qwen3-VL captioning."""

__all__ = ["AppConfig", "load_config", "run_inference"]


def __getattr__(name: str):
    if name in {"AppConfig", "load_config"}:
        from .config import AppConfig, load_config

        return {"AppConfig": AppConfig, "load_config": load_config}[name]
    if name == "run_inference":
        from .runner import run_inference

        return run_inference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
