"""Prompt loading: inline text, prompt files, and per-sample sidecar overrides."""

from __future__ import annotations

from pathlib import Path

from .config import AppConfig, JobConfig


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def resolve_path(base: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    # Prefer project root (cwd) then config directory
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.is_file():
        return cwd_candidate.resolve()
    return (base / p).resolve()


def load_job_prompt(cfg: AppConfig, job: JobConfig) -> str:
    """Resolve prompt with priority: job prompt > job prompt_file > global prompt > global prompt_file."""
    if job.prompt:
        return job.prompt.strip()
    if job.prompt_file:
        return _read_text(resolve_path(cfg.config_dir, job.prompt_file))
    if cfg.prompt:
        return cfg.prompt.strip()
    if cfg.prompt_file:
        return _read_text(resolve_path(cfg.config_dir, cfg.prompt_file))
    raise ValueError(
        f"No prompt configured for job '{job.name}'. "
        "Set prompt / prompt_file on the job or globally."
    )


def load_sample_prompt(
    default_prompt: str,
    sample_id: str,
    sidecar_dirs: list[Path],
) -> str:
    """Use `{sample_id}.prompt.txt` next to the sample if present."""
    for directory in sidecar_dirs:
        candidate = directory / f"{sample_id}.prompt.txt"
        if candidate.is_file():
            return _read_text(candidate)
    return default_prompt
