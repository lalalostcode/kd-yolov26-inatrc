"""Small, explicit gates for supported knowledge-distillation methods."""

from __future__ import annotations


def configure_kd(config: dict) -> dict:
    """Validate method/teacher and return pinned trainer kwargs without baseline fallback."""
    method = config.get("method", "none")
    if method == "none":
        return {}
    if config.get("stage", "student") != "student":
        raise ValueError("Knowledge distillation is only valid for the student stage.")
    if method == "native":
        from .native import native_train_kwargs

        return native_train_kwargs(config)
    if method == "crosskd":
        from .crosskd import configure_crosskd

        return configure_crosskd(config)
    if method == "csakd":
        from .csakd import configure_csakd

        return configure_csakd(config)
    raise ValueError(f"Unknown KD method: {method!r}. Choose none/native/crosskd/csakd.")
