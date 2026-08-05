from __future__ import annotations

import gc
from threading import Lock
from types import SimpleNamespace
from typing import Any, cast

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

    class _FallbackModule:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def parameters(self) -> list[Any]:
            return []

    class _FallbackNN:
        Module = _FallbackModule

    nn = _FallbackNN()


class SimpleSingleton(type):
    _instances: dict[type, object] = {}
    _lock = Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def destroy_instance(cls, model_class: type) -> bool:
        with cls._lock:
            instance = cls._instances.pop(model_class, None)
        if instance is None:
            return False
        close = getattr(instance, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        elif hasattr(instance, "model") and hasattr(instance.model, "cpu"):
            instance.model.cpu()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return True

    @classmethod
    def destroy_all_instances(cls) -> None:
        with cls._lock:
            instances = list(cls._instances.values())
            cls._instances.clear()
        for instance in instances:
            close = getattr(instance, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            elif hasattr(instance, "model") and hasattr(instance.model, "cpu"):
                instance.model.cpu()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class SingletonModel(nn.Module, metaclass=SimpleSingleton):
    default_conf: dict[str, Any] = {}

    def __init__(self, **conf: Any) -> None:
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.warming_up = False
        self.warmed_up = False

    def warmup(self, iterations: int = 1) -> None:
        raise RuntimeError(f"{self.__class__.__name__}.warmup() is not implemented")

    def _disable_grad(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    @classmethod
    def destroy_instance(cls) -> bool:
        return SimpleSingleton.destroy_instance(cls)

    @classmethod
    def destroy_all_instances(cls) -> None:
        SimpleSingleton.destroy_all_instances()


class OcrSingletonModel(metaclass=SimpleSingleton):
    """Paddle and other non-torch recognizers: at most one live instance per
    class per process.

    Subclasses (e.g. ``PaddleOcrVlRecognizer``) are constructed as usual;
    repeat calls return the same instance. Use ``destroy_instance`` after a
    refinement pass to free memory and allow a fresh construction on the next
    run, matching the previous ``close()``-then-``del`` behavior.
    """

    @classmethod
    def destroy_instance(cls) -> bool:
        return SimpleSingleton.destroy_instance(cls)


def load_model(
    model_conf: dict[str, Any], device: Any, warmup_iterations: int = 0
) -> "SingletonModel":
    if torch is None:
        raise ImportError("torch is required to load standalone pipeline models")
    if "name" not in model_conf:
        raise ValueError("model name is required to load it")

    model_name = model_conf["name"]
    if model_name == "metaclip2":
        from pipeline.core.models.metaclip import MetaCLIP

        model = MetaCLIP(**model_conf).eval().to(device)
    elif model_name == "grounding_dino":
        from pipeline.core.models.detection.grounding_dino import GroundingDINOModel

        model = GroundingDINOModel(**model_conf).eval().to(device)
    else:
        raise ValueError(f"Unsupported standalone pipeline model: {model_name!r}")

    model._disable_grad()
    if warmup_iterations and not model.warmed_up and not model.warming_up:
        model.warmup(warmup_iterations)
    return cast("SingletonModel", model)
