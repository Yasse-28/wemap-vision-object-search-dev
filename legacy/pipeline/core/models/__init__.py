"""Pipeline models package.

Use explicit submodules to avoid heavy imports at package load (e.g.
``from pipeline.core.models.metaclip import MetaCLIP``).
"""

from pipeline.core.models.base_model import (
    SimpleSingleton,
    SingletonModel,
    load_model,
)

__all__ = [
    "SimpleSingleton",
    "SingletonModel",
    "load_model",
]
