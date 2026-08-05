from __future__ import annotations

from types import SimpleNamespace

import pipeline.core.models.metaclip as metaclip_module
from pipeline.core.models.metaclip import (
    get_shared_metaclip,
    resolve_metaclip_device,
    set_metaclip_device_override,
)


def test_resolve_metaclip_device_prefers_explicit_device() -> None:
    set_metaclip_device_override(None)
    assert resolve_metaclip_device("cpu") == "cpu"
    assert resolve_metaclip_device("cuda:1") == "cuda:1"


def test_resolve_metaclip_device_defaults_to_cuda_when_available(monkeypatch) -> None:
    set_metaclip_device_override(None)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setattr(metaclip_module, "torch", fake_torch)
    assert resolve_metaclip_device(None) == "cuda"


def test_resolve_metaclip_device_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch,
) -> None:
    set_metaclip_device_override(None)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(metaclip_module, "torch", fake_torch)
    assert resolve_metaclip_device(None) == "cpu"


def test_get_shared_metaclip_falls_back_to_cpu_after_cuda_oom(monkeypatch) -> None:
    set_metaclip_device_override(None)
    calls: list[str] = []

    class FakeOOM(RuntimeError):
        pass

    def fake_load_model(model_conf, device, warmup_iterations: int = 0):
        calls.append(str(device))
        if str(device) == "cuda":
            raise FakeOOM("CUDA out of memory while loading model")
        return SimpleNamespace(
            model=SimpleNamespace(device="cpu"),
            warmed_up=True,
            warming_up=False,
            _disable_grad=lambda: None,
        )

    import pipeline.core.models.base_model as base_model

    monkeypatch.setattr(base_model, "load_model", fake_load_model)
    monkeypatch.setattr(
        metaclip_module,
        "_cleanup_after_failed_cuda_load",
        lambda: None,
    )
    monkeypatch.setattr(
        metaclip_module,
        "resolve_metaclip_device",
        lambda preferred_device=None: "cuda",
    )

    model = get_shared_metaclip()

    assert model.model.device == "cpu"
    assert calls == ["cuda", "cpu"]
    assert metaclip_module.get_metaclip_device_override() == "cpu"
