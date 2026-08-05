import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pipeline.core.types import ObjectSearchResult  # noqa: E402
from pipeline.online import app as app_module  # noqa: E402


class _DummyRegistry:
    def map_ids(self):
        return ["station-a"]

    def start_background_loading(self):
        pass

    def get_map_state(self, map_id):
        return "ready" if map_id == "station-a" else None

    def get_service(self, map_id):
        return None

    def get_history(self, map_id):
        return None

    def close(self):
        pass


class _DummyHistory:
    def __init__(self):
        self.requests = []

    def store_request(
        self,
        prompt,
        search_type,
        enforced,
        timestamp,
        time_router_ms,
        time_embedding_ms,
        time_db_ms,
    ):
        self.requests.append(
            {
                "prompt": prompt,
                "search_type": search_type,
                "enforced": enforced,
                "timestamp": timestamp,
                "time_router_ms": time_router_ms,
                "time_embedding_ms": time_embedding_ms,
                "time_db_ms": time_db_ms,
            }
        )


class _DummySearchService:
    def search(self, text, num_results, search_type="cutout"):
        return ObjectSearchResult(
            results=[("12", 0.9)],
            router_object_type=None,
            time_router_ms=0,
            time_embedding_ms=11,
            time_retrieval_ms=22,
        )

    def search_objects_localized(
        self,
        text,
        num_results,
        min_similarity=0.2,
        max_observations_per_cluster=10,
    ):
        return app_module.ObjectLocalizationResponse(
            localizations=[],
            time_embedding_ms=33,
            time_retrieval_ms=44,
        )

    def search_objects_localized_by_image(
        self,
        image,
        num_results,
        min_similarity=0.2,
        *,
        log_query="[image]",
        max_observations_per_cluster=10,
    ):
        self.last_image_query = {
            "mode": image.mode,
            "size": image.size,
            "num_results": num_results,
            "min_similarity": min_similarity,
            "log_query": log_query,
        }
        return app_module.ObjectLocalizationResponse(
            localizations=[],
            time_embedding_ms=77,
            time_retrieval_ms=88,
        )

    def search_objects_localized_online(
        self,
        text,
        num_results,
        *,
        localization_params,
        include_debug=False,
    ):
        self.last_online_localization_params = localization_params
        return app_module.ObjectLocalizationResponse(
            localizations=[],
            time_embedding_ms=55,
            time_retrieval_ms=66,
        )


class _DummyObjectSearchRegistry:
    def __init__(self):
        self.service = _DummySearchService()
        self.history = _DummyHistory()

    def map_ids(self):
        return ["station-a"]

    def start_background_loading(self):
        pass

    def get_map_state(self, map_id):
        return "ready" if map_id == "station-a" else None

    def get_service(self, map_id):
        return self.service if map_id == "station-a" else None

    def get_history(self, map_id):
        return self.history if map_id == "station-a" else None

    def close(self):
        pass


def test_ui_maps_endpoint_returns_config_maps_with_id_display_name(
    tmp_path: Path,
    monkeypatch,
):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "maps": [
                    {
                        "id": "station-a",
                        "path": "maps/station-a",
                        "display_name": "Ignored display name",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(app_module, "get_shared_metaclip", lambda _device: object())
    monkeypatch.setattr(
        app_module.ObjectSearchMapRegistry,
        "from_config_file",
        lambda _config_path, device=None: _DummyRegistry(),
    )

    app = app_module.create_app(cfg)

    with TestClient(app) as client:
        response = client.get("/ui/api/maps")

    assert response.status_code == 200
    assert response.json() == {
        "maps": [
            {
                "id": "station-a",
                "display_name": "station-a",
                "path": str((tmp_path / "maps" / "station-a").resolve()),
                "emmid": None,
                "object_search_index_path": None,
            }
        ]
    }


def test_ui_route_returns_clear_message_when_build_is_missing(
    tmp_path: Path,
    monkeypatch,
):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"maps": [{"id": "station-a"}]}), encoding="utf-8")

    monkeypatch.setattr(app_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(app_module, "get_shared_metaclip", lambda _device: object())
    monkeypatch.setattr(
        app_module.ObjectSearchMapRegistry,
        "from_config_file",
        lambda _config_path, device=None: _DummyRegistry(),
    )
    monkeypatch.setattr(app_module, "_UI_DIST_DIR", tmp_path / "missing-dist")

    app = app_module.create_app(cfg)

    with TestClient(app) as client:
        response = client.get("/ui")

    assert response.status_code == 404
    assert "Object-search UI build not found" in response.text


def test_text_search_endpoint_logs_history_to_map_history_service(
    tmp_path: Path, monkeypatch
):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"maps": [{"id": "station-a"}]}), encoding="utf-8")
    registry = _DummyObjectSearchRegistry()

    monkeypatch.setattr(app_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(app_module, "get_shared_metaclip", lambda _device: object())
    monkeypatch.setattr(
        app_module.ObjectSearchMapRegistry,
        "from_config_file",
        lambda _config_path, device=None: registry,
    )

    app = app_module.create_app(cfg)

    with TestClient(app) as client:
        response = client.post(
            "/station-a/object-search/text",
            json={"text": "ticket machine", "num_results": 5, "search_type": "cutout"},
        )

    assert response.status_code == 200
    assert registry.history.requests == [
        {
            "prompt": "ticket machine",
            "search_type": "cutout",
            "enforced": True,
            "timestamp": registry.history.requests[0]["timestamp"],
            "time_router_ms": 0,
            "time_embedding_ms": 11,
            "time_db_ms": 22,
        }
    ]


def test_localization_endpoints_log_history_to_map_history_service(
    tmp_path: Path,
    monkeypatch,
):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"maps": [{"id": "station-a"}]}), encoding="utf-8")
    registry = _DummyObjectSearchRegistry()

    monkeypatch.setattr(app_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(app_module, "get_shared_metaclip", lambda _device: object())
    monkeypatch.setattr(
        app_module.ObjectSearchMapRegistry,
        "from_config_file",
        lambda _config_path, device=None: registry,
    )

    app = app_module.create_app(cfg)

    with TestClient(app) as client:
        localized = client.post(
            "/station-a/object-search/localize-offline",
            json={"text": "exit sign", "num_results": 3, "search_type": "object"},
        )
        online = client.post(
            "/station-a/object-search/localize",
            json={"text": "bench", "num_results": 3, "search_type": "object"},
        )

    assert localized.status_code == 200
    assert online.status_code == 200
    assert [
        {
            "prompt": req["prompt"],
            "search_type": req["search_type"],
            "enforced": req["enforced"],
            "time_router_ms": req["time_router_ms"],
            "time_embedding_ms": req["time_embedding_ms"],
            "time_db_ms": req["time_db_ms"],
        }
        for req in registry.history.requests
    ] == [
        {
            "prompt": "exit sign",
            "search_type": "object",
            "enforced": True,
            "time_router_ms": 0,
            "time_embedding_ms": 33,
            "time_db_ms": 44,
        },
        {
            "prompt": "bench",
            "search_type": "object",
            "enforced": True,
            "time_router_ms": 0,
            "time_embedding_ms": 55,
            "time_db_ms": 66,
        },
    ]
    assert registry.service.last_online_localization_params.clustering_eps_m == 2.0


def test_localization_endpoint_accepts_single_image_upload(
    tmp_path: Path,
    monkeypatch,
):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"maps": [{"id": "station-a"}]}), encoding="utf-8")
    registry = _DummyObjectSearchRegistry()

    monkeypatch.setattr(app_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(app_module, "get_shared_metaclip", lambda _device: object())
    monkeypatch.setattr(
        app_module.ObjectSearchMapRegistry,
        "from_config_file",
        lambda _config_path, device=None: registry,
    )

    app = app_module.create_app(cfg)
    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02"
        b"\xfeA\xe2%\xb3\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    with TestClient(app) as client:
        response = client.post(
            "/station-a/object-search/localize-offline",
            data={"num_results": "4", "min_similarity": "0.35"},
            files={"image": ("query.png", png_1x1, "image/png")},
        )

    assert response.status_code == 200
    assert registry.service.last_image_query == {
        "mode": "RGB",
        "size": (1, 1),
        "num_results": 4,
        "min_similarity": 0.35,
        "log_query": "[image] query.png",
    }
    assert registry.history.requests[-1]["prompt"] == "[image] query.png"
    assert registry.history.requests[-1]["time_embedding_ms"] == 77
    assert registry.history.requests[-1]["time_db_ms"] == 88


def test_localization_endpoint_rejects_mixed_text_and_image(
    tmp_path: Path,
    monkeypatch,
):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"maps": [{"id": "station-a"}]}), encoding="utf-8")
    registry = _DummyObjectSearchRegistry()

    monkeypatch.setattr(app_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(app_module, "get_shared_metaclip", lambda _device: object())
    monkeypatch.setattr(
        app_module.ObjectSearchMapRegistry,
        "from_config_file",
        lambda _config_path, device=None: registry,
    )

    app = app_module.create_app(cfg)

    with TestClient(app) as client:
        response = client.post(
            "/station-a/object-search/localize-offline",
            data={"text": "chair"},
            files={"image": ("query.png", b"not important", "image/png")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Provide either text or image, not both"
