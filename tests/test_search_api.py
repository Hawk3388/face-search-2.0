from dataclasses import replace
from io import BytesIO
import json
import sqlite3

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from face_search.api import create_app
from face_search.config import DLIB_MODEL, Settings
from face_search.search import SearchIndex
from face_search.storage import ensure_schema, save_article


def vector(index=0):
    out = np.zeros(128, dtype=np.float32)
    out[index] = 1
    return out


class FakeEncoder:
    model_id = DLIB_MODEL
    dimension = 128

    def extract(self, image):
        return [(0, 10, 10, 0), (10, 20, 20, 10)], [vector(), vector(1)]


@pytest.fixture(params=["json", "sqlite"])
def api(tmp_path, request):
    rows = [
        {
            "image_url": f"https://example.org/{i}.jpg",
            "page_url": "https://example.org/source",
            "embedding": vector(i).tolist(),
        }
        for i in range(2)
    ]
    path = tmp_path / ("faces.json" if request.param == "json" else "faces.db")
    if request.param == "json":
        path.write_text(json.dumps(rows))
    else:
        ensure_schema(path)
        with sqlite3.connect(path) as conn:
            save_article(conn, rows)
    settings = Settings(database=path, backend=request.param, token="test-token")
    app = create_app(settings=settings, encoder=FakeEncoder(), stats_path=tmp_path / "stats.db")
    with TestClient(app) as client:
        yield client, path


def post(app, payload):
    return app.post(
        "/search",
        content=json.dumps(payload),
        headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"},
    )


def test_embedding_and_second_uploaded_face_match(api):
    app, _ = api
    result = post(app, {"encoding": vector(1).tolist(), "format": "objects"})
    assert result.status_code == 200
    assert result.json()["matches"][0]["image_url"].endswith("1.jpg")
    assert "embedding" not in result.json()["matches"][0]
    image = BytesIO()
    Image.new("RGB", (20, 20)).save(image, "PNG")
    result = app.post(
        "/search",
        files={"image": ("test.png", image.getvalue(), "image/png")},
        data={"selected_face": "1", "format": "objects"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert result.status_code == 200
    assert result.json()["face_count"] == 2
    assert result.json()["matches"][0]["image_url"].endswith("1.jpg")


@pytest.mark.parametrize(
    "change",
    [
        {"encoding": [0] * 128},
        {"encoding": [float("nan")] * 128},
        {"encoding": ["1"] * 128},
        {"encoding": [[1]] * 128},
        {"encoding": [1] * 127},
        {"encoding": None},
        {"tolerance": -1},
        {"tolerance": "nan"},
        {"max_results": 0},
        {"max_results": -2},
        {"max_results": 101},
        {"max_results": 1.5},
        {"dedup": "banana"},
        {"metric": "unknown"},
        {"model_id": "wrong-model"},
    ],
)
def test_invalid_inputs_are_client_errors(api, change):
    app, _ = api
    response = post(app, dict({"encoding": vector().tolist()}, **change))
    assert response.status_code == 400


def test_auth_and_reload_failure_preserve_working_index(api):
    app, path = api
    assert app.post("/search", json={}).status_code == 401
    before = app.app.state.face_search.index
    # Change only test configuration; the test database is not removed.
    service = app.app.state.face_search
    service.settings = replace(service.settings, database=path.parent / "missing")
    response = app.post("/reload", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 503
    assert service.index is before
    assert post(app, {"encoding": vector().tolist()}).status_code == 200


def test_unavailable_database_reports_503(tmp_path):
    app = create_app(settings=Settings(tmp_path / "missing.json", token="test-token"), encoder=FakeEncoder())
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        assert post(client, {"encoding": vector().tolist()}).status_code == 503


def test_missing_token_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="TOKEN"):
        create_app(settings=Settings(tmp_path / "missing"))


def test_dedup_before_limit_and_metrics_agree_with_reference():
    rng = np.random.default_rng(123)
    matrix = rng.normal(size=(25, 128)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    records = [{"image_url": f"https://example.org/{i // 2}", "embedding": row} for i, row in enumerate(matrix)]
    index = SearchIndex.build(records)
    query = matrix[0]
    for metric in ("euclidean", "cosine"):
        results, _ = index.search(query, 2, 3, metric)
        assert len(results) == 3
        assert len({r["image_url"] for r in results}) == 3
        ref = np.linalg.norm(matrix - query, axis=1) if metric == "euclidean" else 1 - matrix @ query
        all_results, _ = index.search(query, 2, 100, metric, dedup=False)
        assert [r["image_url"] for r in all_results] == [
            records[i]["image_url"] for i in np.argsort(ref, kind="stable")
        ]
        np.testing.assert_allclose([r["distance"] for r in all_results], np.sort(ref), atol=0.001)


def test_exact_self_match_at_zero_tolerance():
    query = np.random.default_rng(4).normal(size=128).astype(np.float32)
    index = SearchIndex.build([{"embedding": query, "image_url": "https://example.org/a"}])
    matches, _ = index.search(query, tolerance=0)
    assert len(matches) == 1
    assert matches[0]["distance"] == 0


def test_analysis_invalid_image_and_request_limits(tmp_path):
    path = tmp_path / "faces.json"
    path.write_text(json.dumps([{"embedding": vector().tolist(), "image_url": "https://example.org/a"}]))
    settings = Settings(path, token="test-token", requests_per_minute=2, max_results=1)
    app = create_app(settings=settings, encoder=FakeEncoder(), stats_path=tmp_path / "stats.db")
    with TestClient(app) as client:
        assert (
            client.post(
                "/analyze",
                files={"image": ("bad.jpg", b"not an image", "image/jpeg")},
                headers={"Authorization": "Bearer test-token"},
            ).status_code
            == 400
        )
        assert post(client, {"encoding": vector().tolist()}).status_code == 200
        assert post(client, {"encoding": vector().tolist()}).status_code == 429


def test_statistics_preserve_legacy_count_and_concurrent_updates(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from face_search.api import RequestStats

    (tmp_path / "stats.txt").write_text("15")
    stats = RequestStats(tmp_path / "stats.db")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: stats.update(1), range(20)))
    assert stats.update() == 35
    assert (tmp_path / "stats.txt").read_text() == "15"


def test_legacy_and_object_default_formats_remain_compatible(api):
    client, _ = api
    result = post(client, {"encoding": vector().tolist()}).json()
    first = result["matches"][0]
    if client.app.state.face_search.settings.backend == "json":
        assert isinstance(first, list) and len(first) == 2
        assert "embedding" not in first[0]
        assert isinstance(first[1], float)
    else:
        assert isinstance(first, dict) and "distance" in first
    stats = client.get("/stats", headers={"Authorization": "Bearer test-token"}).json()
    assert stats["requests"] == stats["total_requests"] == 1
    assert stats["total_images"] == 2


def test_analyze_and_documentation(api):
    client, _ = api
    image = BytesIO()
    Image.new("RGB", (20, 20)).save(image, "PNG")
    result = client.post(
        "/analyze", files={"image": ("test.png", image.getvalue())}, headers={"Authorization": "Bearer test-token"}
    )
    assert result.status_code == 200
    assert result.json()["face_count"] == 2
    assert len(result.json()["encodings"]) == 2
    assert client.head("/health").status_code == 200
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert schema["paths"]["/health"]["get"]["security"] == []
    assert schema["paths"]["/search"]["post"]["security"]
    content = schema["paths"]["/search"]["post"]["requestBody"]["content"]
    assert set(content) == {"application/json", "multipart/form-data"}


def small_app(tmp_path, encoder=None, **options):
    path = tmp_path / "faces.json"
    path.write_text(json.dumps([{"embedding": vector().tolist(), "image_url": "https://example.org/a"}]))
    return create_app(
        settings=Settings(path, token="test-token", **options),
        encoder=encoder or FakeEncoder(),
        stats_path=tmp_path / "stats.db",
    )


def test_lifespan_loads_once_before_accepting_requests(tmp_path, monkeypatch):
    app = small_app(tmp_path)
    service = app.state.face_search
    assert service.index is None
    original, calls = service.reload, []

    def reload():
        calls.append(True)
        return original()

    monkeypatch.setattr(service, "reload", reload)
    with TestClient(app) as client:
        assert service.index is not None
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200
        assert len(calls) == 1


@pytest.mark.parametrize("mode", ["length", "stream", "file"])
def test_upload_limits_cover_files_and_chunked_bodies(tmp_path, mode):
    app = small_app(tmp_path, max_upload_bytes=1024)
    with TestClient(app) as client:
        kwargs = {"headers": {"Authorization": "Bearer test-token"}}
        if mode == "file":
            kwargs["files"] = {"image": ("large.jpg", b"x" * 1025)}
        else:
            kwargs["headers"]["Content-Type"] = "application/json"
            data = b"x" * (1024 + 64 * 1024 + 1)
            kwargs["content"] = data if mode == "length" else iter([data[:200], data[200:]])
        response = client.post("/analyze" if mode == "file" else "/search", **kwargs)
        assert response.status_code == 413
        assert "error" in response.json()


def test_auth_rejects_body_before_parsing_and_limits_ignore_forwarded_headers(tmp_path):
    app = small_app(tmp_path, requests_per_minute=1, max_upload_bytes=1024)
    with TestClient(app) as client:
        unauthorized = client.post("/analyze", content=b"x" * 100000)
        assert unauthorized.status_code == 401
        assert post(client, {"encoding": vector().tolist()}).status_code == 200
        blocked = client.post(
            "/search",
            json={"encoding": vector().tolist()},
            headers={"Authorization": "Bearer test-token", "X-Forwarded-For": "203.0.113.3"},
        )
        assert blocked.status_code == 429
        assert blocked.headers["Retry-After"] == "60"


def test_blocked_encoder_does_not_block_health_and_rejects_second_analysis(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    started, release = Event(), Event()

    class SlowEncoder(FakeEncoder):
        def extract(self, image):
            started.set()
            assert release.wait(10), "Test did not release the encoder"
            return super().extract(image)

    app = small_app(tmp_path, encoder=SlowEncoder())
    image = BytesIO()
    Image.new("RGB", (20, 20)).save(image, "PNG")

    def analyze(client):
        return client.post(
            "/analyze", files={"image": ("test.png", image.getvalue())}, headers={"Authorization": "Bearer test-token"}
        )

    with TestClient(app) as client, ThreadPoolExecutor(2) as pool:
        first = pool.submit(analyze, client)
        try:
            assert started.wait(5)
            health = pool.submit(client.get, "/health").result(timeout=3)
            assert health.status_code == 200
            second = pool.submit(analyze, client).result(timeout=3)
            assert second.status_code == 429
            assert second.headers["Retry-After"] == "1"
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200


def test_malformed_requests_and_http_errors_keep_error_envelope(api):
    client, _ = api
    headers = {"Authorization": "Bearer test-token", "Content-Type": "application/json"}
    for body in ("{", "[]", "null"):
        response = client.post("/search", content=body, headers=headers)
        assert response.status_code == 400 and "error" in response.json()
    unknown = client.get("/unknown", headers=headers)
    assert unknown.status_code == 404 and "error" in unknown.json()
    multiple = client.post(
        "/analyze",
        files=[("image", ("a.jpg", b"a")), ("extra", ("b.jpg", b"b"))],
        headers={"Authorization": "Bearer test-token"},
    )
    assert multiple.status_code == 400 and "error" in multiple.json()
