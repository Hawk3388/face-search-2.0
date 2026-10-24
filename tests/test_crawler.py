from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import sqlite3
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import requests

from crawler.web_runner import discover_html
from crawler.wiki_api import HttpClient, WikiApi, retry_delay
from crawler.wiki_runner import Crawler
from crawler.wiki_store import CrawlStore
from face_search.config import DLIB_MODEL, Settings
from face_search.storage import ensure_schema


class FakeHttp:
    def __init__(self, replies=()):
        self.replies = iter(replies)
        self.calls = []
        self.cooldowns = []
        self.stop = threading.Event()

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        result = next(self.replies)
        if isinstance(result, Exception):
            raise result
        return json.dumps(result).encode() if isinstance(result, dict) else result

    def cooldown(self, url, seconds):
        self.cooldowns.append(seconds)


def info(sha1="hash"):
    return {
        "url": "https://upload.wikimedia.org/photo.jpg",
        "sha1": sha1,
        "title": "File:Photo.jpg",
        "download_url": "https://upload.wikimedia.org/photo.jpg",
        "mime": "image/jpeg",
    }


@pytest.fixture
def store(tmp_path):
    instance = CrawlStore(tmp_path / "new.db", min_free_bytes=0)
    yield instance
    instance.close()


def article(store, profile="p", metadata=None, page_url="https://example.org/Person", page_id=1):
    store.enqueue_batch("scope", [{"pageid": page_id, "title": "Person"}], None, 86400)
    row = next(row for row in store.due_articles("scope") if row["page_id"] == page_id)
    store.discover("scope", row, page_url, [metadata or info()], profile)


def test_api_finds_commons_files_and_all_continuations():
    first = {"title": "File:A.jpg", "missing": True, "imageinfo": [info()]}
    second = {"title": "File:B.webp", "imageinfo": [{**info(), "url": "https://upload.wikimedia.org/b.webp"}]}
    http = FakeHttp(
        [
            {"query": {"pages": [first]}, "continue": {"continue": "||", "gimcontinue": "42|A"}},
            {"query": {"pages": [second]}},
        ]
    )
    images = WikiApi(http).images(42)
    assert len(images) == 2
    params = http.calls[1][1]["params"]
    assert params["gimcontinue"] == "42|A"
    assert params["continue"] == "||"
    assert params["gimlimit"] == 50
    assert params["maxlag"] == 5


def test_empty_page_and_category_are_successful():
    api = WikiApi(FakeHttp([{"batchcomplete": True}, {"query": {"categorymembers": []}}]))
    assert api.images(1) == []
    assert api.category_batch("Category:Empty") == ([], None)


def test_category_uses_api_cursor_and_main_namespace():
    http = FakeHttp([{"query": {"categorymembers": [{"pageid": 9, "title": "B"}]}}])
    WikiApi(http).category_batch("Category:People", {"cmcontinue": "page|B", "continue": "-||"})
    params = http.calls[0][1]["params"]
    assert params["cmcontinue"] == "page|B"
    assert params["cmnamespace"] == 0
    assert params["cmtype"] == "page"


def test_http_200_api_errors_are_not_empty_articles():
    http = FakeHttp([{"error": {"code": "maxlag", "lag": 8}}, {"batchcomplete": True}])
    assert WikiApi(http).images(1) == []
    assert http.cooldowns == [8]
    with pytest.raises(ValueError, match="permissiondenied"):
        WikiApi(FakeHttp([{"error": {"code": "permissiondenied"}}])).images(1)


def test_repeated_image_cursor_fails_instead_of_looping():
    reply = {"query": {}, "continue": {"gimcontinue": "same"}}
    with pytest.raises(ValueError, match="Repeated"):
        WikiApi(FakeHttp([reply, reply])).images(1)


def test_html_lazy_picture_srcset_noscript_and_extensionless_images():
    images, links = discover_html(
        """
        <base href="https://example.org/photos/">
        <img src="data:image/gif;base64,placeholder" data-src="portrait?id=1">
        <picture><source srcset="small.webp 320w, large.webp 1200w"></picture>
        <img data-srcset="one.jpg 1x, two.jpg 2x">
        <noscript><img src="hidden.jpg"></noscript>
        <meta property="og:image" content="social.png">
        <a href="../person#photo">Person</a><a href="https://outside.org/">Outside</a>
    """,
        "https://example.org/entry",
    )
    assert set(images) == {
        "https://example.org/photos/" + name
        for name in ("portrait?id=1", "large.webp", "two.jpg", "hidden.jpg", "social.png")
    }
    assert links == ["https://example.org/person"]


def test_failed_image_survives_article_checkpoint_and_restart(tmp_path):
    path = tmp_path / "resume.db"
    store = CrawlStore(path, min_free_bytes=0)
    article(store)
    job = store.due_images("p")[0]
    store.fail_image(job, "timeout")
    assert not store.due_articles("scope")
    store.close()
    reopened = CrawlStore(path, min_free_bytes=0)
    try:
        row = reopened.conn.execute("SELECT * FROM crawler_images").fetchone()
        assert row["status"] == "retry" and row["attempts"] == 1
        assert not reopened.due_images("p")
        with reopened.conn:
            reopened.conn.execute("UPDATE crawler_images SET retry_at=0")
        assert len(reopened.due_images("p")) == 1
        assert list(tmp_path.glob("resume.db.backup-*"))
    finally:
        reopened.close()


def test_negative_cache_invalidates_on_content_or_profile_change(store):
    article(store)
    store.complete_image(store.due_images("p")[0], [], {})
    article(store)
    assert not store.due_images("p")
    article(store, metadata=info("new-hash"))
    assert len(store.due_images("p")) == 1
    article(store, profile="different-model")
    assert len(store.due_images("different-model")) == 1


def test_multiface_commit_preserves_provenance_and_rolls_back_bad_vectors(store):
    article(store)
    article(store, page_url="https://example.org/Other", page_id=2)
    job = store.due_images("p")[0]
    good = {"embedding": np.ones(128), "bbox": [0, 10, 10, 0]}
    with pytest.raises(ValueError):
        store.complete_image(job, [good, {"embedding": [float("nan")] * 128}], {})
    assert store.conn.execute("SELECT count(*) FROM faces").fetchone()[0] == 0
    assert len(store.due_images("p")) == 1
    store.complete_image(job, [good, {"embedding": np.arange(128)}], {})
    assert store.conn.execute("SELECT count(*) FROM faces").fetchone()[0] == 2
    assert store.conn.execute("SELECT count(*) FROM wiki_pages").fetchone()[0] == 2
    assert not store.due_images("p")


def test_category_cursor_cannot_advance_without_queued_articles(store):
    with pytest.raises(KeyError):
        store.enqueue_batch("scope", [{"pageid": 1, "title": "A"}, {"title": "Broken"}], {"cmcontinue": "next"}, 86400)
    assert store.scan("scope")["cursor"] is None
    assert not store.due_articles("scope")


def test_failed_backup_prevents_any_existing_database_mutation(tmp_path, monkeypatch):
    path = tmp_path / "existing.db"
    ensure_schema(path)
    before = path.read_bytes()

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr("crawler.wiki_store.backup_database", fail)
    with pytest.raises(OSError, match="disk full"):
        CrawlStore(path, min_free_bytes=0)
    assert path.read_bytes() == before
    with sqlite3.connect(path) as conn:
        assert not conn.execute("SELECT name FROM sqlite_master WHERE name='crawler_images'").fetchall()


def test_low_disk_stops_before_creating_database(tmp_path, monkeypatch):
    monkeypatch.setattr("crawler.wiki_store.shutil.disk_usage", lambda _: SimpleNamespace(free=0))
    path = tmp_path / "new.db"
    with pytest.raises(OSError, match="disk space"):
        CrawlStore(path)
    assert not path.exists()


def test_periodic_backup_requires_data_changes_and_elapsed_time(store, monkeypatch):
    calls = []
    monkeypatch.setattr("crawler.wiki_store.backup_database", lambda path: calls.append(path))
    store.last_backup -= store.backup_seconds + 1
    article(store)
    store.complete_image(store.due_images("p")[0], [], {})
    store.maintain()
    assert not calls
    article(store, metadata=info("new"))
    store.complete_image(store.due_images("p")[0], [{"embedding": np.ones(128)}], {})
    store.maintain()
    assert len(calls) == 1
    store.maintain()
    assert len(calls) == 1


def image_bytes(width=100):
    out = BytesIO()
    Image.new("RGB", (width, width)).save(out, "PNG")
    return out.getvalue()


class Encoder:
    model_id = DLIB_MODEL
    preprocessing = "dlib-rgb-large-v1"
    dimension = 128

    def extract(self, image):
        if image.shape[0] < 200:
            return [], []
        return [(0, 100, 100, 0)], [np.ones(128)]


def test_preview_without_face_gets_full_resolution_fallback(store):
    http = FakeHttp([image_bytes(400)])
    crawler = Crawler(WikiApi(http), store, Encoder(), Settings(store.path))
    metadata = {**info(), "download_url": "https://upload.wikimedia.org/small.png", "width": 400, "height": 400}
    entries, quality = crawler.analyze(metadata, image_bytes(100))
    assert len(entries) == 1
    assert quality["width"] == 400
    assert http.calls[0][0] == metadata["url"]


def test_pipeline_persists_download_failure_then_recovers_without_recrawling_article(store):
    http = FakeHttp([requests.Timeout("simulated"), image_bytes(400)])
    crawler = Crawler(WikiApi(http), store, Encoder(), Settings(store.path), workers=1)
    article(store, profile=crawler.profile)
    with ThreadPoolExecutor(1) as pool:
        crawler.process_images(pool)
        assert store.counts(crawler.profile) == {"retry": 1}
        with store.conn:
            store.conn.execute("UPDATE crawler_images SET retry_at=0")
        crawler.process_images(pool)
    assert store.counts(crawler.profile) == {"done": 1}
    assert not store.due_articles("scope")


def test_one_pass_processes_final_category_batch(store):
    http = FakeHttp(
        [
            {"query": {"categorymembers": [{"pageid": 4, "title": "Person"}]}},
            {"query": {"pages": [{"title": "File:Photo.jpg", "imageinfo": [info()]}]}},
            image_bytes(400),
        ]
    )
    crawler = Crawler(WikiApi(http), store, Encoder(), Settings(store.path), workers=1)
    crawler.run("Category:People", once=True)
    assert store.counts(crawler.profile) == {"done": 1}


def test_retry_after_http_date_and_seconds(monkeypatch):
    monkeypatch.setattr("crawler.wiki_api.time.time", lambda: 0)
    assert retry_delay("Thu, 01 Jan 1970 00:02:00 GMT", 0) == 120
    assert retry_delay("600", 0) == 600  # Never retry earlier than the server asks.
    assert retry_delay("invalid", 0) == 2


def test_sessions_reused_and_download_size_bounded(monkeypatch):
    class Response:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b"12345"
            yield b"67890"

    http = HttpClient("test", interval=0)
    session = http.session()
    assert http.session() is session
    monkeypatch.setattr(session, "get", lambda *args, **kwargs: Response())
    try:
        with pytest.raises(ValueError, match="byte limit"):
            http.get("https://example.org/image", max_bytes=8)
    finally:
        http.close()


def test_429_sets_shared_cooldown_and_retries(monkeypatch):
    class Response:
        def __init__(self, status):
            self.status_code = status
            self.headers = {"Retry-After": "90"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            if self.status_code == 429:
                raise requests.HTTPError(response=self)

        def iter_content(self, size):
            yield b"image"

    http = HttpClient("test", interval=0)
    responses = iter([Response(429), Response(200)])
    cooldowns = []
    monkeypatch.setattr(http.session(), "get", lambda *a, **k: next(responses))
    monkeypatch.setattr(http, "pace", lambda _: None)
    monkeypatch.setattr(http, "wait", lambda _: None)
    monkeypatch.setattr(http, "cooldown", lambda url, delay: cooldowns.append(delay))
    try:
        assert http.get("https://example.org/photo") == b"image"
        assert cooldowns == [90]
    finally:
        http.close()


def test_downloads_overlap_and_use_only_configured_workers(store):
    import time

    class ParallelHttp(FakeHttp):
        active = 0
        maximum = 0
        downloads = 0
        lock = threading.Lock()

        def get(self, url, **kwargs):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                self.downloads += 1
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return image_bytes(400)

    http = ParallelHttp()
    crawler = Crawler(WikiApi(http), store, Encoder(), Settings(store.path), workers=3)
    for i in range(6):
        article(store, profile=crawler.profile, metadata={**info(), "url": f"https://example.org/{i}.png"})
    with ThreadPoolExecutor(3) as pool:
        crawler.process_images(pool)
        crawler.process_images(pool)
    assert http.maximum == 3
    assert http.downloads == 6  # The second pass does not download completed jobs.
    assert store.counts(crawler.profile) == {"done": 6}


def test_stop_retains_pending_work(store):
    http = FakeHttp()
    crawler = Crawler(WikiApi(http), store, Encoder(), Settings(store.path), workers=1)
    article(store, profile=crawler.profile)
    http.stop.set()
    with ThreadPoolExecutor(1) as pool:
        crawler.process_images(pool)
    assert store.counts(crawler.profile) == {"pending": 1}
    assert http.calls == []


def test_second_crawler_cannot_write_same_database(store):
    from filelock import Timeout

    with pytest.raises(Timeout):
        CrawlStore(store.path, min_free_bytes=0)


def test_server_defaults_to_sqlite_with_explicit_json_mode(monkeypatch):
    import main_server

    monkeypatch.setattr(main_server, "load_dotenv", lambda *a: None)
    monkeypatch.setattr(main_server, "_create_app", lambda backend: backend)
    monkeypatch.delenv("SERVER_BACKEND", raising=False)
    assert main_server.create_app() == "sqlite"
    monkeypatch.setenv("SERVER_BACKEND", "json")
    assert main_server.create_app() == "json"


def test_system_trust_keeps_tls_verification_and_explicit_bundle():
    import ssl
    http = HttpClient("test")
    try:
        adapter = http.session().get_adapter("https://example.org")
        request = requests.Request("GET", "https://example.org").prepare()
        _, pool = adapter.build_connection_pool_key_attributes(request, True)
        assert pool["ssl_context"].verify_mode == ssl.CERT_REQUIRED
        assert pool["ssl_context"].check_hostname
        _, custom = adapter.build_connection_pool_key_attributes(request, "custom-ca.pem")
        assert custom["ca_certs"] == "custom-ca.pem"
        assert custom.get("ssl_context") is not http.tls_context
    finally:
        http.close()
