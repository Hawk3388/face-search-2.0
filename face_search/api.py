import hmac
import json
import logging
import math
import sqlite3
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager, closing
from pathlib import Path
from threading import Lock

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

from .config import ROOT, DLIB_MODEL, Settings
from .encoding import FaceEncoder, load_image
from .search import SearchIndex
from .storage import read_records

log = logging.getLogger(__name__)


def boolean(value, name):
    if isinstance(value, bool):
        return value
    if value in ("true", "1", "false", "0"):
        return value in ("true", "1")
    raise ValueError(f"{name} must be a boolean")


def integer(value, name, lower, upper):
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return result


class SearchService:
    def __init__(self, settings, encoder=None):
        self.settings = settings
        self.encoder = encoder or FaceEncoder(settings)
        self.index = None
        self.reload_lock = Lock()
        self.encode_lock = Lock()
        self.last_reload_failed = False

    def reload(self):
        with self.reload_lock:
            try:
                index = SearchIndex.build(
                    read_records(self.settings.database, self.settings.backend),
                    self.encoder.model_id,
                    self.encoder.dimension,
                )
                if not index.entries:
                    raise ValueError("No embeddings for the configured model in this database")
            except Exception:
                self.last_reload_failed = True
                raise
            self.index = index
            self.last_reload_failed = False
        return len(index.entries)

    def analyze(self, data):
        if not self.encode_lock.acquire(blocking=False):
            raise HTTPException(429, "Face analysis is busy; please retry shortly", headers={"Retry-After": "1"})
        try:
            image = load_image(data, self.settings.max_upload_bytes, self.settings.max_pixels)
            locations, vectors = self.encoder.extract(image)
        finally:
            self.encode_lock.release()
        return {
            "face_locations": [list(map(int, loc)) for loc in locations],
            "encodings": [vector.tolist() for vector in vectors],
            "face_count": len(vectors),
            "model_id": self.encoder.model_id,
        }


class RequestStats:
    """A separate counter DB, safe across threads and server workers."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = Lock()

    def update(self, increment=0):
        with self.lock, closing(sqlite3.connect(self.path, timeout=10)) as conn:
            with conn:
                conn.execute("CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL)")
                initial = 0
                legacy = self.path.parent / "stats.txt"
                if legacy.exists():
                    try:
                        initial = int(legacy.read_text().strip())
                    except (OSError, ValueError):
                        pass
                conn.execute("INSERT OR IGNORE INTO counters VALUES ('requests', ?)", (initial,))
                conn.execute("UPDATE counters SET value=value+? WHERE name='requests'", (increment,))
                return conn.execute("SELECT value FROM counters WHERE name='requests'").fetchone()[0]


async def request_inputs(request, settings, require_image=False):
    """Bound bytes before multipart parsing, including bodies without Content-Length.

    Only one file and a small set of fields are accepted. The form context closes
    spooled files on both success and failure; encoder work receives plain bytes.
    """
    limit = settings.max_upload_bytes + 64 * 1024
    length = request.headers.get("content-length")
    if length is not None:
        try:
            length = int(length)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length < 0:
            raise ValueError("Invalid Content-Length")
        if length > limit:
            raise HTTPException(413, "Request exceeds the upload limit")
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower().strip()
    multipart = content_type == "multipart/form-data"
    if not multipart and (require_image or not (content_type == "application/json" or content_type.endswith("+json"))):
        raise ValueError(
            "An image upload is required" if require_image else "Expected a JSON object or an image upload"
        )
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
            raise HTTPException(413, "Request exceeds the upload limit")
        body.extend(chunk)
    if multipart:

        async def receive():
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        parsed = Request(request.scope, receive)
        async with parsed.form(max_files=1, max_fields=20, max_part_size=64 * 1024) as form:
            upload = form.get("image")
            if not isinstance(upload, UploadFile) or not upload.filename:
                raise ValueError("An image upload is required")
            data = await upload.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, "Image exceeds the upload limit")
            return {key: value for key, value in form.items() if key != "image"}, data
    try:
        params = json.loads(body)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Expected a JSON object or an image upload") from exc
    if not isinstance(params, dict):
        raise ValueError("Expected a JSON object or an image upload")
    return params, None


def create_app(backend="sqlite", settings=None, encoder=None, stats_path=None):
    settings = settings or Settings.from_env(backend)
    if not settings.token and not settings.allow_unauthenticated:
        raise RuntimeError("Set TOKEN or explicitly enable ALLOW_UNAUTHENTICATED=1 for local development")
    service = SearchService(settings, encoder)
    stats = RequestStats(stats_path or ROOT / "request_stats.db")

    @asynccontextmanager
    async def lifespan(app):
        try:
            await run_in_threadpool(service.reload)
        except Exception:
            log.exception("Search database not ready; /reload can retry after repair")
        yield

    app = FastAPI(
        title="Face Search API",
        version="2.2.0",
        lifespan=lifespan,
    )
    secured = [Depends(HTTPBearer(auto_error=False))]
    app.state.face_search = service
    rate_lock, rate_clients = Lock(), OrderedDict()

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        # Documentation contains no database contents; data endpoints remain protected.
        public = request.method in ("GET", "HEAD") and request.url.path in (
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        )
        if public:
            return await call_next(request)
        if settings.token:
            token = request.headers.get("Authorization", "").removeprefix("Bearer ")
            if not hmac.compare_digest(token.encode(), settings.token.encode()):
                return JSONResponse({"error": "Unauthorized"}, 401)
        if (
            settings.requests_per_minute
            and request.method == "POST"
            and request.url.path in ("/analyze", "/search", "/reload")
        ):
            now = time.monotonic()
            # Trust the peer address, never a user-supplied forwarding header.
            client = request.client.host if request.client else "unknown"
            with rate_lock:
                history = rate_clients.setdefault(client, deque())
                rate_clients.move_to_end(client)
                while history and history[0] <= now - 60:
                    history.popleft()
                if len(history) >= settings.requests_per_minute:
                    return JSONResponse(
                        {"error": "Request limit reached; retry in one minute"}, 429, headers={"Retry-After": "60"}
                    )
                history.append(now)
                while len(rate_clients) > 10000:
                    rate_clients.popitem(last=False)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid_input(request, error):
        return JSONResponse({"error": str(error)}, 400)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse({"error": "Invalid request"}, 400)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse({"error": str(error.detail)}, error.status_code, headers=error.headers)

    @app.exception_handler(Exception)
    async def internal_error(request, error):
        log.exception("Request failed")
        return JSONResponse({"error": "Internal server error"}, 500)

    image_schema = {
        "type": "object",
        "required": ["image"],
        "properties": {
            "image": {"type": "string", "format": "binary"},
            "selected_face": {"type": "integer", "minimum": 0, "default": 0},
            "format": {"type": "string", "enum": ["objects", "legacy"]},
        },
    }
    upload_docs = {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": image_schema}}}}

    @app.post("/analyze", openapi_extra=upload_docs, dependencies=secured)
    async def analyze(request: Request):
        _, data = await request_inputs(request, settings, require_image=True)
        return {"success": True, **await run_in_threadpool(service.analyze, data)}

    search_docs = {
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {"schema": image_schema},
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["encoding"],
                        "properties": {
                            "encoding": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": service.encoder.dimension,
                                "maxItems": service.encoder.dimension,
                            },
                            "model_id": {"type": "string", "default": service.encoder.model_id},
                            "metric": {"type": "string", "enum": ["cosine", "euclidean"], "default": settings.metric},
                            "tolerance": {"type": "number", "minimum": 0, "maximum": 2},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": settings.max_results},
                            "format": {"type": "string", "enum": ["objects", "legacy"]},
                            "dedup": {"type": "boolean", "default": True},
                            "debug": {"type": "boolean", "default": False},
                            "full_scan": {"type": "boolean", "default": False},
                        },
                    }
                },
            },
        }
    }

    @app.post("/search", openapi_extra=search_docs, dependencies=secured)
    async def search(request: Request):
        index = service.index
        if index is None:
            return JSONResponse({"error": "Search database is not ready"}, 503)
        params, data = await request_inputs(request, settings)
        return await run_in_threadpool(search_sync, index, params, data)

    def search_sync(index, params, data):
        analysis = {}
        if data is not None:
            analysis = service.analyze(data)
            encodings = analysis.pop("encodings")
            selected = integer(params.get("selected_face", 0), "selected_face", 0, max(0, len(encodings) - 1))
            encoding = encodings[selected] if encodings else None
        else:
            if "encoding" not in params:
                raise ValueError("Missing face encoding")
            encoding = params["encoding"]
        model_id = params.get("model_id", service.encoder.model_id if analysis else DLIB_MODEL)
        if model_id != index.model_id:
            raise ValueError("Query model does not match the database model")
        metric = params.get("metric", settings.metric)
        if metric not in ("euclidean", "cosine"):
            raise ValueError("metric must be euclidean or cosine")
        default_tolerance = settings.tolerance if metric == settings.metric else (0.07 if metric == "cosine" else 0.5)
        try:
            tolerance = float(params.get("tolerance", default_tolerance))
        except (TypeError, ValueError) as exc:
            raise ValueError("tolerance must be numeric") from exc
        if not math.isfinite(tolerance) or not 0 <= tolerance <= 2:
            raise ValueError("tolerance must be between 0 and 2")
        maximum = integer(
            params.get("max_results", min(20, settings.max_results)), "max_results", 1, settings.max_results
        )
        full_scan = boolean(params.get("full_scan", False), "full_scan")
        dedup = boolean(params.get("dedup", True), "dedup")
        debug = boolean(params.get("debug", False), "debug")
        if full_scan:
            maximum = settings.max_results
        if encoding is None and analysis:
            matches, meta = [], {}
        else:
            matches, meta = index.search(encoding, tolerance, maximum, metric, dedup)
        response_format = params.get("format", "legacy" if settings.backend == "json" else "objects")
        if response_format not in ("legacy", "objects"):
            raise ValueError("format must be legacy or objects")
        output = (
            [
                [{k: v for k, v in item.items() if k not in ("distance", "similarity")}, item["distance"]]
                for item in matches
            ]
            if response_format == "legacy"
            else matches
        )
        result = dict(
            success=True,
            matches=output,
            total_results=len(matches),
            metric=metric,
            tolerance=tolerance,
            model_id=index.model_id,
            **{k: v for k, v in analysis.items() if k != "model_id"},
        )
        if debug:
            result["meta"] = meta
        try:
            stats.update(1)
        except sqlite3.Error:
            log.exception("Unable to persist request statistics")
        return result

    @app.get("/health", openapi_extra={"security": []})
    @app.head("/health", include_in_schema=False)
    def health():
        index = service.index
        last_page = next((e["page_url"] for e in reversed(index.entries) if e.get("page_url")), None) if index else None
        return JSONResponse(
            dict(
                status="healthy" if index else "unavailable",
                database_loaded=index is not None,
                total_entries=len(index.entries) if index else 0,
                last_page_url=last_page,
                model_id=service.encoder.model_id,
                metric=settings.metric,
                tolerance=settings.tolerance,
                last_reload_failed=service.last_reload_failed,
                capabilities=["image_search", "embedding_search", "analyze"],
            ),
            status_code=200 if index else 503,
        )

    @app.get("/stats", dependencies=secured)
    def statistics():
        index = service.index
        count = stats.update()
        entries = index.entries if index else ()
        size = settings.database.stat().st_size if settings.database.exists() else 0
        return dict(
            total_entries=len(entries),
            total_images=len({e.get("image_url") for e in entries}),
            total_pages=len({p for e in entries for p in e.get("page_urls", [])}),
            requests=count,
            total_requests=count,
            database_size=size,
            database_file_size=size,
        )

    @app.post("/reload", dependencies=secured)
    def reload_database():
        try:
            count = service.reload()
        except Exception:
            log.exception("Database reload failed; keeping previous snapshot")
            return JSONResponse({"error": "Database reload failed; previous index retained"}, 503)
        return dict(success=True, entries=count)

    return app


def run(backend):
    import os
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    app = create_app(backend)
    host = "127.0.0.1" if app.state.face_search.settings.allow_unauthenticated else "0.0.0.0"
    # Never accept user-supplied forwarded client addresses for rate limiting.
    uvicorn.run(app, host=host, port=int(os.getenv("PORT", "7403")), proxy_headers=False)
