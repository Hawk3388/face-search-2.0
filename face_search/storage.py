"""Non-destructive legacy readers and backed-up, additive SQLite migrations."""

import hashlib
import json
import sqlite3
import shutil
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .config import DLIB_MODEL
from .search import validate_embedding

SCHEMA_VERSION = 2


def connect_readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)


def backup_database(path):
    """Include committed WAL changes, never overwrite any previous backup."""
    path = Path(path).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = path.with_name(f"{path.name}.backup-{stamp}-{uuid4().hex[:8]}")
    with closing(connect_readonly(path)) as source, closing(sqlite3.connect(target)) as dest:
        source.backup(dest)
        if dest.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"Backup verification failed: {target}; original unchanged")
    return target


def read_json(path):
    """Stream large arrays, also accepting an otherwise complete legacy open array.

    Truncated objects still fail. No repairs are ever written to the source.
    """
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8-sig") as source:
        buffer, eof = "", False

        def refill():
            nonlocal buffer, eof
            chunk = source.read(1024 * 1024)
            eof = not chunk
            buffer += chunk.replace("\x00", "")

        def whitespace():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                refill()
                buffer = buffer.lstrip()

        whitespace()
        if not buffer.startswith("["):
            raise ValueError("Database must contain a JSON array")
        buffer = buffer[1:]
        need_separator = False
        while True:
            whitespace()
            if not buffer:
                return  # Legacy file without a closing bracket, after a complete entry.
            if buffer.startswith("]"):
                buffer = buffer[1:]
                whitespace()
                if buffer:
                    raise ValueError("Unexpected content after JSON array")
                return
            if need_separator:
                if not buffer.startswith(","):
                    raise ValueError("Missing comma between database records")
                buffer = buffer[1:]
                whitespace()
                if not buffer:
                    return
            while True:
                try:
                    row, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
            if not isinstance(row, dict):
                raise ValueError("Every database entry must be an object")
            yield row
            buffer = buffer[end:]
            need_separator = True


def backup_file(path):
    """Unique, byte-verified backup for legacy JSON maintenance tools."""
    path = Path(path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = path.with_name(f"{path.name}.backup-{stamp}-{uuid4().hex[:8]}")
    shutil.copy2(path, target)

    def digest(filename):
        result = hashlib.sha256()
        with filename.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                result.update(chunk)
        return result.digest()

    if digest(path) != digest(target):
        raise RuntimeError("Backup verification failed; stop writers and retry")
    return target


def decode_legacy(blob):
    if blob is None or len(blob) not in (128 * 4, 128 * 8):
        raise ValueError("Invalid legacy embedding; expected 128 float32 or float64 values")
    dtype = "<f4" if len(blob) == 512 else "<f8"
    return validate_embedding(np.frombuffer(blob, dtype=dtype), 128)


def read_sqlite(path):
    with closing(connect_readonly(path)) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        pages = {}
        for image_id, url in conn.execute("SELECT image_id, page_url FROM wiki_pages ORDER BY rowid"):
            if url:
                pages.setdefault(image_id, []).append(url)
        if "faces" in tables:
            query = """SELECT f.id,i.image_url,i.phash,f.embedding,f.model_id,f.dimension,
                       f.bbox,i.id,f.preprocessing FROM faces f JOIN images i ON i.id=f.image_id ORDER BY f.id"""
            for face_id, url, phash, blob, model_id, dim, bbox, image_id, preprocessing in conn.execute(query):
                yield {
                    "id": face_id,
                    "image_id": image_id,
                    "image_url": url,
                    "phash": phash,
                    "embedding": validate_embedding(np.frombuffer(blob, dtype="<f4"), dim),
                    "model_id": model_id,
                    "bbox": json.loads(bbox) if bbox else None,
                    "preprocessing": preprocessing,
                    "page_urls": pages.get(image_id, []),
                }
        else:
            for image_id, url, phash, blob in conn.execute(
                "SELECT id,image_url,phash,embedding FROM images ORDER BY id"
            ):
                if blob is not None:
                    yield {
                        "id": image_id,
                        "image_id": image_id,
                        "image_url": url,
                        "phash": phash,
                        "embedding": decode_legacy(blob),
                        "model_id": DLIB_MODEL,
                        "page_urls": pages.get(image_id, []),
                    }


def read_records(path, backend):
    return read_sqlite(path) if backend == "sqlite" else read_json(path)


def ensure_schema(path):
    """Retain all legacy tables/columns/rows. Back up before the first mutation."""
    path = Path(path)
    backup = None
    if path.exists():
        with closing(connect_readonly(path)) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            return None
        if version > SCHEMA_VERSION:
            raise ValueError("Database schema is newer than this application")
        backup = backup_database(path)
    with closing(sqlite3.connect(path, timeout=30)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT, image_url TEXT UNIQUE, phash TEXT, embedding BLOB)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS wiki_pages (
                image_id INTEGER, page_url TEXT, PRIMARY KEY(image_id,page_url),
                FOREIGN KEY(image_id) REFERENCES images(id))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY, image_id INTEGER NOT NULL REFERENCES images(id),
                model_id TEXT NOT NULL, dimension INTEGER NOT NULL,
                preprocessing TEXT NOT NULL, bbox TEXT, embedding BLOB NOT NULL,
                embedding_hash TEXT NOT NULL, UNIQUE(image_id,model_id,embedding_hash))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS crawl_progress (
                name TEXT PRIMARY KEY, page_url TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_model ON faces(model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_images_phash ON images(phash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pages_url ON wiki_pages(page_url)")
            for image_id, blob in conn.execute("SELECT id,embedding FROM images WHERE embedding IS NOT NULL"):
                insert_face(conn, image_id, {"embedding": decode_legacy(blob)})
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return backup


def insert_face(conn, image_id, entry):
    model_id = entry.get("model_id", DLIB_MODEL)
    dim = int(entry.get("dimension", 128 if model_id == DLIB_MODEL else len(entry["embedding"])))
    vector = validate_embedding(entry["embedding"], dim)
    blob = vector.astype("<f4").tobytes()
    conn.execute(
        """INSERT OR IGNORE INTO faces
        (image_id,model_id,dimension,preprocessing,bbox,embedding,embedding_hash)
        VALUES (?,?,?,?,?,?,?)""",
        (
            image_id,
            model_id,
            dim,
            entry.get("preprocessing", "dlib-rgb-large-v1"),
            json.dumps(entry["bbox"]) if entry.get("bbox") is not None else None,
            blob,
            hashlib.sha256(blob).hexdigest(),
        ),
    )


def save_article(conn, entries, page_url=None, category_url=None):
    """The checkpoint advances only if ALL image/face writes commit successfully."""
    with conn:
        for entry in entries:
            url = entry.get("image_url")
            if not url:
                raise ValueError("Missing image_url")
            conn.execute("INSERT OR IGNORE INTO images(image_url,phash) VALUES (?,?)", (url, entry.get("phash")))
            image_id = conn.execute("SELECT id FROM images WHERE image_url=?", (url,)).fetchone()[0]
            if entry.get("embedding") is not None:
                insert_face(conn, image_id, entry)
            urls = list(entry.get("page_urls") or [])
            urls.extend([entry.get("page_url"), page_url])
            for source in dict.fromkeys(u for u in urls if u):
                conn.execute("INSERT OR IGNORE INTO wiki_pages VALUES (?,?)", (image_id, source))
        if page_url:
            conn.execute(
                "INSERT OR REPLACE INTO crawl_progress VALUES (?,?,?)",
                ("wikipedia", page_url, datetime.now(timezone.utc).isoformat()),
            )
            if category_url:
                conn.execute(
                    "INSERT OR REPLACE INTO crawl_progress VALUES (?,?,?)",
                    ("wikipedia-category", category_url, datetime.now(timezone.utc).isoformat()),
                )


def get_progress(path, name="wikipedia"):
    with closing(connect_readonly(path)) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='crawl_progress'").fetchone():
            return None
        row = conn.execute("SELECT page_url FROM crawl_progress WHERE name=?", (name,)).fetchone()
        return row[0] if row else None
