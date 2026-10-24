import sqlite3
from contextlib import closing

import numpy as np
import pytest

from face_search.storage import backup_database, ensure_schema, read_sqlite, save_article


def legacy_database(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE images(id INTEGER PRIMARY KEY, image_url TEXT UNIQUE, phash TEXT, embedding BLOB)")
    conn.execute("CREATE TABLE wiki_pages(image_id INTEGER, page_url TEXT, PRIMARY KEY(image_id,page_url))")
    conn.execute(
        "INSERT INTO images VALUES (1,'https://example.org/a.jpg','hash',?)",
        (np.ones(128, dtype=np.float64).tobytes(),),
    )
    conn.execute("INSERT INTO wiki_pages VALUES (1,'https://example.org/source')")
    conn.commit()
    return conn


def test_wal_backup_contains_committed_changes(tmp_path):
    path = tmp_path / "legacy.db"
    with closing(legacy_database(path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("INSERT INTO wiki_pages VALUES (1,'https://example.org/new')")
        conn.commit()
        backup = backup_database(path)
        second = backup_database(path)
        assert backup != second
        with closing(sqlite3.connect(backup)) as copy:
            assert copy.execute("SELECT COUNT(*) FROM wiki_pages").fetchone()[0] == 2


def test_additive_migration_and_multiple_faces(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_database(path).close()
    backup = ensure_schema(path)
    assert backup.exists()
    assert ensure_schema(path) is None
    assert len(list(read_sqlite(path))) == 1
    with closing(sqlite3.connect(path)) as conn:
        before = conn.execute("SELECT embedding FROM images").fetchone()[0]
        entries = [{"image_url": "https://example.org/a.jpg", "embedding": np.full(128, n)} for n in (2.0, 3.0)]
        save_article(conn, entries, "https://example.org/other")
        save_article(conn, entries, "https://example.org/other")
        assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        assert conn.execute("SELECT embedding FROM images").fetchone()[0] == before
        assert conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 3
    assert len(list(read_sqlite(path))) == 3


def test_failed_article_does_not_advance_checkpoint(tmp_path):
    path = tmp_path / "new.db"
    ensure_schema(path)
    with closing(sqlite3.connect(path)) as conn:
        save_article(conn, [], "old")
        with pytest.raises(ValueError):
            save_article(conn, [{"image_url": "new", "embedding": [0] * 128}], "new")
        assert conn.execute("SELECT page_url FROM crawl_progress").fetchone()[0] == "old"
        assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0


def test_invalid_legacy_migration_rolls_back_and_keeps_backup(tmp_path):
    path = tmp_path / "legacy.db"
    with closing(legacy_database(path)) as conn:
        conn.execute("UPDATE images SET embedding=?", (b"bad",))
        conn.commit()
    with pytest.raises(ValueError):
        ensure_schema(path)
    assert list(tmp_path.glob("legacy.db.backup-*"))
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT embedding FROM images").fetchone()[0] == b"bad"
        assert not conn.execute("SELECT name FROM sqlite_master WHERE name='faces'").fetchone()


def test_backup_failure_blocks_migration(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    legacy_database(path).close()
    original = path.read_bytes()

    def fail_backup(path):
        raise OSError("disk full")

    monkeypatch.setattr("face_search.storage.backup_database", fail_backup)
    with pytest.raises(OSError, match="disk full"):
        ensure_schema(path)
    assert path.read_bytes() == original


def test_image_merge_keeps_faces_and_sources(tmp_path):
    from tools.merge_sql import merge_database

    path = tmp_path / "faces.db"
    ensure_schema(path)
    original = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Test.jpg"
    thumb = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Test.jpg/100px-Test.jpg"
    with closing(sqlite3.connect(path)) as conn:
        save_article(conn, [{"image_url": original, "embedding": np.ones(128)}], "source-one")
        save_article(conn, [{"image_url": thumb, "embedding": np.full(128, 2.0)}], "source-two")
    merge_database(path)
    records = list(read_sqlite(path))
    assert len(records) == 2
    assert all(set(row["page_urls"]) == {"source-one", "source-two"} for row in records)
    assert all(row["image_url"] == original for row in records)
