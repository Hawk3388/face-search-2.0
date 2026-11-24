"""Durable crawl queue, additive tables and verified backups before writes."""

import json
import logging
import re
import shutil
import sqlite3
import time
from pathlib import Path
from subprocess import DEVNULL, SubprocessError, run as run_subprocess

from filelock import FileLock

from face_search.storage import backup_database, ensure_schema, insert_face

LOG = logging.getLogger(__name__)


class CrawlStore:
    def __init__(self, path, min_free_bytes=512 * 1024 * 1024, backup_hours=24, *, snapshot_target=None):
        self.path = Path(path).resolve()
        self.snapshot_target = snapshot_target
        self.pending_upload = None
        self.next_upload = 0
        if snapshot_target and not shutil.which("rsync"):
            raise RuntimeError("Install rsync to use CRAWLER_SNAPSHOT_TARGET")
        self.lock = FileLock(str(self.path) + ".crawler.lock", timeout=0)
        self.lock.acquire()
        try:
            self._open(min_free_bytes, backup_hours)
        except BaseException:
            if hasattr(self, "conn"):
                self.conn.close()
            self.lock.release()
            raise

    def _open(self, min_free_bytes, backup_hours):
        self.min_free_bytes = min_free_bytes
        self.backup_seconds = backup_hours * 3600
        self.last_backup = time.monotonic()
        self.dirty = False
        self.check_space(backup=self.path.exists())
        existed = self.path.exists()
        backup = ensure_schema(self.path)
        if existed and backup is None:
            backup = backup_database(self.path)
        if backup:
            self._retain_backup(backup)
            print(f"Verified database backup: {backup}", flush=True)
            self.queue_upload(backup)
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        with self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS crawler_scans (
                    scope TEXT PRIMARY KEY, cursor TEXT, finished INTEGER NOT NULL DEFAULT 0,
                    next_scan REAL NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS crawler_articles (
                    scope TEXT, page_id INTEGER, title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                    retry_at REAL NOT NULL DEFAULT 0, error TEXT,
                    PRIMARY KEY(scope,page_id));
                CREATE INDEX IF NOT EXISTS idx_crawler_articles_due
                    ON crawler_articles(scope,status,retry_at);
                CREATE TABLE IF NOT EXISTS crawler_images (
                    url TEXT, profile TEXT, sha1 TEXT NOT NULL, metadata TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                    retry_at REAL NOT NULL DEFAULT 0, error TEXT, quality TEXT,
                    PRIMARY KEY(url,profile));
                CREATE INDEX IF NOT EXISTS idx_crawler_images_due
                    ON crawler_images(profile,status,retry_at);
                CREATE TABLE IF NOT EXISTS crawler_sources (
                    url TEXT, profile TEXT, page_url TEXT, PRIMARY KEY(url,profile,page_url));
            """)

    def _retain_backup(self, backup):
        """Keep the newly verified snapshot; prune this database's older snapshots.

        Called only after backup_database/ensure_schema succeeds. Never prune
        before a replacement exists, or match other databases/custom archives.
        """
        backup = Path(backup)
        pattern = re.compile(re.escape(self.path.name) + r"\.backup-\d{8}T\d{12}Z-[0-9a-f]{8}")
        if (
            backup.parent != self.path.parent
            or not pattern.fullmatch(backup.name)
            or backup.is_symlink()
            or not backup.is_file()
        ):
            raise ValueError("Verified backup must be a regular snapshot beside the crawler database")
        removed = 0
        for candidate in self.path.parent.iterdir():
            if (
                candidate == backup
                or not pattern.fullmatch(candidate.name)
                or candidate.is_symlink()
                or not candidate.is_file()
            ):
                continue
            try:
                candidate.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise OSError(
                    f"Verified backup retained at {backup}; cannot remove old backup {candidate}: {exc}"
                ) from exc
            removed += 1
        if removed:
            print(f"Removed {removed} older database backups; retained: {backup}", flush=True)

    def check_space(self, backup=False):
        needed = self.min_free_bytes
        if backup:
            needed += self.path.stat().st_size
            wal = Path(str(self.path) + "-wal")
            if wal.exists():
                needed += wal.stat().st_size
        if shutil.disk_usage(self.path.parent).free < needed:
            raise OSError("Insufficient free disk space; stopping without deleting databases or backups")

    def maintain(self):
        self.check_space()
        if self.dirty and time.monotonic() - self.last_backup >= self.backup_seconds:
            backup = self.create_snapshot()
            print(f"Verified periodic backup: {backup}", flush=True)
            self.queue_upload(backup)
        if self.pending_upload is not None and time.monotonic() >= self.next_upload:
            try:
                self.upload_snapshot(self.pending_upload, "daily")
            except (OSError, SubprocessError) as exc:
                self.next_upload = time.monotonic() + 300
                LOG.warning("Daily backup transfer failed; retrying in 5 minutes: %s", exc)
            else:
                self.pending_upload = None

    def create_snapshot(self):
        self.check_space(backup=True)
        backup = backup_database(self.path)
        self._retain_backup(backup)
        self.dirty = False
        self.last_backup = time.monotonic()
        return backup

    def queue_upload(self, backup):
        if self.snapshot_target:
            self.pending_upload = backup
            self.next_upload = 0

    def upload_snapshot(self, backup, kind):
        destination = f"{self.snapshot_target.rstrip('/')}/crawler-{kind}.db"
        # Default rsync publication uses a temporary file and a rename after transfer.
        run_subprocess(
            [
                "rsync",
                "--ignore-times",
                "-s",
                "--timeout=60",
                "-e",
                "ssh -o BatchMode=yes -o ConnectTimeout=10",
                "--",
                str(backup),
                destination,
            ],
            check=True,
            stdin=DEVNULL,
        )
        LOG.info("Published %s crawler snapshot", kind)

    def publish_final(self):
        if self.snapshot_target:
            backup = self.create_snapshot()
            try:
                self.upload_snapshot(backup, "final")
            except (OSError, SubprocessError) as exc:
                raise RuntimeError(f"Final snapshot saved at {backup}, but transfer failed; restart to retry") from exc
            self.pending_upload = None

    def scan(self, scope):
        row = self.conn.execute("SELECT * FROM crawler_scans WHERE scope=?", (scope,)).fetchone()
        return dict(row) if row else {"cursor": None, "finished": 0, "next_scan": 0}

    def enqueue_batch(self, scope, members, cursor, refresh_seconds):
        """Queue article work and move category cursor in the SAME transaction."""
        with self.conn:
            for member in members:
                self.conn.execute(
                    """INSERT INTO crawler_articles(scope,page_id,title) VALUES (?,?,?)
                    ON CONFLICT(scope,page_id) DO UPDATE SET title=excluded.title,
                    status=CASE WHEN status='done' THEN 'pending' ELSE status END""",
                    (scope, member["pageid"], member["title"]),
                )
            self.conn.execute(
                "INSERT OR REPLACE INTO crawler_scans VALUES (?,?,?,?)",
                (
                    scope,
                    json.dumps(cursor) if cursor else None,
                    int(cursor is None),
                    time.time() + refresh_seconds if cursor is None else 0,
                ),
            )

    def restart_scan(self, scope):
        with self.conn:
            self.conn.execute("UPDATE crawler_scans SET cursor=NULL,finished=0 WHERE scope=?", (scope,))

    def due_articles(self, scope, limit=50, retry_limit=None):
        """New work first; optionally allow a separate bounded retry budget."""
        pending = self.conn.execute(
            """SELECT * FROM crawler_articles WHERE scope=?
            AND status='pending' AND retry_at<=? ORDER BY retry_at,page_id LIMIT ?""",
            (scope, time.time(), limit),
        ).fetchall()
        if retry_limit is None:
            retry_limit = max(0, limit - len(pending))
        retries = (
            self.conn.execute(
                """SELECT * FROM crawler_articles WHERE scope=?
            AND status='retry' AND retry_at<=? ORDER BY retry_at,page_id LIMIT ?""",
                (scope, time.time(), retry_limit),
            ).fetchall()
            if retry_limit
            else []
        )
        return pending + retries

    def discover(self, scope, article, page_url, images, profile):
        """No-face and failed images stay in the queue independently of article progress."""
        with self.conn:
            for info in images:
                url, sha1 = info["url"], info["sha1"]
                self.conn.execute(
                    """INSERT INTO crawler_images(url,profile,sha1,metadata)
                    VALUES (?,?,?,?) ON CONFLICT(url,profile) DO UPDATE SET
                    metadata=excluded.metadata,
                    status=CASE WHEN sha1!=excluded.sha1 THEN 'pending' ELSE status END,
                    attempts=CASE WHEN sha1!=excluded.sha1 THEN 0 ELSE attempts END,
                    retry_at=CASE WHEN sha1!=excluded.sha1 THEN 0 ELSE retry_at END,
                    sha1=excluded.sha1""",
                    (url, profile, sha1, json.dumps(info)),
                )
                self.conn.execute("INSERT OR IGNORE INTO crawler_sources VALUES (?,?,?)", (url, profile, page_url))
                added = self.conn.execute(
                    """INSERT OR IGNORE INTO wiki_pages(image_id,page_url)
                    SELECT id,? FROM images WHERE image_url=?""",
                    (page_url, url),
                )
                self.dirty |= added.rowcount > 0
            self.conn.execute(
                """UPDATE crawler_articles SET status='done',attempts=0,
                error=NULL,retry_at=0 WHERE scope=? AND page_id=?""",
                (scope, article["page_id"]),
            )

    def due_images(self, profile, limit=8):
        return self.conn.execute(
            """SELECT * FROM crawler_images WHERE profile=?
            AND status IN ('pending','retry') AND retry_at<=? ORDER BY retry_at,url LIMIT ?""",
            (profile, time.time(), limit),
        ).fetchall()

    def fail_article(self, scope, article, error):
        self._fail("crawler_articles", "scope=? AND page_id=?", (scope, article["page_id"]), article["attempts"], error)

    def fail_image(self, job, error):
        self._fail("crawler_images", "url=? AND profile=?", (job["url"], job["profile"]), job["attempts"], error)

    def _fail(self, table, where, keys, attempts, error):
        # Internal constant SQL identifiers only. Failed jobs never disappear after a retry limit.
        delay = min(86400, 60 * 2 ** min(attempts, 11))
        with self.conn:
            self.conn.execute(
                f"UPDATE {table} SET status='retry',attempts=attempts+1,retry_at=?,error=? WHERE {where}",
                (time.time() + delay, str(error)[:2000], *keys),
            )

    def complete_image(self, job, entries, quality):
        """All faces, provenance and successful status commit together; errors roll back."""
        before = self.conn.total_changes
        with self.conn:
            if entries:
                self.conn.execute(
                    "INSERT OR IGNORE INTO images(image_url,phash) VALUES (?,?)", (job["url"], entries[0].get("phash"))
                )
                image_id = self.conn.execute("SELECT id FROM images WHERE image_url=?", (job["url"],)).fetchone()[0]
                for entry in entries:
                    insert_face(self.conn, image_id, entry)
                self.conn.execute(
                    """INSERT OR IGNORE INTO wiki_pages(image_id,page_url)
                    SELECT ?,page_url FROM crawler_sources WHERE url=? AND profile=?""",
                    (image_id, job["url"], job["profile"]),
                )
            data_changed = self.conn.total_changes > before
            self.conn.execute(
                """UPDATE crawler_images SET status=?,attempts=0,retry_at=0,error=NULL,
                quality=? WHERE url=? AND profile=? AND sha1=?""",
                ("done" if entries else "no_faces", json.dumps(quality), job["url"], job["profile"], job["sha1"]),
            )
        self.dirty |= data_changed

    def counts(self, profile):
        return dict(
            self.conn.execute("SELECT status,count(*) FROM crawler_images WHERE profile=? GROUP BY status", (profile,))
        )

    def close(self):
        self.conn.close()
        self.lock.release()
