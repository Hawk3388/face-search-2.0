"""Audit SQLite duplicates; merge only canonical URL aliases after a verified backup.

pHash equality is a review hint, never evidence that two face records are redundant.
Run without --apply for a read-only report. Stop all database writers before --apply.
"""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock
from face_search.config import Settings
from face_search.search import normalize_image_url
from face_search.storage import backup_database, connect_readonly, ensure_schema


def inspect_connection(conn):
    first, groups = {}, {}
    images = normalized = 0
    for row in conn.execute("SELECT id,image_url FROM images ORDER BY id"):
        image_id, url = row
        images += 1
        canonical = normalize_image_url(url)
        if not canonical:
            continue
        normalized += canonical != url
        if canonical in first:
            groups.setdefault(canonical, [first[canonical]]).append((image_id, url))
        else:
            first[canonical] = (image_id, url)
    hash_groups = conn.execute("""SELECT count(*) FROM (
        SELECT phash FROM images WHERE phash IS NOT NULL AND phash!=''
        GROUP BY phash HAVING count(*)>1)""").fetchone()[0]
    report = {
        "images": images,
        "url_alias_groups": len(groups),
        "possible_image_merges": sum(len(rows) - 1 for rows in groups.values()),
        "urls_to_normalize": normalized,
        "equal_phash_groups_review_only": hash_groups,
    }
    return report, first, groups


def audit_database(path):
    with closing(connect_readonly(path)) as conn:
        conn.execute("BEGIN")  # Consistent report even with a WAL writer.
        return inspect_connection(conn)[0]


def check_schema(conn):
    allowed = {
        "images": {"id", "image_url", "phash", "embedding"},
        "faces": {"id", "image_id", "model_id", "dimension", "preprocessing", "bbox", "embedding", "embedding_hash"},
        "wiki_pages": {"image_id", "page_url"},
    }
    for table, columns in allowed.items():
        actual = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if actual - columns:
            raise ValueError(f"Unsupported extra columns in {table}; refusing to discard metadata")
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        quoted = '"' + table.replace('"', '""') + '"'
        for fk in conn.execute(f"PRAGMA foreign_key_list({quoted})"):
            if fk[2] in ("images", "faces") and table not in ("wiki_pages", "faces"):
                raise ValueError(f"Additional references from {table}; cleanup requires an explicit migration")


def face_rows(conn, image_id):
    return conn.execute(
        """SELECT id,model_id,embedding_hash,dimension,preprocessing,bbox,embedding
        FROM faces WHERE image_id=? ORDER BY id""",
        (image_id,),
    ).fetchall()


def merge_groups(conn, first, groups):
    merged = removed_faces = conflicts = 0
    for canonical, first_row in first.items():
        rows = groups.get(canonical, [first_row])
        if len(rows) == 1:
            if rows[0][1] != canonical:
                conn.execute("UPDATE images SET image_url=? WHERE id=?", (canonical, rows[0][0]))
            continue
        keeper = next((row for row in rows if row[1] == canonical), rows[0])[0]
        faces = {image_id: face_rows(conn, image_id) for image_id, _ in rows}
        signatures = {}
        conflict = False
        for values in faces.values():
            for face in values:
                key, payload = (face[1], face[2]), face[3:]
                if key in signatures and signatures[key] != payload:
                    conflict = True
                signatures[key] = payload
        if conflict:
            # The existing UNIQUE constraint cannot represent these metadata variants
            # under one image. Keep the entire group rather than losing a face/box.
            conflicts += 1
            continue
        present = {(face[1], face[2]) for face in faces[keeper]}
        for duplicate, _ in rows:
            if duplicate == keeper:
                continue
            for face in faces[duplicate]:
                key = (face[1], face[2])
                if key in present:
                    conn.execute("DELETE FROM faces WHERE id=?", (face[0],))
                    removed_faces += 1
                else:
                    conn.execute("UPDATE faces SET image_id=? WHERE id=?", (keeper, face[0]))
                    present.add(key)
            conn.execute(
                """INSERT OR IGNORE INTO wiki_pages(image_id,page_url)
                SELECT ?,page_url FROM wiki_pages WHERE image_id=?""",
                (keeper, duplicate),
            )
            conn.execute(
                """UPDATE images SET phash=COALESCE(phash,
                (SELECT phash FROM images WHERE id=?)) WHERE id=?""",
                (duplicate, keeper),
            )
            conn.execute("DELETE FROM wiki_pages WHERE image_id=?", (duplicate,))
            conn.execute("DELETE FROM images WHERE id=?", (duplicate,))
            merged += 1
        conn.execute("UPDATE images SET image_url=? WHERE id=?", (canonical, keeper))
    return {
        "merged_images": merged,
        "removed_identical_face_rows": removed_faces,
        "metadata_conflict_groups_kept": conflicts,
    }


def clean_database(path, apply=False):
    path = Path(path).resolve()
    if not apply:
        return {"applied": False, **audit_database(path)}
    # Share the crawler lock; another crawler must not undo a cleanup in parallel.
    with FileLock(str(path) + ".crawler.lock", timeout=0):
        before = audit_database(path)  # Refuse missing/invalid databases before any writes.
        if not before["urls_to_normalize"] and not before["possible_image_merges"]:
            return {"applied": False, "reason": "No URL aliases to clean", **before}
        backup = backup_database(path)  # A backup failure must abort before migration or cleanup.
        ensure_schema(path)
        with closing(sqlite3.connect(path, timeout=30)) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                conn.execute("BEGIN IMMEDIATE")
                check_schema(conn)
                report, first, groups = inspect_connection(conn)
                result = merge_groups(conn, first, groups)
                if conn.execute("PRAGMA foreign_key_check").fetchone():
                    raise ValueError("Foreign key check failed; cleanup rolled back")
                if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("Database integrity check failed; cleanup rolled back")
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return {"applied": True, "backup": str(backup), **report, **result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, help="Defaults to DATABASE_PATH or face_embeddings.db")
    parser.add_argument(
        "--apply", action="store_true", help="Back up, then merge canonical URL aliases transactionally"
    )
    args = parser.parse_args(argv)
    path = args.database or Settings.from_env("sqlite").database
    if path.suffix.lower() == ".json":
        parser.error("Import JSON with tools/json_to_sql.py first; this cleaner preserves all SQLite faces and sources")
    print(json.dumps(clean_database(path, apply=args.apply), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
