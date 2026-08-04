import os
import shutil
import sqlite3
import urllib.parse


DB_FILE = "face_embeddings.db"


def normalize_image_url(url):
    if not url:
        return None

    url = str(url).strip()
    if not url:
        return None

    parsed = urllib.parse.urlsplit(url)
    path = parsed.path or ""

    if "/thumb/" in path:
        prefix, thumb_path = path.split("/thumb/", 1)
        segments = thumb_path.split("/")

        if len(segments) >= 4:
            filename = segments[-2]
            normalized_path = f"{prefix}/{'/'.join(segments[:-2])}/{filename}"
            parsed = parsed._replace(path=normalized_path, query="", fragment="")
            return urllib.parse.urlunsplit(parsed)

    return urllib.parse.urlunsplit(parsed._replace(query="", fragment=""))


def backup_database(db_path):
    backup_path = f"{db_path}.bak"
    shutil.copy2(db_path, backup_path)
    return backup_path


def fetch_images(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, image_url, phash, embedding FROM images ORDER BY id")
    return cur.fetchall()


def fetch_page_urls(conn, image_id):
    cur = conn.cursor()
    cur.execute(
        "SELECT page_url FROM wiki_pages WHERE image_id=? ORDER BY rowid",
        (image_id,)
    )
    return [row[0] for row in cur.fetchall() if row[0]]


def choose_canonical_row(rows, normalized_url):
    for row in rows:
        if row[1] == normalized_url:
            return row

    for row in rows:
        if normalize_image_url(row[1]) == normalized_url:
            return row

    return rows[0]


def merge_database(db_path=DB_FILE):
    if not os.path.exists(db_path):
        print(f"❌ '{db_path}' existiert nicht!")
        return

    backup_path = backup_database(db_path)
    print(f"💾 Backup erstellt: {backup_path}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        rows = fetch_images(conn)
        if not rows:
            print("ℹ️ Keine Bilder in der Datenbank gefunden.")
            return

        groups = {}
        group_order = []

        for image_id, image_url, phash, embedding in rows:
            normalized_url = normalize_image_url(image_url) or image_url

            if normalized_url not in groups:
                groups[normalized_url] = []
                group_order.append(normalized_url)

            groups[normalized_url].append((image_id, image_url, phash, embedding))

        before_images = len(rows)
        before_pages = conn.execute("SELECT COUNT(*) FROM wiki_pages").fetchone()[0]

        merged_images = 0
        moved_pages = 0

        for normalized_url in group_order:
            group_rows = groups[normalized_url]
            if len(group_rows) == 1:
                image_id, image_url, phash, embedding = group_rows[0]
                if image_url != normalized_url:
                    conn.execute(
                        "UPDATE images SET image_url=? WHERE id=?",
                        (normalized_url, image_id)
                    )
                continue

            canonical_row = choose_canonical_row(group_rows, normalized_url)
            canonical_id, canonical_url, canonical_phash, canonical_embedding = canonical_row

            if canonical_url != normalized_url:
                conn.execute(
                    "UPDATE images SET image_url=? WHERE id=?",
                    (normalized_url, canonical_id)
                )

            for duplicate_row in group_rows:
                duplicate_id, duplicate_url, duplicate_phash, duplicate_embedding = duplicate_row
                if duplicate_id == canonical_id:
                    continue

                page_urls = fetch_page_urls(conn, duplicate_id)
                for page_url in page_urls:
                    conn.execute(
                        "INSERT OR IGNORE INTO wiki_pages(image_id, page_url) VALUES (?, ?)",
                        (canonical_id, page_url)
                    )

                moved_pages += len(page_urls)

                conn.execute(
                    "DELETE FROM wiki_pages WHERE image_id=?",
                    (duplicate_id,)
                )
                conn.execute(
                    "DELETE FROM images WHERE id=?",
                    (duplicate_id,)
                )
                merged_images += 1

        conn.commit()

        after_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        after_pages = conn.execute("SELECT COUNT(*) FROM wiki_pages").fetchone()[0]

        print(f"📊 Bilder vorher: {before_images}")
        print(f"📊 Bilder nachher: {after_images}")
        print(f"📊 Wiki-Seiten vorher: {before_pages}")
        print(f"📊 Wiki-Seiten nachher: {after_pages}")
        print(f"🔁 Zusammengeführte Bildduplikate: {merged_images}")
        print(f"🔗 Verschobene Wiki-Seiten: {moved_pages}")
        print("✅ Fertig.")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    merge_database()