import json
import sqlite3
import numpy as np
from pathlib import Path

JSON_FILE = "face_embeddings_original.json"
DB_FILE = "face_embeddings.db"
BATCH_SIZE = 10000


def create_database(conn):
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_url TEXT UNIQUE,
        phash TEXT,
        embedding BLOB
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS wiki_pages (
        image_id INTEGER,
        page_url TEXT,
        PRIMARY KEY(image_id, page_url),
        FOREIGN KEY(image_id) REFERENCES images(id)
    )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_image_url ON images(image_url)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_phash ON images(phash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_page_url ON wiki_pages(page_url)")

    conn.commit()


def main():
    if not Path(JSON_FILE).exists():
        print("JSON-Datei nicht gefunden.")
        return

    print("📂 Lade JSON...")
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📊 {len(data)} Einträge gefunden")

    conn = sqlite3.connect(DB_FILE)
    create_database(conn)
    cur = conn.cursor()

    count = 0

    for entry in data:

        embedding = None
        if "embedding" in entry and entry["embedding"] is not None:
            embedding = np.asarray(entry["embedding"], dtype=np.float32).tobytes()

        image_url = entry.get("image_url")

        cur.execute(
            "SELECT id FROM images WHERE image_url=?",
            (image_url,)
        )

        row = cur.fetchone()

        if row is None:
            cur.execute("""
                INSERT INTO images(image_url, phash, embedding)
                VALUES (?, ?, ?)
            """, (
                image_url,
                entry.get("phash"),
                embedding
            ))
            image_id = cur.lastrowid
        else:
            image_id = row[0]

        urls = []

        if entry.get("page_url"):
            urls.append(entry["page_url"])

        if entry.get("page_urls"):
            urls.extend(entry["page_urls"])

        urls = list(dict.fromkeys(urls))

        for url in urls:
            cur.execute("""
                INSERT OR IGNORE INTO wiki_pages(image_id, page_url)
                VALUES (?, ?)
            """, (image_id, url))

        count += 1

        if count % BATCH_SIZE == 0:
            conn.commit()
            print(f"✅ {count}/{len(data)} importiert")

    conn.commit()
    conn.close()

    print("🎉 Fertig!")
    print(f"SQLite-Datenbank gespeichert als: {DB_FILE}")


if __name__ == "__main__":
    main()