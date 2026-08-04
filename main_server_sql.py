from flask import Flask, request, jsonify
import sqlite3
import numpy as np
import os
import urllib.parse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

AUTH_TOKEN = os.getenv("TOKEN")

DB_FILE = "face_embeddings.db"

db_conn = None
db = []
db_embeddings_matrix = None
db_norms_sq = None

request_count = 0


def verify_token():
    if not AUTH_TOKEN:
        return True

    token = request.headers.get("Authorization")
    if not token:
        return False

    if token.startswith("Bearer "):
        token = token[7:]

    return token == AUTH_TOKEN


def get_connection():
    global db_conn

    if db_conn is None:
        db_conn = sqlite3.connect(DB_FILE, check_same_thread=False)

        db_conn.execute("PRAGMA journal_mode=WAL")
        db_conn.execute("PRAGMA synchronous=NORMAL")
        db_conn.execute("PRAGMA temp_store=MEMORY")
        db_conn.execute("PRAGMA cache_size=-200000")

    return db_conn


def load_stats():
    global request_count

    try:
        if os.path.exists("stats.txt"):
            with open("stats.txt") as f:
                request_count = int(f.read().strip())
        else:
            request_count = 0
    except:
        request_count = 0


def save_stats():
    with open("stats.txt", "w") as f:
        f.write(str(request_count))


def increment_request_count():
    global request_count
    request_count += 1
    save_stats()


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


def deduplicate_results(results):
    seen = {}
    deduped = []

    for item in results:
        image_url = item.get("image_url")
        key = normalize_image_url(image_url) or image_url or item.get("phash") or ""

        if not key:
            deduped.append(item)
            continue

        if key in seen:
            existing = seen[key]
            if item.get("distance", float("inf")) < existing.get("distance", float("inf")):
                seen[key] = item
                for idx, candidate in enumerate(deduped):
                    if normalize_image_url(candidate.get("image_url")) == key or candidate.get("image_url") == image_url:
                        deduped[idx] = item
                        break
            continue

        seen[key] = item
        deduped.append(item)

    return deduped


def load_database():
    global db
    global db_embeddings_matrix
    global db_norms_sq

    print(f"[{datetime.now()}] Lade SQLite...")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            i.id,
            i.image_url,
            i.phash,
            i.embedding,
            w.page_url
        FROM images i
        LEFT JOIN wiki_pages w
        ON i.id = w.image_id
        ORDER BY i.id
    """)

    rows = cur.fetchall()

    db = []
    embeddings = []
    entry_lookup = {}

    for image_id, image_url, phash, blob, page_url in rows:
        normalized_url = normalize_image_url(image_url)
        key = normalized_url or image_url

        if key not in entry_lookup:
            embedding = np.frombuffer(blob, dtype=np.float32)

            current_entry = {
                "id": image_id,
                "image_url": normalized_url or image_url,
                "phash": phash,
                "page_urls": []
            }

            db.append(current_entry)
            embeddings.append(embedding)
            entry_lookup[key] = current_entry

        current_entry = entry_lookup[key]

        if page_url is not None and page_url not in current_entry["page_urls"]:
            current_entry["page_urls"].append(page_url)

    if embeddings:
        db_embeddings_matrix = np.vstack(embeddings)
        db_norms_sq = np.sum(db_embeddings_matrix ** 2, axis=1)
    else:
        db_embeddings_matrix = None
        db_norms_sq = None

    print(f"✅ {len(db)} Bilder geladen")


def find_matches(query_embedding,
                 tolerance=0.5,
                 max_results=10):

    global db
    global db_embeddings_matrix
    global db_norms_sq

    if db_embeddings_matrix is None:
        return []

    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    query_norm_sq = np.sum(query_embedding ** 2)

    dot_products = np.dot(
        db_embeddings_matrix,
        query_embedding
    )

    squared_distances = np.maximum(
        0,
        db_norms_sq +
        query_norm_sq -
        2 * dot_products
    )

    tolerance_sq = tolerance ** 2

    valid_indices = np.where(
        squared_distances <= tolerance_sq
    )[0]

    if len(valid_indices) == 0:
        return []

    valid_distances = squared_distances[valid_indices]

    if len(valid_indices) > max_results:

        top = np.argpartition(
            valid_distances,
            max_results - 1
        )[:max_results]

        top = top[np.argsort(valid_distances[top])]

        indices = valid_indices[top]

    else:

        indices = valid_indices[
            np.argsort(valid_distances)
        ]

    results = []

    for idx in indices:
        db_entry = db[idx]
        page_urls = db_entry.get("page_urls", [])

        distance = float(np.sqrt(squared_distances[idx]))
        similarity = max(0.0, 1.0 - distance)

        results.append({
            "distance": distance,
            "similarity": similarity,
            "image_url": db_entry["image_url"],
            "phash": db_entry["phash"],
            "page_url": page_urls[0] if page_urls else None,
            "page_urls": page_urls
        })

    return deduplicate_results(results)

@app.route("/search", methods=["POST"])
def search_faces():
    try:

        if not verify_token():
            return jsonify({
                "error": "Unauthorized"
            }), 401

        increment_request_count()

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data"
            }), 400

        encoding = data.get("encoding")

        if encoding is None:
            return jsonify({
                "error": "Missing encoding"
            }), 400

        if not isinstance(encoding, list) or len(encoding) != 128:
            return jsonify({
                "error": "Encoding must contain 128 values"
            }), 400

        matches = find_matches(encoding)

        return jsonify({
            "success": True,
            "matches": matches,
            "total_results": len(matches)
        })

    except Exception as e:
        print(e)
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM images")
    total = cur.fetchone()[0]

    cur.execute("""
        SELECT page_url
        FROM wiki_pages
        ORDER BY rowid DESC
        LIMIT 1
    """)

    row = cur.fetchone()

    return jsonify({
        "status": "healthy",
        "database_loaded": db_embeddings_matrix is not None,
        "total_entries": total,
        "last_page_url": row[0] if row else None
    })


@app.route("/stats", methods=["GET"])
def stats():

    if not verify_token():
        return jsonify({
            "error": "Unauthorized"
        }), 401

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM images")
    total_images = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM wiki_pages")
    total_pages = cur.fetchone()[0]

    return jsonify({
        "total_images": total_images,
        "total_pages": total_pages,
        "database_size": os.path.getsize(DB_FILE),
        "requests": request_count
    })


@app.route("/reload", methods=["POST"])
def reload_database():

    if not verify_token():
        return jsonify({
            "error": "Unauthorized"
        }), 401

    load_database()

    return jsonify({
        "success": True,
        "entries": len(db)
    })


if __name__ == "__main__":

    load_stats()
    load_database()

    port = int(os.getenv("PORT", 7403))
    debug = os.getenv("FLASK_ENV") == "development"

    print(f"SQLite-Datei: {DB_FILE}")
    print(f"Geladene Bilder: {len(db)}")
    print(f"Port: {port}")

    if AUTH_TOKEN:
        print("🔒 Token aktiviert")
    else:
        print("⚠️ Keine Authentifizierung")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug
    )