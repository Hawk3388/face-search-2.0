import os
import json
import sqlite3
import requests
from bs4 import BeautifulSoup
import face_recognition
import urllib.parse
from collections import deque
from io import BytesIO
from PIL import Image
import imagehash
import shutil
from dotenv import load_dotenv
import signal
import sys
import re
import gc  # For garbage collection
import time  # For error waiting times
import urllib.robotparser  # For robots.txt respect
import numpy as np

load_dotenv()  # Load environment variables from .env file

# Ethical scraping - respect Wikipedia's guidelines
CRAWL_DELAY = 1.0  # 1 second between requests (more than recommended 100ms)
USER_AGENT = os.environ.get("USER_AGENT", "FaceSearchBot/1.0 (Educational Research; Contact: github.com/Hawk3388/face-search-2.0)")

# Global counter for consecutive 429 errors
consecutive_429_errors = 0
MAX_CONSECUTIVE_429 = 10  # Stop after 10 consecutive 429 errors
robots_cache = {}

# Robots.txt parser for Wikipedia
def check_robots_txt(url):
    try:
        domain = urllib.parse.urlparse(url).netloc

        if domain not in robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"https://{domain}/robots.txt")
            rp.read()
            robots_cache[domain] = rp

        return robots_cache[domain].can_fetch(USER_AGENT, url)

    except Exception as e:
        print(f"robots.txt error: {e}")
        return True

queue = deque()  # Global queue for signal handler

DB_FILE = "face_embeddings.db"
db_conn = None


def get_connection():
    global db_conn

    if db_conn is None:
        db_conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        db_conn.execute("PRAGMA journal_mode=WAL")
        db_conn.execute("PRAGMA synchronous=NORMAL")
        db_conn.execute("PRAGMA temp_store=MEMORY")
        db_conn.execute("PRAGMA cache_size=-200000")
        db_conn.execute("PRAGMA foreign_keys=ON")

    return db_conn


def initialize_database():
    conn = get_connection()
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
        PRIMARY KEY (image_id, page_url),
        FOREIGN KEY(image_id) REFERENCES images(id)
    )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_images_image_url ON images(image_url)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_images_phash ON images(phash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_wiki_pages_page_url ON wiki_pages(page_url)")

    conn.commit()
    return conn


def save_database():
    """Persist the current article data into SQLite."""
    global total_entries_saved

    if not current_article_data:
        return

    try:
        conn = initialize_database()
        cur = conn.cursor()

        for entry in current_article_data:
            image_url = entry.get("image_url")
            if not image_url:
                continue

            embedding = entry.get("embedding")
            embedding_blob = None
            if embedding is not None:
                embedding_blob = np.asarray(embedding, dtype=np.float32).tobytes()

            cur.execute(
                "SELECT id FROM images WHERE image_url=?",
                (image_url,)
            )
            row = cur.fetchone()

            if row is None:
                cur.execute(
                    "INSERT INTO images(image_url, phash, embedding) VALUES (?, ?, ?)",
                    (image_url, entry.get("phash"), embedding_blob)
                )
                image_id = cur.lastrowid
            else:
                image_id = row[0]

            page_url = entry.get("page_url")
            if page_url:
                cur.execute(
                    "INSERT OR IGNORE INTO wiki_pages(image_id, page_url) VALUES (?, ?)",
                    (image_id, page_url)
                )

        conn.commit()

        entries_added = len(current_article_data)
        total_entries_saved += entries_added
        create_periodic_backup(total_entries_saved)

        print(f"{entries_added} new entries added to database. (Total: {total_entries_saved})")

        current_article_data.clear()
    except Exception as e:
        print(f"Error saving database: {e}")


def close_database():
    """Commit and close the SQLite connection."""
    global db_conn
    try:
        if db_conn is not None:
            db_conn.commit()
            db_conn.close()
            db_conn = None
        print("Database properly closed.")
    except Exception as e:
        print(f"Error closing database: {e}")

def get_visited_pages_from_db():
    """Since we don't do reading - return empty set."""
    return set()

def get_last_category_page():
    """Reads the last crawled category page from a small separate file."""
    try:
        if os.path.exists("last_category_page.txt"):
            with open("last_category_page.txt", "r", encoding="utf-8") as f:
                last_page = json.load(f)
                if last_page:
                    print(f"📄 Last crawled category page: {last_page}")
                    return last_page["category_url"], last_page["last_article_url"]
    except Exception as e:
        print(f"Error reading last category page: {e}")
    return None


def get_last_crawled_page():
    """Reads the last crawled article from a small separate file (compat with threaded crawler)."""
    try:
        if os.path.exists("last_crawled_page.txt"):
            with open("last_crawled_page.txt", "r", encoding="utf-8") as f:
                content = f.read()
                if content is None:
                    return None
                last_page = content.strip()
                if last_page:
                    print(f"📄 Last crawled article: {last_page}")
                    return last_page
    except Exception as e:
        print(f"Error reading last crawled page: {e}")
    return None


def wikipedia_url_to_category_link(article_url, category="Living people"):
    """
    Convert a Wikipedia article URL into a category page URL with a pagefrom parameter
    so crawling can resume close to that article instead of starting from the beginning.
    """
    # Title from URL
    title = article_url.split("/")[-1]
    title = title.replace("_", " ")

    # Remove parenthetical disambiguation like " (volleyball)"
    title = re.sub(r"\s*\([^)]*\)", "", title).strip()

    parts = title.split(" ")
    if len(parts) == 1:
        lastname = parts[0]
        firstname = ""
    else:
        lastname = parts[-1]
        firstname = " ".join(parts[:-1])

    pagefrom_raw = f"{lastname}, {firstname}\n{title}"
    pagefrom_encoded = urllib.parse.quote_plus(pagefrom_raw)

    category_url = (
        f"https://en.wikipedia.org/w/index.php?title=Category:{category.replace(' ', '_')}&pagefrom={pagefrom_encoded}#mw-pages"
    )
    return category_url

def save_last_crawled_page(page_url):
    """Saves the current article URL to a small separate file."""
    try:
        with open("last_crawled_page.txt", "w", encoding="utf-8") as f:
            f.write(page_url)
    except Exception as e:
        print(f"Error saving last page: {e}")

def signal_handler(sig, frame):
    """Handler for interrupt signals (Ctrl+C)."""
    print("\nInterrupt detected! Saving database...")
    save_database()
    close_database()
    print("Crawler terminated.")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def create_backup():
    """Creates a backup of the SQLite database at startup."""
    try:
        if os.path.exists(DB_FILE):
            backup_file = f"{DB_FILE}.bak"
            shutil.copy2(DB_FILE, backup_file)
            print(f"📁 Startup backup created: {backup_file}")
        else:
            print("No existing database found - no startup backup created.")
    except Exception as e:
        print(f"Error creating startup backup: {e}")


def create_periodic_backup(entries_count):
    """Creates a backup every 1000 entries."""
    try:
        if entries_count > 0 and entries_count % 1000 == 0:
            if os.path.exists(DB_FILE):
                backup_file = f"{DB_FILE}.bak"
                shutil.copy2(DB_FILE, backup_file)
                print(f"📁 Backup created after {entries_count} entries: {backup_file}")
    except Exception as e:
        print(f"Error creating periodic backup: {e}")

# Initialize the SQLite database and create backups
create_backup()
initialize_database()

# Global variable for current article (RAM-efficient)
current_article_data = []

# Counter for backup system
total_entries_saved = 0

def is_internal_link(base_url, link):
    base_domain = urllib.parse.urlparse(base_url).netloc
    link_domain = urllib.parse.urlparse(link).netloc
    
    # Create complete URLs for comparison
    full_link = urllib.parse.urljoin(base_url, link)
    
    # URLs without fragment (part after #) for comparison
    base_parsed = urllib.parse.urlparse(base_url)
    link_parsed = urllib.parse.urlparse(full_link)
    
    base_path = base_parsed.path
    link_path = link_parsed.path
    
    # Exclude self-links (links pointing to the same page, even with # anchors)
    if base_path == link_path:
        return False
    
    # Check if it's Wikipedia
    if "wikipedia.org" in base_domain:
        # For Wikipedia: only allow links to other Wikipedia articles
        if link_domain == "" or link_domain == base_domain:
            # Check if it's a Wikipedia article (not Talk, User, Special, etc.)
            path = link_path
            
            # Wikipedia articles have the format /wiki/ArticleName
            if path.startswith("/wiki/") and ":" not in path.split("/wiki/")[1]:
                # Exclude special pages
                excluded_prefixes = [
                    "Category:", "File:", "Template:", "Help:", "Special:", 
                    "User:", "Talk:", "User_talk:", "Wikipedia:", "MediaWiki:",
                    "Portal:", "Draft:", "Module:"
                ]
                article_name = path.split("/wiki/")[1]
                if not any(article_name.startswith(prefix) for prefix in excluded_prefixes):
                    return True
            return False
    else:
        # For other domains: normal domain check
        return base_domain == link_domain or link_domain == ""

def download_image(img_url):
    def skip(reason, url):
        print(f"SKIPPED [{reason}]")
        print(url)
        return None

    global consecutive_429_errors
    
    # Only allow compatible image formats
    allowed_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    if not any(img_url.lower().endswith(ext) for ext in allowed_exts):
        print(f"Skipping incompatible image format: {img_url}")
        return skip("EXTENSION", img_url)
    if ".svg" in img_url.lower():
        print(f"Skipping incompatible image format: {img_url}")
        return skip("EXTENSION", img_url)
    if img_url.lower().endswith("wikipedia.png"):
        print(f"Skipping Wikipedia logo: {img_url}")
        return skip("WIKIPEDIA_LOGO", img_url)
    
    # Maximum 3 attempts per image
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            # Respect robots.txt
            if "upload.wikimedia.org" not in img_url:
                if not check_robots_txt(img_url):
                    print(f"⚠️ robots.txt forbids access to: {img_url}")
                    return skip("ROBOTS", img_url)
            
            headers = {
                "User-Agent": USER_AGENT,
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Connection": "close"
            }
            
            print(f"📥 Downloading image (attempt {attempt}/{max_attempts}): {img_url}")
            
            response = requests.get(img_url, timeout=30, headers=headers, stream=True)
            response.raise_for_status()
            
            # Size check before complete download
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
                print(f"⚠️ Image too large ({int(content_length)/1024/1024:.1f}MB): {img_url}")
                return skip("SIZE", img_url)
            
            # Complete download
            image_bytes = response.content
            if len(image_bytes) > 10 * 1024 * 1024:
                size_mb = len(image_bytes) / 1024 / 1024
                print(f"⚠️ Image too large ({size_mb:.1f}MB): {img_url}")
                return None
            
            print(f"✅ Image successfully downloaded ({len(image_bytes)/1024:.1f}KB)")
            
            # Successful download - reset 429 counter
            consecutive_429_errors = 0
            
            # Ethical crawling: Respect crawl delay
            time.sleep(CRAWL_DELAY)
            return image_bytes
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                print(f"⚠️ Rate limit (429) reached for {img_url} (attempt {attempt}/{max_attempts})")
                
                if attempt < max_attempts:
                    wait_time = min(300, 30 * (2 ** attempt))
                    print(f"⏳ Waiting {wait_time} seconds before next attempt...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Only increase 429 counter on final failure of the image
                    consecutive_429_errors += 1
                    print(f"❌ Skipping image after {max_attempts} attempts with 429 errors")
                    print(f"📊 Consecutive images with 429 errors: {consecutive_429_errors}/{MAX_CONSECUTIVE_429}")
                    
                    # Check if we've reached the limit for consecutive 429 errors
                    if consecutive_429_errors >= MAX_CONSECUTIVE_429:
                        print(f"🚨 CRITICAL: {MAX_CONSECUTIVE_429} consecutive images with 429 errors!")
                        print("🛑 This indicates a permanent block - stopping crawler!")
                        save_database()
                        close_database()
                        sys.exit(1)
                    
                    return None
            else:
                print(f"⚠️ HTTP error {e.response.status_code} for {img_url}")
                if attempt < max_attempts:
                    wait_time = min(30 * attempt, 300)  # Max 5 minutes
                    print(f"⏳ Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Skipping image after {max_attempts} attempts")
                    return None
                    
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network error (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                wait_time = min(30 * attempt, 300)  # Max 5 minutes
                print(f"⏳ Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(f"❌ Skipping image after {max_attempts} attempts")
                return None
        
        except Exception as e:
            print(f"⚠️ Unexpected error (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                wait_time = min(30 * attempt, 300)  # Max 5 minutes
                print(f"⏳ Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(f"❌ Skipping image after {max_attempts} attempts")
                return None
    
    # Falls alle Versuche fehlschlagen
    print(f"❌ All {max_attempts} attempts failed for {img_url}")
    return None

def process_image(image, image_bytes, img_url, page_url):
    try:
        encodings = face_recognition.face_encodings(image, model="large")
        if encodings:
            phash = get_phash(image_bytes)
            embedding = encodings[0].tolist()
            entry = {
                "image_url": img_url,
                "page_url": page_url,
                "embedding": embedding,
                "phash": str(phash)
            }
            # Add to current article data (memory-efficient)
            current_article_data.append(entry)
            print(f"Face found in {img_url}")
    except Exception as e:
        print(f"Error processing image {img_url}: {e}")

def image_bytes_contains_face(image):
    try:
        faces = face_recognition.face_locations(image)
        return len(faces) > 0
    except Exception as e:
        print(f"Error checking image: {e}")
        return False

def get_phash(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    hash_value = imagehash.phash(image)
    return hash_value

def str_to_phash(phash_str):
    try:
        return imagehash.hex_to_hash(phash_str)
    except ValueError as e:
        print(f"Error converting phash: {e}")
        return None

def compare_hashes(phash):
    """Da wir keine Datei lesen - immer False (keine Duplikate erkennen)."""
    # Nur in aktuellen Artikel-Daten prüfen
    for entry in current_article_data:
        existing_phash = str_to_phash(entry["phash"])
        if existing_phash and existing_phash - phash < 5:
            return True
    return False

def get_articles_from_single_category_page(category_url):
    """Collects all articles from a single category page."""
    # Infinite retry loop - never give up!
    while True:
        try:
            print(f"Loading articles from category page: {category_url}")
            headers = {
                "User-Agent": USER_AGENT
            }
            resp = requests.get(category_url, timeout=60, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Ethical crawling: respect crawl delay
            time.sleep(CRAWL_DELAY)
            
            page_articles = []
            
            # Collect links to person articles
            content_div = soup.find("div", {"id": "mw-content-text"})
            if content_div:
                links = content_div.find_all("a", href=lambda x: x and x.startswith("/wiki/") and ":" not in x.split("/wiki/")[1])
                for link in links:
                    article_url = urllib.parse.urljoin("https://en.wikipedia.org", link['href'])
                    page_articles.append(article_url)
            
            # Find next category page URL (robust search)
            next_category_url = None
            
            # Method 1: Search for "(next page)" link
            next_links = soup.find_all("a", string=lambda text: text and text.strip() == "(next page)")
            if next_links:
                next_page_link = next_links[0]
                if next_page_link.get('href'):
                    next_category_url = urllib.parse.urljoin("https://en.wikipedia.org", next_page_link['href'])
                    print(f"    Next-page link found: {next_page_link.get('href')}")
            
            # Method 2: Fallback - Search for link with "next" text  
            if not next_category_url:
                all_links = soup.find_all("a", href=True)
                for link in all_links:
                    link_text = link.get_text().strip().lower()
                    if "next page" in link_text or link_text == "next":
                        next_category_url = urllib.parse.urljoin("https://en.wikipedia.org", link['href'])
                        print(f"    Fallback next-link found: {link['href']}")
                        break
            
            # Method 3: Search for pagefrom parameter in URLs
            if not next_category_url:
                pagefrom_links = soup.find_all("a", href=lambda x: x and "pagefrom=" in x)
                if pagefrom_links:
                    # Filter links that come after the current page
                    for link in pagefrom_links:
                        if "(next page)" in str(link.parent) or "next" in link.get_text().lower():
                            next_category_url = urllib.parse.urljoin("https://en.wikipedia.org", link['href'])
                            print(f"    Pagefrom next-link found: {link['href']}")
                            break
                    # If no explicit "next" text, take the first pagefrom link
                    if not next_category_url and pagefrom_links:
                        next_category_url = urllib.parse.urljoin("https://en.wikipedia.org", pagefrom_links[0]['href'])
                        print(f"    First pagefrom-link found: {pagefrom_links[0]['href']}")
            
            # Method 4: Search in navigation elements
            if not next_category_url:
                nav_elements = soup.find_all(['div', 'span'], class_=lambda x: x and ('mw-category-group' in x or 'pager' in x))
                for nav in nav_elements:
                    links = nav.find_all("a", href=True)
                    for link in links:
                        if "pagefrom=" in link.get('href', ''):
                            next_category_url = urllib.parse.urljoin("https://en.wikipedia.org", link['href'])
                            print(f"    Navigation next-link found: {link['href']}")
                            break
                    if next_category_url:
                        break
            
            if next_category_url:
                print(f"✓ {len(page_articles)} articles loaded from this category page, next page available")
            else:
                print(f"✓ {len(page_articles)} articles loaded from this category page, no further page found")
            
            return page_articles, next_category_url
            
        except Exception as e:
            print(f"⚠️ Network error while loading category page {category_url}: {e}")
            print("⏳ Waiting 60 seconds and retrying...")
            time.sleep(60)
            # Loop continues - never give up!

def get_original_image_url(filename):
    """
    Holt die Original-Bild-URL einer Wikipedia-Datei über die MediaWiki API.
    filename: z.B. "File:Albert_Einstein_1921.jpg"
    """

    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": filename,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }

    try:
        data = requests.get(api, params=params, timeout=30).json()
    except Exception as e:
        print(f"⚠️ API error for {filename}: {e}")
        return None

    pages = data.get("query", {}).get("pages", {})

    for page in pages.values():
        if "imageinfo" in page:
            return page["imageinfo"][0]["url"]

    return None


def crawl_images():
    global queue, current_article_data
    processed_count = 0
    entries_saved = 0
    current_category_url = None

    # Resume logic
    last_article = get_last_crawled_page()
    if last_article:
        current_category_url = wikipedia_url_to_category_link(last_article)
        print("📄 Resuming from last crawled article via category link")
    else:
        print("🚀 Starting from first category page")
        current_category_url = "https://en.wikipedia.org/wiki/Category:Living_people"

    while current_category_url:
        print(f"\n--- Processing category page: {current_category_url} ---")

        page_articles, next_category_url = get_articles_from_single_category_page(current_category_url)

        if not page_articles:
            print("No articles found on this page - going to next")
            current_category_url = next_category_url
            continue

        queue = deque(page_articles)
        remaining_articles = len(queue)
        print(f"Processing {remaining_articles} remaining articles from this category page")

        while queue:
            url = queue.popleft()

            if url == "https://en.wikipedia.org/wiki/Family_name":
                continue

            processed_count += 1
            print(f"Crawling page: {url} (#{processed_count})")

            current_article_data.clear()

            # Load article with retry
            try:
                while True:
                    try:
                        headers = {"User-Agent": USER_AGENT}
                        resp = requests.get(url, timeout=60, headers=headers)
                        resp.raise_for_status()
                        time.sleep(CRAWL_DELAY)
                        break
                    except Exception as e:
                        print(f"⚠️ Network error while loading page {url}: {e}")
                        print("⏳ Waiting 30 seconds and retrying...")
                        time.sleep(30)
            except Exception as e:
                print(f"Unexpected error while loading page {url}: {e}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # ------------------------------------------------------------
            # 🔥 NEU: Datei-Links extrahieren (findet ALLE Bilder)
            # ------------------------------------------------------------
            file_links = [
                a.get("href")
                for a in soup.find_all("a")
                if a.get("href", "").startswith("/wiki/File:")
            ]

            # ------------------------------------------------------------
            # 🔥 NEU: Originalbilder über MediaWiki API holen
            # ------------------------------------------------------------
            for link in file_links:
                filename = link.split("/wiki/")[1]
                original_url = get_original_image_url(filename)

                if not original_url:
                    continue

                # Nur echte Fotos behalten
                if not original_url.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                image_bytes = download_image(original_url)
                if not image_bytes:
                    continue

                try:
                    image = face_recognition.load_image_file(BytesIO(image_bytes))

                    if image_bytes_contains_face(image):
                        if not compare_hashes(get_phash(image_bytes)):
                            process_image(image, image_bytes, original_url, url)
                        else:
                            print(f"Image already present in current article: {original_url}")
                    else:
                        print(f"No face found: {original_url}")

                    del image, image_bytes

                except Exception as e:
                    print(f"Error loading image {original_url}: {e}")
                    continue

            # ------------------------------------------------------------
            # Dein alter IMG-Tag-Code bleibt bestehen (Backup)
            # ------------------------------------------------------------
            images = soup.find_all("img")
            for img in images:
                img_url = img.get("src")
                if not img_url:
                    continue
                img_url = urllib.parse.urljoin(url, img_url)

                if "/thumb/" in img_url:
                    parts = img_url.split("/thumb/")
                    if len(parts) == 2:
                        thumb_path = parts[1]
                        segments = thumb_path.split("/")
                        if len(segments) >= 4:
                            original_filename = segments[-2]
                            img_url = (
                                parts[0]
                                + "/"
                                + "/".join(segments[:-2])
                                + "/"
                                + original_filename
                            )

                if "upload.wikimedia.org" not in img_url:
                    continue

                image_bytes = download_image(img_url)
                if not image_bytes:
                    continue

                try:
                    image = face_recognition.load_image_file(BytesIO(image_bytes))

                    if image_bytes_contains_face(image):
                        if not compare_hashes(get_phash(image_bytes)):
                            process_image(image, image_bytes, img_url, url)
                        else:
                            print(f"Image already present in current article: {img_url}")
                    else:
                        print(f"No face found: {img_url}")

                    del image, image_bytes

                except Exception as e:
                    print(f"Error loading image {img_url}: {e}")
                    continue

            # Save results
            if current_article_data:
                entries_saved += len(current_article_data)
                save_database()

            save_last_crawled_page(url)

            if processed_count % 10 == 0:
                gc.collect()
                print(f"🧹 Garbage collection performed after {processed_count} articles")

            print(f"✓ Category page completed. Processed articles: {processed_count}")

            current_category_url = next_category_url
            if not current_category_url:
                print("⚠️ No more category pages found - starting from beginning...")
                current_category_url = "https://en.wikipedia.org/wiki/Category:Living_people"
                time.sleep(300)

    print(f"Crawling completed. {entries_saved} new entries added to database.")

if __name__ == "__main__":
    print("Starting Living People Crawler (append-only)...")
    try:
        crawl_images()  # adjust max_pages
    except KeyboardInterrupt:
        # In case the signal handler is not triggered
        print("\nInterrupt detected! Saving database...")
        save_database()
        close_database()
        print("Crawler terminated.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        save_database()
        close_database()
        print("Database was saved despite error.")
    finally:
        # Ensure database is saved and closed in any case
        save_database()
        close_database()
    