import requests
from bs4 import BeautifulSoup
import face_recognition
import urllib.parse
import json
from collections import deque
from io import BytesIO
from PIL import Image
import imagehash
from warcio.archiveiterator import ArchiveIterator

visited_pages = set()

with open("face_embeddings.json", "r") as f:
    try:
        face_db = json.load(f)
    except json.JSONDecodeError:
        print("Error loading face database. Make sure the file is correctly formatted.")
        face_db = []

# --- Common Crawl Einstellungen ---
CC_SERVER = 'http://index.commoncrawl.org/'
CC_INDEX = 'CC-MAIN-2025-30'  # Aktuellsten Index anpassen
CC_USER_AGENT = {'User-Agent': 'cc-get-started/1.0 (deinname@example.com)'}

def is_internal_link(base_url, link):
    base_domain = urllib.parse.urlparse(base_url).netloc
    link_domain = urllib.parse.urlparse(link).netloc
    return base_domain == link_domain or link_domain == ""

def search_cc_index(url):
    encoded_url = urllib.parse.quote_plus(url)
    index_url = f'{CC_SERVER}{CC_INDEX}-index?url={encoded_url}&output=json'
    try:
        resp = requests.get(index_url, headers=CC_USER_AGENT, timeout=15)
        if resp.status_code == 200:
            records = resp.text.strip().split('\n')
            return [json.loads(r) for r in records]
    except Exception as e:
        print(f"Error in Common Crawl index search for {url}: {e}")
    return []

def fetch_page_from_cc(records):
    for record in records:
        offset, length = int(record['offset']), int(record['length'])
        s3_url = f'https://data.commoncrawl.org/{record["filename"]}'
        byte_range = f'bytes={offset}-{offset+length-1}'
        try:
            resp = requests.get(s3_url, headers={**CC_USER_AGENT, 'Range': byte_range}, stream=True, timeout=20)
            if resp.status_code == 206:
                stream = ArchiveIterator(resp.raw)
                for warc_record in stream:
                    if warc_record.rec_type == 'response':
                        return warc_record.content_stream().read()
        except Exception as e:
            print(f"Error loading page from Common Crawl: {e}")
    return None

def download_image(img_url):
    allowed_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    if not any(img_url.lower().endswith(ext) for ext in allowed_exts):
        print(f"Skipping incompatible image format: {img_url}")
        return None
    try:
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")
        return None

def process_image(image, image_bytes, img_url, page_url):
    try:
        encodings = face_recognition.face_encodings(image)
        if encodings:
            phash = get_phash(image_bytes)
            embedding = encodings[0].tolist()
            entry = {
                "image_url": img_url,
                "page_url": page_url,
                "embedding": embedding,
                "phash": str(phash)
            }
            face_db.append(entry)
            print(f"Face found in {img_url}")
    except Exception as e:
        print(f"Error processing image {img_url}: {e}")

def bild_bytes_enthält_gesicht(image):
    try:
        gesichter = face_recognition.face_locations(image)
        return len(gesichter) > 0
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
    for entry in face_db:
        existing_phash = str_to_phash(entry["phash"])
        if existing_phash and existing_phash - phash < 5:
            return True
    return False

def crawl_images(start_url, max_pages=1000):
    queue = deque([start_url])
    while queue and len(visited_pages) < max_pages:
        url = queue.popleft()
        if url in visited_pages:
            continue
        print(f"Crawling page: {url} ({len(visited_pages)+1}/{max_pages})")
        visited_pages.add(url)

        # Suche Seite im Common Crawl Index
        records = search_cc_index(url)
        if not records:
            print(f"No data found for {url} in Common Crawl.")
            continue

        # Lade Seite von Common Crawl
        page_bytes = fetch_page_from_cc(records)
        if not page_bytes:
            print(f"Page {url} could not be loaded.")
            continue

        # Parse HTML mit BeautifulSoup
        try:
            soup = BeautifulSoup(page_bytes, "html.parser")
        except Exception as e:
            print(f"Error parsing page {url}: {e}")
            continue

        # Bilder finden und herunterladen
        images = soup.find_all("img")
        for img in images:
            img_url = img.get("src")
            if not img_url:
                continue
            img_url = urllib.parse.urljoin(url, img_url)

            image_bytes = download_image(img_url)
            if not image_bytes:
                continue
            image = face_recognition.load_image_file(BytesIO(image_bytes))
            if image_bytes and bild_bytes_enthält_gesicht(image) and not compare_hashes(get_phash(image_bytes)):
                process_image(image, image_bytes, img_url, url)

        # Interne Links sammeln und hinzufügen
        links = soup.find_all("a", href=True)
        for link in links:
            next_url = urllib.parse.urljoin(url, link['href'])
            if is_internal_link(start_url, next_url) and next_url not in visited_pages:
                queue.append(next_url)

        # Ergebnisse speichern
        with open("face_embeddings.json", "w") as f:
            json.dump(face_db, f, indent=2)

if __name__ == "__main__":
    start_url = "https://www.imdb.com/list/ls524618334/"  # Your starting page
    crawl_images(start_url, max_pages=100)  # Adjust max_pages
