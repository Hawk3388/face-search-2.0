import os
# CUDA-Pfade für dlib/face_recognition setzen
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
cudnn_path = r"C:\Program Files\NVIDIA\CUDNN\v9.6\bin\12.6"

# CUDA-Bibliotheken zum PATH hinzufügen
if os.path.exists(cuda_path):
    os.environ['PATH'] = cuda_path + r'\bin;' + os.environ.get('PATH', '')

if os.path.exists(cudnn_path):
    os.environ['PATH'] = cudnn_path + ';' + os.environ.get('PATH', '')

import requests
from bs4 import BeautifulSoup
import face_recognition
import urllib.parse
import json
from collections import deque
from io import BytesIO
from PIL import Image
import imagehash

visited_pages = set()

# Gesichts-Datenbank laden oder erstellen
try:
    with open("face_embeddings.json", "r") as f:
        face_db = json.load(f)
except FileNotFoundError:
    print("face_embeddings.json nicht gefunden. Erstelle neue Datenbank.")
    face_db = []
except json.JSONDecodeError:
    print("Fehler beim Laden der Gesichts-Datenbank. Stelle sicher, dass die Datei korrekt formatiert ist.")
    face_db = []

def is_internal_link(base_url, link):
    base_domain = urllib.parse.urlparse(base_url).netloc
    link_domain = urllib.parse.urlparse(link).netloc
    
    # Vollständige URLs für Vergleich erstellen
    full_link = urllib.parse.urljoin(base_url, link)
    
    # URLs ohne Fragment (Teil nach #) für Vergleich
    base_parsed = urllib.parse.urlparse(base_url)
    link_parsed = urllib.parse.urlparse(full_link)
    
    base_path = base_parsed.path
    link_path = link_parsed.path
    
    # Self-Links ausschließen (Links die auf die gleiche Seite zeigen, auch mit #-Ankern)
    if base_path == link_path:
        return False
    
    # Prüfen ob es sich um Wikipedia handelt
    if "wikipedia.org" in base_domain:
        # Für Wikipedia: nur Links zu anderen Wikipedia-Artikeln erlauben
        if link_domain == "" or link_domain == base_domain:
            # Prüfen ob es ein Wikipedia-Artikel ist (nicht Talk, User, Special, etc.)
            path = link_path
            
            # Wikipedia-Artikel haben das Format /wiki/Artikelname
            if path.startswith("/wiki/") and ":" not in path.split("/wiki/")[1]:
                # Ausschließen von speziellen Seiten
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
        # Für andere Domains: normale Domain-Prüfung
        return base_domain == link_domain or link_domain == ""

def download_image(img_url):
    # Nur kompatible Bildformate zulassen
    allowed_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    if not any(img_url.lower().endswith(ext) for ext in allowed_exts):
        print(f"Überspringe inkompatibles Bildformat: {img_url}")
        return None
    try:
        # Gleichen User-Agent wie beim Crawlen verwenden
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        }
        response = requests.get(img_url, timeout=10, headers=headers)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Fehler beim Herunterladen von {img_url}: {e}")
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
            face_db.append(entry)
            print(f"Gesicht gefunden in {img_url}")
    except Exception as e:
        print(f"Fehler beim Verarbeiten von Bild {img_url}: {e}")

def bild_bytes_enthält_gesicht(image):
    try:
        gesichter = face_recognition.face_locations(image)
        return len(gesichter) > 0
    except Exception as e:
        print(f"Fehler beim Prüfen des Bilds: {e}")
        return False

def get_phash(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    hash_value = imagehash.phash(image)
    return hash_value

def str_to_phash(phash_str):
    try:
        return imagehash.hex_to_hash(phash_str)
    except ValueError as e:
        print(f"Fehler beim Konvertieren des phash: {e}")
        return None

def compare_hashes(phash):
    for entry in face_db:
        existing_phash = str_to_phash(entry["phash"])
        if existing_phash and existing_phash - phash < 5:  # Toleranzwert für Ähnlichkeit
            return True
    return False

def crawl_images(start_url, max_pages=1000):
    queue = deque([start_url])
    while queue and len(visited_pages) < max_pages:
        url = queue.popleft()
        if url in visited_pages:
            continue
        print(f"Crawle Seite: {url} ({len(visited_pages)+1}/{max_pages})")
        visited_pages.add(url)

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
            }
            resp = requests.get(url, timeout=10, headers=headers)
            resp.raise_for_status()
        except Exception as e:
            print(f"Fehler beim Laden der Seite {url}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

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

        # Interne Links sammeln
        links = soup.find_all("a", href=True)
        for link in links:
            next_url = urllib.parse.urljoin(url, link['href'])
            if is_internal_link(start_url, next_url) and next_url not in visited_pages:
                queue.append(next_url)
        
        # Ergebnisse speichern
        with open("face_embeddings.json", "w") as f:
                json.dump(face_db, f, indent=2)

if __name__ == "__main__":
    start_url = "https://en.wikipedia.org/wiki/Barack_Obama"  # Hier deine Startseite eintragen
    crawl_images(start_url, max_pages=100)  # max_pages anpassen
    