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
import shutil

visited_pages = set()

def create_backup():
    """Erstellt eine Sicherheitskopie der face_embeddings.json Datei"""
    db_file = "face_embeddings.json"
    if os.path.exists(db_file):
        # Zeitstempel für Backup-Namen
        backup_file = f"face_embeddings_backup.json"
        
        try:
            shutil.copy2(db_file, backup_file)
            print(f"Sicherheitskopie erstellt: {backup_file}")
        except Exception as e:
            print(f"Fehler beim Erstellen der Sicherheitskopie: {e}")
    else:
        print("Keine bestehende Datenbank gefunden - keine Sicherheitskopie erstellt.")

# Sicherheitskopie beim Start erstellen
create_backup()

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

def is_person_article(url):
    """Prüft ob ein Wikipedia-Artikel wahrscheinlich über eine Person ist."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        }
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Indikatoren für Personenartikel
        person_indicators = [
            # Kategorien
            "Category:Living people", "Category:People", "Category:Births", "Category:Deaths",
            "Category:American people", "Category:British people", "Category:German people",
            "Category:Politicians", "Category:Actors", "Category:Musicians", "Category:Athletes",
            "Category:Scientists", "Category:Writers", "Category:Artists", "Category:Presidents",
            
            # Text-Indikatoren
            "born", "died", "birth", "death", "married", "spouse", "children",
            "early life", "personal life", "career", "education", "biography"
        ]
        
        # Infobox prüfen - Personen haben oft spezielle Infoboxen
        infobox = soup.find("table", class_=lambda x: x and "infobox" in x.lower())
        if infobox:
            infobox_text = infobox.get_text().lower()
            if any(indicator in infobox_text for indicator in ["born", "died", "birth", "occupation", "spouse"]):
                return True
        
        # Kategorien prüfen
        categories = soup.find_all("a", href=lambda x: x and "/wiki/Category:" in x)
        for cat in categories:
            cat_text = cat.get_text().lower()
            if any(indicator.lower() in cat_text for indicator in person_indicators):
                return True
        
        # Seiteninhalt prüfen
        content = soup.find("div", {"id": "mw-content-text"})
        if content:
            content_text = content.get_text().lower()
            person_count = sum(1 for indicator in person_indicators if indicator.lower() in content_text)
            if person_count >= 3:  # Mindestens 3 Indikatoren müssen vorhanden sein
                return True
        
        return False
    except Exception as e:
        print(f"Fehler beim Prüfen des Artikels {url}: {e}")
        return False

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
    if ".svg" in img_url.lower():
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

def get_len_images(db):
    """Gibt die Anzahl der Bilder in der Datenbank zurück."""
    return len(db)

def get_person_articles_from_categories():
    """Sammelt Wikipedia-Artikel von Personen aus bekannten Kategorien."""
    person_categories = [
        "https://en.wikipedia.org/wiki/Category:Living_people",
    ]
    
    person_articles = []
    
    for category_url in person_categories:
        try:
            print(f"Durchsuche Kategorie: {category_url}")
            
            # Mehrere Seiten der Kategorie durchsuchen
            current_url = category_url
            pages_crawled = 0
            max_pages_per_category = 3  # Maximal 3 Seiten pro Kategorie
            
            while current_url and pages_crawled < max_pages_per_category:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
                }
                resp = requests.get(current_url, timeout=10, headers=headers)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Links zu Personen-Artikeln sammeln
                content_div = soup.find("div", {"id": "mw-content-text"})
                if content_div:
                    links = content_div.find_all("a", href=lambda x: x and x.startswith("/wiki/") and ":" not in x.split("/wiki/")[1])
                    page_articles = 0
                    for link in links:
                        article_url = urllib.parse.urljoin("https://en.wikipedia.org", link['href'])
                        if article_url not in person_articles:
                            person_articles.append(article_url)
                            page_articles += 1
                    
                    print(f"  Seite {pages_crawled + 1}: {page_articles} Artikel gefunden")
                
                # Nach "next page" Link suchen
                next_page_link = None
                nav_links = soup.find_all("a", string=lambda text: text and "next" in text.lower())
                if not nav_links:
                    # Alternative: Suche nach numerischen Seitenzahlen
                    page_links = soup.find("div", {"id": "mw-pages"})
                    if page_links:
                        next_links = page_links.find_all("a", href=lambda x: x and "pagefrom=" in x)
                        if next_links:
                            next_page_link = next_links[0]
                
                if nav_links:
                    next_page_link = nav_links[0]
                
                if next_page_link and next_page_link.get('href'):
                    current_url = urllib.parse.urljoin("https://en.wikipedia.org", next_page_link['href'])
                    pages_crawled += 1
                else:
                    # Keine weitere Seite gefunden
                    break
                
                # Begrenzen um nicht zu viele Artikel auf einmal zu sammeln
                if len(person_articles) >= 500:
                    print(f"  Maximale Artikelanzahl erreicht ({len(person_articles)})")
                    return person_articles
                    
        except Exception as e:
            print(f"Fehler beim Durchsuchen der Kategorie {category_url}: {e}")
            continue
    
    print(f"Insgesamt {len(person_articles)} Artikel aus Kategorien gesammelt")
    return person_articles

def is_page_already_crawled(page_url):
    """Prüft ob eine Seite bereits gecrawlt wurde anhand der page_url in der Datenbank."""
    for entry in face_db:
        if entry.get("page_url") == page_url:
            return True
    return False

def crawl_images(max_pages=1000):
    old_db_len = get_len_images(face_db)
    
    # Personen-Artikel aus Living People Kategorie sammeln
    print("Sammle Living People Artikel...")
    category_articles = get_person_articles_from_categories()
    
    # Queue mit gefundenen Artikeln erstellen
    queue = deque(category_articles)
    
    print(f"Insgesamt {len(queue)} Artikel in der Warteschlange")
    
    while queue and len(visited_pages) < max_pages:
        url = queue.popleft()
        if url in visited_pages:
            continue
        
        # Prüfen ob diese Seite bereits gecrawlt wurde (basierend auf der Datenbank)
        if is_page_already_crawled(url):
            print(f"Seite bereits gecrawlt (überspringe): {url}")
            visited_pages.add(url)
            continue
        
        # Für Wikipedia: Prüfen ob es sich um einen Personenartikel handelt
        if "wikipedia.org" in url and not is_person_article(url):
            print(f"Kein Personenartikel (überspringe): {url}")
            visited_pages.add(url)
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
            if bild_bytes_enthält_gesicht(image):
                if not compare_hashes(get_phash(image_bytes)):
                    process_image(image, image_bytes, img_url, url)
                else:
                    print(f"Bild bereits vorhanden: {img_url}")
            else:
                print(f"Kein Gesicht gefunden: {img_url}")

        # Interne Links sammeln
        links = soup.find_all("a", href=True)
        for link in links:
            next_url = urllib.parse.urljoin(url, link['href'])
            if is_internal_link(url, next_url) and next_url not in visited_pages:
                queue.append(next_url)
        
        # Ergebnisse speichern
        with open("face_embeddings.json", "w") as f:
                json.dump(face_db, f, indent=2)

    print(f"Insgesamt sind {get_len_images(face_db)} Bilder in der Datenbank, davon wurden {get_len_images(face_db) - old_db_len} neue Bilder gespeichert.")

if __name__ == "__main__":
    print("Starte Living People Crawler...")
    crawl_images(max_pages=100)  # max_pages anpassen
    