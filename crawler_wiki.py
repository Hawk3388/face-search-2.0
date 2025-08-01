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
import signal
import sys

queue = deque()  # Globale Queue für Signal-Handler

def save_database():
    """Speichert die Gesichts-Datenbank in die JSON-Datei."""
    try:
        with open("face_embeddings.json", "w") as f:
            json.dump(face_db, f, indent=2)
        print("Datenbank erfolgreich gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern der Datenbank: {e}")

def get_visited_pages_from_db():
    """Extrahiert alle bereits besuchten Seiten aus der Datenbank."""
    visited_pages = set()
    for entry in face_db:
        page_url = entry.get("page_url")
        if page_url:
            visited_pages.add(page_url)
    return visited_pages

def get_last_crawled_page():
    """Gibt die letzte gecrawlte Seite zurück."""
    if not face_db:
        return None
    # Das letzte Element in der Datenbank ist die zuletzt bearbeitete Seite
    last_entry = face_db[-1]
    return last_entry.get("page_url")

def signal_handler(sig, frame):
    """Handler für Unterbrechungssignale (Ctrl+C)."""
    print("\nUnterbrechung erkannt! Speichere Datenbank...")
    save_database()
    print("Crawler beendet.")
    sys.exit(0)

# Signal-Handler registrieren
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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

def get_current_category_page_url():
    """Findet die aktuelle Kategorie-Seiten-URL basierend auf der letzten bearbeiteten Seite."""
    last_page = get_last_crawled_page()
    
    if not last_page:
        # Wenn keine letzte Seite, starte mit der ersten Kategorie-Seite
        return "https://en.wikipedia.org/wiki/Category:Living_people"
    
    # Versuche die Kategorie-Seite zu finden, auf der sich die letzte Seite befindet
    print(f"Suche Kategorie-Seite für letzte bearbeitete Seite: {last_page}")
    
    category_url = "https://en.wikipedia.org/wiki/Category:Living_people"
    pages_searched = 0
    max_search_pages = 50  # Begrenzte Suche um nicht ewig zu suchen
    
    while category_url and pages_searched < max_search_pages:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
            }
            resp = requests.get(category_url, timeout=10, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Artikel auf dieser Seite sammeln
            content_div = soup.find("div", {"id": "mw-content-text"})
            if content_div:
                links = content_div.find_all("a", href=lambda x: x and x.startswith("/wiki/") and ":" not in x.split("/wiki/")[1])
                for link in links:
                    article_url = urllib.parse.urljoin("https://en.wikipedia.org", link['href'])
                    if article_url == last_page:
                        print(f"✓ Letzte Seite gefunden auf Kategorie-Seite: {category_url}")
                        return category_url
            
            # Zur nächsten Kategorie-Seite (robuste Suche)
            next_page_link = None
            
            # Methode 1: Suche nach "(next page)" Link
            next_links = soup.find_all("a", string=lambda text: text and text.strip() == "(next page)")
            if next_links:
                next_page_link = next_links[0]
            
            # Methode 2: Fallback - Suche nach Link mit "next" Text  
            if not next_page_link:
                all_links = soup.find_all("a", href=True)
                for link in all_links:
                    link_text = link.get_text().strip().lower()
                    if "next page" in link_text or link_text == "next":
                        next_page_link = link
                        break
            
            # Methode 3: Suche nach pagefrom Parameter in URLs
            if not next_page_link:
                pagefrom_links = soup.find_all("a", href=lambda x: x and "pagefrom=" in x)
                if pagefrom_links:
                    # Filtere Links die nach der aktuellen Seite kommen
                    for link in pagefrom_links:
                        if "(next page)" in str(link.parent) or "next" in link.get_text().lower():
                            next_page_link = link
                            break
                    # Falls kein expliziter "next" Text, nimm den ersten pagefrom Link
                    if not next_page_link and pagefrom_links:
                        next_page_link = pagefrom_links[0]
            
            if next_page_link and next_page_link.get('href'):
                category_url = urllib.parse.urljoin("https://en.wikipedia.org", next_page_link['href'])
                pages_searched += 1
                print(f"    Suche weiter auf nächster Kategorie-Seite ({pages_searched})")
            else:
                print(f"    Keine weitere Kategorie-Seite gefunden nach {pages_searched} Seiten")
                break
                
        except Exception as e:
            print(f"Fehler beim Suchen der Kategorie-Seite: {e}")
            break
    
    print(f"Letzte Seite nicht in den letzten {pages_searched} Kategorie-Seiten gefunden - starte von vorne")
    return "https://en.wikipedia.org/wiki/Category:Living_people"

def get_articles_from_single_category_page(category_url):
    """Sammelt alle Artikel von einer einzelnen Kategorie-Seite."""
    try:
        print(f"Lade Artikel von Kategorie-Seite: {category_url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        }
        resp = requests.get(category_url, timeout=10, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        page_articles = []
        
        # Links zu Personen-Artikeln sammeln
        content_div = soup.find("div", {"id": "mw-content-text"})
        if content_div:
            links = content_div.find_all("a", href=lambda x: x and x.startswith("/wiki/") and ":" not in x.split("/wiki/")[1])
            for link in links:
                article_url = urllib.parse.urljoin("https://en.wikipedia.org", link['href'])
                page_articles.append(article_url)
        
        # Nächste Kategorie-Seiten-URL finden (robuste Suche)
        next_category_url = None
        
        # Methode 1: Suche nach "(next page)" Link
        next_links = soup.find_all("a", string=lambda text: text and text.strip() == "(next page)")
        if next_links:
            next_page_link = next_links[0]
            if next_page_link.get('href'):
                next_category_url = urllib.parse.urljoin("https://en.wikipedia.org", next_page_link['href'])
                print(f"    Next-Page Link gefunden: {next_page_link.get('href')}")
        
        # Methode 2: Fallback - Suche nach Link mit "next" Text  
        if not next_category_url:
            all_links = soup.find_all("a", href=True)
            for link in all_links:
                link_text = link.get_text().strip().lower()
                if "next page" in link_text or link_text == "next":
                    next_category_url = urllib.parse.urljoin("https://en.wikipedia.org", link['href'])
                    print(f"    Fallback Next-Link gefunden: {link['href']}")
                    break
        
        # Methode 3: Suche nach pagefrom Parameter in URLs
        if not next_category_url:
            pagefrom_links = soup.find_all("a", href=lambda x: x and "pagefrom=" in x)
            if pagefrom_links:
                # Filtere Links die nach der aktuellen Seite kommen
                for link in pagefrom_links:
                    if "(next page)" in str(link.parent) or "next" in link.get_text().lower():
                        next_category_url = urllib.parse.urljoin("https://en.wikipedia.org", link['href'])
                        print(f"    Pagefrom Next-Link gefunden: {link['href']}")
                        break
                # Falls kein expliziter "next" Text, nimm den ersten pagefrom Link
                if not next_category_url and pagefrom_links:
                    next_category_url = urllib.parse.urljoin("https://en.wikipedia.org", pagefrom_links[0]['href'])
                    print(f"    Erster Pagefrom-Link gefunden: {pagefrom_links[0]['href']}")
        
        # Methode 4: Suche in Navigation-Elementen
        if not next_category_url:
            nav_elements = soup.find_all(['div', 'span'], class_=lambda x: x and ('mw-category-group' in x or 'pager' in x))
            for nav in nav_elements:
                links = nav.find_all("a", href=True)
                for link in links:
                    if "pagefrom=" in link.get('href', ''):
                        next_category_url = urllib.parse.urljoin("https://en.wikipedia.org", link['href'])
                        print(f"    Navigation Next-Link gefunden: {link['href']}")
                        break
                if next_category_url:
                    break
        
        if next_category_url:
            print(f"✓ {len(page_articles)} Artikel von dieser Kategorie-Seite geladen, nächste Seite verfügbar")
        else:
            print(f"✓ {len(page_articles)} Artikel von dieser Kategorie-Seite geladen, keine weitere Seite gefunden")
        
        return page_articles, next_category_url
        
    except Exception as e:
        print(f"Fehler beim Laden der Kategorie-Seite {category_url}: {e}")
        return [], None

def is_page_already_crawled(page_url):
    """Prüft ob eine Seite bereits gecrawlt wurde anhand der page_url in der Datenbank."""
    for entry in face_db:
        if entry.get("page_url") == page_url:
            return True
    return False

def crawl_images(max_pages=1000):
    global queue
    old_db_len = get_len_images(face_db)
    processed_count = 0
    
    # Letzte gecrawlte Seite finden
    last_page = get_last_crawled_page()
    
    if last_page:
        print(f"Letzte bearbeitete Seite gefunden: {last_page}")
        print("Setze Crawling nach der letzten Seite fort...")
    else:
        print("Keine vorherigen Daten gefunden - starte neu...")
    
    # Aktuelle Kategorie-Seiten-URL finden
    current_category_url = get_current_category_page_url()
    
    while current_category_url and processed_count < max_pages:
        print(f"\n--- Bearbeite Kategorie-Seite: {current_category_url} ---")
        
        # Artikel von der aktuellen Kategorie-Seite laden
        page_articles, next_category_url = get_articles_from_single_category_page(current_category_url)
        
        if not page_articles:
            print("Keine Artikel auf dieser Seite gefunden - gehe zur nächsten")
            current_category_url = next_category_url
            continue
        
        # Queue mit Artikeln von dieser Kategorie-Seite füllen
        queue = deque()
        
        # Wenn wir eine letzte Seite haben und sie in dieser Liste ist
        if last_page and last_page in page_articles:
            # Finde Position der letzten Seite in der Liste
            last_index = page_articles.index(last_page)
            # Starte mit den Artikeln NACH der letzten bearbeiteten Seite
            remaining_articles = page_articles[last_index + 1:]
            queue = deque(remaining_articles)
            print(f"✓ Letzte Seite in aktueller Liste gefunden - setze ab nächster Seite fort ({len(remaining_articles)} verbleibend)")
            # Nach dem ersten Durchlauf die letzte Seite zurücksetzen
            last_page = None
        else:
            # Alle Artikel dieser Seite bearbeiten
            queue = deque(page_articles)
            print(f"Bearbeite alle {len(page_articles)} Artikel dieser Kategorie-Seite")
        
        # Artikel von der aktuellen Kategorie-Seite bearbeiten
        while queue and processed_count < max_pages:
            url = queue.popleft()
            
            # Prüfen ob diese Seite bereits gecrawlt wurde
            if is_page_already_crawled(url):
                print(f"Seite bereits gecrawlt (überspringe): {url}")
                continue
                
            processed_count += 1
            print(f"Crawle Seite: {url} ({processed_count}/{max_pages})")

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
            
            # Ergebnisse speichern
            save_database()
        
        # Zur nächsten Kategorie-Seite
        print(f"✓ Kategorie-Seite abgeschlossen. Verarbeitete Artikel: {processed_count}")
        if processed_count >= max_pages:
            print(f"Maximale Anzahl erreicht ({max_pages})")
            break
            
        current_category_url = next_category_url
        if not current_category_url:
            print("Keine weiteren Kategorie-Seiten gefunden - Crawling abgeschlossen")
            break

    print(f"Insgesamt sind {get_len_images(face_db)} Bilder in der Datenbank, davon wurden {get_len_images(face_db) - old_db_len} neue Bilder gespeichert.")

if __name__ == "__main__":
    print("Starte Living People Crawler...")
    try:
        crawl_images()  # max_pages anpassen
    except KeyboardInterrupt:
        # Falls der Signal-Handler nicht ausgelöst wird
        print("\nUnterbrechung erkannt! Speichere Datenbank...")
        save_database()
        print("Crawler beendet.")
    except Exception as e:
        print(f"Unerwarteter Fehler: {e}")
        save_database()
        print("Datenbank wurde trotz Fehler gespeichert.")
    finally:
        # Sicherstellen, dass die Datenbank auf jeden Fall gespeichert wird
        save_database()
    