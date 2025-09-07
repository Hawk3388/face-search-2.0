import os
import dlib

# Set CUDA paths for dlib/face_recognition
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
cudnn_path = r"C:\Program Files\NVIDIA\CUDNN\v9.6\bin\12.6"

cuda = True if os.path.exists(cuda_path) and os.path.exists(cudnn_path) and dlib.DLIB_USE_CUDA else False

# Add CUDA libraries to PATH
if cuda:
    os.environ['PATH'] = cuda_path + r'\bin;' + os.environ.get('PATH', '')
    os.environ['PATH'] = cudnn_path + ';' + os.environ.get('PATH', '')

cuda = False

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
import gc  # For garbage collection
import time  # For error waiting times
import urllib.robotparser  # For robots.txt respect
import concurrent.futures
import re
import multiprocessing

# Ethical scraping - respect Wikipedia's guidelines
CRAWL_DELAY = 1.0  # 1 second between requests (more than recommended 100ms)
USER_AGENT = "FaceSearchBot/1.0 (Educational Research; Contact: github.com/Hawk3388/face-search-2.0)"

# Global counter for consecutive 429 errors
consecutive_429_errors = 0
MAX_CONSECUTIVE_429 = 10  # Stop after 10 consecutive 429 errors

model = None

# Check for LOW_RAM environment variable to disable PyTorch model
low_ram = os.environ.get('LOW_RAM', '0').lower() in ('1', 'true', 'yes')

if os.path.exists("tinyfacenet_best.pth") and not low_ram:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import transforms

        # Gleiche Transforms wie im Training!
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))
        ])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class TinyFaceNet_inference(nn.Module):
            def __init__(self):
                super(TinyFaceNet_inference, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
                self.bn1   = nn.BatchNorm2d(16)
                self.pool1 = nn.MaxPool2d(2, 2)   # 64x64

                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
                self.bn2   = nn.BatchNorm2d(32)
                self.pool2 = nn.MaxPool2d(2, 2)   # 32x32

                self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
                self.bn3   = nn.BatchNorm2d(64)
                self.pool3 = nn.MaxPool2d(2, 2)   # 16x16

                self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
                self.bn4   = nn.BatchNorm2d(128)
                self.pool4 = nn.MaxPool2d(2, 2)   # 8x8

                self.fc1 = nn.Linear(128 * 8 * 8, 256)
                self.dropout = nn.Dropout(0.5)  # 50% Dropout
                self.fc2 = nn.Linear(256, 1)

            def forward(self, x):
                x = self.pool1(F.relu(self.bn1(self.conv1(x))))
                x = self.pool2(F.relu(self.bn2(self.conv2(x))))
                x = self.pool3(F.relu(self.bn3(self.conv3(x))))
                x = self.pool4(F.relu(self.bn4(self.conv4(x))))
                x = x.view(-1, 128 * 8 * 8)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.sigmoid(self.fc2(x))
                x = x.squeeze().item()
                x = x >= 0.5
                x = not x
                return x

        model = TinyFaceNet_inference().to(device)
        model.load_state_dict(torch.load("tinyfacenet_best.pth", map_location=device))
        model.eval()
        print("PyTorch model loaded successfully.")
    except Exception as e:
        print(f"Error loading PyTorch model: {e}, running face detection with face_recognition only.")
        model = None
elif low_ram:
    print("LOW_RAM mode enabled - skipping PyTorch model loading.")
else:
    print("No model file found, running face detection with face_recognition only.")

# Robots.txt parser for Wikipedia
def check_robots_txt(url):
    """Checks if the URL is allowed according to robots.txt."""
    try:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(urllib.parse.urljoin(url, '/robots.txt'))
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except:
        # On errors reading robots.txt - allow access
        return True

queue = deque()  # Global queue for signal handler

def save_database():
    """Saves the current article data with append-only (absolutely no reading!)."""
    global json_initialized, total_entries_saved
    
    if not current_article_data:
        return
        
    try:
        db_file = "face_embeddings.json"
        
        # If JSON not yet initialized
        if not json_initialized:
            if not os.path.exists(db_file):
                # New file - write JSON array beginning
                with open(db_file, "w") as f:
                    f.write("[\n")
                is_first_entry = True
            else:
                # Existing file - remove closing ] at the end
                remove_closing_bracket(db_file)
                is_first_entry = False  # File already has entries
            
            # Mark file as initialized
            json_initialized = True
        else:
            is_first_entry = False
        
        # Append data (append-only!)
        with open(db_file, "a") as f:
            for i, entry in enumerate(current_article_data):
                if not is_first_entry or i > 0:
                    f.write(",\n")
                # Remove null bytes before writing
                entry_str = "  " + json.dumps(entry, indent=2).replace('\n', '\n  ')
                entry_str = entry_str.replace('\x00', '')
                f.write(entry_str)
                is_first_entry = False
        
        # Update counter and check periodic backup
        entries_added = len(current_article_data)
        total_entries_saved += entries_added
        create_periodic_backup(total_entries_saved)
        
        print(f"{entries_added} new entries added to database. (Total: {total_entries_saved})")
        
        # Clear current article data after saving
        current_article_data.clear()
    except Exception as e:
        print(f"Error saving database: {e}")

def remove_closing_bracket(filename):
    """Removes the closing ] at the end of the file without reading the file."""
    try:
        # Open file in read+write mode
        with open(filename, "r+b") as f:
            # Go to end
            f.seek(0, 2)
            file_size = f.tell()
            
            if file_size < 10:
                return  # File too small
            
            # Go to last bytes and remove ] and whitespace
            max_chars_to_check = min(10, file_size)
            f.seek(file_size - max_chars_to_check)
            
            # Read the last bytes
            end_content = f.read().decode('utf-8', errors='ignore')
            
            # Find position of last ]
            bracket_pos = end_content.rfind(']')
            if bracket_pos != -1:
                # Truncate from position of ]
                new_end_pos = file_size - max_chars_to_check + bracket_pos
                f.seek(new_end_pos)
                f.truncate()
                print("Closing ] removed for append mode.")
            
    except Exception as e:
        print(f"Error removing closing ]: {e}")

def close_database():
    """Properly closes the JSON array."""
    global json_initialized
    try:
        db_file = "face_embeddings.json"
        if os.path.exists(db_file) and json_initialized:
            with open(db_file, "a") as f:
                f.write("\n]")
            print("Database properly closed.")
            # Reset flag to avoid double closing
            json_initialized = False
    except Exception as e:
        print(f"Error closing database: {e}")

def get_visited_pages_from_db():
    """Since we don't do reading - return empty set."""
    return set()

def extract_last_page_url(file_path, max_read_bytes=50000):
    """
    Reads the file from the end and extracts the last 'page_url'.
    max_read_bytes: Bytes to read backwards (increase if needed).
    """
    try:
        with open(file_path, 'rb') as f:
            f.seek(0, 2)  # Go to the end
            file_size = f.tell()
            read_size = min(max_read_bytes, file_size)
            f.seek(file_size - read_size)
            data_bytes = f.read()
            if data_bytes is None:
                print("⚠️ File read returned None - possible file corruption")
                return None
            data = data_bytes.decode('utf-8', errors='ignore')
        
        # Search for the last "page_url" in the data
        # Regex: "page_url": "VALUE"
        match = re.search(r'"page_url"\s*:\s*"([^"]+)"', data)
        if match:
            return match.group(1)  # Extract the value
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_last_crawled_page():
    """Reads the last crawled article from a small separate file."""
    try:
        if os.path.exists("last_crawled_page.txt"):
            with open("last_crawled_page.txt", "r", encoding="utf-8") as f:
                content = f.read()
                if content is None:
                    print("⚠️ File read returned None - possible file corruption")
                    return None
                last_page = content.strip()
                if last_page:
                    print(f"📄 Last crawled article: {last_page}")
                    return last_page
                else:
                    if os.path.exists("face_embeddings.json"):
                        print("⚠️ last_crawled_page.txt is empty - trying to read last page from database.")
                        last_page = extract_last_page_url("face_embeddings.json")
                        if last_page:
                            save_last_crawled_page(last_page)
                            print(f"📄 Last crawled article from database: {last_page}")
                            return last_page
                        else:
                            print("⚠️ No last page found in database - starting fresh.")
    except Exception as e:
        print(f"Error reading last crawled page: {e}")
    return None

def get_last_crawled_pages(filename="last_crawled_page.txt"):
    """
    Reads all URLs and thread indices from the file and returns a list of tuples: (url, thread_id)
    """
    urls = []
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        url = parts[0]
                        urls.append(url)
        return urls
    except Exception as e:
        print(f"Error loading last crawled pages: {e}")
        return []

def save_last_crawled_pages(page_url, thread_id, filename="last_crawled_page.txt"):
    """
    Save only the page URL and thread index for each thread.
    Each thread writes only its own line: <page_url> <thread_id>
    """
    try:
        # Read all lines (if file exists)
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = []
        # Ensure enough lines
        while len(lines) <= thread_id:
            lines.append("\n")
        # Update only the line for this thread
        lines[thread_id] = f"{page_url} {thread_id}\n"
        # Write back all lines
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error saving last page for thread {thread_id}: {e}")

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
    """Creates a backup at startup."""
    try:
        db_file = "face_embeddings.json"
        if os.path.exists(db_file):
            backup_file = f"face_embeddings_backup.json"
            
            shutil.copy2(db_file, backup_file)
            print(f"📁 Startup backup created: {backup_file}")
        else:
            print("No existing database found - no startup backup created.")
    except Exception as e:
        print(f"Error creating startup backup: {e}")

def create_periodic_backup(entries_count):
    """Creates a backup every 1000 entries."""
    try:
        if entries_count > 0 and entries_count % 1000 == 0:
            import datetime
            db_file = "face_embeddings.json"
            if os.path.exists(db_file):
                backup_file = f"face_embeddings_backup.json"
                
                shutil.copy2(db_file, backup_file)
                print(f"📁 Backup created after {entries_count} entries: {backup_file}")
    except Exception as e:
        print(f"Error creating periodic backup: {e}")

# Skip backup at startup
create_backup()

# Global variable for current article (RAM-efficient)
current_article_data = []

# Flag whether JSON file has been initialized
json_initialized = False

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
    global consecutive_429_errors
    
    # Only allow compatible image formats
    allowed_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    if not any(img_url.lower().endswith(ext) for ext in allowed_exts):
        print(f"Skipping incompatible image format: {img_url}")
        return None
    if ".svg" in img_url.lower():
        print(f"Skipping incompatible image format: {img_url}")
        return None
    if img_url.lower().endswith("wikipedia.png"):
        print(f"Skipping Wikipedia logo: {img_url}")
        return None
    
    # Maximum 3 attempts per image
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            # # Respect robots.txt
            # if not check_robots_txt(img_url):
            #     print(f"⚠️ robots.txt forbids access to: {img_url}")
            #     return None
            
            headers = {
                "User-Agent": USER_AGENT,
                "Accept": "image/*",
                "Connection": "close"
            }
            
            print(f"📥 Downloading image (attempt {attempt}/{max_attempts}): {img_url}")
            
            response = requests.get(img_url, timeout=30, headers=headers, stream=True)
            response.raise_for_status()
            
            # Size check before complete download
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 2 * 1024 * 1024:  # 2MB
                print(f"⚠️ Image too large ({int(content_length)/1024/1024:.1f}MB): {img_url}")
                return None
            
            # Complete download
            image_bytes = response.content
            response.close()  # Close response immediately
            
            if len(image_bytes) > 2 * 1024 * 1024:
                size_mb = len(image_bytes) / 1024 / 1024
                print(f"⚠️ Image too large ({size_mb:.1f}MB): {img_url}")
                return None
            
            # Reduce image size to save memory (for low RAM systems)
            try:
                img = Image.open(BytesIO(image_bytes)).convert("RGB")
                max_size = (256, 256)  # Reduce to 256x256 max
                img.thumbnail(max_size, Image.LANCZOS)
                out_io = BytesIO()
                img.save(out_io, format="JPEG", quality=85)
                image_bytes = out_io.getvalue()
                out_io.close()
                del img
                print(f"✅ Image resized and downloaded ({len(image_bytes)/1024:.1f}KB)")
            except Exception:
                # If resize fails, continue with original bytes
                print(f"✅ Image downloaded ({len(image_bytes)/1024:.1f}KB)")
            
            # Successful download - reset 429 counter
            consecutive_429_errors = 0
            
            # Ethical crawling: Respect crawl delay
            time.sleep(CRAWL_DELAY)
            return image_bytes
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                print(f"⚠️ Rate limit (429) reached for {img_url} (attempt {attempt}/{max_attempts})")
                
                if attempt < max_attempts:
                    wait_time = 60 * attempt  # 1, 2, 3, 4 minutes
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

def prepare_image_for_model(np_image, device="cpu"):
    # NumPy (H, W, 3) -> PIL.Image
    pil_image = Image.fromarray(np_image)

    # PIL -> Tensor (1, 3, 128, 128)
    tensor_image = transform(pil_image).unsqueeze(0).to(device)

    return tensor_image

def image_bytes_contains_face(image):
    try:
        if model:
            tensor_image = prepare_image_for_model(image, device)
            with torch.no_grad():
                output = model(tensor_image)
                return output
        else:
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

def get_current_category_page_url(last_page):
    """Finds the current category page URL based on the last processed page."""
    if not last_page:
        # If no last page, start with the first category page
        return "https://en.wikipedia.org/wiki/Category:Living_people"
    
    # Try to find the category page where the last page is located
    print(f"Searching category page for last processed page: {last_page}")
    
    category_url = "https://en.wikipedia.org/wiki/Category:Living_people"
    pages_searched = 0
    
    while category_url:
        # Infinite retry loop for network errors
        while True:
            try:
                headers = {
                    "User-Agent": USER_AGENT
                }
                resp = requests.get(category_url, timeout=60, headers=headers)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Ethical crawling: respect crawl delay
                time.sleep(CRAWL_DELAY)
                break  # Successfully loaded, exit retry loop
            except Exception as e:
                print(f"⚠️ Network error while searching category page {category_url}: {e}")
                print("⏳ Waiting 60 seconds and retrying...")
                time.sleep(60)
                # Retry loop continues
        
        # Artikel auf dieser Seite sammeln
        content_div = soup.find("div", {"id": "mw-content-text"})
        if content_div:
            links = content_div.find_all("a", href=lambda x: x and x.startswith("/wiki/") and ":" not in x.split("/wiki/")[1])
            for link in links:
                article_url = urllib.parse.urljoin("https://en.wikipedia.org", link['href'])
                if article_url == last_page:
                    print(f"✓ Last page found on category page: {category_url}")
                    return category_url
        
        # Go to next category page (robust search)
        next_page_link = None
        
        # Method 1: Search for "(next page)" link
        next_links = soup.find_all("a", string=lambda text: text and text.strip() == "(next page)")
        if next_links:
            next_page_link = next_links[0]
        
        # Method 2: Fallback - Search for link with "next" text  
        if not next_page_link:
            all_links = soup.find_all("a", href=True)
            for link in all_links:
                link_text = link.get_text().strip().lower()
                if "next page" in link_text or link_text == "next":
                    next_page_link = link
                    break
        
        # Method 3: Search for pagefrom parameter in URLs
        if not next_page_link:
            pagefrom_links = soup.find_all("a", href=lambda x: x and "pagefrom=" in x)
            if pagefrom_links:
                # Filter links that come after the current page
                for link in pagefrom_links:
                    if "(next page)" in str(link.parent) or "next" in link.get_text().lower():
                        next_page_link = link
                        break
                # If no explicit "next" text, take the first pagefrom link
                if not next_page_link and pagefrom_links:
                    next_page_link = pagefrom_links[0]
        
        if next_page_link and next_page_link.get('href'):
            category_url = urllib.parse.urljoin("https://en.wikipedia.org", next_page_link['href'])
            pages_searched += 1
            print(f"    Continuing search on next category page ({pages_searched})")
        else:
            print(f"    No further category page found after {pages_searched} pages")
            break
    
    print(f"Last page not found after {pages_searched} category pages - continuing from current position")
    return category_url  # Continue from current category page

def wikipedia_url_to_category_link(article_url, category="Living people"):
    """
    Wandelt eine Wikipedia-Artikel-URL in die exakte Kategorie-Unterseiten-URL um.
    """
    # Titel aus URL extrahieren
    title = article_url.split("/")[-1]  # z.B. "Jane_Mary_Doe"
    title = title.replace("_", " ")     # "Jane Mary Doe"

    # Nachname = letztes Wort, alle anderen Wörter = Vorname(n)
    parts = title.split(" ")
    if len(parts) == 1:
        lastname = parts[0]
        firstname = ""
    else:
        lastname = parts[-1]
        firstname = " ".join(parts[:-1])

    # pagefrom = "Nachname, Vorname(n)\nOriginalname"
    pagefrom_raw = f"{lastname}, {firstname}\n{title}"

    # URL-encode
    pagefrom_encoded = urllib.parse.quote_plus(pagefrom_raw)

    # Kategorie-Link zusammenbauen
    category_url = f"https://en.wikipedia.org/w/index.php?title=Category:{category.replace(' ', '_')}&pagefrom={pagefrom_encoded}#mw-pages"
    return category_url

def get_current_category_pages_url(last_pages, num_threads=8):
    """Finds the current category page URL based on the last processed page."""
    if not last_pages:
        # If no last page, start with the first category page
        return "https://en.wikipedia.org/wiki/Category:Living_people"
    
    # Try to find the category page where the last page is located
    print(f"Searching category page for last processed pages: {last_pages}")
    API_URL = "https://en.wikipedia.org/w/api.php"
    CATEGORY = "Living people"
    target_urls = []
    limit = last_pages // 10 if last_pages > 1000 else last_pages

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{CATEGORY}",
        "cmlimit": limit,
        "format": "json"
    }

    USER_AGENT = "FaceSearchBot/1.0 (Educational Research; Contact: github.com/Hawk3388/face-search-2.0)"

    headers = {
        "User-Agent": USER_AGENT
    }

    count = 0
    cmcontinue = None

    while True:
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        response = requests.get(API_URL, params=params, headers=headers)

        # Ensure the response is JSON
        if response.headers.get("content-type", "").startswith("application/json"):
            data = response.json()
        else:
            print("⚠️ Response was not JSON!")
            print("HTTP Status:", response.status_code)
            print(response.text[:500])  # print first 500 characters
            return None

        members = data["query"]["categorymembers"]

        for m in members:
            count += 1
            if count in last_pages:
                print(f"{count}th article: {m['title']}")
                print(f"https://en.wikipedia.org/wiki/{m['title'].replace(' ', '_')}")
                target_urls.append(f"https://en.wikipedia.org/wiki/{m['title'].replace(' ', '_')}")
                if len(target_urls) == num_threads:
                    return target_urls

        if "continue" not in data:
            print("Index too large, category too short.")
            return None

        cmcontinue = data["continue"]["cmcontinue"]

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

def is_page_already_crawled(page_url):
    """Da wir keine Datei lesen - immer False (keine Duplikate erkennen)."""
    return False

def get_article_count(category_url="https://en.wikipedia.org/wiki/Category:Living_people"):
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(category_url, timeout=30, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Suche nach dem Satz mit der Gesamtanzahl
        for p in soup.find_all("p"):
            text = p.get_text()
            match = re.search(r"out of approximately ([\d,]+) total", text)
            if match:
                count_str = match.group(1).replace(",", "")
                return int(count_str)
        # Fallback: Suche im gesamten Text
        text = soup.get_text()
        match = re.search(r"out of approximately ([\d,]+) total", text)
        if match:
            count_str = match.group(1).replace(",", "")
            return int(count_str)
        print("⚠️ Keine Artikelanzahl gefunden.")
        return None
    except Exception as e:
        print(f"Fehler beim Auslesen der Artikelanzahl: {e}")
        return None

def crawl_images_thread(num_threads=8):
    global queue, current_article_data
    processed_count = 0
    entries_saved = 0

    article_count = get_article_count()

    if not article_count:
        print("⚠️ Artikelanzahl konnte nicht ermittelt werden.")
        return

    # Check if we have a resume point
    last_pages = get_last_crawled_pages()
    if last_pages:
        print("Resume: Searching start point based on last article...")
    else:
        article_per_thread = int(article_count / num_threads)
        start_threads = [article_per_thread * i for i in range(num_threads)]
        last_pages = get_current_category_pages_url(start_threads, num_threads)
        print("Starting new crawling session (append-only)...")

    current_category_urls = []
    # Determine current category page URL
    for page in last_pages:
        current_category_urls.append(wikipedia_url_to_category_link(page))

    for i, page in enumerate(current_category_urls):
        process = multiprocessing.Process(target=thread_wiki, args=(page, i))
        process.start()

def thread_wiki(current_category_url, i):
    while current_category_url:
        print(f"\n--- Processing category page: {current_category_url} ---")
        
        # Load articles from current category page
        page_articles, next_category_url = get_articles_from_single_category_page(current_category_url)
        
        if not page_articles:
            print("No articles found on this page - going to next")
            current_category_url = next_category_url
            continue
        
        # Fill queue with articles from this category page
        queue = deque(page_articles)
        
        # If we have a resume point, skip all articles up to last + 1
        if last_page:
            articles_to_skip = []
            found_last_page = False
            
            for article in page_articles:
                if article == last_page:
                    found_last_page = True
                    print(f"🎯 Last article found: {article} - skipping up to here")
                    break
                else:
                    articles_to_skip.append(article)
            
            if found_last_page:
                # Skip all articles up to the last one (inclusive)
                for _ in range(len(articles_to_skip) + 1):  # +1 to skip the last article itself
                    if queue:
                        queue.popleft()
                print(f"⏭️ {len(articles_to_skip) + 1} already processed articles skipped")
                # Reset last_page after successful resume
                last_page = None
        
        remaining_articles = len(queue)
        print(f"Processing {remaining_articles} remaining articles from this category page")
        
        # Process articles from current category page
        while queue:
            url = queue.popleft()
                
            processed_count += 1
            print(f"Crawling page: {url} (#{processed_count})")

            # Clear current article data for this article
            current_article_data.clear()

            try:
                # Infinite retry loop for network errors
                while True:
                    try:
                        headers = {
                            "User-Agent": USER_AGENT
                        }
                        resp = requests.get(url, timeout=60, headers=headers)
                        resp.raise_for_status()
                        
                        # Ethical crawling: respect crawl delay
                        time.sleep(CRAWL_DELAY)
                        break  # Successfully loaded, exit retry loop
                    except Exception as e:
                        print(f"⚠️ Network error while loading page {url}: {e}")
                        print("⏳ Waiting 30 seconds and retrying...")
                        time.sleep(30)
                        # Retry loop continues - never give up!
            except Exception as e:
                print(f"Unexpected error while loading page {url}: {e}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Bilder finden und herunterladen
            images = soup.find_all("img")
            process_images_in_article(images, url)
            
            # Save results for this article (only if data exists)
            if current_article_data:
                entries_saved += len(current_article_data)
                save_database()
                
            # Save last crawled article (for resume function)
            save_last_crawled_pages(url, i)

            # Regular garbage collection every 10 articles to avoid memory leaks
            if processed_count % 10 == 0:
                gc.collect()
                print(f"🧹 Garbage collection performed after {processed_count} articles")
        
        # Go to next category page
        print(f"✓ Category page completed. Processed articles: {processed_count}")
            
        current_category_url = next_category_url
        if not current_category_url:
            print("⚠️ No more category pages found - starting from beginning...")
            # Instead of stopping, start from the beginning
            current_category_url = "https://en.wikipedia.org/wiki/Category:Living_people"
            time.sleep(300)  # Wait 5 minutes before restart

    print(f"Crawling completed. {entries_saved} new entries added to database.")

def crawl_images():
    global queue, current_article_data
    processed_count = 0
    entries_saved = 0
    
    # Check if we have a resume point
    last_page = get_last_crawled_page()
    if last_page:
        print("Resume: Searching start point based on last article...")
    else:
        print("Starting new crawling session (append-only)...")
    
    # Determine current category page URL
    current_category_url = wikipedia_url_to_category_link(last_page)
    print(f"Current category page URL: {current_category_url}")
    
    while current_category_url:
        print(f"\n--- Processing category page: {current_category_url} ---")
        
        # Load articles from current category page
        page_articles, next_category_url = get_articles_from_single_category_page(current_category_url)
        
        if not page_articles:
            print("No articles found on this page - going to next")
            current_category_url = next_category_url
            continue
        
        # Fill queue with articles from this category page
        queue = deque(page_articles)
        
        # If we have a resume point, skip all articles up to last + 1
        if last_page:
            articles_to_skip = []
            found_last_page = False
            
            for article in page_articles:
                if article == last_page:
                    found_last_page = True
                    print(f"🎯 Last article found: {article} - skipping up to here")
                    break
                else:
                    articles_to_skip.append(article)
            
            if found_last_page:
                # Skip all articles up to the last one (inclusive)
                for _ in range(len(articles_to_skip) + 1):  # +1 to skip the last article itself
                    if queue:
                        queue.popleft()
                print(f"⏭️ {len(articles_to_skip) + 1} already processed articles skipped")
                # Reset last_page after successful resume
                last_page = None
        
        remaining_articles = len(queue)
        print(f"Processing {remaining_articles} remaining articles from this category page")
        
        # Process articles from current category page
        while queue:
            url = queue.popleft()
                
            processed_count += 1
            print(f"Crawling page: {url} (#{processed_count})")

            # Clear current article data for this article
            current_article_data.clear()

            try:
                # Infinite retry loop for network errors
                while True:
                    try:
                        headers = {
                            "User-Agent": USER_AGENT
                        }
                        resp = requests.get(url, timeout=60, headers=headers)
                        resp.raise_for_status()
                        
                        # Ethical crawling: respect crawl delay
                        time.sleep(CRAWL_DELAY)
                        break  # Successfully loaded, exit retry loop
                    except Exception as e:
                        print(f"⚠️ Network error while loading page {url}: {e}")
                        print("⏳ Waiting 30 seconds and retrying...")
                        time.sleep(30)
                        # Retry loop continues - never give up!
            except Exception as e:
                print(f"Unexpected error while loading page {url}: {e}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Bilder finden und herunterladen
            images = soup.find_all("img")
            process_images_in_article(images, url)
            
            # Save results for this article (only if data exists)
            if current_article_data:
                entries_saved += len(current_article_data)
                save_database()
                
            # Save last crawled article (for resume function)
            save_last_crawled_page(url)

            # Regular garbage collection every 2 articles to avoid memory leaks (more frequent for low RAM)
            if processed_count % 2 == 0:
                gc.collect()
                print(f"🧹 Garbage collection performed after {processed_count} articles")
        
        # Go to next category page
        print(f"✓ Category page completed. Processed articles: {processed_count}")
            
        current_category_url = next_category_url
        # if not current_category_url:
        #     print("⚠️ No more category pages found - starting from beginning...")
        #     # Instead of stopping, start from the beginning
        #     current_category_url = "https://en.wikipedia.org/wiki/Category:Living_people"
        #     time.sleep(300)  # Wait 5 minutes before restart

    print(f"Crawling completed. {entries_saved} new entries added to database.")

def process_images_in_article(images, url):
    for img in images:
        img_url = img.get("src")
        if not img_url:
            continue
        img_url = urllib.parse.urljoin(url, img_url)
        process_single_image(img_url, url)

def process_single_image(img_url, page_url):
    image_bytes = download_image(img_url)
    if not image_bytes:
        return None
    try:
        image = face_recognition.load_image_file(BytesIO(image_bytes))
        if image_bytes_contains_face(image):
            if not compare_hashes(get_phash(image_bytes)):
                process_image(image, image_bytes, img_url, page_url)
                result = True
            else:
                print(f"Image already present in current article: {img_url}")
                result = False
        else:
            print(f"No face found: {img_url}")
            result = False
        
        # Immediate memory cleanup after processing
        del image, image_bytes
        gc.collect()
        return result
    except Exception as e:
        print(f"Error loading image {img_url}: {e}")
        # Cleanup on error
        if 'image' in locals():
            del image
        if 'image_bytes' in locals():
            del image_bytes
        gc.collect()
    return None

if __name__ == "__main__":
    # Auto-restart configuration
    max_restarts_per_hour = 10
    restart_count = 0
    start_time = time.time()
    restart_delays = [30, 60, 120, 300, 600]  # Progressive delays: 30s, 1m, 2m, 5m, 10m

    print("Starting Living People Crawler (append-only) with auto-restart...")

    while True:
        try:
            # Check if we've exceeded restart limit per hour
            current_time = time.time()
            if current_time - start_time > 3600:  # Reset counter every hour
                restart_count = 0
                start_time = current_time

            if restart_count >= max_restarts_per_hour:
                print(f"\n🚨 Too many restarts ({restart_count}) in the last hour. Stopping crawler.")
                print("Please check for persistent issues.")
                break

            print(f"\n{'='*50}")
            print(f"Starting crawler session #{restart_count + 1}")
            print(f"{'='*50}")

            if cuda:
                print("Using CUDA for image processing.")
                crawl_images_thread()
            else:
                print("CUDA not available - using CPU for image processing.")
                crawl_images()  # adjust max_pages

        except KeyboardInterrupt:
            # In case the signal handler is not triggered
            print("\nInterrupt detected! Saving database...")
            save_database()
            close_database()
            print("Crawler terminated by user.")
            break

        except OSError as e:
            if "swap" in str(e).lower() or "read-error" in str(e).lower():
                print(f"\nSwap error detected: {e}")
                print("Attempting to continue after memory cleanup...")
                gc.collect()
                time.sleep(10)  # Wait for system to recover
                # Try to continue
                try:
                    if cuda:
                        print("Using CUDA for image processing.")
                        crawl_images_thread()
                    else:
                        print("CUDA not available - using CPU for image processing.")
                        crawl_images()
                    continue  # Success, continue main loop
                except Exception as e2:
                    print(f"Failed to recover from swap error: {e2}")
                    save_database()
                    close_database()
            else:
                raise

        except Exception as e:
            print(f"\nUnexpected error: {e}")
            print("Saving database and preparing for restart...")
            save_database()
            close_database()

        # If we reach here, there was an error - prepare for restart
        restart_count += 1

        # Calculate delay with progressive backoff
        delay_index = min(restart_count - 1, len(restart_delays) - 1)
        delay = restart_delays[delay_index]

        print(f"\n🔄 Restart #{restart_count} scheduled in {delay} seconds...")
        print(f"Total restarts in last hour: {restart_count}/{max_restarts_per_hour}")

        # Wait before restart
        time.sleep(delay)

        # Force garbage collection before restart
        gc.collect()
        print("🧹 Memory cleaned up. Restarting crawler...\n")

    print("\nCrawler stopped.")
