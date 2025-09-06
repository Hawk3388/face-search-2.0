import os
import re
from bs4 import BeautifulSoup
import requests

USER_AGENT = "FaceSearchBot/1.0 (Educational Research; Contact: github.com/Hawk3388/face-search-2.0)"

def get_last_crawled_page():
    """Reads the last crawled article from a small separate file."""
    try:
        if os.path.exists("last_crawled_page.txt"):
            with open("last_crawled_page.txt", "r", encoding="utf-8") as f:
                last_page = f.read().strip()
                if last_page:
                    print(f"📄 Last crawled article: {last_page}")
                    return last_page
    except Exception as e:
        print(f"Error reading last page: {e}")
    return None

def get_article_index_in_category(article_url, category="Living people"):
    title_to_find = article_url.split("/")[-1].replace("_", " ")
    API_URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": 10000,  # maximal 10000 pro Anfrage für normale Benutzer
        "format": "json"
    }

    index = 0
    cmcontinue = None

    while True:
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(API_URL, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        members = data["query"]["categorymembers"]

        for m in members:
            index += 1
            if m["title"] == title_to_find:
                return index  # Index des Artikels in der Kategorie

        if "continue" not in data:
            break
        cmcontinue = data["continue"]["cmcontinue"]

    return None  # Artikel nicht gefunden

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
    limit = last_pages // 10 if last_pages[-1] > 1000 else last_pages

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

if __name__ == "__main__":
    article_url = get_last_crawled_page()
    base_index = get_article_index_in_category(article_url)
    article_count = get_article_count()
    to_go = article_count - base_index
    while True:
        threads = input("Threads: ")
        if threads.isdigit() and 1 <= int(threads) <= 25:
            threads = int(threads)
            break
        if not threads:
            threads = 8
            break
        print("Please enter a number between 1 and 25.")
    start_threads = [to_go // threads * i + base_index for i in range(threads)]
    last_pages = get_current_category_pages_url(start_threads, threads)
    for i, page in enumerate(last_pages):
        save_last_crawled_pages(page, i)
    print("Preprocessing for threads is complete.")
