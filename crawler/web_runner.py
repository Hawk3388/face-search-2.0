"""URL and Common Crawl sources feeding the same SQLite image pipeline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urldefrag, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
import requests

from face_search.config import ROOT, Settings
from face_search.encoding import FaceEncoder
from .wiki_api import HttpClient, Stopped, WikiApi
from .wiki_runner import Crawler
from .wiki_store import CrawlStore

LOG = logging.getLogger(__name__)


def http_url(base, value):
    url = urldefrag(urljoin(base, value or ""))[0]
    return url if value and urlsplit(url).scheme in ("http", "https") else None


def best_srcset(value):
    candidates = []
    for item in (value or "").split(","):
        fields = item.strip().split()
        if fields:
            try:
                score = float(fields[1][:-1]) if len(fields) > 1 else 1
            except ValueError:
                score = 1
            candidates.append((score, fields[0]))
    return max(candidates, default=(0, None))[1]


def discover_html(html, page_url):
    soup = BeautifulSoup(html, "html.parser")
    base_tag = soup.find("base", href=True)
    base = urljoin(page_url, base_tag["href"]) if base_tag else page_url
    images = set()
    for tag in soup.find_all(["img", "source"]):
        # Keep lazy originals even when src is just a placeholder or tracking pixel.
        values = [tag.get(key) for key in ("src", "data-src", "data-original", "data-lazy-src")]
        values += [best_srcset(tag.get(key)) for key in ("srcset", "data-srcset")]
        for value in values:
            url = http_url(base, value)
            if url:
                images.add(url)
    for tag in soup.find_all("meta"):
        if tag.get("property", tag.get("name", "")).lower() in ("og:image", "og:image:url", "twitter:image"):
            url = http_url(base, tag.get("content"))
            if url:
                images.add(url)
    for tag in soup.find_all("link", href=True):
        if "image_src" in tag.get("rel", []):
            url = http_url(base, tag["href"])
            if url:
                images.add(url)
    links = {
        url
        for tag in soup.find_all("a", href=True)
        if (url := http_url(base, tag["href"])) and urlsplit(url).netloc == urlsplit(page_url).netloc
    }
    return sorted(images), sorted(links)


class WebSource:
    def __init__(self, http):
        self.http = http
        self.robots = {}

    def allowed(self, url):
        parsed = urlsplit(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self.robots:
            parser = RobotFileParser(origin + "/robots.txt")
            try:
                data = self.http.get(parser.url, max_bytes=1024 * 1024)
                parser.parse(data.decode("utf-8", errors="replace").splitlines())
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 404:
                    parser.parse([])
                elif status in (401, 403):
                    parser.disallow_all = True
                else:
                    raise
            self.robots[origin] = parser
        return self.robots[origin].can_fetch(self.http.user_agent, url)

    def page(self, url):
        if not self.allowed(url):
            raise ValueError("robots.txt disallows this page")
        return self.http.get(url)


class CommonCrawlSource(WebSource):
    def __init__(self, http, index):
        super().__init__(http)
        self.index = index

    def page(self, url):
        from warcio.archiveiterator import ArchiveIterator

        data = self.http.get(
            f"https://index.commoncrawl.org/{self.index}-index",
            params={"url": url, "output": "json", "filter": "status:200"},
        )
        records = [json.loads(line) for line in data.splitlines() if line.strip()]
        for record in reversed(records):
            offset, length = int(record["offset"]), int(record["length"])
            filename = record["filename"]
            if not filename.startswith("crawl-data/") or ".." in filename:
                raise ValueError("Invalid Common Crawl archive path")
            archive = self.http.get(
                "https://data.commoncrawl.org/" + filename,
                headers={"Range": f"bytes={offset}-{offset + length - 1}"},
                expected_status=206,
            )
            for item in ArchiveIterator(BytesIO(archive)):
                if item.rec_type == "response":
                    page = item.content_stream().read(10 * 1024 * 1024 + 1)
                    if len(page) > 10 * 1024 * 1024:
                        raise ValueError("Archived page too large")
                    return page
        raise ValueError("No archived page available")


def member(url):
    return {"pageid": int.from_bytes(hashlib.sha256(url.encode()).digest()[:8], "big") & ((1 << 63) - 1), "title": url}


def run_web(crawler, source, start_url, max_pages):
    scope = json.dumps(["web", start_url, getattr(source, "index", "live"), crawler.profile])
    # Previously queued failures survive restarting; the seed is checked again.
    crawler.store.enqueue_batch(scope, [member(start_url)], None, 86400)
    count = 0
    visited = set()
    with ThreadPoolExecutor(max_workers=crawler.workers) as pool:
        crawler.process_images(pool)
        while count < max_pages and not crawler.stop.is_set():
            articles = crawler.store.due_articles(scope, min(50, max_pages - count))
            if not articles:
                break
            for article in articles:
                url = article["title"]
                try:
                    html = source.page(url)
                    images, links = discover_html(html, url)
                    # A daily refresh also retries previously negative detections on changed web images.
                    infos = [
                        {"url": image, "download_url": image, "sha1": f"web-day-{int(time.time() // 86400)}"}
                        for image in images
                        if source.allowed(image)
                    ]
                except requests.exceptions.SSLError:
                    raise
                except (requests.RequestException, ValueError) as exc:
                    crawler.store.fail_article(scope, article, exc)
                    LOG.warning("Page queued for retry: %s: %s", url, exc)
                    count += 1
                    continue
                crawler.store.discover(scope, article, url, infos, crawler.profile)
                visited.add(url)
                crawler.store.enqueue_batch(
                    scope, [member(link) for link in links if link not in visited and link != url], None, 86400
                )
                LOG.info("%s: %d images discovered", url, len(infos))
                crawler.process_images(pool)
                count += 1


def main(argv=None, archive=False):
    parser = argparse.ArgumentParser(description="SQLite URL crawler" + (" using Common Crawl" if archive else ""))
    parser.add_argument("url", nargs="?")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--index", default=os.getenv("COMMON_CRAWL_INDEX", "CC-MAIN-2025-30"))
    args = parser.parse_args(argv)
    settings = Settings.from_env("sqlite")
    settings = replace(settings, num_jitters=int(os.getenv("CRAWLER_NUM_JITTERS", str(settings.num_jitters))))
    workers = int(os.getenv("CRAWLER_DOWNLOAD_WORKERS", "3"))
    interval = float(os.getenv("CRAWLER_REQUEST_INTERVAL", "0.5"))
    backup_hours = float(os.getenv("CRAWLER_BACKUP_HOURS", "24"))
    free_mb = int(os.getenv("CRAWLER_MIN_FREE_MB", "512"))
    if not 1 <= workers <= 8 or interval < 0.1 or backup_hours <= 0 or free_mb < 0:
        parser.error("Invalid crawler worker, request interval, backup or free-space settings")
    url = args.url or input("Start URL: ").strip()
    if not http_url(url, url) or args.max_pages < 1:
        parser.error("Provide an HTTP(S) URL and positive --max-pages")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    http = HttpClient(
        os.getenv("USER_AGENT", "FaceSearchBot/2.0 (https://github.com/Hawk3388/face-search-2.0)"), stop, interval
    )
    store = None
    try:
        store = CrawlStore(
            args.database or Path(os.getenv("CRAWLER_DATABASE_PATH", str(ROOT / "face_embeddings.db"))),
            free_mb * 1024 * 1024,
            backup_hours,
        )
        crawler = Crawler(WikiApi(http), store, FaceEncoder(settings), settings, workers=workers, stop=stop)
        source = CommonCrawlSource(http, args.index) if archive else WebSource(http)
        run_web(crawler, source, url, args.max_pages)
    except (Stopped, KeyboardInterrupt):
        LOG.info("Stopped; pending work remains in SQLite")
    finally:
        http.close()
        if store:
            store.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
