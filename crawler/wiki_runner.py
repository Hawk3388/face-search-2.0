"""Shared SQLite crawler used by all three Wikipedia entry points."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlsplit

import requests
from PIL import Image

from face_search.config import ROOT, Settings
from face_search.encoding import FaceEncoder, load_image
from .wiki_api import HttpClient, Stopped, WikiApi
from .wiki_store import CrawlStore

LOG = logging.getLogger(__name__)


def profile_id(encoder, settings, width):
    config = [
        encoder.model_id,
        encoder.preprocessing,
        settings.num_jitters,
        settings.max_faces,
        settings.max_pixels,
        settings.max_upload_bytes,
        width,
        "wiki-preview-v1",
    ]
    return hashlib.sha256(json.dumps(config).encode()).hexdigest()


class Crawler:
    def __init__(self, api, store, encoder, settings, workers=3, stop=None):
        self.api, self.store, self.encoder, self.settings = api, store, encoder, settings
        self.workers = workers
        self.stop = stop or api.http.stop
        self.profile = profile_id(encoder, settings, api.width)
        self.processed = 0

    def analyze(self, info, data):
        import imagehash

        image = load_image(data, self.settings.max_upload_bytes, self.settings.max_pixels)
        locations, embeddings = self.encoder.extract(image)
        # Small faces / no detections get another chance at a higher resolution.
        small = any(min(b - t, r - l) < 80 for t, r, b, l in locations)
        if (not locations or small) and info["download_url"] != info["url"]:
            original_fits = (
                info.get("mime") in ("image/jpeg", "image/png", "image/webp", "image/gif", "image/tiff", "image/bmp")
                and info.get("width", 0) * info.get("height", 0) <= self.settings.max_pixels
                and info.get("size", 0) <= self.settings.max_upload_bytes
            )
            larger = info["url"] if original_fits else self.api.larger_preview(info["title"])
            if larger != info["download_url"]:
                larger_data = self.api.http.get(larger, max_bytes=self.settings.max_upload_bytes)
                larger_image = load_image(larger_data, self.settings.max_upload_bytes, self.settings.max_pixels)
                larger_locations, larger_embeddings = self.encoder.extract(larger_image)
                # Keep the initial detection if the second pass detects fewer faces.
                if len(larger_locations) >= len(locations):
                    image, locations, embeddings = larger_image, larger_locations, larger_embeddings
        if len(locations) != len(embeddings):
            raise ValueError("Encoder did not return an embedding for every face")
        phash = str(imagehash.phash(Image.fromarray(image))) if locations else None
        entries, quality = [], {"width": int(image.shape[1]), "height": int(image.shape[0]), "faces": []}
        for box, embedding in zip(locations, embeddings):
            top, right, bottom, left = map(int, box)
            crop = image[top:bottom, left:right].astype("float32")
            quality["faces"].append(
                {
                    "bbox": list(map(int, box)),
                    "face_pixels": (bottom - top) * (right - left),
                    "contrast_std": round(float(crop.std()), 3),
                }
            )
            entries.append(
                {
                    "embedding": embedding,
                    "bbox": list(map(int, box)),
                    "phash": phash,
                    "model_id": self.encoder.model_id,
                    "dimension": self.encoder.dimension,
                    "preprocessing": self.encoder.preprocessing,
                }
            )
        return entries, quality

    def process_images(self, pool):
        """Bounded prefetch, a single encoder and exactly one database writer."""
        while not self.stop.is_set():
            self.store.maintain()
            jobs = self.store.due_images(self.profile, self.workers * 2)
            if not jobs:
                return
            futures = [
                (
                    job,
                    json.loads(job["metadata"]),
                    pool.submit(
                        self.api.http.get,
                        json.loads(job["metadata"])["download_url"],
                        max_bytes=self.settings.max_upload_bytes,
                    ),
                )
                for job in jobs
            ]
            for job, info, future in futures:
                if self.stop.is_set():
                    raise Stopped()
                try:
                    data = future.result()
                    entries, quality = self.analyze(info, data)
                except requests.exceptions.SSLError:
                    raise
                except (requests.RequestException, ValueError, OSError) as exc:
                    self.store.fail_image(job, exc)
                    LOG.warning("Image queued for retry: %s: %s", job["url"], exc)
                    continue
                # Storage errors must stop the run, never turn into a skipped image.
                self.store.complete_image(job, entries, quality)
                self.processed += 1
            LOG.info("Images: %s", self.store.counts(self.profile))

    def run(self, category, once=False, max_articles=0, refresh_seconds=86400, pages=None):
        scope = json.dumps([self.api.wiki, category, pages or [], self.profile])
        count, started = 0, time.monotonic()
        if pages:
            data = self.api.query(titles="|".join(pages), redirects=1, prop="info")
            members = data["query"].get("pages", [])
            if any("pageid" not in p for p in members):
                raise ValueError("At least one requested article does not exist")
            self.store.enqueue_batch(scope, members, None, refresh_seconds)
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="wiki-download") as pool:
            while not self.stop.is_set():
                self.process_images(pool)
                self.store.maintain()
                for article in self.store.due_articles(scope):
                    if self.stop.is_set():
                        raise Stopped()
                    try:
                        images = self.api.images(article["page_id"])
                    except requests.exceptions.SSLError:
                        raise
                    except (requests.RequestException, ValueError) as exc:
                        self.store.fail_article(scope, article, exc)
                        LOG.warning("Article queued for retry: %s: %s", article["title"], exc)
                        continue
                    self.store.discover(scope, article, self.api.page_url(article["title"]), images, self.profile)
                    count += 1
                    LOG.info("%s: %d image files discovered", article["title"], len(images))
                    self.process_images(pool)
                    if max_articles and count >= max_articles:
                        return
                scan = self.store.scan(scope)
                if not scan["finished"]:
                    cursor = json.loads(scan["cursor"]) if scan["cursor"] else None
                    members, next_cursor = self.api.category_batch(category, cursor)
                    if cursor and cursor == next_cursor:
                        raise ValueError("Category API returned the same continuation token")
                    self.store.enqueue_batch(scope, members, next_cursor, refresh_seconds)
                elif once or pages:
                    break
                elif time.time() >= scan["next_scan"]:
                    self.store.restart_scan(scope)
                else:
                    # Failed jobs are checked every 30s, independent of the next full pass.
                    self.api.http.wait(min(30, max(0, scan["next_scan"] - time.time())))
        LOG.info(
            "Run finished: %d articles, %d images analyzed in %.1fs; queue: %s",
            count,
            self.processed,
            time.monotonic() - started,
            self.store.counts(self.profile),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resumable Wikipedia face crawler (SQLite)")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--wiki", default="https://en.wikipedia.org")
    parser.add_argument("--category", default="Category:Living people")
    parser.add_argument("--page", action="append", help="Article title; repeat for up to 50 titles")
    parser.add_argument("--once", action="store_true", help="One discovery pass; delayed retries remain queued")
    parser.add_argument("--max-articles", type=int, default=0, help="Stop after N successful article discoveries")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    settings = Settings.from_env("sqlite")
    workers = args.workers if args.workers is not None else int(os.getenv("CRAWLER_DOWNLOAD_WORKERS", "3"))
    width = int(os.getenv("CRAWLER_PREVIEW_WIDTH", "960"))
    interval = float(os.getenv("CRAWLER_REQUEST_INTERVAL", "0.5"))
    refresh = float(os.getenv("CRAWLER_REFRESH_HOURS", "24")) * 3600
    backup_hours = float(os.getenv("CRAWLER_BACKUP_HOURS", "24"))
    free_mb = int(os.getenv("CRAWLER_MIN_FREE_MB", "512"))
    if not 1 <= workers <= 8 or width < 320 or interval < 0.1 or refresh < 300 or backup_hours <= 0 or free_mb < 0:
        parser.error("Invalid crawler limits (workers 1..8, width >=320, interval >=0.1, refresh >=5min)")
    wiki = urlsplit(args.wiki)
    if wiki.scheme != "https" or not (wiki.hostname or "").endswith(".wikipedia.org") or wiki.path not in ("", "/"):
        parser.error("--wiki must be an https://LANG.wikipedia.org origin")
    if args.max_articles < 0 or (args.page and len(args.page) > 50):
        parser.error("Use nonnegative --max-articles and at most 50 --page titles")
    settings = replace(settings, num_jitters=int(os.getenv("CRAWLER_NUM_JITTERS", str(settings.num_jitters))))
    database = args.database or Path(os.getenv("CRAWLER_DATABASE_PATH", str(ROOT / "face_embeddings.db")))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = threading.Event()
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda *_: stop.set())
    http = HttpClient(
        os.getenv("USER_AGENT", "FaceSearchBot/2.0 (https://github.com/Hawk3388/face-search-2.0)"), stop, interval
    )
    store = None
    try:
        encoder = FaceEncoder(settings)
        store = CrawlStore(database, free_mb * 1024 * 1024, backup_hours)
        LOG.info(
            "SQLite: %s; %d download workers; preview %dpx; num_jitters=%d",
            database,
            workers,
            width,
            settings.num_jitters,
        )
        crawler = Crawler(WikiApi(http, args.wiki, width), store, encoder, settings, workers, stop)
        crawler.run(args.category, args.once, args.max_articles, refresh, args.page)
    except (Stopped, KeyboardInterrupt):
        LOG.info("Stopped. Committed results and pending work are retained.")
    finally:
        http.close()
        if store:
            store.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
