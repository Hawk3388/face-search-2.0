"""MediaWiki discovery without depending on skin, lazy-loading or HTML pagination."""

import threading
import time
import ssl
from email.utils import parsedate_to_datetime
from urllib.parse import quote, urlsplit

import requests
from requests.adapters import HTTPAdapter


class SystemTrustAdapter(HTTPAdapter):
    """Use OS trust roots as well as Requests' CA bundle, including on Windows."""

    def __init__(self, context):
        self.context = context
        super().__init__()

    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        host, pool = super().build_connection_pool_key_attributes(request, verify, cert)
        if verify is True:
            pool["ssl_context"] = self.context
        # Explicit REQUESTS_CA_BUNDLE / client-certificate settings remain authoritative.
        return host, pool


class Stopped(Exception):
    pass


def retry_delay(value, attempt):
    try:
        return max(0, float(value))
    except (ValueError, TypeError):
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return min(300, 2 ** (attempt + 1))


class HttpClient:
    """One reusable Session per worker; shared host pacing and Retry-After cooldown."""

    def __init__(self, user_agent, stop=None, interval=0.5):
        self.user_agent = user_agent
        self.stop = stop or threading.Event()
        self.interval = interval
        self.local = threading.local()
        self.lock = threading.Lock()
        self.next_request = {}
        self.sessions = []
        self.tls_context = ssl.create_default_context()
        self.tls_context.load_verify_locations(requests.certs.where())

    def wait(self, seconds):
        if self.stop.wait(max(0, seconds)):
            raise Stopped()

    def session(self):
        if not hasattr(self.local, "session"):
            session = requests.Session()
            session.mount("https://", SystemTrustAdapter(self.tls_context))
            session.headers["User-Agent"] = self.user_agent
            self.local.session = session
            with self.lock:
                self.sessions.append(session)
        return self.local.session

    def pace(self, host):
        while True:
            if self.stop.is_set():
                raise Stopped()
            with self.lock:
                remaining = self.next_request.get(host, 0) - time.monotonic()
                if remaining <= 0:
                    self.next_request[host] = time.monotonic() + self.interval
                    return
            self.wait(remaining)

    def cooldown(self, url, delay):
        host = urlsplit(url).netloc
        with self.lock:
            self.next_request[host] = max(self.next_request.get(host, 0), time.monotonic() + delay)

    def get(self, url, params=None, max_bytes=10 * 1024 * 1024, headers=None, expected_status=None):
        for attempt in range(3):
            self.pace(urlsplit(url).netloc)
            try:
                with self.session().get(url, params=params, headers=headers, timeout=(10, 30), stream=True) as response:
                    if response.status_code == 429 or response.status_code >= 500:
                        self.cooldown(url, retry_delay(response.headers.get("Retry-After"), attempt))
                    response.raise_for_status()
                    if expected_status is not None and response.status_code != expected_status:
                        raise ValueError(f"Expected HTTP {expected_status}, received {response.status_code}")
                    if int(response.headers.get("Content-Length", 0)) > max_bytes:
                        raise ValueError("Download exceeds byte limit")
                    chunks, length = [], 0
                    for chunk in response.iter_content(64 * 1024):
                        if self.stop.is_set():
                            raise Stopped()
                        length += len(chunk)
                        if length > max_bytes:
                            raise ValueError("Download exceeds byte limit")
                        chunks.append(chunk)
                    return b"".join(chunks)
            except requests.exceptions.SSLError:
                # A broken trust configuration cannot be repaired by repeated requests.
                raise
            except requests.RequestException as exc:
                status = exc.response.status_code if exc.response is not None else None
                if attempt == 2 or (status and 400 <= status < 500 and status != 429):
                    raise
                self.wait(2 ** (attempt + 1))

    def close(self):
        for session in self.sessions:
            session.close()


class WikiApi:
    def __init__(self, http, wiki="https://en.wikipedia.org", width=960):
        self.http = http
        self.wiki = wiki.rstrip("/")
        self.endpoint = self.wiki + "/w/api.php"
        self.width = width

    def query(self, **params):
        import json

        for attempt in range(3):
            data = json.loads(
                self.http.get(
                    self.endpoint,
                    params={
                        "action": "query",
                        "format": "json",
                        "formatversion": 2,
                        "maxlag": 5,
                        **params,
                    },
                )
            )
            error = data.get("error")
            if not error:
                if "query" not in data and "batchcomplete" not in data:
                    raise ValueError("MediaWiki response has no query result")
                data.setdefault("query", {})
                return data
            if error.get("code") not in ("maxlag", "ratelimited", "readonly") or attempt == 2:
                raise ValueError(f"MediaWiki: {error}")
            self.http.cooldown(self.endpoint, max(5, retry_delay(error.get("lag"), attempt)))
        raise RuntimeError("MediaWiki retry exhausted")

    def category_batch(self, category, cursor=None):
        data = self.query(
            list="categorymembers", cmtitle=category, cmnamespace=0, cmtype="page", cmlimit=50, **(cursor or {})
        )
        return data["query"].get("categorymembers", []), data.get("continue")

    def page_url(self, title):
        return self.wiki + "/wiki/" + quote(title.replace(" ", "_"), safe="()")

    def larger_preview(self, title):
        data = self.query(titles=title, prop="imageinfo", iiprop="url", iiurlwidth=self.width * 2)
        for page in data["query"].get("pages", []):
            for info in page.get("imageinfo", []):
                return info.get("thumburl", info.get("url"))
        raise ValueError("Larger image preview unavailable")

    def images(self, page_id):
        """Generator includes template/gallery files and Commons imageinfo.

        Commons files can carry 'missing' on the local wiki despite valid imageinfo.
        Never discard those. Honour *all* continuation keys, not just gimcontinue.
        """
        cursor, seen, images = {}, set(), {}
        while True:
            data = self.query(
                pageids=page_id,
                generator="images",
                gimlimit=50,
                prop="imageinfo",
                iiprop="url|sha1|timestamp|mime|size",
                iiurlwidth=self.width,
                redirects=1,
                **cursor,
            )
            for page in data["query"].get("pages", []):
                infos = page.get("imageinfo", [])
                if not infos:
                    # Truly unavailable/deleted files have no bytes to fetch.
                    continue
                info = infos[0]
                if not info.get("mime", "").startswith("image/"):
                    continue
                url = info.get("url")
                if not url or not info.get("sha1"):
                    raise ValueError("Incomplete image metadata; article remains pending")
                images[url] = {**info, "title": page["title"], "download_url": info.get("thumburl", url)}
            cursor = data.get("continue")
            if not cursor:
                return list(images.values())
            marker = repr(sorted(cursor.items()))
            if marker in seen:
                raise ValueError("Repeated MediaWiki continuation token")
            seen.add(marker)
