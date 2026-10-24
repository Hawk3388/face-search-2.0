"""Bounded downloads and retries shared by crawler and reindexing tools."""

import time
import requests


def download_image(url, session=None, max_bytes=10 * 1024 * 1024, attempts=3):
    client = session or requests
    for attempt in range(attempts):
        try:
            with client.get(url, timeout=(10, 30), stream=True) as response:
                response.raise_for_status()
                if int(response.headers.get("Content-Length", 0)) > max_bytes:
                    raise ValueError("Remote image exceeds download limit")
                chunks, length = [], 0
                for chunk in response.iter_content(64 * 1024):
                    length += len(chunk)
                    if length > max_bytes:
                        raise ValueError("Remote image exceeds download limit")
                    chunks.append(chunk)
                return b"".join(chunks)
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            if attempt == attempts - 1 or (status and 400 <= status < 500 and status != 429):
                raise
            retry_after = exc.response.headers.get("Retry-After", "") if exc.response is not None else ""
            delay = min(float(retry_after), 60) if retry_after.isdigit() else 2 ** (attempt + 1)
            time.sleep(delay)
