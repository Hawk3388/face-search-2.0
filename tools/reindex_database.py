"""Re-encode source images into a NEW database; the source remains read-only."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from contextlib import closing
from pathlib import Path
import sqlite3

import requests

from face_search.config import Settings
from face_search.download import download_image
from face_search.encoding import FaceEncoder, load_image
from face_search.storage import read_records, ensure_schema, save_article


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.source.resolve() == args.output.resolve():
        parser.error("Output must be a new database path")
    settings = Settings(database=args.output, backend="sqlite", encoder="siamese", model_path=str(args.checkpoint))
    encoder = FaceEncoder(settings)
    images = {}
    for entry in read_records(args.source, "json" if args.source.suffix == ".json" else "sqlite"):
        url = entry["image_url"]
        record = images.setdefault(url, {"pages": set(), "phash": entry.get("phash")})
        record["pages"].update(entry.get("page_urls") or ([entry["page_url"]] if entry.get("page_url") else []))
    ensure_schema(args.output)
    # Stop on the first failed download/inference; never silently claim a complete index.
    with closing(sqlite3.connect(args.output)) as conn, requests.Session() as session:
        session.headers["User-Agent"] = "FaceSearchBot/2.0 (github.com/Hawk3388/face-search-2.0)"
        for number, (url, record) in enumerate(images.items(), 1):
            image = load_image(download_image(url, session))
            boxes, vectors = encoder.extract(image)
            entries = [
                {
                    "image_url": url,
                    "page_urls": sorted(record["pages"]),
                    "phash": record["phash"],
                    "embedding": vector,
                    "bbox": box,
                    "model_id": encoder.model_id,
                    "dimension": encoder.dimension,
                    "preprocessing": encoder.preprocessing,
                }
                for box, vector in zip(boxes, vectors)
            ]
            save_article(conn, entries)
            print(f"{number}/{len(images)}: {len(entries)} faces")
    print(f"Reindex complete: {args.output}; source unchanged")


if __name__ == "__main__":
    main()
