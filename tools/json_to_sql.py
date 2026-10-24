"""Import every face from JSON, backing up an existing destination first."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
from face_search.storage import backup_database, ensure_schema, read_json, save_article


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("face_embeddings.json"))
    parser.add_argument("--output", type=Path, default=Path("face_embeddings.db"))
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Source and output must differ")
    if args.output.exists():
        print(f"Verified backup: {backup_database(args.output)}")
    ensure_schema(args.output)
    with closing(sqlite3.connect(args.output)) as conn:
        batch = []
        count = 0
        for entry in read_json(args.source):
            batch.append(entry)
            if len(batch) == 1000:
                save_article(conn, batch)
                count += len(batch)
                batch.clear()
                print(f"Imported {count} records")
        save_article(conn, batch)
        print(f"Complete: {count + len(batch)} records; source unchanged")


if __name__ == "__main__":
    main()
