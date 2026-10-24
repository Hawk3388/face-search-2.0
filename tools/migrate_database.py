"""python -m tools.migrate_database [database]; never deletes legacy data."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from face_search.storage import ensure_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", nargs="?", type=Path, default=Path("face_embeddings.db"))
    args = parser.parse_args()
    if not args.database.exists():
        parser.error("Database does not exist")
    backup = ensure_schema(args.database)
    print(f"Migration complete. Verified backup: {backup}" if backup else "Schema already up to date; no changes.")


if __name__ == "__main__":
    main()
