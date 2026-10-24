"""Repair NUL bytes only after creating a verified backup."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path
from face_search.storage import backup_file


def main():
    path = Path("face_embeddings.json")
    print(f"Backup: {backup_file(path)}")
    data = path.read_bytes().replace(b"\x00", b"")
    temporary = path.with_suffix(".json.repaired")
    with temporary.open("xb") as target:
        target.write(data)
    temporary.replace(path)


if __name__ == "__main__":
    main()
