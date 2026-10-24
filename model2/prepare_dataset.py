"""Create a NEW dataset of single-face crops without modifying source images."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import face_recognition

from face_search.encoding import load_image
from model2.data import load_people


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("model2/dataset"))
    parser.add_argument("--output", type=Path, default=Path("model2/prepared"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seen, skipped, saved = {}, [], 0
    for person, paths in load_people(args.input).items():
        for path in paths:
            try:
                image = load_image(path.read_bytes())
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest in seen:
                    skipped.append({"file": str(path), "reason": f"Duplicate of {seen[digest]}"})
                    continue
                seen[digest] = str(path)
                locations = face_recognition.face_locations(image, model="hog")
                if len(locations) != 1:
                    skipped.append({"file": str(path), "reason": f"Expected one face, found {len(locations)}"})
                    continue
                top, right, bottom, left = locations[0]
                target = args.output / person
                target.mkdir(exist_ok=True)
                Image.fromarray(image[top:bottom, left:right]).save(target / (digest + ".png"))
                saved += 1
            except (ValueError, OSError) as exc:
                skipped.append({"file": str(path), "reason": str(exc)})
    (args.output / "preprocessing.json").write_text(
        json.dumps(
            {
                "preprocessing": "face-crop-rgb224-imagenet-v1",
                "source": str(args.input.resolve()),
                "saved": saved,
                "skipped": skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {saved} face crops; skipped {len(skipped)}. Source images unchanged.")


if __name__ == "__main__":
    main()
