"""Compare two images using your own checkpoint, without hard-coded paths."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import numpy as np
from face_search.config import Settings
from face_search.encoding import FaceEncoder, load_image
from face_search.evaluation import pair_distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("model2/siamese_model.pth"))
    args = parser.parse_args()
    encoder = FaceEncoder(Settings(Path("unused"), encoder="siamese", model_path=str(args.checkpoint)))
    vectors = []
    for path in (args.first, args.second):
        _, faces = encoder.extract(load_image(path.read_bytes()))
        if len(faces) != 1:
            parser.error(f"{path}: expected one face, found {len(faces)}")
        vectors.append(faces[0])
    distance = pair_distances(np.array(vectors[:1]), np.array(vectors[1:]), "cosine")[0]
    print(f"Model: {encoder.model_id}; cosine distance: {distance:.6f}")
    print("Calibrate a threshold on held-out identities before interpreting this as a match.")


if __name__ == "__main__":
    main()
