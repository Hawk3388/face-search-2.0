"""Compare identity retrieval and verification using held-out person folders."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
from pathlib import Path
import time

import numpy as np

from face_search.config import Settings
from face_search.encoding import FaceEncoder, load_image
from face_search.evaluation import calibrate_threshold, pair_distances, retrieval_metrics, verification_metrics
from model2.data import create_pairs, load_people, split_people


def evaluate(vectors, people, metric, threshold=None, target_far=0.01):
    valid = {name: [str(p) for p in paths if str(p) in vectors] for name, paths in people.items()}
    valid = {name: paths for name, paths in valid.items() if len(paths) >= 2}
    pairs = create_pairs(valid, 50, seed=314)
    distances = pair_distances(
        np.stack([vectors[str(a)] for a, _, _ in pairs]), np.stack([vectors[str(b)] for _, b, _ in pairs]), metric
    )
    labels = np.array([label for _, _, label in pairs])
    threshold = calibrate_threshold(distances, labels, target_far) if threshold is None else threshold
    matrix, identities = [], []
    for identity, paths in valid.items():
        matrix.extend(vectors[p] for p in paths)
        identities.extend([identity] * len(paths))
    return dict(
        verification_metrics(distances, labels, threshold), **retrieval_metrics(np.stack(matrix), identities, metric)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", type=Path, help="Training split.json; uses only validation and test identities")
    parser.add_argument("--target-far", type=float, default=0.01)
    parser.add_argument("--max-people", type=int, default=0, help="Optional total cap for a reproducible pilot evaluation")
    parser.add_argument("--images-per-person", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new report path")
    settings = Settings(
        database=Path("unused"),
        encoder="siamese" if args.checkpoint else "dlib",
        model_path=str(args.checkpoint) if args.checkpoint else "",
    )
    encoder = FaceEncoder(settings)
    people = load_people(args.dataset)
    if args.split:
        split = json.loads(args.split.read_text())
        calibration = {n: people[n] for n in split["validation"] if n in people}
        testing = {n: people[n] for n in split["test"] if n in people}
        if set(calibration) & set(testing) or set(split["train"]) & (set(calibration) | set(testing)):
            parser.error("Train, validation and test identities must be disjoint")
    elif args.checkpoint:
        parser.error("For a trained checkpoint supply --split to avoid evaluating on training identities")
    else:
        calibration, testing = split_people(people, 0.5)
    if args.max_people and args.max_people < 4:
        parser.error("max-people must be zero or at least four")
    if args.images_per_person and args.images_per_person < 2:
        parser.error("images-per-person must be zero or at least two")
    rng = random.Random(args.seed)
    def subset(group):
        names = sorted(group)
        if args.max_people:
            names = sorted(rng.sample(names, min(len(names), args.max_people // 2)))
        return {name: sorted(rng.sample(group[name], min(len(group[name]), args.images_per_person)))
                if args.images_per_person else group[name] for name in names}
    calibration, testing = subset(calibration), subset(testing)
    vectors, skipped = {}, []
    start = time.perf_counter()
    prepared = (args.dataset / "preprocessing.json").exists()
    for paths in {**calibration, **testing}.values():
        for path in paths:
            try:
                image = load_image(path.read_bytes())
                if prepared:
                    encoded = encoder.encode_locations(image, [(0, image.shape[1], image.shape[0], 0)])
                else:
                    _, encoded = encoder.extract(image)
                if len(encoded) != 1:
                    raise ValueError(f"Expected one face, found {len(encoded)}")
                vectors[str(path)] = encoded[0]
            except (ValueError, OSError) as exc:
                skipped.append({"path": str(path), "reason": str(exc)})
    report = {
        "model_id": encoder.model_id,
        "target_far": args.target_far,
        "skipped": skipped,
        "images_encoded": len(vectors),
        "encoding_seconds": time.perf_counter() - start,
        "calibration_people": sorted(calibration),
        "test_people": sorted(testing),
        "metrics": {},
        "sampling": {"seed": args.seed, "max_people": args.max_people, "images_per_person": args.images_per_person},
    }
    for metric in ("euclidean", "cosine"):
        calibration_result = evaluate(vectors, calibration, metric, target_far=args.target_far)
        report["metrics"][metric] = {
            "calibration": calibration_result,
            "test": evaluate(vectors, testing, metric, threshold=calibration_result["threshold"]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
