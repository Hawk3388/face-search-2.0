from pathlib import Path
import random


def load_people(path):
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    people = {}
    for person in sorted(root.iterdir()):
        if person.is_dir():
            images = sorted(p for p in person.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
            if len(images) >= 2:
                people[person.name] = images
    return people


def split_people(people, val_split=0.2, seed=42):
    if len(people) < 4:
        raise ValueError("At least four people with two images each are required for train/validation pairs")
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")
    names = sorted(people)
    random.Random(seed).shuffle(names)
    n_val = min(len(names) - 2, max(2, int(len(names) * val_split)))
    return ({n: people[n] for n in names[n_val:]}, {n: people[n] for n in names[:n_val]})


def create_pairs(people, pairs_per_person=20, seed=42):
    if len(people) < 2 or any(len(images) < 2 for images in people.values()):
        raise ValueError("Pair generation needs two people and two images per person")
    if pairs_per_person < 1:
        raise ValueError("pairs_per_person must be positive")
    rng = random.Random(seed)
    pairs = []
    names = sorted(people)
    for name in names:
        for _ in range(pairs_per_person):
            first, second = rng.sample(people[name], 2)
            pairs.append((first, second, 1))
            other = rng.choice([n for n in names if n != name])
            pairs.append((rng.choice(people[name]), rng.choice(people[other]), 0))
    rng.shuffle(pairs)
    return pairs
