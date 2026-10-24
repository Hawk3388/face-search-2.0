"""Identity verification metrics; calibration and held-out testing are separate."""

import numpy as np


def pair_distances(first, second, metric):
    if metric == "euclidean":
        return np.linalg.norm(first - second, axis=1)
    if metric == "cosine":
        a = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
        b = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
        return np.clip(1 - np.sum(a * b, axis=1), 0, 2)
    raise ValueError("Unknown metric")


def verification_metrics(distances, labels, threshold):
    distances, labels = np.asarray(distances, dtype=np.float64), np.asarray(labels, dtype=bool)
    if not labels.any() or labels.all():
        raise ValueError("Evaluation requires positive and negative pairs")
    accepted = distances <= threshold
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(accepted == labels)),
        "far": float(np.mean(accepted[~labels])),
        "frr": float(np.mean(~accepted[labels])),
        "positive_pairs": int(labels.sum()),
        "negative_pairs": int((~labels).sum()),
    }


def calibrate_threshold(distances, labels, target_far=0.01):
    if not 0 <= target_far < 1:
        raise ValueError("target_far must be in [0, 1)")
    distances, labels = np.asarray(distances, dtype=np.float64), np.asarray(labels, dtype=bool)
    negative = np.sort(distances[~labels])
    if not len(negative) or not labels.any():
        raise ValueError("Calibration requires positive and negative pairs")
    allowed = int(np.floor(target_far * len(negative)))
    threshold = float(np.nextafter(float(negative[allowed]), -np.inf))
    return threshold


def retrieval_metrics(vectors, labels, metric):
    """First image of each person is gallery, remaining images are queries."""
    labels = np.asarray(labels)
    gallery = np.array([np.flatnonzero(labels == identity)[0] for identity in sorted(set(labels))])
    queries = np.setdiff1d(np.arange(len(labels)), gallery)
    if not len(queries):
        raise ValueError("Retrieval needs at least two images per identity")
    hits = {1: 0, 10: 0}
    for query in queries:
        distances = pair_distances(
            np.repeat(vectors[query : query + 1], len(gallery), axis=0), vectors[gallery], metric
        )
        order = np.argsort(distances, kind="stable")
        for k in hits:
            hits[k] += int(labels[query] in labels[gallery[order[:k]]])
    return {
        "recall_at_1": hits[1] / len(queries),
        "recall_at_10": hits[10] / len(queries),
        "gallery_size": len(gallery),
        "query_count": len(queries),
    }
