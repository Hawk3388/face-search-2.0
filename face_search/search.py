"""Immutable search snapshots: metadata and vectors always change together."""

from dataclasses import dataclass
import math
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .config import DLIB_MODEL


def normalize_image_url(url):
    if not url:
        return None
    parsed = urlsplit(str(url).strip())
    # Only Wikimedia has this thumbnail path convention.
    if parsed.hostname == "upload.wikimedia.org" and "/thumb/" in parsed.path:
        prefix, tail = parsed.path.split("/thumb/", 1)
        parts = tail.split("/")
        if len(parts) >= 4:
            parsed = parsed._replace(path=prefix + "/" + "/".join(parts[:-1]), query="")
    return urlunsplit(parsed._replace(fragment=""))


def validate_embedding(value, dimension):
    array = np.asarray(value)
    if array.shape != (dimension,) or array.dtype.kind not in "fiu":
        raise ValueError(f"Encoding must contain exactly {dimension} numeric values")
    array = array.astype(np.float32)
    if not np.isfinite(array).all() or np.max(np.abs(array), initial=0) > 1e6:
        raise ValueError("Encoding values must be finite and within [-1000000, 1000000]")
    if np.linalg.norm(array) == 0:
        raise ValueError("Encoding must not be a zero vector")
    return array


@dataclass(frozen=True)
class SearchIndex:
    entries: tuple
    matrix: np.ndarray
    normalized: np.ndarray
    norms_sq: np.ndarray
    model_id: str
    dimension: int

    @classmethod
    def build(cls, records, model_id=DLIB_MODEL, dimension=128):
        entries, vectors = [], []
        for row in records:
            if row.get("model_id", DLIB_MODEL) != model_id:
                continue
            vector = validate_embedding(row["embedding"], dimension)
            entry = {key: value for key, value in row.items() if key != "embedding"}
            entry["image_url"] = normalize_image_url(entry.get("image_url"))
            pages = list(dict.fromkeys(row.get("page_urls") or ([row["page_url"]] if row.get("page_url") else [])))
            entry.update(page_urls=pages, page_url=pages[0] if pages else None)
            entries.append(entry)
            vectors.append(vector)
        matrix = np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1, dimension)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = matrix / np.maximum(norms, 1e-12)
        norms_sq = np.einsum("ij,ij->i", matrix, matrix)
        for array in (matrix, normalized, norms_sq):
            array.setflags(write=False)
        return cls(tuple(entries), matrix, normalized, norms_sq, model_id, dimension)

    def search(self, encoding, tolerance=0.5, max_results=20, metric="euclidean", dedup=True):
        query = validate_embedding(encoding, self.dimension)
        if not math.isfinite(tolerance) or tolerance < 0 or tolerance > 2:
            raise ValueError("Tolerance must be between 0 and 2")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise ValueError("max_results must be a positive integer")
        if metric == "cosine":
            distances = np.clip(1 - self.normalized @ (query / np.linalg.norm(query)), 0, 2)
        elif metric == "euclidean":
            squared = np.maximum(self.norms_sq + np.dot(query, query) - 2 * (self.matrix @ query), 0)
            # Refine near zero/the decision boundary to avoid float32 cancellation
            # rejecting an exact self-match or changing a threshold decision.
            error_bound = 16 * np.finfo(np.float32).eps * (self.norms_sq + np.dot(query, query))
            uncertain = np.flatnonzero((squared <= error_bound) | (np.abs(squared - tolerance**2) <= error_bound))
            for start in range(0, len(uncertain), 4096):
                selected = uncertain[start : start + 4096]
                difference = self.matrix[selected].astype(np.float64) - query
                squared[selected] = np.einsum("ij,ij->i", difference, difference)
            distances = np.sqrt(squared)
        else:
            raise ValueError("metric must be euclidean or cosine")
        indices = np.flatnonzero(distances <= tolerance)
        # Stable ordering also resolves ties at the top-k boundary.
        indices = indices[np.argsort(distances[indices], kind="stable")]
        results, seen = [], set()
        for i in indices:
            entry = self.entries[int(i)]
            key = entry.get("image_url") or ("face", int(i))
            if dedup and key in seen:
                continue
            seen.add(key)
            distance = float(distances[i])
            results.append(dict(entry, distance=distance, similarity=max(0.0, 1.0 - distance)))
            if len(results) >= max_results:
                break
        return results, {
            "valid_candidates": len(indices),
            "returned_candidates": len(results),
            "total_db_entries": len(self.entries),
            "embeddings_loaded": len(self.entries),
        }
