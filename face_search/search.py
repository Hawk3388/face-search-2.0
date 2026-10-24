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
    if parsed.hostname in ("upload.wikimedia.org", "thumb.wikimedia.org") and "/thumb/" in parsed.path:
        prefix, tail = parsed.path.split("/thumb/", 1)
        parts = tail.split("/")
        # Multipage documents/video thumbnails may depict different pages/frames.
        # Only collapse formats with an unambiguous original still image.
        suffix = parts[2].rsplit(".", 1)[-1].lower() if len(parts) >= 4 else ""
        if suffix in ("jpg", "jpeg", "png", "webp", "bmp", "svg"):
            parsed = parsed._replace(
                scheme=parsed.scheme or "https",
                netloc="upload.wikimedia.org",
                path=prefix + "/" + "/".join(parts[:-1]),
                query="",
            )
    return urlunsplit(parsed._replace(fragment=""))


def validate_embedding(value, dimension):
    array = np.asarray(value)
    if array.shape != (dimension,) or array.dtype.kind not in "fiu":
        raise ValueError(f"Encoding must contain exactly {dimension} numeric values")
    array = array.astype(np.float32)
    if not np.isfinite(array).all() or np.max(np.abs(array), initial=0) > 1e6:
        raise ValueError("Encoding values must be finite and within [-1000000, 1000000]")
    if not np.any(array):
        raise ValueError("Encoding must not be a zero vector")
    return array


def nearest_positions(indices, distances, count):
    """Partial selection with deterministic row-order ties, including the cutoff."""
    if len(indices) > count:
        cutoff = np.partition(distances, count - 1)[count - 1]
        below = np.flatnonzero(distances < cutoff)
        tied = np.flatnonzero(distances == cutoff)[: count - len(below)]
        positions = np.concatenate((below, tied))
    else:
        positions = np.arange(len(indices))
    return positions[np.lexsort((indices[positions], distances[positions]))]


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
        norms_sq = np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64)
        normalized = np.empty_like(matrix)
        np.divide(matrix, np.sqrt(norms_sq)[:, None], out=normalized)
        for array in (matrix, normalized, norms_sq):
            array.setflags(write=False)
        return cls(tuple(entries), matrix, normalized, norms_sq, model_id, dimension)

    def search(self, encoding, tolerance=0.5, max_results=20, metric="euclidean", dedup=True):
        query = validate_embedding(encoding, self.dimension)
        if not math.isfinite(tolerance) or tolerance < 0 or tolerance > 2:
            raise ValueError("Tolerance must be between 0 and 2")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise ValueError("max_results must be a positive integer")
        query64 = query.astype(np.float64)
        query_norm_sq = np.dot(query64, query64)
        roundoff = 4 * (self.dimension + 4) * np.finfo(np.float32).eps
        if metric == "cosine":
            unit_query = query64 / np.sqrt(query_norm_sq)
            approximate = np.clip(1 - self.normalized @ unit_query.astype(np.float32), 0, 2)
            indices = np.flatnonzero(approximate <= tolerance + roundoff)
            distances = approximate[indices].astype(np.float64)
            uncertain = (
                np.arange(len(indices))
                if len(indices) <= 4096
                else np.flatnonzero((distances <= roundoff) | (np.abs(distances - tolerance) <= roundoff))
            )
            for start in range(0, len(uncertain), 4096):
                positions = uncertain[start : start + 4096]
                selected = indices[positions]
                block = self.matrix[selected].astype(np.float64)
                identical = np.all(block == query64, axis=1)
                # Half the squared unit-vector distance avoids cancellation at cosine=1.
                difference = block / np.sqrt(self.norms_sq[selected])[:, None] - unit_query
                distances[positions] = np.clip(0.5 * np.einsum("ij,ij->i", difference, difference), 0, 2)
                distances[positions[identical]] = 0
        elif metric == "euclidean":
            norm_sums = self.norms_sq.astype(np.float32) + np.float32(query_norm_sq)
            approximate = np.maximum(norm_sums - 2 * (self.matrix @ query), 0)
            # Refine near zero/the decision boundary to avoid float32 cancellation
            # rejecting an exact self-match or changing a threshold decision.
            error_bound = roundoff * norm_sums + self.dimension * np.finfo(np.float32).tiny
            indices = np.flatnonzero(approximate <= tolerance**2 + error_bound)
            squared = approximate[indices].astype(np.float64)
            error_bound = error_bound[indices]
            uncertain = (
                np.arange(len(indices))
                if len(indices) <= 4096
                else np.flatnonzero((squared <= error_bound) | (np.abs(squared - tolerance**2) <= error_bound))
            )
            for start in range(0, len(uncertain), 4096):
                positions = uncertain[start : start + 4096]
                selected = indices[positions]
                difference = self.matrix[selected].astype(np.float64) - query
                squared[positions] = np.einsum("ij,ij->i", difference, difference)
            distances = np.sqrt(squared)
        else:
            raise ValueError("metric must be euclidean or cosine")
        valid = distances <= tolerance
        indices, distances = indices[valid], distances[valid]
        # Only sort a small leading window. Expand it if image duplicates consume
        # slots, so deduplication never hides otherwise eligible results.
        count = min(len(indices), max_results * 2 if dedup else max_results)
        while True:
            selected = nearest_positions(indices, distances, count)
            results, seen = [], set()
            for position in selected:
                i = indices[position]
                entry = self.entries[int(i)]
                key = entry.get("image_url") or ("face", int(i))
                if dedup and key in seen:
                    continue
                seen.add(key)
                distance = float(distances[position])
                results.append(dict(entry, distance=distance, similarity=max(0.0, 1.0 - distance)))
                if len(results) >= max_results:
                    break
            if len(results) >= max_results or count == len(indices):
                break
            count = min(len(indices), count * 2)
        return results, {
            "valid_candidates": len(indices),
            "returned_candidates": len(results),
            "total_db_entries": len(self.entries),
            "embeddings_loaded": len(self.entries),
        }
