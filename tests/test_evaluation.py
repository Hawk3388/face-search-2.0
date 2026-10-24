import numpy as np
import pytest

from face_search.evaluation import calibrate_threshold, verification_metrics, retrieval_metrics
from model2.data import split_people, create_pairs
from face_search.storage import read_json


def test_small_person_split_has_both_labels_and_no_identity_overlap():
    people = {str(i): [f"{i}-a", f"{i}-b"] for i in range(5)}
    train, validation = split_people(people)
    assert len(train) >= 2 and len(validation) >= 2
    assert not set(train) & set(validation)
    assert {p[2] for p in create_pairs(validation)} == {0, 1}
    assert create_pairs(train, seed=1) != create_pairs(train, seed=2)
    with pytest.raises(ValueError):
        split_people({"one": ["a", "b"]})


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_calibration_respects_false_acceptance_with_ties(dtype):
    distances = np.array([0.1, 0.2, 0.3, 0.3, 0.7, 0.8], dtype=dtype)
    labels = [1, 1, 0, 0, 0, 0]
    threshold = calibrate_threshold(distances, labels, 0.25)
    metrics = verification_metrics(distances, labels, threshold)
    assert metrics["far"] <= 0.25
    assert metrics["frr"] == 0


def test_retrieval_uses_separate_query_and_gallery_images():
    vectors = np.array([[1.0, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
    metrics = retrieval_metrics(vectors, ["a", "a", "b", "b"], "euclidean")
    assert metrics["query_count"] == 2
    assert metrics["gallery_size"] == 2
    assert metrics["recall_at_1"] == 1


@pytest.mark.parametrize("content", ['[{"embedding": [1]}]', '[{"embedding": [1]}', '[{"embedding": [1]},'])
def test_legacy_json_read_never_modifies_source(tmp_path, content):
    path = tmp_path / "legacy.json"
    path.write_text(content)
    assert list(read_json(path)) == [{"embedding": [1]}]
    assert path.read_text() == content


def test_truncated_object_is_not_silently_accepted(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('[{"embedding": [1')
    with pytest.raises(ValueError):
        list(read_json(path))
