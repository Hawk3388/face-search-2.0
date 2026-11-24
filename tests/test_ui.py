from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import requests
from PIL import Image
from streamlit.testing.v1 import AppTest

from face_search.people import suggest_person, wikipedia_people
from main import render_results


def test_selecting_another_face_does_not_repeat_analysis(monkeypatch):
    monkeypatch.setenv("API_URL", "https://test.invalid")
    monkeypatch.setenv("TOKEN", "test-token")
    data = BytesIO()
    Image.new("RGB", (20, 20)).save(data, "PNG")
    upload = SimpleNamespace(name="test.png", type="image/png", getvalue=lambda: data.getvalue())
    health = {
        "database_loaded": True,
        "total_entries": 2,
        "last_page_url": None,
        "model_id": "test",
        "capabilities": ["analyze"],
    }
    calls = []

    class Response:
        ok = True

        def __init__(self, data):
            self.data = data

        def json(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def post(url, **kwargs):
        calls.append(url)
        if url.endswith("analyze"):
            return Response(
                {
                    "success": True,
                    "face_count": 2,
                    "face_locations": [[0, 10, 10, 0], [10, 20, 20, 10]],
                    "encodings": [[1], [2]],
                    "model_id": "test",
                }
            )
        return Response({"success": True, "matches": []})

    with (
        patch("requests.get", return_value=Response(health)),
        patch("requests.post", side_effect=post),
        patch("streamlit.file_uploader", return_value=upload),
    ):
        app = AppTest.from_file("main.py", default_timeout=15).run()
        assert not app.exception
        assert app.title[0].value == "🔍 Face Search"
        assert app.session_state.selected_face is None
        assert sum(url.endswith("search") for url in calls) == 0
        app.button(key="face_1").click().run()
        assert not app.exception
        assert app.session_state.selected_face == 1
        assert sum(url.endswith("analyze") for url in calls) == 1
        assert sum(url.endswith("search") for url in calls) == 1
        app.button(key="face_0").click().run()
        assert sum(url.endswith("search") for url in calls) == 2
        app.button(key="face_1").click().run()
        assert sum(url.endswith("search") for url in calls) == 2


def result(number, *pages, distance=0.04):
    return {"image_url": f"https://example.invalid/{number}.jpg", "page_urls": pages, "distance": distance}


EN = "https://en.wikipedia.org/wiki/Example_Person"
DE = "https://de.wikipedia.org/wiki/Beispielperson"
OTHER = "https://en.wikipedia.org/wiki/Another_Person"


def people(articles):
    return {
        ("en", "Example Person"): ("Q1", "Example Person"),
        ("de", "Beispielperson"): ("Q1", "Example Person"),
        ("en", "Another Person"): ("Q2", "Another Person"),
    }


def test_name_votes_merge_language_versions_without_counting_an_image_twice():
    matches = [result(1, EN, DE), result(1, EN), result(2, DE), result(3, OTHER)]
    assert suggest_person(matches, lookup=people) == ("Example Person", 2)
    assert suggest_person(matches + [result(4, OTHER)], lookup=people) is None


def test_closer_images_can_outweigh_more_distant_images():
    matches = [
        [result(1, EN), 0.02],
        result(2, DE, distance=0.02),
        *(result(i, OTHER, distance=0.06) for i in (3, 4, 5)),
    ]
    assert suggest_person(matches, lookup=people) == ("Example Person", 2)


def test_weighting_uses_only_best_distance_per_person_and_image():
    matches = [
        result(1, EN, distance=0.06),
        result(1, DE, distance=0.01),
        result(2, EN, distance=0.06),
        result(3, OTHER, distance=0.03),
        result(4, OTHER, distance=0.03),
    ]
    assert suggest_person(matches, lookup=people) == ("Example Person", 2)
    # Duplicating A's stronger image cannot inflate A's score.
    assert suggest_person(
        matches[:3] + [matches[1], result(3, OTHER, distance=0.02), result(4, OTHER, distance=0.02)], lookup=people
    ) == ("Another Person", 2)


def test_one_distinct_image_needs_no_lookup():
    def unexpected_lookup(_):
        raise AssertionError("One distinct image must not trigger a lookup")

    assert suggest_person([result(1, EN, DE), result(1, EN)], lookup=unexpected_lookup) is None


def test_only_human_items_are_returned_and_article_redirects_resolve(monkeypatch):
    def fake_api(url, **params):
        if url == "https://en.wikipedia.org/w/api.php":
            assert params["redirects"] == 1
            return {
                "query": {
                    "redirects": [{"from": "Alias", "to": "Example Person"}],
                    "pages": [
                        {"title": "Example Person", "ns": 0, "pageprops": {"wikibase_item": "Q1"}},
                        {"title": "Band", "ns": 0, "pageprops": {"wikibase_item": "Q2"}},
                    ],
                }
            }
        assert url == "https://www.wikidata.org/w/api.php"
        assert params["ids"] == "Q1|Q2"

        def item(kind):
            return {"claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": kind}}}}]}}

        return {"entities": {"Q1": item("Q5"), "Q2": item("Q215380")}}

    monkeypatch.setattr("face_search.people.wiki_json", fake_api)
    assert wikipedia_people((("en", "Alias"), ("en", "Band"))) == {("en", "Alias"): ("Q1", "Example Person")}


def test_render_uses_only_visible_hits_and_continues_if_wikidata_is_down(monkeypatch):
    image_urls, notices = [], []
    monkeypatch.setattr("main.st.image", lambda url, **_kwargs: image_urls.append(url))
    monkeypatch.setattr("main.st.info", notices.append)
    for widget in ("success", "divider", "markdown"):
        monkeypatch.setattr(f"main.st.{widget}", lambda *_args: None)
    monkeypatch.setattr("main.cached_wikipedia_people", people)
    response = {"matches": [result(1, EN), result(2, DE, distance=0.06), result(3, OTHER, distance=0.0644)]}
    render_results(response, use_server=True)
    assert len(image_urls) == 2
    assert len(notices) == 1 and "Example Person** · 2 matching images" in notices[0]
    notices.clear()

    def unavailable(_):
        raise requests.Timeout()

    monkeypatch.setattr("main.cached_wikipedia_people", unavailable)
    render_results(response, use_server=True)
    assert len(image_urls) == 4
    assert not notices
