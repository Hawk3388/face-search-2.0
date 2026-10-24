from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest


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
        app.button(key="face_1").click().run()
        assert not app.exception
        assert app.session_state.selected_face == 1
        assert sum(url.endswith("analyze") for url in calls) == 1
        assert sum(url.endswith("search") for url in calls) == 2
        app.button(key="face_0").click().run()
        assert sum(url.endswith("search") for url in calls) == 2
