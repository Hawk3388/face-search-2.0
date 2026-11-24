"""Streamlit frontend; private query state stays in the current user session."""

import hashlib
import os
from pathlib import Path
from threading import Lock
from urllib.parse import quote, unquote, urlsplit

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from face_search.config import Settings
from face_search.encoding import FaceEncoder, load_image
from face_search.search import SearchIndex
from face_search.storage import read_records


def configuration():
    settings = Settings.from_env(os.getenv("LOCAL_BACKEND", "sqlite"))
    api_url, token = os.getenv("API_URL", ""), os.getenv("TOKEN", "")
    try:
        api_url = api_url or st.secrets.get("API_URL", "")
        token = token or st.secrets.get("TOKEN", "")
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    return settings, api_url.rstrip("/"), token


@st.cache_data(ttl=20, show_spinner=False)
def get_health_stats(api_url):
    if not api_url:
        return None
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        if response.ok:
            data = response.json()
            if data.get("database_loaded", True):
                return data
    except (requests.RequestException, ValueError):
        pass
    return None


def database_stamp(path):
    files = [Path(path), Path(str(path) + "-wal")]
    return tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in files)


@st.cache_resource(show_spinner="Loading local search index...", max_entries=2)
def local_resources(settings, stamp):
    encoder = FaceEncoder(settings)
    index = SearchIndex.build(read_records(settings.database, settings.backend), encoder.model_id, encoder.dimension)
    if not index.entries:
        raise ValueError("No faces for the selected model in the local database")
    return encoder, index, Lock()


def draw_face_boxes(image, face_locations):
    result = Image.fromarray(image).copy()
    draw = ImageDraw.Draw(result)
    for i, (top, right, bottom, left) in enumerate(face_locations):
        font_size = max(12, min(bottom - top, right - left) // 4)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default(size=font_size)
        draw.rectangle((left, top, right, bottom), outline="red", width=3)
        draw.text((left, max(0, top - font_size - 5)), f"Face {i + 1}", fill="red", font=font)
    return result


def api_call(api_url, token, endpoint, **kwargs):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.post(f"{api_url}/{endpoint}", headers=headers, timeout=(5, 120), **kwargs) as response:
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise ValueError("API returned an invalid response")
        if not response.ok or not data.get("success"):
            raise ValueError(data.get("error", f"API error {response.status_code}"))
        return data


def render_results(response, use_server):
    matches = [
        match
        for match in response.get("matches", [])
        if (match[1] if isinstance(match, (list, tuple)) else match["distance"]) <= 0.6
    ]
    if not matches:
        st.error("😕 No similar faces found.")
        return
    st.success(f"✅ {len(matches)} matches found!")
    for match in matches:
        if isinstance(match, (list, tuple)):
            entry, distance = match
        else:
            entry, distance = match, match["distance"]
        score = f"**Distance**: {distance:.4f}"
        if use_server and not isinstance(match, (list, tuple)) and match.get("similarity") is not None:
            score += f" | **Similarity**: {match['similarity']:.3f}"
        st.markdown(score)
        image_url = entry.get("image_url")
        pages = entry.get("page_urls") or [entry.get("page_url")]
        links = []
        for label, url in [("View image", image_url), *(("Visit page", page) for page in pages)]:
            if isinstance(url, str) and url.lower().startswith(("https://", "http://")):
                target = quote(url, safe=":/?#[]@!$&'()*+,;=%")
                links.append(f"[{label}](<{target}>)")
        if links:
            st.markdown("  \n".join(links))
        if image_url:
            st.image(image_url, width=250)


def main():
    st.set_page_config(page_title="Face Search", layout="centered")
    title_column, stats_column = st.columns([3, 1])
    with title_column:
        st.title("🔍 Face Search")
    settings, api_url, token = configuration()
    health = get_health_stats(api_url)
    use_server = health is not None
    if not use_server and not settings.database.exists():
        st.error("API unavailable and no local database found. Configure API_URL or DATABASE_PATH.")
        st.stop()
    encoder = index = encode_lock = None
    stamp = None
    if not use_server:
        stamp = database_stamp(settings.database)
        try:
            encoder, index, encode_lock = local_resources(settings, stamp)
        except Exception as exc:
            st.error(f"Unable to load local database: {exc}")
            st.stop()
    with stats_column, st.popover("📊 Stats", help="Database statistics"):
        st.markdown("**🌐 API Database**" if use_server else "**💻 Local Database**")
        st.write(f"**Entries:** {health.get('total_entries', 0) if use_server else len(index.entries):,}")
        last_page = (
            health.get("last_page_url")
            if use_server
            else next((entry["page_url"] for entry in reversed(index.entries) if entry.get("page_url")), None)
        )
        if last_page:
            page_name = unquote(urlsplit(last_page).path.rstrip("/").rsplit("/", 1)[-1]).replace("_", " ")
            st.write(f"**Last Source Page:** {page_name}")
            st.link_button("🔗 View Page", last_page)
    st.info(
        "🌐 Using API server (uploaded images are processed for this search)"
        if use_server
        else "💻 Using local database"
    )
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], key="image_upload")
    if uploaded is None:
        for key in ("query", "query_key", "selected_face", "search_results"):
            st.session_state.pop(key, None)
        return
    data = uploaded.getvalue()
    model_id = health.get("model_id", "legacy") if use_server else encoder.model_id
    key = (hashlib.sha256(data).hexdigest(), api_url if use_server else str(settings.database), model_id, stamp)
    try:
        if st.session_state.get("query_key") != key:
            image = load_image(data, settings.max_upload_bytes, settings.max_pixels)
            with st.spinner("Detecting faces..."):
                if use_server:
                    endpoint = "analyze" if "analyze" in health.get("capabilities", []) else "search"
                    query = api_call(
                        api_url,
                        token,
                        endpoint,
                        files={"image": (uploaded.name, data, uploaded.type or "application/octet-stream")},
                    )
                    if endpoint == "search":
                        query["legacy_response"] = dict(query)
                else:
                    with encode_lock:
                        locations, vectors = encoder.extract(image)
                    query = dict(
                        face_locations=locations,
                        encodings=[v.tolist() for v in vectors],
                        face_count=len(vectors),
                        model_id=encoder.model_id,
                    )
            query["image"] = image
            st.session_state.query = query
            st.session_state.query_key = key
            st.session_state.selected_face = 0 if query.get("face_count", 0) <= 1 else None
            st.session_state.search_results = {0: query["legacy_response"]} if "legacy_response" in query else {}
        query = st.session_state.query
        count = query.get("face_count", 0)
        caption = f"Uploaded image - {count} face(s) detected" if count else "Uploaded image - No faces detected"
        st.image(draw_face_boxes(query["image"], query.get("face_locations", [])), caption=caption, width="stretch")
        if not count:
            st.warning("❌ No face detected in the image.")
            return
        if count > 1:
            st.info(f"🔍 Found {count} faces in the image. Select which face to search for:")
            for i, column in enumerate(st.columns(count)):
                with column:
                    if st.button(f"Search Face {i + 1}", key=f"face_{i}"):
                        st.session_state.selected_face = i
        selected = st.session_state.selected_face
        if selected is None:
            return
        st.info(f"🔎 Searching for Face {selected + 1}..." if count > 1 else "🔎 Searching for similar faces...")
        if selected not in st.session_state.search_results:
            with st.spinner("Searching..."):
                if use_server and "encodings" in query:
                    response = api_call(
                        api_url,
                        token,
                        "search",
                        json={
                            "encoding": query["encodings"][selected],
                            "model_id": query["model_id"],
                            "format": "objects",
                        },
                    )
                elif use_server:
                    response = api_call(
                        api_url,
                        token,
                        "search",
                        data={"selected_face": selected},
                        files={"image": (uploaded.name, data, uploaded.type)},
                    )
                else:
                    matches, _ = index.search(
                        query["encodings"][selected], settings.tolerance, min(20, settings.max_results), settings.metric
                    )
                    response = {"matches": matches}
                st.session_state.search_results[selected] = response
        render_results(st.session_state.search_results[selected], use_server)
    except (requests.RequestException, ValueError, OSError, RuntimeError) as exc:
        st.error(f"Search failed: {exc}")


if __name__ == "__main__":
    main()
