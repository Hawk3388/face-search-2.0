"""Streamlit frontend; private query state stays in the current user session."""

import hashlib
import os
from pathlib import Path
from threading import Lock

import requests
import streamlit as st
from PIL import Image, ImageDraw

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
        draw.rectangle((left, top, right, bottom), outline="red", width=3)
        draw.text((left, max(0, top - 15)), f"Face {i + 1}", fill="red")
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


def render_results(response):
    matches = response.get("matches", [])
    if not matches:
        st.info("No matching faces found at this distance threshold.")
        return
    st.success(f"{len(matches)} matches found")
    st.caption("Smaller distance means a closer match. Results are candidates, not proof of identity.")
    for match in matches:
        if isinstance(match, (list, tuple)):
            entry, distance = match
        else:
            entry, distance = match, match["distance"]
        if entry.get("image_url"):
            st.image(entry["image_url"], width=250)
        st.write(f"Distance: {distance:.4f}")
        if entry.get("image_url"):
            st.link_button("View image", entry["image_url"])
        for page in entry.get("page_urls") or ([entry["page_url"]] if entry.get("page_url") else []):
            st.link_button("Source page", page)
        st.divider()


def main():
    st.set_page_config(page_title="Face Search", layout="centered")
    st.title("Face Search")
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
    with st.popover("Database statistics"):
        st.write(f"Faces: {health.get('total_entries', 0) if use_server else len(index.entries):,}")
        last_page = health.get("last_page_url") if use_server else None
        if last_page:
            st.link_button("Last source page", last_page)
    st.info("Using API server" if use_server else "Using local database")
    st.caption("Uploaded images are processed for this search. Use Clear search to discard this session's query.")
    generation = st.session_state.get("upload_generation", 0)
    if st.button("Clear search"):
        for key in ("query", "query_key", "selected_face", "search_results"):
            st.session_state.pop(key, None)
        st.session_state.upload_generation = generation + 1
        st.rerun()
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], key=f"upload_{generation}")
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
            st.session_state.selected_face = 0
            st.session_state.search_results = {0: query["legacy_response"]} if "legacy_response" in query else {}
        query = st.session_state.query
        count = query.get("face_count", 0)
        st.image(draw_face_boxes(query["image"], query.get("face_locations", [])), use_container_width=True)
        if not count:
            st.warning("No face detected in the image.")
            return
        if count > 1:
            st.write("Select a face to search:")
            columns = st.columns(min(count, 4))
            for i, (top, right, bottom, left) in enumerate(query["face_locations"]):
                with columns[i % len(columns)]:
                    st.image(query["image"][top:bottom, left:right], width=100)
                    if st.button(f"Search Face {i + 1}", key=f"face_{i}"):
                        st.session_state.selected_face = i
        selected = st.session_state.selected_face
        st.write(f"Results for Face {selected + 1}")
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
        render_results(st.session_state.search_results[selected])
    except (requests.RequestException, ValueError, OSError, RuntimeError) as exc:
        st.error(f"Search failed: {exc}")


if __name__ == "__main__":
    main()
