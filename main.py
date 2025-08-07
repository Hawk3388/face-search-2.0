import streamlit as st
import face_recognition
import json
from PIL import Image
import numpy as np
from io import BytesIO
import requests

# Lade Embedding-Datenbank
def load_database(path="face_embeddings.json"):
    with open(path, "r") as f:
        return json.load(f)

# Vergleiche Embedding mit allen bekannten
def find_matches(query_embedding, db, tolerance=0.5):
    matches = []
    for entry in db:
        db_embedding = np.array(entry["embedding"])
        distance = np.linalg.norm(query_embedding - db_embedding)
        if distance <= tolerance:
            matches.append((entry, distance))
    matches.sort(key=lambda x: x[1])  # sortieren nach Ähnlichkeit
    return matches

# App
st.set_page_config(page_title="Face Search", layout="centered")
st.title("🔍 Gesichtssuche")

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    # Prüfe Dateityp und lade ggf. als PIL Image
    if uploaded_file.type == "image/webp" or uploaded_file.name.lower().endswith(".webp"):
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_bytes = BytesIO()
        pil_img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        image = face_recognition.load_image_file(img_bytes)
    else:
        image = face_recognition.load_image_file(uploaded_file)

    encodings = face_recognition.face_encodings(image, model="large", num_jitters=10)

    if not encodings:
        st.warning("❌ Kein Gesicht im Bild erkannt.")
    else:
        st.image(uploaded_file, caption="Hochgeladenes Bild", use_container_width=True)
        st.info("🔎 Suche nach ähnlichen Gesichtern...")
        db = load_database()
        results = find_matches(encodings[0], db)

        if not results:
            st.error("😕 Keine ähnlichen Gesichter gefunden.")
        else:
            st.success(f"✅ {len(results)} Treffer gefunden!")
            for entry, dist in results[:10]:  # zeige die Top 10
                st.markdown(f"**Ähnlichkeit**: {dist:.4f}")
                st.markdown(f"[Bild anzeigen]({entry['image_url']})  \n[Seite besuchen]({entry['page_url']})")
                try:
                    img_bytes = requests.get(entry['image_url'], timeout=60).content
                    st.image(Image.open(BytesIO(img_bytes)), width=250)
                except Exception:
                    st.warning("Bild konnte nicht geladen werden.")

