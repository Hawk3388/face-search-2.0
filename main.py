import streamlit as st
import face_recognition
import json
from PIL import Image
import numpy as np
from io import BytesIO
import requests
import os
import gdown

# Load embedding database
def load_database(path="face_embeddings.json"):
    with open(path, "r") as f:
        return json.load(f)

# Compare embedding with all known faces
def find_matches(query_embedding, db, tolerance=0.5):
    matches = []
    for entry in db:
        db_embedding = np.array(entry["embedding"])
        distance = np.linalg.norm(query_embedding - db_embedding)
        if distance <= tolerance:
            matches.append((entry, distance))
    matches.sort(key=lambda x: x[1])  # sort by similarity
    return matches

def main():
    # App
    st.set_page_config(page_title="Face Search", layout="centered")
    st.title("🔍 Face Search")

    path = "face_embeddings.json"

    if not os.path.exists(path):
        file_id = st.secrets["gdrive"]["file_id"]
        gdown.download(file_id, path, quiet=False)

    db = load_database()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        # Check file type and load as PIL Image if necessary
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
            st.warning("❌ No face detected in the image.")
        else:
            st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
            st.info("🔎 Searching for similar faces...")
            
            results = find_matches(encodings[0], db)

            if not results:
                st.error("😕 No similar faces found.")
            else:
                st.success(f"✅ {len(results)} matches found!")
                for entry, dist in results[:10]:  # show top 10
                    st.markdown(f"**Similarity**: {dist:.4f}")
                    st.markdown(f"[View image]({entry['image_url']})  \n[Visit page]({entry['page_url']})")
                    try:
                        img_bytes = requests.get(entry['image_url'], timeout=60).content
                        st.image(Image.open(BytesIO(img_bytes)), width=250)
                    except Exception:
                        st.warning("Image could not be loaded.")

if __name__ == "__main__":
    main()
