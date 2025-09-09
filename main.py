import streamlit as st
import face_recognition
import json
from PIL import Image
import numpy as np
from io import BytesIO
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")

# Load embedding database
def load_database(path="face_embeddings.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return []

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

def serverless(encodings, db):
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

def server(encodings):
    try:
        payload = {"encoding": encodings[0].tolist()}
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        api_response = response.json()
        
        if api_response.get("success"):
            results = api_response.get("matches", [])
            
            if not results:
                st.error("😕 No similar faces found.")
            else:
                st.success(f"✅ {len(results)} matches found!")
                for match in results[:10]:  # show top 10
                    st.markdown(f"**Distance**: {match['distance']:.4f} | **Similarity**: {match['similarity']:.3f}")
                    st.markdown(f"[View image]({match['image_url']})  \n[Visit page]({match['page_url']})")
                    try:
                        img_bytes = requests.get(match['image_url'], timeout=60).content
                        st.image(Image.open(BytesIO(img_bytes)), width=250)
                    except Exception:
                        st.warning("Image could not be loaded.")
        else:
            st.error(f"API Error: {api_response.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error contacting API: {e}")

def main():
    # App
    st.set_page_config(page_title="Face Search", layout="centered")
    st.title("🔍 Face Search")

    path = "face_embeddings.json"

    if os.path.exists(path):
        db = load_database(path)

    # Sidebar for settings
    st.sidebar.header("⚙️ Search Settings")
    
    # Check API health
    def check_api_health():
        if not API_URL:
            return False
        try:
            response = requests.get(f"{API_URL.rstrip('/')}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    # Check availability
    api_available = check_api_health()
    local_available = os.path.exists(path)
    
    # Mode selection if both are available
    if api_available and local_available:
        st.sidebar.subheader("🔄 Search Mode")
        search_mode = st.sidebar.radio(
            "Choose search method:",
            ["🌐 Server Mode (API)", "💻 Local Mode"],
            help="Server mode uses the API, Local mode uses the local database file"
        )
        use_server = "Server" in search_mode
        
        if use_server:
            st.sidebar.success("🌐 Using Server Mode")
            st.sidebar.write(f"API: {API_URL}")
        else:
            st.sidebar.info("💻 Using Local Mode")
            st.sidebar.write(f"Database: {len(db) if 'db' in locals() else 0} entries")
    
    # Single option available
    elif api_available:
        st.sidebar.success("🌐 Server Mode (Only)")
        st.sidebar.write(f"API: {API_URL}")
        use_server = True
    elif local_available:
        st.sidebar.info("💻 Local Mode (Only)")
        st.sidebar.write(f"Database: {len(db) if 'db' in locals() else 0} entries")
        use_server = False
    else:
        st.sidebar.error("❌ No database or API configured")
        use_server = False

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
            
            # Use selected mode
            if use_server and api_available:
                st.info("🔎 Searching for similar faces (Server Mode)...")
                server(encodings)
            elif not use_server and local_available:
                st.info("🔎 Searching for similar faces (Local Mode)...")
                serverless(encodings, db)
            else:
                st.error("❌ Selected mode not available!")

if __name__ == "__main__":
    main()
