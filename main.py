import streamlit as st
import face_recognition
import json
from PIL import Image
import numpy as np
from io import BytesIO
import requests
import os

# Load configuration from Streamlit secrets
try:
    API_URL = st.secrets["API_URL"]
    TOKEN = st.secrets["TOKEN"]
except KeyError as e:
    st.error(f"Missing configuration in Streamlit secrets: {e}")
    API_URL = None
    TOKEN = None

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
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(entry['image_url'], timeout=60, headers=headers)
                response.raise_for_status()
                img_bytes = response.content
                st.image(Image.open(BytesIO(img_bytes)), width=250)
            except requests.exceptions.Timeout:
                st.warning("⏰ Image loading timeout (>60s)")
            except requests.exceptions.RequestException as e:
                st.warning(f"🌐 Network error loading image: {str(e)[:50]}...")
            except Exception as e:
                st.warning(f"🖼️ Image processing error: {str(e)[:50]}...")

def server(encodings):
    try:
        payload = {"encoding": encodings[0].tolist()}
        headers = {'Content-Type': 'application/json'}
        if TOKEN:
            headers['Authorization'] = f'Bearer {TOKEN}'
        response = requests.post(f"{API_URL}/search", json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        api_response = response.json()
        
        if api_response.get("success"):
            results = api_response.get("matches", [])
            
            if not results:
                st.error("😕 No similar faces found.")
            else:
                st.success(f"✅ {len(results)} matches found!")
                for match in results[:10]:  # show top 10
                    # Handle different response formats from server
                    if isinstance(match, list) and len(match) >= 2:
                        # Format: [entry_dict, distance]
                        entry, distance = match[0], match[1]
                        st.markdown(f"**Distance**: {distance:.4f}")
                        st.markdown(f"[View image]({entry['image_url']})  \n[Visit page]({entry['page_url']})")
                        img_url = entry['image_url']
                    elif isinstance(match, dict):
                        # Format: {"distance": x, "similarity": y, "image_url": z, ...}
                        st.markdown(f"**Distance**: {match['distance']:.4f} | **Similarity**: {match['similarity']:.3f}")
                        st.markdown(f"[View image]({match['image_url']})  \n[Visit page]({match['page_url']})")
                        img_url = match['image_url']
                    else:
                        st.warning("Unknown response format from server")
                        continue
                        
                    try:
                        USER_AGENT = "FaceSearchBot/1.0 (Educational Research; Contact: github.com/Hawk3388/face-search-2.0)"
                        headers = {'User-Agent': USER_AGENT}
                        response = requests.get(img_url, timeout=60, headers=headers)
                        response.raise_for_status()
                        img_bytes = response.content
                        st.image(Image.open(BytesIO(img_bytes)), width=250)
                    except requests.exceptions.Timeout:
                        st.warning("⏰ Image loading timeout (>60s)")
                    except requests.exceptions.RequestException as e:
                        st.warning(f"🌐 Network error loading image: {str(e)[:50]}...")
                    except Exception as e:
                        st.warning(f"🖼️ Image processing error: {str(e)[:50]}...")
        else:
            st.error(f"API Error: {api_response.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error contacting API: {e}")

# Check API health
def check_api_health():
    if not API_URL or not TOKEN:
        return False
    try:
        headers = {'Authorization': f'Bearer {TOKEN}'}
        response = requests.get(f"{API_URL}/health", headers=headers, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def main():
    # App
    st.set_page_config(page_title="Face Search", layout="centered")
    st.title("🔍 Face Search")

    path = "face_embeddings.json"

    if os.path.exists(path):
        db = load_database(path)

    # Determine mode automatically
    local_available = os.path.exists(path)
    
    if local_available:
        use_server = False
        st.info(f"💻 Using local database ({len(db) if 'db' in locals() else 0} entries)")
    elif API_URL and TOKEN:
        # Only check API health if no local DB is available
        api_available = check_api_health()
        if api_available:
            use_server = True
            st.info("🌐 Using API server")
        else:
            st.error("❌ No local database found and API server is not reachable!")
            st.stop()
            use_server = False
    else:
        st.error("❌ No database file found and no API configured!")
        st.stop()
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
