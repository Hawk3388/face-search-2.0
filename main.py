import streamlit as st
import face_recognition
import json
from PIL import Image, ImageDraw, ImageFont
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

def draw_face_boxes(image, face_locations):
    """Draw boxes around detected faces"""
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    draw = ImageDraw.Draw(pil_image)
    
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Calculate face box dimensions
        face_height = bottom - top
        face_width = right - left
        
        # Dynamic font size: 1/4 of the smaller dimension (height or width)
        font_size = max(12, min(face_height, face_width) // 4)  # Minimum 12px font
        
        # Draw rectangle around face
        draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
        
        # Add face number with dynamic font size
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
                # Try to scale default font if possible
                font = font.font_variant(size=font_size)
            except:
                font = ImageFont.load_default()
        
        # Position text above the face box
        text_y = max(0, top - font_size - 5)  # 5px padding, don't go above image
        draw.text((left, text_y), f"Face {i+1}", fill="red", font=font)
    
    return pil_image

def serverless(encodings, db):
    results = find_matches(encodings[0], db)

    if not results:
        st.error("😕 No similar faces found.")
    else:
        st.success(f"✅ {len(results)} matches found!")
        for entry, dist in results[:10]:  # show top 10
            st.markdown(f"<small>**Similarity**: {dist:.4f}</small>", unsafe_allow_html=True)
            st.markdown(f"<small>[View image]({entry['image_url']})  \n[Visit page]({entry['page_url']})</small>", unsafe_allow_html=True)
            # Always try to show image, show placeholder if failed
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(entry['image_url'], timeout=30, headers=headers)
                response.raise_for_status()
                img_bytes = response.content
                st.image(Image.open(BytesIO(img_bytes)), width=250)
            except Exception:
                # Show placeholder image with error message
                placeholder_img = Image.new('RGB', (250, 200), color='lightgray')
                draw = ImageDraw.Draw(placeholder_img)
                draw.text((10, 90), "Image not\navailable", fill='darkgray')
                st.image(placeholder_img, width=250)

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
                        st.markdown(f"<small>**Distance**: {distance:.4f}</small>", unsafe_allow_html=True)
                        st.markdown(f"<small>[View image]({entry['image_url']})  \n[Visit page]({entry['page_url']})</small>", unsafe_allow_html=True)
                        img_url = entry['image_url']
                    elif isinstance(match, dict):
                        # Format: {"distance": x, "similarity": y, "image_url": z, ...}
                        st.markdown(f"<small>**Distance**: {match['distance']:.4f} | **Similarity**: {match['similarity']:.3f}</small>", unsafe_allow_html=True)
                        st.markdown(f"<small>[View image]({match['image_url']})  \n[Visit page]({match['page_url']})</small>", unsafe_allow_html=True)
                        img_url = match['image_url']
                    else:
                        st.warning("Unknown response format from server")
                        continue
                        
                    # Always try to show image, show placeholder if failed
                    try:
                        USER_AGENT = "FaceSearchBot/1.0 (Educational Research; Contact: github.com/Hawk3388/face-search-2.0)"
                        headers = {'User-Agent': USER_AGENT}
                        response = requests.get(img_url, timeout=30, headers=headers)
                        response.raise_for_status()
                        img_bytes = response.content
                        st.image(Image.open(BytesIO(img_bytes)), width=250)
                    except Exception:
                        # Show placeholder image with error message
                        placeholder_img = Image.new('RGB', (250, 200), color='lightgray')
                        draw = ImageDraw.Draw(placeholder_img)
                        draw.text((10, 90), "Image not\navailable", fill='darkgray')
                        st.image(placeholder_img, width=250)
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

# Get detailed health information
def get_health_info():
    """Get detailed server health information"""
    if not API_URL:
        return None
    try:
        headers = {}
        if TOKEN:
            headers['Authorization'] = f'Bearer {TOKEN}'
        response = requests.get(f"{API_URL}/health", headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}", "status": "error"}
    except Exception as e:
        return {"error": str(e), "status": "error"}

def main():
    # App
    st.set_page_config(page_title="Face Search", layout="centered")
    st.title("🔍 Face Search")

    # Initialize session state for health info
    if 'show_health' not in st.session_state:
        st.session_state.show_health = False
    if 'health_info' not in st.session_state:
        st.session_state.health_info = None

    # Health Check Section - elegant design without sidebar
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🏥 Check Server Health", type="secondary", use_container_width=True):
            with st.spinner("Checking server health..."):
                st.session_state.health_info = get_health_info()
                st.session_state.show_health = True

    path = "face_embeddings.json"

    if os.path.exists(path):
        db = load_database(path)

    # Determine mode automatically
    local_available = os.path.exists(path)
    
    if API_URL and TOKEN:
        # Only check API health if no local DB is available
        api_available = check_api_health()
        if api_available:
            use_server = True
            st.info("🌐 Using API server (all uploaded images are deleted immediately after usage)")
        else:
            st.error("❌ No local database found and API server is not reachable!")
            use_server = False
    elif local_available:
        use_server = False
        st.info(f"💻 Using local database ({len(db) if 'db' in locals() else 0} entries)")
    else:
        st.error("❌ No database file found and no API configured!")
        st.stop()
        use_server = False

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    # Hide health info whenever an image is uploaded (any image)
    if uploaded_file is not None:
        st.session_state.show_health = False

    # Display health info if available and should be shown (AFTER file uploader check)
    if st.session_state.show_health and st.session_state.health_info:
        health_info = st.session_state.health_info
        
        if health_info is None:
            st.error("❌ No API URL configured")
        elif "error" in health_info:
            st.error(f"❌ Server Error: {health_info['error']}")
        else:
            # Create a nice info box
            with st.container():
                st.markdown("---")
                
                # Server status with colored indicator
                if health_info.get("status") == "healthy":
                    st.markdown("🟢 **Server Status:** Healthy")
                else:
                    st.markdown(f"🟡 **Server Status:** {health_info.get('status', 'unknown')}")
                
                # Database info in columns
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    if health_info.get("database_loaded"):
                        entries = health_info.get('total_entries', 0)
                        st.metric(
                            label="📊 Database Entries", 
                            value=f"{entries:,}"
                        )
                    else:
                        st.metric(
                            label="📊 Database Status", 
                            value="Not Loaded"
                        )
                
                with info_col2:
                    last_page = health_info.get("last_page_url")
                    if last_page:
                        # Extract just the page title from URL
                        page_title = last_page.split("/")[-1].replace("_", " ")
                        if len(page_title) > 20:
                            page_title = page_title[:20] + "..."
                        st.metric(
                            label="📄 Last Page", 
                            value=page_title
                        )
                
                st.markdown("---")

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

        # Detect faces and get locations
        face_locations = face_recognition.face_locations(image, model="hog")
        encodings = face_recognition.face_encodings(image, face_locations, model="large", num_jitters=10)

        # Always show image with face detection boxes
        if face_locations:
            image_with_boxes = draw_face_boxes(image, face_locations)
            st.image(image_with_boxes, caption=f"Uploaded image - {len(face_locations)} face(s) detected", use_container_width=True)
        else:
            st.image(uploaded_file, caption="Uploaded image - No faces detected", use_container_width=True)

        if not encodings:
            st.warning("❌ No face detected in the image.")
        else:
            # Handle multiple faces
            if len(encodings) > 1:
                st.info(f"🔍 Found {len(encodings)} faces in the image. Select which face to search for:")
                
                cols = st.columns(len(encodings))
                selected_face = None
                
                for i, encoding in enumerate(encodings):
                    with cols[i]:
                        if st.button(f"Search Face {i+1}", key=f"face_{i}"):
                            selected_face = i
                
                if selected_face is not None:
                    st.info(f"🔎 Searching for Face {selected_face + 1}...")
                    # Use selected mode
                    if use_server and 'api_available' in locals() and api_available:
                        server([encodings[selected_face]])
                    elif not use_server and local_available:
                        serverless([encodings[selected_face]], db)
                    else:
                        st.error("❌ Selected mode not available!")
            else:
                # Single face - search automatically
                st.info("🔎 Searching for similar faces...")
                # Use selected mode
                if use_server and 'api_available' in locals() and api_available:
                    server(encodings)
                elif not use_server and local_available:
                    serverless(encodings, db)
                else:
                    st.error("❌ Selected mode not available!")

if __name__ == "__main__":
    main()
