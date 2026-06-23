import streamlit as st
import face_recognition
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
import requests
import os

if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()
    API_URL = os.getenv("API_URL")
    TOKEN = os.getenv("TOKEN")
else:
    # Load configuration from Streamlit secrets
    try:
        API_URL = st.secrets["API_URL"]
        TOKEN = st.secrets["TOKEN"]
    except KeyError as e:
        st.error(f"Missing configuration in Streamlit secrets: {e}")
        API_URL = None
        TOKEN = None

# Load embedding database
@st.cache_data(show_spinner="Loading database into memory...")
def load_database(path="face_embeddings.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        
        # Convert embeddings to numpy arrays once during load (float32 for memory efficiency)
        for entry in data:
            if isinstance(entry["embedding"], list):
                entry["embedding"] = np.array(entry["embedding"], dtype=np.float32)
        
        return data
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return []

# Optimized vectorized search - 10-50x faster than loop
def find_matches(query_embedding, db, tolerance=0.5, max_results=10):
    if not db:
        return []
    
    # Convert query to float32 to match database
    query_embedding = query_embedding.astype(np.float32)
    
    # Stack all embeddings into a single matrix for vectorized computation
    db_embeddings = np.vstack([entry["embedding"] for entry in db])
    
    # Compute all distances at once (much faster than loop)
    distances = np.linalg.norm(db_embeddings - query_embedding, axis=1)
    
    # Filter by tolerance and get indices
    valid_indices = np.where(distances <= tolerance)[0]
    
    if len(valid_indices) == 0:
        return []
    
    # Sort by distance and take top results
    sorted_indices = valid_indices[np.argsort(distances[valid_indices])][:max_results]
    
    # Return results
    return [(db[i], float(distances[i])) for i in sorted_indices]

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
            # Direkt die URL an Streamlit übergeben, der Browser des Nutzers lädt dann das Bild.
            st.image(entry['image_url'], width=250)
            st.markdown(f"**Similarity**: {dist:.4f}")
            st.markdown(f"[View image]({entry['image_url']})  \n[Visit page]({entry['page_url']})")
            st.markdown("---")

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
                for match in results:  # show all results
                    # Handle different response formats from server
                    if isinstance(match, list) and len(match) >= 2:
                        # Format: [entry_dict, distance]
                        entry, distance = match[0], match[1]
                        img_url = entry['image_url']
                        # Direkt die URL an Streamlit übergeben, der Browser des Nutzers lädt dann das Bild.
                        st.image(img_url, width=250)
                        st.markdown(f"**Distance**: {distance:.4f}")
                        st.markdown(f"[View image]({entry['image_url']})  \n[Visit page]({entry['page_url']})")
                    elif isinstance(match, dict):
                        # Format: {"distance": x, "similarity": y, "image_url": z, ...}
                        img_url = match['image_url']
                        # Direkt die URL an Streamlit übergeben, der Browser des Nutzers lädt dann das Bild.
                        st.image(img_url, width=250)
                        st.markdown(f"**Distance**: {match['distance']:.4f} | **Similarity**: {match['similarity']:.3f}")
                        st.markdown(f"[View image]({match['image_url']})  \n[Visit page]({match['page_url']})")
                    else:
                        st.warning("Unknown response format from server")
                        continue
                        
                    st.markdown("---")
        else:
            st.error(f"API Error: {api_response.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error contacting API: {e}")

# Check API health
def check_api_health():
    if not API_URL:
        return False
    try:
        response = requests.get(f"{API_URL}/health", timeout=20)
        return response.status_code == 200
    except Exception:
        return False

# Get database stats from health endpoint
def get_health_stats():
    if not API_URL:
        return None
    try:
        response = requests.get(f"{API_URL}/health", timeout=20)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def main():
    # App
    st.set_page_config(page_title="Face Search", layout="centered")
    
    # Header with stats in a more elegant way
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔍 Face Search")
    with col2:
        with st.popover("📊 Stats", help="Database statistics"):
            # Use health API to get all info
            api_health_data = get_health_stats()
            
            if api_health_data:
                st.markdown("**🌐 API Database**")
                st.write(f"**Entries:** {api_health_data.get('total_entries', 'N/A'):,}")
                if 'last_page_url' in api_health_data:
                    page_name = api_health_data['last_page_url'].split("/")[-1].replace("_", " ")
                    st.write(f"**Last Crawled Page:** {page_name}")
                    st.link_button("🔗 View Page", api_health_data['last_page_url'])
            else:
                st.markdown("**💻 Local Database**")
                # Fallback to local stats
                path = "face_embeddings.json"
                if os.path.exists(path):
                    try:
                        db_temp = load_database(path)
                        entry_count = len(db_temp)
                        st.write(f"**Entries:** {entry_count:,}")
                    except:
                        st.write("**Entries:** Error loading DB")
                else:
                    st.write("**Entries:** No local DB")
                
                # Always check last crawled page locally
                try:
                    with open("last_crawled_page.txt", "r") as f:
                        last_page = f.read().strip()
                    page_name = last_page.split("/")[-1].replace("_", " ")
                    st.write(f"**Last:** {page_name}")
                    st.link_button("🔗 View Page", last_page)
                except:
                    st.write("**Last:** No data")

    path = "face_embeddings.json"

    # Determine mode automatically
    local_available = os.path.exists(path)
    use_server = False
    api_available = False
    
    if API_URL:
        api_available = check_api_health()
        if api_available:
            use_server = True
            st.info("🌐 Using API server (all uploaded images are deleted immediately after usage)")
        elif local_available:
            use_server = False
            st.info("💻 Using local database (Server not reachable)")
        else:
            st.error("❌ No local database found and API server is not reachable!")
            st.stop()
    elif local_available:
        use_server = False
        st.info("💻 Using local database")
    else:
        st.error("❌ No local database found and API server is not reachable!")
        st.stop()

    # Load local database ONLY if we are actually using it
    if not use_server and local_available:
        db = load_database(path)
        st.info(f"📊 Local Database Loaded: {len(db)} entries")

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

        # Detect faces and get locations
        face_locations = face_recognition.face_locations(image, model="hog")
        encodings = face_recognition.face_encodings(image, face_locations, model="large", num_jitters=10)

        # Always show image with face detection boxes
        if face_locations:
            image_with_boxes = draw_face_boxes(image, face_locations)
            st.image(image_with_boxes, caption=f"Uploaded image - {len(face_locations)} face(s) detected", width='stretch')
        else:
            st.image(uploaded_file, caption="Uploaded image - No faces detected", width='stretch')

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
                    if use_server and api_available:
                        server([encodings[selected_face]])
                    elif not use_server and local_available:
                        serverless([encodings[selected_face]], db)
                    else:
                        st.error("❌ Selected mode not available!")
            else:
                # Single face - search automatically
                st.info("🔎 Searching for similar faces...")
                # Use selected mode
                if use_server and api_available:
                    server(encodings)
                elif not use_server and local_available:
                    serverless(encodings, db)
                else:
                    st.error("❌ Selected mode not available!")

if __name__ == "__main__":
    main()
