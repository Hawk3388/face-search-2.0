from flask import Flask, request, jsonify
import numpy as np
import json
import os
import re
import shutil
import time
import threading
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Get authentication token from environment
AUTH_TOKEN = os.getenv('TOKEN')
if not AUTH_TOKEN:
    print("⚠️ WARNING: No TOKEN found in environment variables. Server will be publicly accessible!")

def verify_token():
    """Verify authentication token from request headers"""
    if not AUTH_TOKEN:
        return True  # If no token is configured, allow access
    
    token = request.headers.get('Authorization')
    if not token:
        return False
    
    # Remove "Bearer " prefix if present
    if token.startswith('Bearer '):
        token = token[7:]
    
    return token == AUTH_TOKEN

# Global database
db = None
request_count = 0

def load_stats():
    """Load request statistics from file"""
    global request_count
    try:
        if os.path.exists('stats.txt'):
            with open('stats.txt', 'r') as f:
                request_count = int(f.read().strip())
                print(f"📊 Loaded request count: {request_count}")
        else:
            request_count = 0
    except Exception as e:
        print(f"⚠️ Error loading stats: {e}")
        request_count = 0

def save_stats():
    """Save request statistics to file"""
    try:
        with open('stats.txt', 'w') as f:
            f.write(str(request_count))
    except Exception as e:
        print(f"⚠️ Error saving stats: {e}")

def increment_request_count():
    """Increment request counter and save to file"""
    global request_count
    request_count += 1
    save_stats()

def remove_nul_bytes():
    with open("face_embeddings_server.json", "rb") as f:
        data = f.read()

    cleaned = data.replace(b"\x00", b"")

    with open("face_embeddings_server.json", "wb") as f:
        f.write(cleaned)

def get_last_character(file_path="face_embeddings_server.json"):
    """Extract only the last character from a JSON file"""
    try:
        with open(file_path, 'rb') as f:
            f.seek(-1, 2)  # Seek to 1 byte before end of file
            last_byte = f.read(1)
            return last_byte.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading last character: {e}")
        return None

def close_list(file_path="face_embeddings_server.json"):
    """Append a closing bracket to the end of a JSON file"""
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(']')
        print(f"✅ Appended ']' to {file_path}")
    except Exception as e:
        print(f"❌ Error appending to file: {e}")

def load_database():
    """Load the face embeddings database"""
    global db
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"📊 [{timestamp}] Loading database...")
        
        shutil.copy2("face_embeddings.json", "face_embeddings_server.json")
        remove_nul_bytes()
        if not get_last_character() == ']':
            close_list()
        with open('face_embeddings_server.json', 'r', encoding='utf-8') as f:
            db = json.load(f)
        print(f"✅ Database loaded: {len(db)} entries")
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        db = []

def database_reload_scheduler():
    """Background thread that reloads database every 24 hours"""
    while True:
        time.sleep(24 * 60 * 60)  # Wait 24 hours (86400 seconds)
        print("🔄 Scheduled database reload (24h timer)")
        load_database()

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

def extract_last_page_url(file_path, max_read_bytes=50000):
    """
    Reads the file from the end and extracts the last 'page_url'.
    max_read_bytes: Bytes to read backwards (increase if needed).
    """
    try:
        with open(file_path, 'rb') as f:
            f.seek(0, 2)  # Go to the end
            file_size = f.tell()
            read_size = min(max_read_bytes, file_size)
            f.seek(file_size - read_size)
            data_bytes = f.read()
            if data_bytes is None:
                print("⚠️ File read returned None - possible file corruption")
                return None
            data = data_bytes.decode('utf-8', errors='ignore')
        
        # Search for the last "page_url" in the data
        # Regex: "page_url": "VALUE"
        match = re.search(r'"page_url"\s*:\s*"([^"]+)"', data)
        if match:
            return match.group(1)  # Extract the value
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

@app.route('/search', methods=['POST'])
def search_faces():
    """Search for similar faces using face encoding"""
    try:
        # Verify authentication
        if not verify_token():
            return jsonify({'error': 'Unauthorized: Invalid or missing token'}), 401
        
        # Increment request counter
        increment_request_count()
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate embedding
        encoding = data.get('encoding')
        if not encoding:
            return jsonify({'error': 'Missing face encoding'}), 400
        
        if not isinstance(encoding, list) or len(encoding) != 128:
            return jsonify({'error': 'Invalid encoding format (must be 128-dimensional array)'}), 400
        
        # Search for matches
        matches = find_matches(encoding, db)
        
        return jsonify({
            'success': True,
            'matches': matches,
            'total_results': len(matches),
        })
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database_loaded': db is not None,
        'total_entries': len(db) if db else 0,
        'last_page_url': extract_last_page_url('face_embeddings_server.json'),
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get database and request statistics"""
    # Verify authentication for stats endpoint
    if not verify_token():
        return jsonify({'error': 'Unauthorized: Invalid or missing token'}), 401
        
    return jsonify({
        'total_entries': len(db) if db else 0,
        'database_file_size': os.path.getsize('face_embeddings_server.json') if os.path.exists('face_embeddings_server.json') else 0,
        'total_requests': request_count,
    })

if __name__ == '__main__':
    # Load database on startup
    load_database()
    load_stats()
    
    # Start background thread for database reloading every 24h
    reload_thread = threading.Thread(target=database_reload_scheduler, daemon=True)
    reload_thread.start()
    print("⏰ Database auto-reload scheduled every 24 hours")
    
    # Run server
    port = int(os.environ.get('PORT', 7403))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"🚀 Starting Face Search API on port {port}")
    print(f"📊 Database entries: {len(db) if db else 0}")
    
    if AUTH_TOKEN:
        print(f"🔒 API is secured with token authentication")
    else:
        print(f"⚠️ API is publicly accessible (no token configured)")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
