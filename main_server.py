from flask import Flask, request, jsonify
import numpy as np
import json
import os
import re
import shutil
import time
# import threading
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
db_embeddings_matrix = None
db_norms_sq = None
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
    global db, db_embeddings_matrix, db_norms_sq
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"📊 [{timestamp}] Loading database...")
        
        shutil.copy2("face_embeddings.json", "face_embeddings_server.json")
        remove_nul_bytes()
        if not get_last_character() == ']':
            close_list()
            
        with open('face_embeddings_server.json', 'r', encoding='utf-8') as f:
            db = json.load(f)
        
        # Convert embeddings to numpy arrays and build global matrix once
        embeddings_list = []
        for entry in db:
            emb = np.array(entry["embedding"], dtype=np.float32)
            # Wir behalten embedding im JSON als Liste für jsonify
            embeddings_list.append(emb)
            
        if embeddings_list:
            db_embeddings_matrix = np.vstack(embeddings_list)
            db_norms_sq = np.sum(db_embeddings_matrix ** 2, axis=1)
        else:
            db_embeddings_matrix = None
            db_norms_sq = None
        
        print(f"✅ Database loaded: {len(db)} entries")
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        db = []
        db_embeddings_matrix = None
        db_norms_sq = None

def database_reload_scheduler():
    """Background thread that reloads database every 24 hours"""
    try:
        while True:
            time.sleep(24 * 60 * 60)  # Wait 24 hours (86400 seconds)
            print("🔄 Scheduled database reload (24h timer)")
            load_database()
    except Exception as e:
        print(f"Error in database reload scheduler: {e}")

# Optimized vectorized search
def find_matches(query_embedding, db, db_matrix, db_norms_sq, tolerance=0.5, max_results=10):
    if not db or db_matrix is None or db_norms_sq is None:
        return []
    
    # Convert query to float32 to match database
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    
    # 1. Use squared euclidean distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2(a.b)
    # Matrix multiplication uses BLAS and is significantly faster than array subtraction
    query_norm_sq = np.sum(query_embedding ** 2)
    dot_products = np.dot(db_matrix, query_embedding)
    
    # Calculate squared distances (use np.maximum to avoid small negative values from floating point errors)
    squared_distances = np.maximum(0, db_norms_sq + query_norm_sq - 2 * dot_products)
    tolerance_sq = tolerance ** 2
    
    # 2. Filter by tolerance and get indices
    valid_indices = np.where(squared_distances <= tolerance_sq)[0]
    
    if len(valid_indices) == 0:
        return []
        
    valid_distances = squared_distances[valid_indices]
    
    # 3. Use argpartition for O(n) partial sort instead of O(n log n) full sort
    if len(valid_indices) > max_results:
        # Get the top K smallest distances (unordered)
        top_k_idx = np.argpartition(valid_distances, max_results - 1)[:max_results]
        # Sort just the top K to have them strictly ordered
        top_k_sorted_idx = top_k_idx[np.argsort(valid_distances[top_k_idx])]
        sorted_indices = valid_indices[top_k_sorted_idx]
    else:
        # If fewer than max_results, just sort them all
        sorted_indices = valid_indices[np.argsort(valid_distances)]
    
    # Return results, taking the square root only for the final matched distances
    return [(db[i], float(np.sqrt(squared_distances[i]))) for i in sorted_indices]

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
        matches = find_matches(encoding, db, db_embeddings_matrix, db_norms_sq)
        
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
    # reload_thread = threading.Thread(target=database_reload_scheduler, daemon=True)
    # reload_thread.start()
    # print("⏰ Database auto-reload scheduled every 24 hours")
    
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
