#!/usr/bin/env python3
"""
Simple script to remove 'face_index' field from face_embeddings.json
"""

import json

def remove_face_index():
    # Load database
    with open('face_embeddings.json.backup_20250913_132342', 'r') as f:
        data = json.load(f)
    
    # Remove face_index from all entries
    for entry in data:
        if 'face_index' in entry:
            del entry['face_index']
    
    # Save database with original formatting
    with open('face_embeddings.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("face_index removed from database")

if __name__ == "__main__":
    remove_face_index()