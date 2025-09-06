import json

with open('face_embeddings.json', 'r') as f:
    face_embeddings = json.load(f)

print("Anzahl der Gesichter:", len(face_embeddings))
