with open("face_embeddings.json", "rb") as f:
    data = f.read()

cleaned = data.replace(b"\x00", b"")

with open("face_embeddings.json", "wb") as f:
    f.write(cleaned)