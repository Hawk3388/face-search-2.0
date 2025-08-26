import os
import random
import shutil
from datasets import load_dataset
import zipfile
import requests
from tqdm import tqdm

# --------------------
# Helper Funktionen
# --------------------
def download_file(url, dest_path):
    """Lädt eine Datei mit Fortschrittsbalken herunter"""
    if os.path.exists(dest_path):
        print(f"✔ {dest_path} existiert schon, überspringe Download.")
        return dest_path

    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    return dest_path

def extract_zip(zip_path, extract_to):
    """Entpackt eine Zip-Datei"""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --------------------
# Hauptskript
# --------------------
def prepare_dataset():
    base_dir = "images"
    face_dir = os.path.join(base_dir, "faces")
    noface_dir = os.path.join(base_dir, "no_faces")

    ensure_dir(face_dir)
    ensure_dir(noface_dir)

    # --- 1. WIDER FACE (Faces)
    wider_url = "https://huggingface.co/datasets/wider_face/resolve/main/WIDER_train.zip"
    wider_zip = "WIDER_train.zip"
    download_file(wider_url, wider_zip)
    extract_zip(wider_zip, "wider")

    # Kopiere ein Subset von Gesichtern
    wider_images = []
    for root, _, files in os.walk("wider/WIDER_train/images"):
        for f in files:
            if f.lower().endswith((".jpg", ".png")):
                wider_images.append(os.path.join(root, f))

    print(f"Gefundene WIDER-FACE Bilder: {len(wider_images)}")
    subset_faces = random.sample(wider_images, 10000)  # Subset anpassen
    for i, src in enumerate(subset_faces):
        shutil.copy(src, os.path.join(face_dir, f"face_{i}.jpg"))

    # --- 2. BG-20k (No-Faces)
    print("Lade BG-20k Dataset für No-Faces...")
    no_faces_dataset = load_dataset("unography/BG-20k-1200px", split="train")

    num_no_faces = 10000  # Subset anpassen
    for i, sample in enumerate(no_faces_dataset):
        if i >= num_no_faces:
            break
        img = sample["image"]  # PIL.Image
        save_path = os.path.join(noface_dir, f"bg_{i:05d}.jpg")
        img.save(save_path, format="JPEG")

    print("✅ Dataset fertig vorbereitet!")
    print(f"- {len(os.listdir(face_dir))} Gesichter in {face_dir}")
    print(f"- {len(os.listdir(noface_dir))} Non-Faces in {noface_dir}")

if __name__ == "__main__":
    prepare_dataset()