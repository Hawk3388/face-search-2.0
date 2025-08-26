import os
import random
import shutil
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

    # --- 1. WIDER FACE (nur Train Split, ~1.3GB gezippt)
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
    subset_faces = random.sample(wider_images, 10000)  # Subset, anpassbar
    for i, src in enumerate(subset_faces):
        shutil.copy(src, os.path.join(face_dir, f"face_{i}.jpg"))

    # --- 2. Places365 Subset für Non-Faces (~1.8GB)
    places_url = "http://data.csail.mit.edu/places/places365/val_256.tar"
    places_tar = "places365_val.tar"

    if not os.path.exists(places_tar):
        download_file(places_url, places_tar)
        os.system(f"tar -xf {places_tar} -C .")

    places_images = []
    for root, _, files in os.walk("val_256"):
        for f in files:
            if f.lower().endswith((".jpg", ".png")):
                places_images.append(os.path.join(root, f))

    print(f"Gefundene Places365 Bilder: {len(places_images)}")
    subset_nofaces = random.sample(places_images, 10000)
    for i, src in enumerate(subset_nofaces):
        shutil.copy(src, os.path.join(noface_dir, f"noface_{i}.jpg"))

    print("✅ Dataset fertig vorbereitet!")
    print(f"- {len(os.listdir(face_dir))} Gesichter in {face_dir}")
    print(f"- {len(os.listdir(noface_dir))} Non-Faces in {noface_dir}")


if __name__ == "__main__":
    prepare_dataset()