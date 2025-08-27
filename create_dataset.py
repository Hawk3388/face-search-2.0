import os
import random
import shutil
import zipfile
from datasets import load_dataset
import requests
from tqdm import tqdm

# --------------------
# Helper Funktionen
# --------------------
def download_file(url, dest_path):
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

    # --- 1. WIDER FACE (Faces) ---
    wider_url = "https://huggingface.co/datasets/wider_face/resolve/main/WIDER_train.zip"
    wider_zip = "WIDER_train.zip"
    download_file(wider_url, wider_zip)
    extract_zip(wider_zip, "wider")

    # Flatten all images from subfolders
    source_faces = os.path.join("wider", "WIDER_train", "images")
    all_images = []
    for root, _, files in os.walk(source_faces):
        for f in files:
            if f.lower().endswith((".jpg", ".png")):
                all_images.append(os.path.join(root, f))

    subset_faces = random.sample(all_images, 10000)
    for i, src in enumerate(subset_faces):
        shutil.copy(src, os.path.join(face_dir, f"face_{i:05d}.jpg"))

    # --- 2. BG-20k (No-Faces) ---
    print("Loading BG-20k dataset for No-Faces...")
    no_faces_dataset = load_dataset("unography/BG-20k-1200px", split="train")

    # Randomly select 10,000 images
    all_no_faces = list(no_faces_dataset)
    subset_no_faces = random.sample(all_no_faces, 10000)

    for i, sample in enumerate(subset_no_faces):
        img = sample["image"]
        save_path = os.path.join(noface_dir, f"bg_{i:05d}.jpg")
        img.save(save_path, format="JPEG")

    print("✅ Dataset prepared!")
    print(f"- {len(os.listdir(face_dir))} Faces in {face_dir}")
    print(f"- {len(os.listdir(noface_dir))} No-Face images in {noface_dir}")

if __name__ == "__main__":
    prepare_dataset()