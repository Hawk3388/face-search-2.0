import cv2
import time
import face_recognition

# Haar-Cascade laden
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def has_face_recognition(image_path):
    img = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(img)
    return len(face_locations) > 0

def has_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gesichter erkennen – großzügiger eingestellt
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # feineres Scannen → mehr kleine/leichte Köpfe
        minNeighbors=2,    # lockerer → mehr Treffer, weniger strenge Filterung
        minSize=(20,20)    # kleinere Gesichter erkennen
    )
    
    return len(faces) > 0

# Beispielaufruf
bild = input("Path to image: ")
start_time = time.time()
if has_face(bild):
    print("True – Gesicht erkannt")
else:
    print("False – kein Gesicht erkannt")
print(f"Time: {time.time() - start_time:.2f} seconds")