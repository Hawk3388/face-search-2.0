import numpy as np
import os
from PIL import Image
import time
import face_recognition

model = None

if os.path.exists("tinyfacenet_best.pth"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms

    # Gleiche Transforms wie im Training!
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class TinyFaceNet_inference(nn.Module):
        def __init__(self):
            super(TinyFaceNet_inference, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.bn1   = nn.BatchNorm2d(16)
            self.pool1 = nn.MaxPool2d(2, 2)   # 64x64

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.bn2   = nn.BatchNorm2d(32)
            self.pool2 = nn.MaxPool2d(2, 2)   # 32x32

            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.bn3   = nn.BatchNorm2d(64)
            self.pool3 = nn.MaxPool2d(2, 2)   # 16x16

            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.bn4   = nn.BatchNorm2d(128)
            self.pool4 = nn.MaxPool2d(2, 2)   # 8x8

            self.fc1 = nn.Linear(128 * 8 * 8, 256)
            self.dropout = nn.Dropout(0.5)  # 50% Dropout
            self.fc2 = nn.Linear(256, 1)

        def forward(self, x):
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = self.pool4(F.relu(self.bn4(self.conv4(x))))
            x = x.view(-1, 128 * 8 * 8)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.sigmoid(self.fc2(x))
            x = x.squeeze().item()
            x = x >= 0.5
            x = not x
            return x

    model = TinyFaceNet_inference().to(device)
    model.load_state_dict(torch.load("tinyfacenet_best.pth", map_location=device))
    model.eval()

def prepare_image_for_model(np_image, device="cpu"):
    # NumPy (H, W, 3) -> PIL.Image
    pil_image = Image.fromarray(np_image)

    # PIL -> Tensor (1, 3, 128, 128)
    tensor_image = transform(pil_image).unsqueeze(0).to(device)

    return tensor_image

def image_bytes_contains_face(image):
    try:
        if model:
            tensor_image = prepare_image_for_model(image, device)
            with torch.no_grad():
                start_time = time.time()
                output = model(tensor_image)
                end_time = time.time()
                print(f"Inference time: {end_time - start_time:.4f} seconds")
                return output
        else:
            return None
    except Exception as e:
        print(f"Error checking image: {e}")
        return False
    
def main():
    img_path = input("Enter the path to the image file to test: ").strip()

    if not os.path.exists(img_path):
        print(f"Image file not found: {img_path}")
        return

    start_time = time.time()

    try:
        pil_image = Image.open(img_path).convert('RGB')
        np_image = np.array(pil_image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    result = image_bytes_contains_face(np_image)
    if result is None:
        print("Model not loaded.")
    elif result:
        print("Face detected in image.")
    else:
        print("No face detected in image.")
    print(f"Total time with model: {time.time() - start_time:.4f} seconds")

    start_time = time.time()

    try:
        image = face_recognition.load_image_file(img_path)
    except Exception as e:
        print(f"Error loading image for face_recognition: {e}")
        return
    face_locations = face_recognition.face_locations(image)

    if face_locations:
        print(f"Face locations found: {face_locations}")
    else:
        print("No faces found.")

    print(f"Total time with face_recognition: {time.time() - start_time:.4f} seconds")

if __name__ == "__main__":
    main()