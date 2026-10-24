from io import BytesIO
import hashlib
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import DLIB_MODEL


def load_image(data, max_bytes=10 * 1024 * 1024, max_pixels=20_000_000):
    if not data or len(data) > max_bytes:
        raise ValueError("Image is empty or exceeds the upload limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.width * image.height > max_pixels:
                    raise ValueError("Image exceeds the pixel limit")
                return np.array(ImageOps.exif_transpose(image).convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid or excessively large image") from exc


class FaceEncoder:
    def __init__(self, settings):
        self.settings = settings
        self.model_id = DLIB_MODEL
        self.dimension = 128
        self.preprocessing = "dlib-rgb-large-v1"
        self.model = None
        if settings.encoder == "siamese":
            self._load_siamese(settings.model_path)
        elif settings.encoder != "dlib":
            raise ValueError("FACE_ENCODER must be dlib or siamese")

    def _load_siamese(self, path):
        import torch
        from model2.network import SiameseNetwork, inference_transform

        if not path:
            raise ValueError("MODEL_PATH is required for the siamese encoder")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.dimension = int(checkpoint.get("embedding_dim", state["projection.4.weight"].shape[0]))
        self.preprocessing = checkpoint.get("preprocessing", "face-crop-rgb224-imagenet-v1")
        if self.preprocessing != "face-crop-rgb224-imagenet-v1":
            raise ValueError("Unsupported checkpoint preprocessing")
        self.model = SiameseNetwork(self.dimension, pretrained=False)
        self.model.load_state_dict(state)
        self.model.eval()
        self.transform = inference_transform()
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        self.model_id = "siamese-" + digest.hexdigest()[:16] + "-crop-v1"

    def extract(self, image):
        import face_recognition

        locations = face_recognition.face_locations(image, model="hog")
        if len(locations) > self.settings.max_faces:
            raise ValueError(f"At most {self.settings.max_faces} faces per image are supported")
        return locations, self.encode_locations(image, locations)

    def encode_locations(self, image, locations):
        if self.model is None:
            import face_recognition

            return face_recognition.face_encodings(
                image, locations, model="large", num_jitters=self.settings.num_jitters
            )
        import torch

        if not locations:
            return []
        crops = [
            self.transform(Image.fromarray(image[top:bottom, left:right])) for top, right, bottom, left in locations
        ]
        with torch.inference_mode():
            return list(self.model.encode(torch.stack(crops)).cpu().numpy())
