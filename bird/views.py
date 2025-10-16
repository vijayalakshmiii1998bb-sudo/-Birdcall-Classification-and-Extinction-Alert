# bird/views.py
import numpy as np
from pathlib import Path
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .forms import ImageUploadForm
import tensorflow as tf
from PIL import Image

# ---- Load model and class names ----
MODEL_PATH = Path(settings.BASE_DIR) / "image_model.keras"
CLASS_PATH = Path(settings.BASE_DIR) / "bird" / "weights" / "class_names.txt"

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_PATH, "r") as f:
    CLASS_NAMES = [c.strip() for c in f.readlines() if c.strip()]

# ---- Predict function (for hybrid model expecting 2 inputs) ----
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    # Duplicate input for 2 branches
    preds = model.predict([arr, arr])[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx])
    return CLASS_NAMES[idx], confidence

# ---- Views ----
def index(request):
    return render(request, "index.html")

def home(request):
    return render(request, "home.html")

def image_upload(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data["image"]
            file_path = default_storage.save("uploads/" + file.name, ContentFile(file.read()))
            full_path = default_storage.path(file_path)

            predicted_class, confidence = predict_image(full_path)
            image_url = settings.MEDIA_URL + file_path

            return render(request, "result_image.html", {
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2),
                "image_url": image_url
            })
    else:
        form = ImageUploadForm()
    return render(request, "image_upload.html", {"form": form})