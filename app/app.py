"""
Flask web app for the Smart Waste Classification model.

Run with:
    python app/app.py

Then open http://127.0.0.1:5000 and upload an image to classify it.
"""

import json
import os
import sys

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.preprocess import get_preprocess_fn

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

model = None
class_names = None
preprocess_fn = get_preprocess_fn()

# Simple disposal-guidance map — extend this as you like for your report/demo.
DISPOSAL_TIPS = {
    "battery": "Hazardous waste — drop off at a designated e-waste/battery collection point.",
    "biological": "Compostable — put in your organic/wet-waste bin.",
    "brown-glass": "Recyclable — rinse and place in glass recycling.",
    "cardboard": "Recyclable — flatten and place in paper/cardboard recycling.",
    "clothes": "Donate if wearable, otherwise use textile recycling.",
    "green-glass": "Recyclable — rinse and place in glass recycling.",
    "metal": "Recyclable — place in metal recycling.",
    "paper": "Recyclable — place in paper recycling.",
    "plastic": "Check local recycling code; rinse before placing in plastic recycling.",
    "shoes": "Donate if usable, otherwise textile/shoe recycling programs.",
    "trash": "General waste — landfill bin.",
    "white-glass": "Recyclable — rinse and place in glass recycling.",
}


def load_model():
    global model, class_names
    model = tf.keras.models.load_model(config.MODEL_PATH)
    with open(config.LABELS_PATH) as f:
        class_names = json.load(f)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(filepath):
    img = tf.keras.utils.load_img(filepath, target_size=config.IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_fn(arr)

    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    label = class_names[top_idx]
    confidence = float(preds[top_idx])

    # Top-3 for a nicer UI
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [(class_names[i], float(preds[i])) for i in top3_idx]

    return label, confidence, top3


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            label, confidence, top3 = predict_image(filepath)
            result = {
                "label": label,
                "confidence": round(confidence * 100, 2),
                "top3": [(name, round(p * 100, 2)) for name, p in top3],
                "tip": DISPOSAL_TIPS.get(label, "No disposal guidance available."),
            }
            image_path = f"uploads/{filename}"

    return render_template("index.html", result=result, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))