from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid
import gdown

"""def download_model():
    model_path = "model/brain_tumor_model.keras"

    # Check if the model file already exists
    if not os.path.exists(model_path):
        file_id = "1xEKPGXrj7Q0ML8jC1kfnCXj0Lzhcsx38"  # Replace with your actual file ID
        url = f"https://drive.google.com/uc?id={file_id}"

        # Create the model folder if it doesn't exist
        os.makedirs("model", exist_ok=True)

        print("Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
    else:
        print("Model file already exists, skipping download.")


# Call this function before using the model
download_model()
"""
# Initialize Flask app
app = Flask(__name__)

# Load trained brain tumor classification model
model = load_model("model/brain_tumor_model.keras")

# Define class labels
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img_file = request.files.get("image")
        if img_file and img_file.filename:
            # Generate a unique filename and save image
            filename = f"{uuid.uuid4().hex}.png"
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            img_file.save(img_path)

            # Preprocess image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)[0]
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = float(np.max(prediction)) * 100

            result_text = (
                "🧠 No Tumor Detected"
                if predicted_class == "no_tumor"
                else f"🧠 Tumor Detected: {predicted_class.capitalize()} ({confidence:.2f}%)"
            )

            # Render result page
            return render_template(
                "result.html",
                image_path=f"uploads/{filename}",  # Relative to /static
                result=result_text,
                confidence=confidence,
                label=predicted_class.capitalize()
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
