import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("model/brain_tumor_model.keras")

# Define class labels
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def predict_tumor(img_path):
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class
        prediction = model.predict(img_array)[0]
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        return predicted_class, confidence
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Example usage
img_path = "test/no.jpg"  # Replace with your image path
label, confidence = predict_tumor(img_path)

if label:
    if label.lower() == "no_tumor":
        print("🧠 No Tumor Detected")
    else:
        print(f"🧠 Tumor Detected: {label.capitalize()} (Confidence: {confidence * 100:.2f}%)")
else:
    print("❌ Prediction failed. Check image path or model files.")
