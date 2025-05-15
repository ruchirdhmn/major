from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model.h5")  # Update with correct filename
classes = ['Ambience', 'Food', 'Menu']  # Adjust if needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    img = image.load_img(file_path, target_size=(224, 224))  # Match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    return f"Predicted Category: {predicted_class}"

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
