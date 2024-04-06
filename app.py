from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib

app = Flask(__name__)
classifier = joblib.load("fake_image_detector_model.pkl")

def predict_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    image_flat = image.reshape(1, -1)
    prediction = classifier.predict(image_flat)[0]
    return "Real" if prediction == 1 else "Fake"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image_file = request.files['image']
    image_np = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Failed to decode image'})

    prediction = predict_image(image)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

