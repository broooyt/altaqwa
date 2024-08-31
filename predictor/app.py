from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import io
from PIL import Image
import json
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model
model = load_model('predictor\\Models\\MobileNetV2_model.h5')

# Open and read the JSON file
with open('predictor\\Models\\classes.json', 'r') as file:
    indices_class = json.load(file)


# Define a route for the default URL, which loads the index page
@app.route('/')
def index():
    return "Image Classification API"

# Route for processing the image
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    if file:
        # Convert the FileStorage object to a PIL Image
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((224, 224))  # Ensure the image is resized to 224x224
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image/255.0
        # Make prediction
        prediction = model.predict(image)
        
        # Assuming you have a softmax output, return the class with the highest score
        predicted_class = np.argmax(prediction, axis=1)[0]
        print(indices_class[str(predicted_class)])
        return jsonify({"prediction": indices_class[str(predicted_class)]})

    return jsonify({"error": "File processing error"}), 500

