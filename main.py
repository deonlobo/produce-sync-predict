from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)

CORS(app)  # Add this line to enable CORS for all routes

# Load the trained model
model = load_model('resnet_model.h5')

# Define class labels
class_labels = {
    0: "Ripened",
    1: "Half Ripened",
    2: "Green"
}

# Define class labels
price_seg = {
    0: 1.5, # "Ripened"
    1: 1.98, # "Half Ripened"
    2: 1.2 # "Green"
}

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']

    # Read the image file
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Reshape the image to match the input shape of the model
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    # Make predictions
    predictions = model.predict(preprocessed_image)

    # Get the predicted class
    predicted_class = np.argmax(predictions)

    # Get the predicted class label
    predicted_label = class_labels.get(predicted_class, "Unknown")
    price_segmentation = price_seg.get(predicted_class, "Unknown")
    
    print("Predicted class is ",predicted_label)
    # Return the predicted class label as JSON response

    return jsonify({'predicted_label': predicted_label, 'price_segmentation': price_segmentation, 'unit': 'lb'})

def preprocess_image(image, target_size=(224, 224)):
    # Calculate aspect ratio
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size
    aspect_ratio = original_width / original_height

    # Determine new dimensions while preserving aspect ratio
    if aspect_ratio > 1:  # Landscape orientation
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:  # Portrait or square orientation
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    # Resize the image
    image = cv2.resize(image, (new_width, new_height))

    # Pad the image if necessary to match the target size
    top_pad = (target_height - new_height) // 2
    bottom_pad = target_height - new_height - top_pad
    left_pad = (target_width - new_width) // 2
    right_pad = target_width - new_width - left_pad
    image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)

    # Normalize image
    image = image / 255.0

    return image

if __name__ == '__main__':
    app.run(debug=True)
