from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

# Create a Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

# Define the translation dictionary
translate = {
    0: "dog",
    1: "horse",
    2: "elephant",
    3: "butterfly",
    4: "chicken",
    5: "cat",
    6: "cow",
    7: "sheep",
    8: "spider",
    9: "squirrel",
}
# Define a route for image classification
@app.route('/', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        # Get the uploaded image from the request
        uploaded_image = request.files['image']

        # Preprocess the image
        img = preprocess_image(uploaded_image)

        # Perform image classification
        prediction = classify(img)

        # Translate the predicted class
        translated_prediction = translate.get(prediction, "Unknown")

        # Encode the image data as a Base64 data URL
        img_data_url = encode_image_as_base64(img)

        # Render the result page with the prediction and image data URL
        return render_template('result.html', prediction=translated_prediction, image=img_data_url)

    # Render the initial upload page
    return render_template('index.html')

# Define a function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image)
    # Perform resizing and preprocessing as needed for your model
    img = img.resize((224, 224))  # Resize to match your model's input size
    img = np.asarray(img)
    img = img / 255.0  # Normalize pixel values

    return img

# Define a function to perform image classification
def classify(img):
    # Preprocess the image further if needed
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions using your model
    predictions = model.predict(img)

    # Get the class index with the highest probability
    prediction = np.argmax(predictions)

    return prediction

# Define a function to encode the image as Base64
def encode_image_as_base64(img):
    # Convert the NumPy array to a PIL image
    pil_image = Image.fromarray(np.uint8(img * 255))

    # Create a buffer to hold the image data
    buffer = io.BytesIO()

    # Save the PIL image to the buffer in JPEG format
    pil_image.save(buffer, format="JPEG")

    # Encode the image data as Base64
    img_data_url = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{img_data_url}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
