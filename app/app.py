from flask import Flask, render_template, request
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploaded_images')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
loaded_model = load_model("model.keras")
img_size = 224

# List of disease names (order must match your model's output)
disease_names = ['Acitinic Keratosis', 'Basal Cell Carcinoma', 'Melanoma', 'Nevus', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis']

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found")
    # Convert BGR to RGB, resize, and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return img / 255.0

def predict_single_image(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    prediction = loaded_model.predict(img)
    predicted_class = np.argmax(prediction)
    class_label = disease_names[predicted_class]
    return class_label

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_label = None
    image_path = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img.jpg')
            file.save(save_path)
            predicted_label = predict_single_image(save_path)
            image_path = '/' + save_path.replace("\\", "/")
    return render_template("index.html", predicted_label=predicted_label, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
