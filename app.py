from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

model = load_model('model_50.h5')

labels = ['animal', 'cartoon', 'floral', 'geometry', 'ikat', 'plain', 'polka dot', 'squares', 'stripes']

def get_model_prediction(image_path):
    img = load_img(image_path, target_size=(255, 255))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalization
    predictions = model.predict(x, verbose=0)
    return labels[predictions.argmax()]

@app.route('/')
def Home():
    return render_template("home.html")

@app.route('/predict_page')
def predict():
    return render_template("predict.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/predict', methods=['POST'])
def prediction():
    img = request.files['ump_image']
    img_path = "static/assets/uploads/" + img.filename
    img.save(img_path)
    p = get_model_prediction(img_path)
    print(f"Predicted: {p}, Probabilities: {model.predict(np.expand_dims(img_to_array(load_img(img_path, target_size=(255, 255))) / 255.0, axis=0), verbose=0)}")
    return render_template("predictionpage.html", img_path=img_path, prediction=p)

if __name__ == "__main__":
    app.run(debug=True)