from flask import Flask, render_template, request
from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from pywebio.input import input_group, input, FLOAT
from pywebio.output import put_text
import numpy as np
from tensorflow import keras

# Initialize Flask app
app = Flask(__name__)

# Load the saved TensorFlow/Keras model
loaded_model = keras.models.load_model('model.h5')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
def predict():
    data = input_group("Food Delivery Time Prediction",
                       inputs=[
                           input("Age of Delivery Partner:", type=FLOAT, name='age'),
                           input("Ratings of Previous Deliveries:", type=FLOAT, name='ratings'),
                           input("Total Distance:", type=FLOAT, name='distance')
                       ])
    
    age = data['age']
    ratings = data['ratings']
    distance = data['distance']

    features = np.array([[age, ratings, distance]])
    predicted_time = loaded_model.predict(features)

    put_text("Predicted Delivery Time in Minutes:", predicted_time[0][0])

app.add_url_rule('/predict', 'webio_view', webio_view(predict),
                 methods=['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
