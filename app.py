import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create a Flask app
app = Flask(__name__)

# Load the pickled model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    return render_template('predict.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        print(prediction)
        output = round(prediction[0], 2)
        print(output)

    if output == 0:
        return render_template('predict.html', prediction_text='The patient has Malignant neoplasm')
    else:
        return render_template('predict.html', prediction_text='The patient has Benign neoplasm')


if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)
