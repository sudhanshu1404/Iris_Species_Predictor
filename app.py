from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_model.joblib')
species = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width']),
        ]
        prediction = model.predict([features])[0]
        result = species[prediction]
        return render_template('result.html', prediction=result)
    except Exception as e:
        return render_template('result.html', prediction="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
