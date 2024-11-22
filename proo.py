from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the datasets
calories = pd.read_csv(r'C:\Users\HP\Downloads\calories.csv')
exercise = pd.read_csv(r'C:\Users\HP\Downloads\exercise.csv')

# Combine and preprocess the dataset
calories_new = pd.concat([exercise, calories['Calories']], axis=1)
calories_new['Gender'] = calories_new['Gender'].replace({'male': 0, 'female': 1}).astype(int)


X = calories_new.drop(['User_ID', 'Calories'], axis=1)
Y = calories_new['Calories']

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        gender = request.form['gender']
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])

        # Convert gender to numeric
        gender = 0 if gender.lower() == 'male' else 1

        # Prepare the input array
        input_data = np.array([gender, age, height, weight, duration, heart_rate, body_temp]).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(input_data)

        return render_template('result.html', prediction=round(prediction[0], 2))
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
