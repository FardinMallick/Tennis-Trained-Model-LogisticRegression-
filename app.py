from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('play_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    outlook = request.form['outlook']
    temp = request.form['temp']
    humidity = request.form['humidity']
    wind = request.form['wind']

    # Encoding maps
    outlook_map = {
        'Overcast': 0,
        'Rain': 1,
        'Sunny': 2
    }

    temp_map = {
        'Cool': 0,
        'Hot': 1,
        'Mild': 2
    }

    humidity_map = {
        'High': 0,
        'Normal': 1
    }

    wind_map = {
        'Strong': 0,
        'Weak': 1
    }

    # Convert input to numbers
    features = [[
        outlook_map[outlook],
        temp_map[temp],
        humidity_map[humidity],
        wind_map[wind]
    ]]

    # Prediction
    prediction = model.predict(features)

    # Final output
    if prediction[0] == 1:
        result = "✅ Yes, Play Tennis"
    else:
        result = "❌ No, Do Not Play Tennis"

    return render_template(
        'index.html',
        prediction=result,
        selected_outlook=outlook,
        selected_temp=temp,
        selected_humidity=humidity,
        selected_wind=wind
    )


if __name__ == '__main__':
    app.run(debug=True)