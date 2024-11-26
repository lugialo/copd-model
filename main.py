from flask import Flask, request, jsonify
from joblib import load

model = load('./random_forest_model.joblib')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['features']
    print("Received features:", features)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)