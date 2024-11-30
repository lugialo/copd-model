from flask import Flask, request, jsonify
from joblib import load
from flask_cors import CORS

model = load('./random_forest_model.joblib')

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": ["https://copd-model-frontend.vercel.app", "http://localhost:3000"]}})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['features']
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)