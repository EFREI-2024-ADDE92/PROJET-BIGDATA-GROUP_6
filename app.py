from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model_path = "knn-model.joblib"

# Chargemenet du modèle
loaded_model = joblib.load(model_path)

@app.route('/')
def index():
    return "Bienvenue sur l'API Iris Prediction!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['feature1'], data['feature2'], data['feature3'], data['feature4']]
    prediction = loaded_model.predict([features])[0]
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(port=8000)



