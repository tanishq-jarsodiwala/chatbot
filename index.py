from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# Load the dropout model
model_path = os.path.join(os.path.dirname(__file__), '../model/drop_out_model.joblib')
dropout_model = joblib.load(model_path)

@app.route('/')
def home():
    return "Student Dropout Prediction API is running!"

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Extract features (update these according to your model's features)
        features = [
            data['marital_status'],
            data['application_mode'],
            data['course'],
            data['previous_qualification'],
            data['nationality'],
            data['mothers_qualification'],
            data['fathers_qualification'],
            data['admission_grade']
            # Add all other features your model expects
        ]
        
        # Make prediction
        prediction = dropout_model.predict([features])
        
        # Convert prediction to response
        result = {
            'prediction': int(prediction[0]),
            'status': 'Dropout' if prediction[0] == 1 else 'Graduate'
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()