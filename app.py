from flask import Flask, request, jsonify
import joblib  # or pickle if you used pickle for saving the model

# Initialize Flask app
app = Flask(__name__)

# Load the ML model
model = joblib.load('drop_out_model.joblib')  # Update with the path to your saved model

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    data = request.get_json(force=True)
    
    # Extract features from data (ensure the structure matches your model's input)
    features = [data['feature1'], data['feature2'], data['feature3']]  # Update as necessary
    
    # Make a prediction
    prediction = model.predict([features])
    
    # Send back the result as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)