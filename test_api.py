import requests
import json

# Replace with your Vercel deployment URL
url = "https://drop-out-predictor.vercel.app/api/predict"

# Test data (update with your actual features)
test_data = {
    "marital_status": 1,
    "application_mode": 1,
    "course": 1,
    "previous_qualification": 1,
    "nationality": 1,
    "mothers_qualification": 1,
    "fathers_qualification": 1,
    "admission_grade": 120
    # Add all other features your model expects
}

# Make prediction request
response = requests.post(url, json=test_data)
print("Prediction Result:", response.json())