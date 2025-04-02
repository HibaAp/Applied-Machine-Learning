import joblib
import pytest
from score import score

# Load the trained model
model = joblib.load(r"C:\Users\Admin\Desktop\AML3\Applied-Machine-Learning\Assignment3\aml3\best_model.pkl")

def test_score():
    """Unit test for score function"""
    spam_text = "Congratulations! You won a free lottery. Click here to claim."
    non_spam_text = "Hello, how are you doing today?"
    
    # Smoke Test
    assert score(spam_text, model, 0.5) is not None
    
    # Format Test
    prediction, propensity = score(spam_text, model, 0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)
    
    # Prediction should be 0 or 1
    assert prediction in [True, False]
    
    # Propensity should be between 0 and 1
    assert 0 <= propensity <= 1
    
    # Edge cases: threshold = 0 (always True), threshold = 1 (always False)
    assert score(spam_text, model, 0.0)[0] == True
    assert score(spam_text, model, 1.0)[0] == False
    
    # Check obvious cases
    assert score(spam_text, model, 0.5)[0] == True  # Spam text
    assert score(non_spam_text, model, 0.5)[0] == False  # Non-spam text


import os
import time
import requests

def test_flask():
    """Integration test for Flask API"""
    # Start Flask app in the background
    os.system("python app.py &")
    time.sleep(2)  # Give server time to start
    
    url = "http://127.0.0.1:5000/score"
    data = {"text": "Congratulations! You've won $1,000,000."}
    response = requests.post(url, json=data)
    
    assert response.status_code == 200
    json_response = response.json()
    
    assert "prediction" in json_response
    assert "propensity" in json_response
    assert isinstance(json_response["prediction"], bool)
    assert 0 <= json_response["propensity"] <= 1
    
    # Kill the Flask app
    os.system("pkill -f app.py")

