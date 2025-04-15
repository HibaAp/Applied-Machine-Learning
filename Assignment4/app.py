import pytest
import joblib
import subprocess
import time
import requests
from score import score
import app

# Paths to the trained model
MODEL_PATH = r"C:\Users\Admin\Desktop\AML3\Applied-Machine-Learning\Assignment3\aml3\best_model.pkl"

# Load the trained model
model = joblib.load(MODEL_PATH)

def test_score():
    """Unit tests for the score function with additional cases."""

    # 1. Basic test - does the function execute properly?
    text = "Get a free vacation package now!"
    prediction, propensity = score(text, model, 0.5)
    
    assert isinstance(prediction, bool), "Prediction should be a boolean."
    assert isinstance(propensity, float), "Propensity should be a float."
    assert 0 <= propensity <= 1, "Propensity score should be between 0 and 1."

    # 2. Edge case - Empty input
    empty_text = ""
    prediction, propensity = score(empty_text, model, 0.5)
    assert prediction == 0, "Empty input should not be classified as spam."
    assert propensity == 0.0, "Propensity for empty input should be 0."

    # 3. Test with special characters and numbers
    text = "Win $$$ NOW!!! 100% Free!!!"
    prediction, _ = score(text, model, 0.5)
    assert prediction == 1, "Text with excessive special characters should be classified as spam."

    # 4. Test with non-English text (if supported)
    foreign_text = "Ganaste un premio! Reclámalo ahora!"
    prediction, _ = score(foreign_text, model, 0.5)
    assert prediction in [0, 1], "Model should handle non-English text gracefully."

    # 5. Long legitimate text (should not be classified as spam)
    long_text = (
        "Dear customer, we are writing to inform you about your recent order. "
        "If you have any questions, please contact our customer support."
    )
    prediction, _ = score(long_text, model, 0.5)
    assert prediction == 0, "Legitimate messages should not be classified as spam."

def test_flask():
    """Integration test for the Flask API with additional cases."""

    # Start the Flask application
    process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the server to initialize
    for _ in range(10):
        try:
            response = requests.get("http://127.0.0.1:5000/")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(1)

    url = "http://127.0.0.1:5000/score"
    headers = {"Content-Type": "application/json"}

    # 1. Valid spam request
    test_data = {"text": "Congratulations! You won $10,000! Claim now."}
    response = requests.post(url, json=test_data, headers=headers)

    assert response.status_code == 200, "API response should be successful."
    data = response.json()
    assert "prediction" in data and "propensity" in data, "Response must contain prediction and propensity."

    # 2. Valid non-spam request
    test_data = {"text": "Hello John, let's meet for coffee tomorrow."}
    response = requests.post(url, json=test_data, headers=headers)
    data = response.json()
    assert data["prediction"] == 0, "Clear non-spam messages should not be flagged."

    # 3. Edge case: Empty input
    test_data = {"text": ""}
    response = requests.post(url, json=test_data, headers=headers)
    assert response.status_code == 400, "Empty input should return a 400 error."

    # 4. Edge case: Missing 'text' field
    test_data = {}
    response = requests.post(url, json=test_data, headers=headers)
    assert response.status_code == 400, "Missing 'text' field should return a 400 error."

    # 5. Stress test: Very long input
    long_text = "Hello! " * 1000  # Repeating text
    test_data = {"text": long_text}
    response = requests.post(url, json=test_data, headers=headers)
    assert response.status_code == 200, "Long inputs should be processed successfully."

    # Stop the Flask app
    process.kill()
    stdout, stderr = process.communicate()
    print(stdout.decode(), stderr.decode())  # Print logs if needed

if __name__ == "__main__":
    pytest.main()
