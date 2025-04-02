import joblib
import re
import numpy as np
from sklearn.base import BaseEstimator
from typing import Tuple

MODEL_PATH = r"C:\Users\Admin\Desktop\AML3\Applied-Machine-Learning\Assignment3\aml3\best_model.pkl"
VECTORIZER_PATH = r"C:\Users\Admin\Desktop\AML3\Applied-Machine-Learning\Assignment3\aml3\vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH) 

import re
from typing import Tuple
from sklearn.base import BaseEstimator

def clean_text(text: str) -> str:
    """Performs basic text cleaning by converting to lowercase and removing non-alphanumeric characters."""
    return re.sub(r'\W+', ' ', text.lower()).strip()

def score(text: str, model: BaseEstimator, threshold: float) -> Tuple[bool, float]:
    """
    Evaluates a given text sample using a trained model.

    Parameters:
        text (str): The input text for classification.
        model (BaseEstimator): A pre-trained scikit-learn model.
        threshold (float): A probability cutoff for classification.

    Returns:
        Tuple[bool, float]: A binary classification result and the associated probability score.
    """
    if not isinstance(text, str):
        raise ValueError("The input must be a string.")
    if not isinstance(threshold, float) or not (0 <= threshold <= 1):
        raise ValueError("Threshold should be a float between 0 and 1.")

    processed_text = clean_text(text)
    text_features = vectorizer.transform([processed_text])
    probability_score = model.predict_proba(text_features)[:, 1][0]  # Get probability of positive class
    classification_result = probability_score >= threshold  # Compare with threshold

    return bool(classification_result), float(probability_score)
