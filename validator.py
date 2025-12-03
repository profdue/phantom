"""
PHANTOM v4.3 - Simple Validator
"""
import pandas as pd
from datetime import datetime
from typing import Dict, List

class ModelValidator:
    """Simple validator for tracking predictions"""
    
    def __init__(self):
        self.predictions = []
    
    def add_prediction(self, prediction: Dict):
        """Store prediction for tracking"""
        self.predictions.append({
            "timestamp": datetime.now(),
            "prediction": prediction
        })
        print(f"✅ Prediction logged to validator ({len(self.predictions)} total)")
    
    def get_predictions_count(self):
        """Get total predictions count"""
        return len(self.predictions)
