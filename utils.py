"""
Utility functions for the prediction system
"""
import json
from datetime import datetime

class PredictionUtils:
    """Utility functions for predictions"""
    
    @staticmethod
    def format_prediction_output(prediction_result):
        """Format prediction result for display"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "match": f"{prediction_result['analysis'].get('home_team', 'Home')} vs {prediction_result['analysis'].get('away_team', 'Away')}",
            **prediction_result
        }
        return output
    
    @staticmethod
    def save_prediction_to_file(prediction, filename="predictions.json"):
        """Save prediction to JSON file"""
        with open(filename, 'a') as f:
            json.dump(prediction, f, indent=2)
            f.write('\n')
    
    @staticmethod
    def calculate_performance_metrics(predictions_history):
        """Calculate model performance metrics"""
        if not predictions_history:
            return {}
        
        metrics = {
            "total_predictions": len(predictions_history),
            "markets": {
                "winner": {"correct": 0, "total": 0},
                "total_goals": {"correct": 0, "total": 0},
                "btts": {"correct": 0, "total": 0}
            }
        }
        
        for pred in predictions_history:
            # This would need actual results to compare
            # Placeholder for implementation
            pass
        
        return metrics
    
    @staticmethod
    def validate_csv_structure(df):
        """Validate CSV structure has required columns"""
        required_columns = [
            'Team', 'Matches', 'Home_Away', 'Wins', 'Draws', 'Losses',
            'Goals', 'Goals_Against', 'Points', 'xG', 'xGA', 'xPTS'
        ]
        
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return True
