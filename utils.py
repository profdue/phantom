"""
Utility functions for the prediction system
"""
import pandas as pd
import os
import json
from datetime import datetime

def get_available_leagues():
    """Get list of available league CSV files"""
    data_dir = "data"
    leagues = {}
    
    if not os.path.exists(data_dir):
        return leagues
    
    for file in os.listdir(data_dir):
        if file.endswith("_home_away.csv"):
            league_name = file.replace("_home_away.csv", "")
            leagues[league_name] = os.path.join(data_dir, file)
    
    return leagues

def load_league_data(league_name):
    """Load CSV data for a specific league"""
    leagues = get_available_leagues()
    
    if league_name not in leagues:
        raise ValueError(f"League {league_name} not found. Available: {list(leagues.keys())}")
    
    file_path = leagues[league_name]
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]
    
    home_teams = df[df['Home_Away'] == 'Home']
    away_teams = df[df['Home_Away'] == 'Away']
    
    return home_teams, away_teams

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
