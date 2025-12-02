"""
Data loading and preprocessing for CSV files
"""
import pandas as pd
import os
from models import TeamProfile

class DataLoader:
    """Loads and processes CSV data files"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.available_leagues = self._detect_leagues()
    
    def _detect_leagues(self):
        """Detect available league files"""
        leagues = {}
        for file in os.listdir(self.data_dir):
            if file.endswith("_home_away.csv"):
                league_name = file.replace("_home_away.csv", "")
                leagues[league_name] = os.path.join(self.data_dir, file)
        return leagues
    
    def load_league_data(self, league_name):
        """Load data for a specific league"""
        if league_name not in self.available_leagues:
            raise ValueError(f"League {league_name} not found. Available: {list(self.available_leagues.keys())}")
        
        file_path = self.available_leagues[league_name]
        df = pd.read_csv(file_path)
        
        # Convert column names to standardized format
        df.columns = [col.strip() for col in df.columns]
        
        # Group by team and home/away
        home_teams = df[df['Home_Away'] == 'Home']
        away_teams = df[df['Home_Away'] == 'Away']
        
        return home_teams, away_teams
    
    def get_team_profile(self, team_name, league_name, is_home=True):
        """Get TeamProfile object for a specific team"""
        home_df, away_df = self.load_league_data(league_name)
        
        if is_home:
            df = home_df
        else:
            df = away_df
        
        team_data = df[df['Team'] == team_name]
        
        if team_data.empty:
            raise ValueError(f"Team {team_name} not found in {league_name} {'home' if is_home else 'away'} data")
        
        # Convert to dictionary
        data_dict = team_data.iloc[0].to_dict()
        
        return TeamProfile(data_dict, is_home=is_home)
    
    def get_all_teams(self, league_name):
        """Get list of all teams in a league"""
        home_df, _ = self.load_league_data(league_name)
        return home_df['Team'].tolist()
