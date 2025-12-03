"""
Data loading and preprocessing for CSV files
"""
import pandas as pd
import os
from models import TeamProfile, LeagueAverages  # Add LeagueAverages import

class DataLoader:
    """Loads and processes CSV data files"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.available_leagues = self._detect_leagues()
    
    def _detect_leagues(self):
        """Detect available league files"""
        leagues = {}
        if not os.path.exists(self.data_dir):
            print(f"⚠️ Warning: Data directory '{self.data_dir}' not found!")
            return leagues
            
        for file in os.listdir(self.data_dir):
            if file.endswith("_home_away.csv"):
                league_name = file.replace("_home_away.csv", "")
                leagues[league_name] = os.path.join(self.data_dir, file)
        return leagues
    
    def calculate_league_averages(self, df: pd.DataFrame) -> LeagueAverages:
        """Calculate actual league averages from CSV data"""
        
        # Separate home and away data
        home_data = df[df['Home_Away'] == 'Home']
        away_data = df[df['Home_Away'] == 'Away']
        
        # Home team statistics
        total_home_goals = home_data['Goals'].sum()
        total_home_matches = home_data['Matches'].sum()
        
        # Away team statistics
        total_away_goals = away_data['Goals'].sum()
        total_away_matches = away_data['Matches'].sum()
        
        # Calculate averages
        avg_home_goals = total_home_goals / total_home_matches if total_home_matches > 0 else 1.5
        avg_away_goals = total_away_goals / total_away_matches if total_away_matches > 0 else 1.2
        
        # Calculate actual outcome rates from data
        total_home_wins = home_data['Wins'].sum()
        total_away_wins = away_data['Wins'].sum()
        total_draws = home_data['Draws'].sum()  # Same as away draws
        
        total_matches = (total_home_matches + total_away_matches) / 2
        
        actual_home_win_rate = total_home_wins / total_matches if total_matches > 0 else 0.45
        actual_away_win_rate = total_away_wins / total_matches if total_matches > 0 else 0.30
        actual_draw_rate = total_draws / total_matches if total_matches > 0 else 0.25
        
        # Create LeagueAverages object
        return LeagueAverages(
            avg_home_goals=round(avg_home_goals, 3),
            avg_away_goals=round(avg_away_goals, 3),
            total_matches=int(total_matches),
            actual_home_win_rate=round(actual_home_win_rate, 3),
            actual_draw_rate=round(actual_draw_rate, 3),
            actual_away_win_rate=round(actual_away_win_rate, 3)
        )
    
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
        
        # Calculate league averages
        league_averages = self.calculate_league_averages(df)
        
        return home_teams, away_teams, league_averages  # Return 3 values, not 2
    
    def get_team_profile(self, team_name, league_name, is_home=True):
        """Get TeamProfile object for a specific team"""
        home_df, away_df, league_averages = self.load_league_data(league_name)  # Get 3 values now
        
        if is_home:
            df = home_df
        else:
            df = away_df
        
        team_data = df[df['Team'] == team_name]
        
        if team_data.empty:
            raise ValueError(f"Team {team_name} not found in {league_name} {'home' if is_home else 'away'} data")
        
        # Convert to dictionary
        data_dict = team_data.iloc[0].to_dict()
        
        return TeamProfile(data_dict, is_home=is_home, league_averages=league_averages)  # Pass league_averages
    
    def get_all_teams(self, league_name):
        """Get list of all teams in a league"""
        home_df, _, _ = self.load_league_data(league_name)  # Get 3 values now
        return home_df['Team'].tolist()