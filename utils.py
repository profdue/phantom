"""
PHANTOM v4.1 - Data Loading & Utilities
"""
import pandas as pd
import os
import json
from typing import Dict, Tuple, Optional
from datetime import datetime
from models import LeagueAverages, TeamProfile

class DataLoader:
    """Load and process CSV data with league average calculations"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.available_leagues = self._detect_leagues()
    
    def _detect_leagues(self) -> Dict[str, str]:
        """Detect available league CSV files"""
        leagues = {}
        if not os.path.exists(self.data_dir):
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
        avg_home_goals = total_home_goals / total_home_matches if total_home_matches > 0 else 1.5
        
        # Away team statistics
        total_away_goals = away_data['Goals'].sum()
        total_away_matches = away_data['Matches'].sum()
        avg_away_goals = total_away_goals / total_away_matches if total_away_matches > 0 else 1.2
        
        # League averages per team per game
        total_goals = total_home_goals + total_away_goals
        total_matches = total_home_matches + total_away_matches
        league_avg_gpg = (total_goals / total_matches) / 2 if total_matches > 0 else 1.4
        
        # Home advantage multiplier
        home_advantage = avg_home_goals / avg_away_goals if avg_away_goals > 0 else 1.15
        
        # Calculate actual outcome rates from data
        total_home_wins = home_data['Wins'].sum()
        total_away_wins = away_data['Wins'].sum()
        total_draws = home_data['Draws'].sum()  # Same as away draws
        
        actual_home_win_rate = total_home_wins / total_matches if total_matches > 0 else 0.45
        actual_away_win_rate = total_away_wins / total_matches if total_matches > 0 else 0.30
        actual_draw_rate = total_draws / total_matches if total_matches > 0 else 0.25
        
        return LeagueAverages(
            avg_home_goals=round(avg_home_goals, 3),
            avg_away_goals=round(avg_away_goals, 3),
            league_avg_gpg=round(league_avg_gpg, 3),
            home_advantage=round(home_advantage, 3),
            total_matches=int(total_matches),
            actual_home_win_rate=round(actual_home_win_rate, 3),
            actual_draw_rate=round(actual_draw_rate, 3),
            actual_away_win_rate=round(actual_away_win_rate, 3)
        )
    
    def load_league_data(self, league_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, LeagueAverages]:
        """Load data for a specific league with calculated averages"""
        if league_name not in self.available_leagues:
            available = list(self.available_leagues.keys())
            raise ValueError(f"League {league_name} not found. Available: {available}")
        
        file_path = self.available_leagues[league_name]
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Validate required columns
        required_columns = ['Team', 'Matches', 'Home_Away', 'Wins', 'Draws', 'Losses',
                           'Goals', 'Goals_Against', 'Points', 'xG', 'xGA']
        
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Separate home and away
        home_teams = df[df['Home_Away'] == 'Home'].copy()
        away_teams = df[df['Home_Away'] == 'Away'].copy()
        
        # Calculate league averages
        league_averages = self.calculate_league_averages(df)
        
        return home_teams, away_teams, league_averages
    
    def get_all_teams(self, league_name: str) -> list:
        """Get list of all teams in a league"""
        home_df, away_df, _ = self.load_league_data(league_name)
        home_teams = home_df['Team'].unique().tolist()
        away_teams = away_df['Team'].unique().tolist()
        
        # Combine and deduplicate
        all_teams = list(set(home_teams + away_teams))
        return sorted(all_teams)
    
    def create_team_profile(self, team_name: str, league_name: str, 
                           is_home: bool = True) -> Optional[TeamProfile]:
        """Create TeamProfile object for a specific team"""
        try:
            home_df, away_df, league_averages = self.load_league_data(league_name)
            
            if is_home:
                df = home_df
            else:
                df = away_df
            
            team_data = df[df['Team'] == team_name]
            
            if team_data.empty:
                raise ValueError(f"Team {team_name} not found in {league_name} {'home' if is_home else 'away'} data")
            
            data_dict = team_data.iloc[0].to_dict()
            
            return TeamProfile(
                data_dict=data_dict,
                is_home=is_home,
                league_avg_gpg=league_averages.league_avg_gpg,
                league_averages=league_averages
            )
            
        except Exception as e:
            print(f"Error creating team profile: {e}")
            return None

class PredictionLogger:
    """Log predictions for validation and analysis"""
    
    def __init__(self, log_file: str = "predictions_log.json"):
        self.log_file = log_file
    
    def log_prediction(self, prediction_data: Dict):
        """Log a prediction with timestamp"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            **prediction_data
        }
        
        try:
            # Append to log file
            with open(self.log_file, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
        except Exception as e:
            print(f"Error logging prediction: {e}")
    
    def load_predictions(self, limit: int = 100) -> list:
        """Load recent predictions from log"""
        if not os.path.exists(self.log_file):
            return []
        
        predictions = []
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()[-limit:]  # Get last N lines
                for line in lines:
                    try:
                        predictions.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error loading predictions: {e}")
        
        return predictions
