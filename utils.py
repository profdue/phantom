"""
PHANTOM v4.2 - Data Loading & Utilities
"""
import pandas as pd
import os
import json
from typing import Dict, Tuple, Optional, List
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
            print(f"⚠️ Warning: Data directory '{self.data_dir}' not found!")
            return leagues
            
        for file in os.listdir(self.data_dir):
            if file.endswith("_home_away.csv"):
                league_name = file.replace("_home_away.csv", "")
                leagues[league_name] = os.path.join(self.data_dir, file)
        
        print(f"📁 Found {len(leagues)} leagues: {list(leagues.keys())}")
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
        
        # League averages per team per game
        total_goals = total_home_goals + total_away_goals
        total_matches = (total_home_matches + total_away_matches) / 2  # Each match counted twice
        
        # Each team plays in each match, so divide by 2
        league_avg_gpg = total_goals / (total_matches * 2) if total_matches > 0 else 1.45
        
        # Calculate actual outcome rates from data
        total_home_wins = home_data['Wins'].sum()
        total_away_wins = away_data['Wins'].sum()
        total_draws = home_data['Draws'].sum()  # Same as away draws
        
        actual_home_win_rate = total_home_wins / total_matches if total_matches > 0 else 0.45
        actual_away_win_rate = total_away_wins / total_matches if total_matches > 0 else 0.30
        actual_draw_rate = total_draws / total_matches if total_matches > 0 else 0.25
        
        # Debug output
        print(f"\n📊 LEAGUE STATISTICS CALCULATED:")
        print(f"  Total Home Goals: {total_home_goals} in {total_home_matches} matches")
        print(f"  Total Away Goals: {total_away_goals} in {total_away_matches} matches")
        print(f"  Avg Home Goals: {avg_home_goals:.2f}")
        print(f"  Avg Away Goals: {avg_away_goals:.2f}")
        print(f"  League Avg per Team: {league_avg_gpg:.2f}")
        print(f"  Total Matches Analyzed: {int(total_matches)}")
        print(f"  Home Win Rate: {actual_home_win_rate:.1%}")
        print(f"  Draw Rate: {actual_draw_rate:.1%}")
        print(f"  Away Win Rate: {actual_away_win_rate:.1%}")
        
        return LeagueAverages(
            avg_home_goals=round(avg_home_goals, 3),
            avg_away_goals=round(avg_away_goals, 3),
            league_avg_gpg=round(league_avg_gpg, 3),
            total_matches=int(total_matches),
            actual_home_win_rate=round(actual_home_win_rate, 3),
            actual_draw_rate=round(actual_draw_rate, 3),
            actual_away_win_rate=round(actual_away_win_rate, 3)
        )
    
    def load_league_data(self, league_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, LeagueAverages]:
        """Load data for a specific league with calculated averages"""
        if league_name not in self.available_leagues:
            available = list(self.available_leagues.keys())
            raise ValueError(f"League '{league_name}' not found. Available: {available}")
        
        file_path = self.available_leagues[league_name]
        print(f"📂 Loading {league_name} from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Validate required columns
        required_columns = [
            'Team', 'Matches', 'Home_Away', 'Wins', 'Draws', 'Losses',
            'Goals', 'Goals_Against', 'Points', 'xG', 'xGA'
        ]
        
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Separate home and away
        home_teams = df[df['Home_Away'] == 'Home'].copy()
        away_teams = df[df['Home_Away'] == 'Away'].copy()
        
        # Calculate league averages
        league_averages = self.calculate_league_averages(df)
        
        print(f"✅ Successfully loaded {len(home_teams)} home teams and {len(away_teams)} away teams")
        
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
                           is_home: bool = True, debug: bool = False) -> Optional[TeamProfile]:
        """Create TeamProfile object for a specific team with debug option"""
        try:
            home_df, away_df, league_averages = self.load_league_data(league_name)
            
            if is_home:
                df = home_df
            else:
                df = away_df
            
            team_data = df[df['Team'] == team_name]
            
            if team_data.empty:
                raise ValueError(f"Team '{team_name}' not found in {league_name} {'home' if is_home else 'away'} data")
            
            data_dict = team_data.iloc[0].to_dict()
            
            return TeamProfile(
                data_dict=data_dict,
                is_home=is_home,
                league_averages=league_averages,
                debug=debug
            )
            
        except Exception as e:
            print(f"❌ Error creating team profile: {e}")
            return None
    
    def validate_data_integrity(self, league_name: str) -> Dict:
        """Validate data integrity for a league"""
        try:
            home_df, away_df, league_averages = self.load_league_data(league_name)
            
            issues = []
            
            # Check for negative values
            for df, venue in [(home_df, "home"), (away_df, "away")]:
                if (df['Goals'] < 0).any():
                    issues.append(f"Negative goals in {venue} data")
                if (df['Goals_Against'] < 0).any():
                    issues.append(f"Negative goals against in {venue} data")
                if (df['Matches'] < 0).any():
                    issues.append(f"Negative matches in {venue} data")
            
            # Check for unrealistic values
            if league_averages.avg_home_goals > 3.0:
                issues.append(f"Unrealistically high home goals average: {league_averages.avg_home_goals:.2f}")
            if league_averages.avg_away_goals > 2.5:
                issues.append(f"Unrealistically high away goals average: {league_averages.avg_away_goals:.2f}")
            
            return {
                "league": league_name,
                "total_teams": len(home_df) + len(away_df),
                "total_matches": league_averages.total_matches,
                "avg_home_goals": league_averages.avg_home_goals,
                "avg_away_goals": league_averages.avg_away_goals,
                "issues": issues,
                "status": "PASS" if not issues else "WARNINGS"
            }
            
        except Exception as e:
            return {
                "league": league_name,
                "status": "ERROR",
                "error": str(e)
            }

class PredictionLogger:
    """Log predictions for validation and analysis"""
    
    def __init__(self, log_file: str = "predictions_log.json"):
        self.log_file = log_file
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """Ensure the log directory exists"""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
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
            return True
        except Exception as e:
            print(f"❌ Error logging prediction: {e}")
            return False
    
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
            print(f"❌ Error loading predictions: {e}")
        
        return predictions
    
    def get_performance_metrics(self, min_samples: int = 10) -> Dict:
        """Calculate performance metrics from logged predictions"""
        predictions = self.load_predictions(limit=1000)
        
        if len(predictions) < min_samples:
            return {"status": "insufficient_data", "samples": len(predictions)}
        
        # Calculate metrics
        correct_wins = 0
        total_wins = 0
        correct_totals = 0
        total_totals = 0
        correct_btts = 0
        total_btts = 0
        
        for pred in predictions:
            if 'outcome' in pred:
                # Simplified check - in practice you'd need actual outcomes
                pass
        
        return {
            "status": "calculated",
            "total_predictions": len(predictions),
            "win_accuracy": correct_wins / total_wins if total_wins > 0 else None,
            "total_accuracy": correct_totals / total_totals if total_totals > 0 else None,
            "btts_accuracy": correct_btts / total_btts if total_btts > 0 else None
        }

class PerformanceTracker:
    """Track model performance over time"""
    
    def __init__(self):
        self.predictions = []
        self.weekly_performance = {}
    
    def add_prediction(self, prediction: Dict, actual_outcome: Dict = None):
        """Add a prediction with optional actual outcome"""
        entry = {
            "timestamp": datetime.now(),
            "prediction": prediction,
            "actual": actual_outcome
        }
        self.predictions.append(entry)
        
        # Update weekly stats
        week_key = datetime.now().strftime("%Y-%U")
        if week_key not in self.weekly_performance:
            self.weekly_performance[week_key] = {
                "total": 0,
                "correct": 0,
                "predictions": []
            }
        
        self.weekly_performance[week_key]["total"] += 1
        if actual_outcome:
            # Check if prediction was correct
            # This would need actual outcome data
            pass
    
    def get_weekly_report(self) -> Dict:
        """Generate weekly performance report"""
        report = {}
        for week, data in self.weekly_performance.items():
            if data["total"] > 0:
                accuracy = data["correct"] / data["total"] if "correct" in data else None
                report[week] = {
                    "total_predictions": data["total"],
                    "accuracy": accuracy,
                    "sample_size": len(data["predictions"])
                }
        return report
