"""
PHANTOM v4.3 - Data Loading & Utilities
Updated to work with corrected models.py
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
        
        # Calculate actual outcome rates from data
        total_home_wins = home_data['Wins'].sum()
        total_away_wins = away_data['Wins'].sum()
        total_draws = home_data['Draws'].sum()  # Same as away draws
        
        total_matches = (total_home_matches + total_away_matches) / 2
        
        actual_home_win_rate = total_home_wins / total_matches if total_matches > 0 else 0.45
        actual_away_win_rate = total_away_wins / total_matches if total_matches > 0 else 0.30
        actual_draw_rate = total_draws / total_matches if total_matches > 0 else 0.25
        
        # Debug output
        print(f"\n📊 LEAGUE STATISTICS CALCULATED:")
        print(f"  Total Home Goals: {total_home_goals} in {total_home_matches} matches")
        print(f"  Total Away Goals: {total_away_goals} in {total_away_matches} matches")
        print(f"  Avg Home Goals: {avg_home_goals:.2f}")
        print(f"  Avg Away Goals: {avg_away_goals:.2f}")
        print(f"  Neutral Baseline: {(avg_home_goals + avg_away_goals) / 2:.2f}")
        print(f"  Total Matches Analyzed: {int(total_matches)}")
        print(f"  Home Win Rate: {actual_home_win_rate:.1%}")
        print(f"  Draw Rate: {actual_draw_rate:.1%}")
        print(f"  Away Win Rate: {actual_away_win_rate:.1%}")
        
        # ✅ CORRECT: Create LeagueAverages without league_avg_gpg parameter
        return LeagueAverages(
            avg_home_goals=round(avg_home_goals, 3),
            avg_away_goals=round(avg_away_goals, 3),
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
        
        # 🔥 UPDATED: Add Last5 columns to required columns
        required_columns = [
            'Team', 'Matches', 'Home_Away', 'Wins', 'Draws', 'Losses',
            'Goals', 'Goals_Against', 'Points', 'xG', 'xGA',
            'Last5_Home_Wins', 'Last5_Home_Draws', 'Last5_Home_Losses',
            'Last5_Home_GF', 'Last5_Home_GA', 'Last5_Home_PTS',
            'Last5_Away_Wins', 'Last5_Away_Draws', 'Last5_Away_Losses',
            'Last5_Away_GF', 'Last5_Away_GA', 'Last5_Away_PTS'
        ]
        
        # Check for missing required columns
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"⚠️ Warning: Missing columns: {missing}")
            print("  Some features may not work correctly")
        
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
                venue = "home"
            else:
                df = away_df
                venue = "away"
            
            team_data = df[df['Team'] == team_name]
            
            if team_data.empty:
                print(f"❌ Team '{team_name}' not found in {league_name} {venue} data")
                print(f"   Available {venue} teams: {sorted(df['Team'].unique().tolist())[:10]}...")
                return None
            
            data_dict = team_data.iloc[0].to_dict()
            
            if debug:
                print(f"\n🔍 Creating TeamProfile for {team_name} ({venue}):")
                print(f"  Matches: {data_dict.get('Matches', 'N/A')}")
                print(f"  Goals: {data_dict.get('Goals', 'N/A')}")
                print(f"  xG: {data_dict.get('xG', 'N/A')}")
                print(f"  Last 5 GF: {data_dict.get(f'Last5_{venue.capitalize()}_GF', 'N/A')}")
            
            return TeamProfile(
                data_dict=data_dict,
                is_home=is_home,
                league_averages=league_averages,
                debug=debug
            )
            
        except Exception as e:
            print(f"❌ Error creating team profile for {team_name}: {e}")
            import traceback
            traceback.print_exc()
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
                
                # Check Last5 data
                last5_cols = [f'Last5_{venue.capitalize()}_{stat}' 
                            for stat in ['Wins', 'Draws', 'Losses', 'GF', 'GA', 'PTS']]
                for col in last5_cols:
                    if col in df.columns and (df[col] < 0).any():
                        issues.append(f"Negative values in {col}")
            
            # Check for unrealistic values
            if league_averages.avg_home_goals > 3.0:
                issues.append(f"Unrealistically high home goals average: {league_averages.avg_home_goals:.2f}")
            if league_averages.avg_away_goals > 2.5:
                issues.append(f"Unrealistically high away goals average: {league_averages.avg_away_goals:.2f}")
            
            # ✅ CORRECT: Access league_avg_gpg as a property
            return {
                "league": league_name,
                "total_teams": len(home_df) + len(away_df),
                "total_matches": league_averages.total_matches,
                "avg_home_goals": league_averages.avg_home_goals,
                "avg_away_goals": league_averages.avg_away_goals,
                "neutral_baseline": league_averages.neutral_baseline,  # ✅ Added
                "league_avg_gpg": league_averages.league_avg_gpg,  # ✅ Access as property
                "home_advantage_ratio": league_averages.home_advantage_ratio,
                "issues": issues,
                "status": "PASS" if not issues else "WARNINGS"
            }
            
        except Exception as e:
            return {
                "league": league_name,
                "status": "ERROR",
                "error": str(e)
            }
    
    def get_team_stats_summary(self, team_name: str, league_name: str) -> Optional[Dict]:
        """Get a summary of team statistics"""
        try:
            home_profile = self.create_team_profile(team_name, league_name, is_home=True, debug=False)
            away_profile = self.create_team_profile(team_name, league_name, is_home=False, debug=False)
            
            if not home_profile or not away_profile:
                return None
            
            return {
                "team": team_name,
                "league": league_name,
                "home_stats": {
                    "matches": home_profile.matches,
                    "wins": home_profile.wins,
                    "draws": home_profile.draws,
                    "losses": home_profile.losses,
                    "goals_for": home_profile.goals_for,
                    "goals_against": home_profile.goals_against,
                    "points": home_profile.points,
                    "goals_pg": round(home_profile.goals_pg, 2),
                    "goals_against_pg": round(home_profile.goals_against_pg, 2),
                    "form_score": round(home_profile.form_score, 2),
                    "attack_strength": round(home_profile.attack_strength, 2),
                    "defense_strength": round(home_profile.defense_strength, 2)
                },
                "away_stats": {
                    "matches": away_profile.matches,
                    "wins": away_profile.wins,
                    "draws": away_profile.draws,
                    "losses": away_profile.losses,
                    "goals_for": away_profile.goals_for,
                    "goals_against": away_profile.goals_against,
                    "points": away_profile.points,
                    "goals_pg": round(away_profile.goals_pg, 2),
                    "goals_against_pg": round(away_profile.goals_against_pg, 2),
                    "form_score": round(away_profile.form_score, 2),
                    "attack_strength": round(away_profile.attack_strength, 2),
                    "defense_strength": round(away_profile.defense_strength, 2)
                }
            }
            
        except Exception as e:
            print(f"❌ Error getting team stats summary: {e}")
            return None

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
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict]:
        """Get recent predictions from log"""
        try:
            if not os.path.exists(self.log_file):
                return []
            
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            predictions = []
            for line in lines[-limit:]:
                try:
                    predictions.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
            
            return predictions[::-1]  # Return most recent first
        except Exception as e:
            print(f"❌ Error reading predictions log: {e}")
            return []
