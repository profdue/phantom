import pandas as pd
import os
from typing import Dict, Optional

class EnhancedDataLoader:
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.team_cache = {}
    
    def load_enhanced_data(self, league_name: str) -> Optional[pd.DataFrame]:
        """Load CSV with enhanced team profile columns"""
        try:
            # Find CSV file
            base_name = league_name.lower().replace(' ', '_')
            possible_files = [
                f"{base_name}_enhanced.csv",
                f"{base_name}.csv",
                f"{base_name}_home_away.csv"
            ]
            
            filepath = None
            for filename in possible_files:
                test_path = os.path.join(self.data_folder, filename)
                if os.path.exists(test_path):
                    filepath = test_path
                    break
            
            if not filepath:
                print(f"No CSV file found for league: {league_name}")
                return None
            
            # Load and validate
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            
            # Check for required columns
            required_columns = ['Team', 'Matches', 'Home_Away', 'Wins', 'Draws', 'Losses',
                               'Goals', 'Goals_Against', 'Points', 'xG', 'xGA', 'xPTS',
                               'Performance_Note']
            
            # Check for enhanced columns
            enhanced_columns = ['Team_Avg_Total_Goals', 'Team_Goals_Conceded_PG', 'Home_Away_Goal_Diff']
            
            missing_required = [col for col in required_columns if col not in df.columns]
            if missing_required:
                print(f"Missing required columns: {missing_required}")
                return None
            
            # Convert numeric columns
            numeric_cols = ['Matches', 'Wins', 'Draws', 'Losses', 'Goals', 'Goals_Against',
                           'Points', 'xG', 'xGA', 'xPTS']
            
            # Add enhanced columns if present
            for col in enhanced_columns:
                if col in df.columns:
                    numeric_cols.append(col)
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Cache team profiles
            self._cache_team_profiles(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading enhanced data: {e}")
            return None
    
    def _cache_team_profiles(self, df: pd.DataFrame):
        """Cache team profile data for quick access"""
        for _, row in df.iterrows():
            team = row['Team']
            if team not in self.team_cache:
                self.team_cache[team] = {}
            
            # Store home and away data separately
            home_away = row.get('Home_Away', 'Home')
            self.team_cache[team][home_away] = row.to_dict()
    
    def get_team_profile(self, team_name: str, home_away: str = None) -> Optional[Dict]:
        """Get enhanced team profile data"""
        if team_name in self.team_cache:
            if home_away and home_away in self.team_cache[team_name]:
                return self.team_cache[team_name][home_away]
            # Return any available data if specific home/away not requested
            for location in ['Home', 'Away', 'Overall']:
                if location in self.team_cache[team_name]:
                    return self.team_cache[team_name][location]
        return None
    
    def calculate_missing_profile_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced profile data if missing from CSV"""
        if 'Team_Avg_Total_Goals' not in df.columns:
            df['Team_Avg_Total_Goals'] = (df['Goals'] + df['Goals_Against']) / df['Matches']
        
        if 'Team_Goals_Conceded_PG' not in df.columns:
            df['Team_Goals_Conceded_PG'] = df['Goals_Against'] / df['Matches']
        
        # Calculate Home/Away goal difference if we have both Home and Away rows
        teams = df['Team'].unique()
        for team in teams:
            team_data = df[df['Team'] == team]
            if len(team_data) >= 2:  # Has both Home and Away
                home_data = team_data[team_data['Home_Away'] == 'Home']
                away_data = team_data[team_data['Home_Away'] == 'Away']
                
                if not home_data.empty and not away_data.empty:
                    home_avg = home_data['Goals'].values[0] / home_data['Matches'].values[0]
                    away_avg = away_data['Goals'].values[0] / away_data['Matches'].values[0]
                    goal_diff = home_avg - away_avg
                    
                    df.loc[df['Team'] == team, 'Home_Away_Goal_Diff'] = goal_diff
        
        # Fill missing values
        if 'Home_Away_Goal_Diff' in df.columns:
            df['Home_Away_Goal_Diff'] = df['Home_Away_Goal_Diff'].fillna(0)
        
        return df
