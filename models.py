import numpy as np
from typing import Dict, List, Tuple
from utils import (
    calculate_btts_poisson_probability,
    calculate_btts_decision,
    get_league_settings,  # FIXED IMPORT
    calculate_team_specific_adjustments
)

class AdvancedUnderstatPredictor:
    def __init__(self):
        self.predictions_history = []
        self.team_profiles = {}
    
    def load_team_profile(self, team_name: str, csv_row: Dict):
        """Extract team-specific profile from CSV data"""
        if team_name not in self.team_profiles:
            goals_conceded_pg = csv_row.get('Team_Goals_Conceded_PG', 
                                           csv_row.get('goals_conceded_pg', 1.3))
            
            matches = max(1, csv_row.get('Matches', 1))
            
            self.team_profiles[team_name] = {
                'avg_total_goals': csv_row.get('Team_Avg_Total_Goals', 2.7),
                'goals_conceded_pg': goals_conceded_pg,
                'home_away_goal_diff': csv_row.get('Home_Away_Goal_Diff', 0.0),
                'offensive_rating': csv_row.get('Goals', 0) / matches,
                'defensive_rating': 2.0 - (csv_row.get('Goals_Against', 0) / matches),
                'xg_per_game': csv_row.get('xG', 0) / matches,
                'xga_per_game': csv_row.get('xGA', 0) / matches,
                'matches': matches
            }
    
    def calculate_advanced_analysis(self, home_data: Dict, away_data: Dict, 
                                   home_team: str, away_team: str, 
                                   games_played: int = 12,
                                   league_name: str = 'Average') -> Dict:
        """Advanced xG-based football predictor with LEAGUE-SPECIFIC adjustments"""
        
        # Get league settings
        league_settings = get_league_settings(league_name)
        league_avg_goals = league_settings['avg_goals']
        league_over_threshold = league_settings['over_threshold']
        league_under_threshold = league_settings['under_threshold']
        league_home_advantage = league_settings['home_advantage']
        
        # Load team profiles
        self.load_team_profile(home_team, home_data)
        self.load_team_profile(away_team, away_data)
        
        home_profile = self.team_profiles[home_team]
        away_profile = self.team_profiles[away_team]
        
        # 1. QUALITY CALCULATION with league context
        home_attack_strength = min(2.5, home_profile['xg_per_game'] * (league_avg_goals / 2.7))
        away_attack_strength = min(2.5, away_profile['xg_per_game'] * (league_avg_goals / 2.7))
        
        home_defense_strength = max(0.5, home_profile['defensive_rating'])
        away_defense_strength = max(0.5, away_profile['defensive_rating'])
        
        home_quality = (home_attack_strength * 0.7) + (home_defense_strength * 0.3)
        away_quality = (away_attack_strength * 0.7) + (away_defense_strength * 0.3)
        
        # 2. EXPECTED GOALS with LEAGUE ADJUSTMENT
        home_expected_raw = (home_attack_strength + (2.0 - away_defense_strength)) / 2
        away_expected_raw = (away_attack_strength + (2.0 - home_defense_strength)) / 2
        
        # League-specific home advantage
        home_advantage = league_home_advantage + home_profile['home_away_goal_diff'] * 0.2
        home_final = home_expected_raw * (1.0 + home_advantage)
        away_final = away_expected_raw * 0.9  # Away disadvantage
        
        # Cap unrealistic values
        home_final = min(3.5, max(0.2, home_final))
        away_final = min(3.0, max(0.2, away_final))
        
        total_expected = home_final + away_final
        expected_goal_diff = home_final - away_final
        
        # 3. MATCH WINNER with league context
        quality_diff = home_quality - away_quality
        
        # League-specific win thresholds
        if league_name in ['Serie A', 'La Liga']:  # Low scoring, tactical leagues
            win_threshold = 0.5
        else:  # Higher scoring leagues
            win_threshold = 0.4
        
        if quality_diff > win_threshold:
            winner = "Home Win"
            win_confidence = min(80, 55 + (quality_diff * 25))
        elif quality_diff < -win_threshold:
            winner = "Away Win"
            win_confidence = min(78, 53 + (abs(quality_diff) * 25))
        else:
            winner = "Draw"
            win_confidence = 45 + (abs(quality_diff) * -10)
        
        # 4. TOTAL GOALS with LEAGUE-SPECIFIC THRESHOLDS
        avg_total_for_matchup = (home_profile['avg_total_goals'] + away_profile['avg_total_goals']) / 2
        
        # Compare to league average, not fixed 2.5
        league_adjusted_total = (total_expected * 0.6) + (league_avg_goals * 0.4)
        
        # Determine over probability
        if league_adjusted_total > league_over_threshold:
            over_prob = 50 + min(30, (league_adjusted_total - league_over_threshold) * 30)
        elif league_adjusted_total < league_under_threshold:
            over_prob = 50 - min(30, (league_under_threshold - league_adjusted_total) * 30)
        else:
            over_prob = 50  # Close to league thresholds
        
        # Apply game script adjustments
        if abs(home_final - away_final) > 1.0:
            league_adjusted_total *= 0.9
            over_prob *= 0.9
        elif abs(home_final - away_final) < 0.3:
            league_adjusted_total *= 1.1
            over_prob *= 1.05
        
        # Final Total Goals decision
        if over_prob >= 55:
            goals_selection = "Over 2.5 Goals"
            goals_confidence = over_prob * 0.9
        elif over_prob <= 45:
            goals_selection = "Under 2.5 Goals"
            goals_confidence = (100 - over_prob) * 0.9
        else:
            goals_selection = "Avoid Total Goals"
            goals_confidence = over_prob
        
        # 5. BTTS with league context
        btts_raw_prob = calculate_btts_poisson_probability(home_final, away_final, league_name)
        
        # Get BTTS decision with league
        btts_selection, btts_confidence, btts_note = calculate_btts_decision(
            btts_raw_prob, league_adjusted_total,
            home_profile['offensive_rating'], away_profile['offensive_rating'],
            league_name,
            home_profile['defensive_rating'], away_profile['defensive_rating']
        )
        
        return {
            "team_names": {"home": home_team, "away": away_team},
            "league": league_name,
            "raw_data": {
                "home_profile": home_profile,
                "away_profile": away_profile,
                "home_overperformance": (home_data["points"] / games_played) - (home_data["xPTS"] / games_played),
                "away_overperformance": (away_data["points"] / games_played) - (away_data["xPTS"] / games_played)
            },
            "analysis": {
                "home_quality": round(home_quality, 1),
                "away_quality": round(away_quality, 1),
                "quality_diff": round(quality_diff, 2),
                "total_advantage": round(quality_diff, 2),
                "home_boost": 1.0 + min(0.3, home_advantage),
                "expected_goals": {
                    "home": round(home_final, 2),
                    "away": round(away_final, 2),
                    "total": round(league_adjusted_total, 2)
                },
                "expected_goal_diff": round(expected_goal_diff, 2),
                "league_settings": {
                    "avg_goals": league_avg_goals,
                    "over_threshold": league_over_threshold,
                    "under_threshold": league_under_threshold
                },
                "probabilities": {
                    "over_25": round(over_prob, 0),
                    "btts_raw": round(btts_raw_prob, 0),
                    "btts_adjust_note": btts_note
                },
                "home_momentum": 1.0 + min(0.15, max(-0.15, home_data["points"] / (games_played * 3) - 0.5)),
                "away_momentum": 1.0 + min(0.15, max(-0.15, away_data["points"] / (games_played * 3) - 0.5)),
                "home_intangible": 0.02 if home_data["points"] > home_data["xPTS"] else -0.01,
                "away_intangible": 0.01 if away_data["points"] > away_data["xPTS"] else -0.02,
                "volatility_note": "High volatility expected" if abs(home_profile['xg_per_game'] - away_profile['xg_per_game']) > 0.8 else ""
            },
            "predictions": [
                {"type": "Match Winner", "selection": winner, "confidence": round(win_confidence, 0)},
                {"type": "Total Goals", "selection": goals_selection, "confidence": round(goals_confidence, 0)},
                {"type": "Both Teams To Score", "selection": btts_selection, "confidence": round(btts_confidence, 0)}
            ]
        }
