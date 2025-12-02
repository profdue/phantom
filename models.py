import numpy as np
from typing import Dict, List, Tuple
from utils import (
    calculate_btts_poisson_probability,
    calculate_btts_decision,
    get_match_context,
    get_league_adjustments,
    calculate_team_specific_adjustments
)

class AdvancedUnderstatPredictor:
    def __init__(self):
        self.predictions_history = []
        self.team_profiles = {}  # Cache for team-specific data
    
    def load_team_profile(self, team_name: str, csv_row: Dict):
        """Extract team-specific profile from CSV data"""
        if team_name not in self.team_profiles:
            # Handle column name variations from your CSV
            goals_conceded_pg = csv_row.get('Team_Goals_Conceded_PG', 
                                           csv_row.get('goals_conceded_pg', 1.3))
            
            self.team_profiles[team_name] = {
                'avg_total_goals': csv_row.get('Team_Avg_Total_Goals', 2.7),
                'goals_conceded_pg': goals_conceded_pg,
                'home_away_goal_diff': csv_row.get('Home_Away_Goal_Diff', 0.0),
                'offensive_rating': csv_row.get('Goals', 0) / max(1, csv_row.get('Matches', 1)),
                'defensive_rating': 2.0 - (csv_row.get('Goals_Against', 0) / max(1, csv_row.get('Matches', 1)))
            }
    
    def calculate_advanced_analysis(self, home_data: Dict, away_data: Dict, 
                                   home_team: str, away_team: str, 
                                   games_played: int = 12) -> Dict:
        """Advanced xG-based football predictor with team-specific adjustments"""
        
        # Load team profiles
        self.load_team_profile(home_team, home_data)
        self.load_team_profile(away_team, away_data)
        
        # 1. PER-GAME AVERAGES
        home_xg_pg = home_data["xG"] / games_played
        away_xg_pg = away_data["xG"] / games_played
        home_xga_pg = home_data["xGA"] / games_played
        away_xga_pg = away_data["xGA"] / games_played
        
        # 2. TEAM-SPECIFIC ADJUSTMENTS
        home_profile = self.team_profiles[home_team]
        away_profile = self.team_profiles[away_team]
        
        # Adjust for team's historical goal-scoring/conceding tendencies
        home_attack_adjustment = home_profile['offensive_rating'] / 1.5  # Normalize to 1.0
        away_attack_adjustment = away_profile['offensive_rating'] / 1.5
        home_defense_adjustment = 2.0 - home_profile['defensive_rating']  # Lower is better defense
        away_defense_adjustment = 2.0 - away_profile['defensive_rating']
        
        # 3. IMPROVED EXPECTED GOALS CALCULATION
        # Matchup-specific: Attack vs Opponent Defense
        home_expected = (home_xg_pg * home_attack_adjustment + away_xga_pg * away_defense_adjustment) / 2
        away_expected = (away_xg_pg * away_attack_adjustment + home_xga_pg * home_defense_adjustment) / 2
        
        # 4. LEAGUE & TEAM BASELINE REGRESSION
        league_avg = 2.7  # Default, will be overridden by league-specific data
        home_team_avg = home_profile['avg_total_goals'] / 2  # Approximate per-team contribution
        away_team_avg = away_profile['avg_total_goals'] / 2
        
        # Blend: 50% matchup, 30% team history, 20% league average
        home_final = (home_expected * 0.5) + (home_team_avg * 0.3) + (league_avg/2 * 0.2)
        away_final = (away_expected * 0.5) + (away_team_avg * 0.3) + (league_avg/2 * 0.2)
        
        total_expected = home_final + away_final
        expected_goal_diff = home_final - away_final
        
        # 5. OVER/UNDER CONFIDENCE WITH TEAM-SPECIFIC THRESHOLDS
        # Use team's historical averages to set thresholds
        avg_total_for_matchup = (home_profile['avg_total_goals'] + away_profile['avg_total_goals']) / 2
        
        if total_expected > avg_total_for_matchup + 0.3:
            over_confidence = min(75, 50 + (total_expected - avg_total_for_matchup) * 20)
        elif total_expected < avg_total_for_matchup - 0.3:
            over_confidence = max(25, 50 - (avg_total_for_matchup - total_expected) * 20)
        else:
            over_confidence = 45  # Close to historical average
        
        # 6. BTTS IMPROVEMENT WITH DEFENSIVE ANALYSIS
        # Consider both teams' defensive weaknesses
        btts_raw_prob = calculate_btts_poisson_probability(home_final, away_final)
        
        # Defensive adjustment: If both teams concede a lot, increase BTTS probability
        defensive_factor = (home_profile['goals_conceded_pg'] + away_profile['goals_conceded_pg']) / 2.6
        if defensive_factor > 1.2:  # Both concede more than average
            btts_raw_prob *= 1.15
        elif defensive_factor < 0.8:  # Both have strong defense
            btts_raw_prob *= 0.85
        
        # 7. GAME SCRIPT PREDICTION
        xg_diff = abs(home_xg_pg - away_xg_pg)
        if xg_diff > 1.0:  # Expected blowout
            total_expected *= 0.85
            btts_raw_prob *= 0.75
        elif xg_diff < 0.3:  # Expected close game
            total_expected *= 1.10
            btts_raw_prob *= 1.15
        
        # 8. MATCH WINNER WITH CONTEXT
        quality_diff = (home_xg_pg * 1.15 + home_profile['defensive_rating'] * 1.1) - \
                      (away_xg_pg * 1.15 + away_profile['defensive_rating'] * 1.1)
        
        # Apply home advantage from profile
        home_advantage = home_profile['home_away_goal_diff'] * 0.3
        total_advantage = quality_diff + home_advantage
        
        # Determine winner
        if total_advantage > 0.3:
            winner = "Home Win"
            confidence = min(75, 60 + total_advantage * 15)
        elif total_advantage < -0.3:
            winner = "Away Win"
            confidence = min(73, 58 + abs(total_advantage) * 15)
        else:
            winner = "Draw"
            confidence = 48
        
        # 9. FINAL PREDICTIONS WITH IMPROVED CONFIDENCE
        # Total Goals decision
        if over_confidence >= 55:
            goals_selection = "Over 2.5 Goals"
            goals_confidence = over_confidence * 0.9
        elif over_confidence <= 45:
            goals_selection = "Under 2.5 Goals"
            goals_confidence = (100 - over_confidence) * 0.9
        else:
            goals_selection = "Avoid Total Goals"
            goals_confidence = 42
        
        # BTTS decision
        btts_selection, btts_confidence, btts_note = calculate_btts_decision(
            btts_raw_prob, total_expected, 
            home_profile['offensive_rating'], away_profile['offensive_rating'],
            home_defense_adjustment, away_defense_adjustment
        )
        
        return {
            "team_names": {"home": home_team, "away": away_team},
            "raw_data": {
                "home_profile": home_profile,
                "away_profile": away_profile,
                "home_overperformance": (home_data["points"] / games_played) - (home_data["xPTS"] / games_played),
                "away_overperformance": (away_data["points"] / games_played) - (away_data["xPTS"] / games_played)
            },
            "analysis": {
                "home_quality": (home_xg_pg * 1.3 + home_profile['defensive_rating'] * 1.2) / 2,
                "away_quality": (away_xg_pg * 1.3 + away_profile['defensive_rating'] * 1.2) / 2,
                "quality_diff": total_advantage,
                "total_advantage": total_advantage,
                "home_boost": 1.0 + min(0.25, max(0.01, home_advantage)),
                "expected_goals": {
                    "home": home_final,
                    "away": away_final,
                    "total": total_expected
                },
                "expected_goal_diff": expected_goal_diff,
                "probabilities": {
                    "over_25": over_confidence,
                    "btts_raw": btts_raw_prob,
                    "btts_adjust_note": btts_note
                },
                "home_momentum": 1.0 + min(0.15, max(-0.15, home_data["points"] / (games_played * 3) - 0.5)),
                "away_momentum": 1.0 + min(0.15, max(-0.15, away_data["points"] / (games_played * 3) - 0.5)),
                "home_intangible": 0.02 if home_data["points"] > home_data["xPTS"] else -0.01,
                "away_intangible": 0.01 if away_data["points"] > away_data["xPTS"] else -0.02,
                "volatility_note": "High volatility expected" if abs(home_xg_pg - away_xg_pg) > 0.8 else ""
            },
            "predictions": [
                {"type": "Match Winner", "selection": winner, "confidence": confidence},
                {"type": "Total Goals", "selection": goals_selection, "confidence": goals_confidence},
                {"type": "Both Teams To Score", "selection": btts_selection, "confidence": btts_confidence}
            ]
        }
