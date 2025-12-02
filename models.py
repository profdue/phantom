"""
Prediction model classes implementing v2.3 logic
"""
import math
import numpy as np
from config import LEAGUE_CONFIGS, MODEL_PARAMS

class TeamProfile:
    """Represents a team's statistical profile"""
    
    def __init__(self, data_dict, is_home=True):
        self.name = data_dict['Team']
        self.is_home = is_home
        
        # Core stats
        self.matches = int(data_dict['Matches'])
        self.wins = int(data_dict['Wins'])
        self.draws = int(data_dict['Draws'])
        self.losses = int(data_dict['Losses'])
        self.goals_for = int(data_dict['Goals'])
        self.goals_against = int(data_dict['Goals_Against'])
        self.points = int(data_dict['Points'])
        
        # Advanced metrics
        self.xg = float(data_dict['xG'])
        self.xga = float(data_dict['xGA'])
        self.xpts = float(data_dict['xPTS'])
        
        # Calculated per-game averages
        self.goals_pg = self.goals_for / self.matches if self.matches > 0 else 0
        self.goals_against_pg = self.goals_against / self.matches if self.matches > 0 else 0
        self.xg_pg = self.xg / self.matches if self.matches > 0 else 0
        self.xga_pg = self.xga / self.matches if self.matches > 0 else 0
        
        # Last 5 form data
        if is_home:
            self.last5_wins = int(data_dict['Last5_Home_Wins'])
            self.last5_draws = int(data_dict['Last5_Home_Draws'])
            self.last5_losses = int(data_dict['Last5_Home_Losses'])
            self.last5_gf = int(data_dict['Last5_Home_GF'])
            self.last5_ga = int(data_dict['Last5_Home_GA'])
            self.last5_gd = int(data_dict['Last5_Home_GD'])
            self.last5_pts = int(data_dict['Last5_Home_PTS'])
        else:
            self.last5_wins = int(data_dict['Last5_Away_Wins'])
            self.last5_draws = int(data_dict['Last5_Away_Draws'])
            self.last5_losses = int(data_dict['Last5_Away_Losses'])
            self.last5_gf = int(data_dict['Last5_Away_GF'])
            self.last5_ga = int(data_dict['Last5_Away_GA'])
            self.last5_gd = int(data_dict['Last5_Away_GD'])
            self.last5_pts = int(data_dict['Last5_Away_PTS'])
        
        # Calculate flags
        self.has_attack_crisis = self._check_attack_crisis()
        self.has_defense_crisis = self._check_defense_crisis()
        self.form_momentum = self._calculate_momentum()
        
    def _check_attack_crisis(self):
        """Detect if team is in attacking crisis"""
        if self.xg_pg < MODEL_PARAMS['attack_crisis_threshold']:
            if self.last5_gf / 5 < 0.5:  # Less than 0.5 goals per game in last 5
                if self.last5_losses >= 2:  # Losing streak
                    return True
        return False
    
    def _check_defense_crisis(self):
        """Detect if team is in defensive crisis"""
        return self.goals_against_pg > MODEL_PARAMS['defense_crisis_threshold']
    
    def _calculate_momentum(self):
        """Calculate form momentum (0 to 0.3 scale)"""
        max_possible_pts = 15  # 5 matches × 3 points
        if max_possible_pts > 0:
            momentum = (self.last5_pts / max_possible_pts) * MODEL_PARAMS.get('momentum_weight', 0.3)
            return min(0.3, momentum)  # Cap at 0.3
        return 0.0
    
    def get_clean_sheet_probability(self, opponent_xg):
        """Calculate probability of keeping clean sheet"""
        if opponent_xg > 0:
            return math.exp(-opponent_xg) * 100
        return 0.0


class MatchPredictor:
    """Main prediction engine implementing v2.3 logic"""
    
    def __init__(self, league_name):
        self.league_config = LEAGUE_CONFIGS.get(league_name.lower())
        if not self.league_config:
            raise ValueError(f"Unknown league: {league_name}")
        
        self.params = MODEL_PARAMS
    
    def predict(self, home_team: TeamProfile, away_team: TeamProfile):
        """Generate predictions for a match"""
        
        # 1. Calculate team qualities
        home_quality = self._calculate_team_quality(home_team, is_home=True)
        away_quality = self._calculate_team_quality(away_team, is_home=False)
        
        # 2. Calculate expected goals
        home_xg, away_xg = self._calculate_expected_goals(home_team, away_team, home_quality, away_quality)
        total_xg = home_xg + away_xg
        
        # 3. Match winner prediction
        winner_pred = self._predict_winner(home_quality, away_quality, home_team, away_team)
        
        # 4. Total goals prediction
        total_pred = self._predict_total_goals(total_xg, home_xg, away_xg, home_quality, away_quality)
        
        # 5. BTTS prediction
        btts_pred = self._predict_btts(home_xg, away_xg, home_team, away_team)
        
        return {
            "analysis": {
                "league": self.league_config['name'],
                "quality_ratings": {
                    "home": round(home_quality, 2),
                    "away": round(away_quality, 2)
                },
                "expected_goals": {
                    "home": round(home_xg, 2),
                    "away": round(away_xg, 2),
                    "total": round(total_xg, 2)
                },
                "team_flags": {
                    "home_attack_crisis": home_team.has_attack_crisis,
                    "away_attack_crisis": away_team.has_attack_crisis,
                    "home_defense_crisis": home_team.has_defense_crisis,
                    "away_defense_crisis": away_team.has_defense_crisis
                }
            },
            "predictions": [winner_pred, total_pred, btts_pred]
        }
    
    def _calculate_team_quality(self, team: TeamProfile, is_home: bool):
        """Calculate team quality score with form momentum"""
        
        # Attack strength
        league_factor = self.league_config['avg_goals'] / 2.7
        attack_raw = min(2.5, team.xg_pg * league_factor)
        
        # Apply attack crisis penalty
        if team.has_attack_crisis:
            attack_strength = attack_raw * 0.6
        else:
            attack_strength = attack_raw
        
        # Defense strength
        if team.has_defense_crisis and team.last5_losses >= 2:
            defense_strength = 0.4  # Crisis mode
        else:
            defense_strength = max(0.4, 2.0 - team.goals_against_pg)
        
        # Base quality
        base_quality = (attack_strength * self.params['attack_weight'] + 
                       defense_strength * self.params['defense_weight'])
        
        # Add form momentum
        final_quality = base_quality + team.form_momentum
        
        return final_quality
    
    def _calculate_expected_goals(self, home: TeamProfile, away: TeamProfile, 
                                 home_quality: float, away_quality: float):
        """Calculate expected goals with crisis adjustments"""
        
        # Raw calculations
        home_raw = (home_quality + (2.0 - away_quality)) / 2
        away_raw = (away_quality + (2.0 - home_quality)) / 2
        
        # Apply home advantage
        home_bonus = 0.05 if home.last5_wins >= 3 else 0
        home_final = home_raw * (1.0 + self.league_config['home_advantage'] + home_bonus)
        
        # Away penalty based on form
        if away.last5_pts < self.params['away_form_penalty_threshold']:
            away_final = away_raw * 0.85  # Strong penalty for poor away form
        else:
            away_final = away_raw * 0.95  # Minimal penalty
        
        # Attack crisis adjustments
        if home.has_attack_crisis:
            home_final *= 0.6
        if away.has_attack_crisis:
            away_final *= 0.6
        
        return home_final, away_final
    
    def _predict_winner(self, home_quality: float, away_quality: float, 
                       home: TeamProfile, away: TeamProfile):
        """Predict match winner with momentum consideration"""
        
        quality_diff = home_quality - away_quality
        win_threshold = self.league_config['win_threshold']
        
        # Form momentum override
        form_diff = home.form_momentum - away.form_momentum
        
        # Adjusted difference
        adjusted_diff = quality_diff + (form_diff * 2)  # Double weight to form
        
        # Determine winner
        if adjusted_diff > win_threshold:
            selection = "Home Win"
            confidence = 55 + (abs(adjusted_diff) * 20)
        elif adjusted_diff < -win_threshold:
            selection = "Away Win"
            confidence = 55 + (abs(adjusted_diff) * 20)
        else:
            selection = "Draw"
            confidence = 50 + (20 - abs(adjusted_diff) * 10)
        
        # Confidence adjustments
        if home.has_attack_crisis or away.has_attack_crisis:
            confidence -= 10
        if abs(home_quality - away_quality) > 1.5:  # Big match
            confidence -= 5
        
        confidence = max(30, min(85, confidence))
        
        return {
            "type": "Match Winner",
            "selection": selection,
            "confidence": round(confidence, 1)
        }
    
    def _predict_total_goals(self, total_xg: float, home_xg: float, away_xg: float,
                            home_quality: float, away_quality: float):
        """Predict total goals with script detection"""
        
        # League blending
        quality_diff = abs(home_quality - away_quality)
        
        if quality_diff > 0.7:  # Mismatch
            total = (total_xg * self.params['mismatch_blend_current'] + 
                    self.league_config['avg_goals'] * self.params['mismatch_blend_league'])
        else:
            total = (total_xg * self.params['league_blend_current'] + 
                    self.league_config['avg_goals'] * self.params['league_blend_league'])
        
        # Script detection
        if 0.7 < quality_diff < 1.2:
            total *= 0.9  # 2-0 or 2-1 script likely
        
        # Big match penalty
        if home_quality > self.params['big_match_quality_threshold'] and \
           away_quality > self.params['big_match_quality_threshold']:
            total *= 0.85
        
        # Compare to thresholds
        over_thresh = self.league_config['over_threshold']
        under_thresh = self.league_config['under_threshold']
        
        if total > over_thresh + 0.3:
            selection = "Over 2.5 Goals"
            confidence = 50 + ((total - over_thresh) * 25)
        elif total < under_thresh - 0.3:
            selection = "Under 2.5 Goals"
            confidence = 50 + ((under_thresh - total) * 25)
        else:
            selection = "Avoid Total Goals"
            confidence = 50
        
        confidence = max(30, min(85, confidence))
        
        return {
            "type": "Total Goals",
            "selection": selection,
            "confidence": round(confidence, 1)
        }
    
    def _predict_btts(self, home_xg: float, away_xg: float, 
                     home: TeamProfile, away: TeamProfile):
        """Predict Both Teams To Score with clean sheet awareness"""
        
        # Base Poisson probability
        home_score_prob = 1 - math.exp(-home_xg)
        away_score_prob = 1 - math.exp(-away_xg)
        btts_raw = home_score_prob * away_score_prob * 100
        
        # Clean sheet risk adjustment
        home_cs_prob = home.get_clean_sheet_probability(away_xg)
        away_cs_prob = away.get_clean_sheet_probability(home_xg)
        
        if home_cs_prob > self.params['clean_sheet_risk_threshold']:
            btts_raw *= (1 - (home_cs_prob - 25) / 100)
        if away_cs_prob > self.params['clean_sheet_risk_threshold']:
            btts_raw *= (1 - (away_cs_prob - 25) / 100)
        
        # Attack crisis penalty
        if home.has_attack_crisis:
            btts_raw *= 0.7
        if away.has_attack_crisis:
            btts_raw *= 0.7
        
        # Compare to league baseline
        baseline = self.league_config['btts_baseline']
        
        if btts_raw > baseline + 8:
            selection = "Yes"
            confidence = min(85, btts_raw)
        elif btts_raw < baseline - 10:
            selection = "No"
            confidence = min(85, 100 - btts_raw)
        else:
            selection = "Avoid BTTS"
            confidence = 50
        
        return {
            "type": "BTTS",
            "selection": selection,
            "confidence": round(confidence, 1)
        }
