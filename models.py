"""
PHANTOM v4.0 - Predictive Logic
No Avoid predictions - Clear Yes/No, Over/Under, Win/Draw/Lose
"""
import math

# ============================================================================
# LEAGUE CONFIGURATIONS - AGGRESSIVE SETTINGS
# ============================================================================

LEAGUE_CONFIGS = {
    "premier_league": {
        "name": "Premier League",
        "avg_goals": 2.93,
        "over_threshold": 2.75,  # LOWER threshold for more Over predictions
        "under_threshold": 2.55,  # HIGHER threshold for more Under predictions
        "home_advantage": 0.25,   # Reduced but meaningful
        "btts_baseline": 52,      # Lower baseline = more Yes predictions
        "win_threshold": 0.25,    # LOWER = more decisive winner predictions
        "form_weight": 0.4        # Higher form weight
    },
    "serie_a": {
        "name": "Serie A",
        "avg_goals": 2.56,
        "over_threshold": 2.40,
        "under_threshold": 2.20,
        "home_advantage": 0.20,
        "btts_baseline": 48,
        "win_threshold": 0.30,
        "form_weight": 0.35
    },
    "la_liga": {
        "name": "La Liga",
        "avg_goals": 2.62,
        "over_threshold": 2.45,
        "under_threshold": 2.25,
        "home_advantage": 0.22,
        "btts_baseline": 50,
        "win_threshold": 0.28,
        "form_weight": 0.38
    },
    "bundesliga": {
        "name": "Bundesliga",
        "avg_goals": 3.14,
        "over_threshold": 2.90,
        "under_threshold": 2.70,
        "home_advantage": 0.30,
        "btts_baseline": 55,
        "win_threshold": 0.22,
        "form_weight": 0.42
    },
    "ligue_1": {
        "name": "Ligue 1",
        "avg_goals": 2.78,
        "over_threshold": 2.60,
        "under_threshold": 2.40,
        "home_advantage": 0.23,
        "btts_baseline": 50,
        "win_threshold": 0.26,
        "form_weight": 0.36
    }
}

# ============================================================================
# TEAM PROFILE - FOCUS ON WHAT MATTERS
# ============================================================================

class TeamProfile:
    def __init__(self, data_dict, is_home=True):
        self.name = data_dict['Team']
        self.is_home = is_home
        
        # Only essential stats
        self.matches = int(data_dict['Matches'])
        self.wins = int(data_dict['Wins'])
        self.draws = int(data_dict['Draws'])
        self.losses = int(data_dict['Losses'])
        self.goals_for = int(data_dict['Goals'])
        self.goals_against = int(data_dict['Goals_Against'])
        self.points = int(data_dict['Points'])
        
        # xG data
        self.xg = float(data_dict['xG'])
        self.xga = float(data_dict['xGA'])
        
        # Per-game averages
        self.goals_pg = self.goals_for / max(1, self.matches)
        self.goals_against_pg = self.goals_against / max(1, self.matches)
        self.xg_pg = self.xg / max(1, self.matches)
        self.xga_pg = self.xga / max(1, self.matches)
        
        # Last 5 form (THE MOST IMPORTANT)
        if is_home:
            self.last5_wins = int(data_dict.get('Last5_Home_Wins', 0))
            self.last5_draws = int(data_dict.get('Last5_Home_Draws', 0))
            self.last5_losses = int(data_dict.get('Last5_Home_Losses', 0))
            self.last5_gf = int(data_dict.get('Last5_Home_GF', 0))
            self.last5_ga = int(data_dict.get('Last5_Home_GA', 0))
            self.last5_pts = int(data_dict.get('Last5_Home_PTS', 0))
        else:
            self.last5_wins = int(data_dict.get('Last5_Away_Wins', 0))
            self.last5_draws = int(data_dict.get('Last5_Away_Draws', 0))
            self.last5_losses = int(data_dict.get('Last5_Away_Losses', 0))
            self.last5_gf = int(data_dict.get('Last5_Away_GF', 0))
            self.last5_ga = int(data_dict.get('Last5_Away_GA', 0))
            self.last5_pts = int(data_dict.get('Last5_Away_PTS', 0))
        
        # Calculate key metrics
        self.form_score = self._calculate_form_score()
        self.attack_strength = self._calculate_attack_strength()
        self.defense_strength = self._calculate_defense_strength()
        self.btts_tendency = self._calculate_btts_tendency()
        
    def _calculate_form_score(self):
        """Form score 0-1, heavily weighted to recent games"""
        if self.last5_pts == 0:
            return 0.3  # Default for no data
        
        # Last 3 games estimated (more weight)
        last3_est = (self.last5_pts / 5) * 1.3  # Assume recent games were better/worse
        last3_est = min(1.0, last3_est)
        
        # Last 5 games
        last5_score = self.last5_pts / 15
        
        # Weight: 70% last 3 games, 30% last 5 games
        return (last3_est * 0.7) + (last5_score * 0.3)
    
    def _calculate_attack_strength(self):
        """Attack strength 0-2 scale"""
        # Recent goals matter MORE than xG
        recent_gpg = self.last5_gf / 5 if self.last5_gf > 0 else 0.5
        season_gpg = self.goals_pg
        
        # Weight: 60% recent form, 40% season average
        gpg = (recent_gpg * 0.6) + (season_gpg * 0.4)
        
        # Cap and scale
        return min(2.0, gpg * 1.2)
    
    def _calculate_defense_strength(self):
        """Defense strength 0-2 scale (higher = better defense)"""
        # Recent goals against matter MORE
        recent_gapg = self.last5_ga / 5 if self.last5_ga > 0 else 1.0
        season_gapg = self.goals_against_pg
        
        # Weight: 70% recent form, 30% season average
        gapg = (recent_gapg * 0.7) + (season_gapg * 0.3)
        
        # Convert to defense strength (lower GA = higher strength)
        return max(0.5, 2.0 - gapg)
    
    def _calculate_btts_tendency(self):
        """1.0 = neutral, >1.0 favors BTTS, <1.0 against BTTS"""
        # Both scoring AND conceding recently = high BTTS tendency
        if self.last5_gf > 0 and self.last5_ga > 0:
            # Calculate ratio of games with both GF and GA
            # Assuming at least 2 games with both = high tendency
            if self.last5_gf >= 3 and self.last5_ga >= 3:
                return 1.3  # Strong BTTS tendency
            else:
                return 1.1  # Moderate BTTS tendency
        elif self.last5_gf == 0 or self.last5_ga == 0:
            return 0.8  # Low BTTS tendency
        return 1.0

# ============================================================================
# MATCH PREDICTOR - BOLD PREDICTIONS
# ============================================================================

class MatchPredictor:
    def __init__(self, league_name):
        self.league_config = LEAGUE_CONFIGS.get(league_name.lower())
        if not self.league_config:
            raise ValueError(f"Unknown league: {league_name}")
    
    def predict(self, home_team: TeamProfile, away_team: TeamProfile):
        """Make BOLD predictions - No Avoids allowed"""
        
        # 1. WINNER PREDICTION (ALWAYS pick winner, no Draw unless very close)
        winner_pred = self._predict_winner(home_team, away_team)
        
        # 2. EXPECTED GOALS
        home_xg, away_xg = self._calculate_expected_goals(home_team, away_team)
        total_xg = home_xg + away_xg
        
        # 3. TOTAL GOALS (ALWAYS Over or Under, no Avoid)
        total_pred = self._predict_total_goals(total_xg, home_team, away_team)
        
        # 4. BTTS (ALWAYS Yes or No, no Avoid)
        btts_pred = self._predict_btts(home_xg, away_xg, home_team, away_team)
        
        return {
            "analysis": {
                "league": self.league_config['name'],
                "form_scores": {
                    "home": round(home_team.form_score, 2),
                    "away": round(away_team.form_score, 2)
                },
                "expected_goals": {
                    "home": round(home_xg, 2),
                    "away": round(away_xg, 2),
                    "total": round(total_xg, 2)
                },
                "attack_strengths": {
                    "home": round(home_team.attack_strength, 2),
                    "away": round(away_team.attack_strength, 2)
                }
            },
            "predictions": [winner_pred, total_pred, btts_pred]
        }
    
    def _predict_winner(self, home: TeamProfile, away: TeamProfile):
        """ALWAYS predict a winner (Draw only if extremely close)"""
        
        # FORM DIFFERENCE (most important)
        form_diff = home.form_score - away.form_score
        
        # ATTACK/DEFENSE DIFFERENCE
        attack_diff = home.attack_strength - away.attack_strength
        defense_diff = home.defense_strength - away.defense_strength
        
        # COMBINED ADVANTAGE
        total_advantage = (form_diff * 0.5) + (attack_diff * 0.3) + (defense_diff * 0.2)
        
        # Apply home advantage
        total_advantage += self.league_config['home_advantage'] * 0.3
        
        # DECISION (very low threshold for draws)
        win_threshold = self.league_config['win_threshold']
        
        if total_advantage > win_threshold:
            selection = "Home Win"
            confidence = 55 + (total_advantage * 25)
            
        elif total_advantage < -win_threshold:
            selection = "Away Win"
            confidence = 55 + (abs(total_advantage) * 25)
            
        else:  # Very close match
            selection = "Draw"
            # Draw confidence based on how close
            closeness = 1.0 - (abs(total_advantage) / win_threshold)
            confidence = 50 + (closeness * 15)
        
        # CONFIDENCE ADJUSTMENTS
        # Recent goal scoring boosts confidence
        if home.last5_gf / 5 > 1.5 and "Home" in selection:
            confidence += 5
        if away.last5_gf / 5 > 1.5 and "Away" in selection:
            confidence += 5
            
        # Cap confidence
        confidence = max(45, min(80, confidence))
        
        return {
            "type": "Match Winner",
            "selection": selection,
            "confidence": round(confidence, 1)
        }
    
    def _calculate_expected_goals(self, home: TeamProfile, away: TeamProfile):
        """Calculate expected goals - SIMPLE and EFFECTIVE"""
        
        # Home xG = Home attack × (2.0 - Away defense) × Home advantage
        home_raw = home.attack_strength * (2.0 - away.defense_strength)
        home_xg = home_raw * (1.0 + self.league_config['home_advantage'])
        
        # Away xG = Away attack × (2.0 - Home defense) × Away factor
        away_raw = away.attack_strength * (2.0 - home.defense_strength)
        away_xg = away_raw * 0.9  # Standard away penalty
        
        # Recent form adjustments
        if home.last5_gf / 5 > 1.5:
            home_xg *= 1.1  # Hot attack
        if away.last5_gf / 5 > 1.5:
            away_xg *= 1.1
            
        return home_xg, away_xg
    
    def _predict_total_goals(self, total_xg: float, home: TeamProfile, away: TeamProfile):
        """ALWAYS predict Over or Under"""
        
        # League context
        league_avg = self.league_config['avg_goals']
        over_thresh = self.league_config['over_threshold']
        under_thresh = self.league_config['under_threshold']
        
        # Recent scoring trend
        home_recent_gpg = home.last5_gf / 5
        away_recent_gpg = away.last5_gf / 5
        recent_scoring = (home_recent_gpg + away_recent_gpg) / 2
        
        # Adjusted total (60% xG, 40% recent form)
        adjusted_total = (total_xg * 0.6) + (recent_scoring * 2.0 * 0.4)
        
        # DECISION - NO AVOID
        if adjusted_total > over_thresh:
            selection = "Over 2.5 Goals"
            # Confidence based on how far above threshold
            excess = (adjusted_total - over_thresh) / over_thresh
            confidence = 55 + (excess * 25)
            
        else:  # MUST be Under
            selection = "Under 2.5 Goals"
            # Confidence based on how far below average
            deficit = (league_avg - adjusted_total) / league_avg
            confidence = 55 + (deficit * 25)
        
        # Cap confidence
        confidence = max(50, min(80, confidence))
        
        return {
            "type": "Total Goals",
            "selection": selection,
            "confidence": round(confidence, 1)
        }
    
    def _predict_btts(self, home_xg: float, away_xg: float, 
                     home: TeamProfile, away: TeamProfile):
        """ALWAYS predict Yes or No"""
        
        # Base probability from xG
        home_score_prob = 1 - math.exp(-home_xg)
        away_score_prob = 1 - math.exp(-away_xg)
        btts_prob = home_score_prob * away_score_prob * 100
        
        # Apply team tendencies
        btts_prob *= ((home.btts_tendency + away.btts_tendency) / 2)
        
        # Recent BTTS trend
        home_btss_games = min(3, home.last5_gf)  # Estimate games they scored
        away_btss_games = min(3, away.last5_gf)
        recent_btss_factor = (home_btss_games + away_btss_games) / 6  # 0-1 scale
        btts_prob *= (0.8 + (recent_btss_factor * 0.4))  # 0.8-1.2 adjustment
        
        # DECISION - NO AVOID
        baseline = self.league_config['btts_baseline']
        
        if btts_prob >= baseline:
            selection = "Yes"
            confidence = min(80, btts_prob)
        else:
            selection = "No"
            confidence = min(80, 100 - btts_prob)
        
        # Minimum confidence
        confidence = max(50, confidence)
        
        return {
            "type": "BTTS",
            "selection": selection,
            "confidence": round(confidence, 1)
        }