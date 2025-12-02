"""
Prediction model classes implementing v3.0 logic
Form-First, xG-Trends Second, Quality-Third Approach
"""
import math

# League configurations
LEAGUE_CONFIGS = {
    "premier_league": {
        "name": "Premier League",
        "avg_goals": 2.93,
        "over_threshold": 2.8,
        "under_threshold": 2.6,
        "home_advantage": 0.30,  # Reduced from 0.35
        "btts_baseline": 56,
        "form_threshold": 0.25,  # Lower threshold for form-based predictions
        "total_goals_adjust": 0.2  # Boost for high-scoring league
    },
    "serie_a": {
        "name": "Serie A",
        "avg_goals": 2.56,
        "over_threshold": 2.45,
        "under_threshold": 2.25,
        "home_advantage": 0.20,  # Reduced from 0.25
        "btts_baseline": 51,
        "form_threshold": 0.30,
        "total_goals_adjust": -0.2  # Reduction for low-scoring league
    },
    "la_liga": {
        "name": "La Liga",
        "avg_goals": 2.62,
        "over_threshold": 2.5,
        "under_threshold": 2.3,
        "home_advantage": 0.25,  # Reduced from 0.30
        "btts_baseline": 56,
        "form_threshold": 0.28,
        "total_goals_adjust": -0.1
    },
    "bundesliga": {
        "name": "Bundesliga",
        "avg_goals": 3.14,
        "over_threshold": 3.0,
        "under_threshold": 2.8,
        "home_advantage": 0.30,  # Reduced from 0.40
        "btts_baseline": 58,
        "form_threshold": 0.25,
        "total_goals_adjust": 0.3
    },
    "ligue_1": {
        "name": "Ligue 1",
        "avg_goals": 2.78,
        "over_threshold": 2.7,
        "under_threshold": 2.5,
        "home_advantage": 0.25,  # Reduced from 0.32
        "btts_baseline": 54,
        "form_threshold": 0.28,
        "total_goals_adjust": 0.1
    },
    "rfpl": {
        "name": "Russian Premier League",
        "avg_goals": 2.68,
        "over_threshold": 2.6,
        "under_threshold": 2.4,
        "home_advantage": 0.22,  # Reduced from 0.28
        "btts_baseline": 53,
        "form_threshold": 0.30,
        "total_goals_adjust": 0.0
    }
}

class TeamProfile:
    """Represents a team's statistical profile with form-first focus"""
    
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
        
        # xG performance ratio (goals/xG)
        self.xg_performance = self.goals_for / self.xg if self.xg > 0 else 1.0
        
        # Last 5 form data
        if is_home:
            self.last5_wins = int(data_dict.get('Last5_Home_Wins', 0))
            self.last5_draws = int(data_dict.get('Last5_Home_Draws', 0))
            self.last5_losses = int(data_dict.get('Last5_Home_Losses', 0))
            self.last5_gf = int(data_dict.get('Last5_Home_GF', 0))
            self.last5_ga = int(data_dict.get('Last5_Home_GA', 0))
            self.last5_gd = int(data_dict.get('Last5_Home_GD', 0))
            self.last5_pts = int(data_dict.get('Last5_Home_PTS', 0))
        else:
            self.last5_wins = int(data_dict.get('Last5_Away_Wins', 0))
            self.last5_draws = int(data_dict.get('Last5_Away_Draws', 0))
            self.last5_losses = int(data_dict.get('Last5_Away_Losses', 0))
            self.last5_gf = int(data_dict.get('Last5_Away_GF', 0))
            self.last5_ga = int(data_dict.get('Last5_Away_GA', 0))
            self.last5_gd = int(data_dict.get('Last5_Away_GD', 0))
            self.last5_pts = int(data_dict.get('Last5_Away_PTS', 0))
        
        # Calculate form momentum (LAST 3 GAMES WEIGHTED HEAVILY)
        self.form_momentum = self._calculate_form_momentum()
        
        # Calculate xG trend (recent vs average)
        self.xg_trend = self._calculate_xg_trend()
        
        # Calculate recent BTTS trend
        self.recent_btts_trend = self._calculate_btts_trend()
        
        # Crisis detection (less aggressive)
        self.has_attack_crisis = self._check_attack_crisis()
        self.has_defense_crisis = self._check_defense_crisis()
    
    def _calculate_form_momentum(self):
        """Calculate form momentum with last 3 games weighted 70%"""
        # Last 5 form (0-1 scale)
        last5_form = self.last5_pts / 15 if self.last5_pts > 0 else 0.33
        
        # Estimate last 3 form (simplified - assuming linear distribution)
        # In production, you'd want actual last 3 game data
        last3_est = min(1.0, (self.last5_pts / 5) * 1.2)  # Approximate
        
        # Weighted: 70% last 3 games, 30% last 5 games
        weighted_form = (last3_est * 0.7) + (last5_form * 0.3)
        
        # Convert to 0-0.3 scale for momentum
        return weighted_form * 0.3
    
    def _calculate_xg_trend(self):
        """Calculate if team is over/underperforming xG recently"""
        # Simplified: compare goals/xG ratio
        if self.xg > 0:
            ratio = self.goals_for / self.xg
            if ratio > 1.2:
                return 0.8  # Overperforming - likely unsustainable
            elif ratio < 0.8:
                return 1.2  # Underperforming - positive regression likely
        return 1.0  # Normal performance
    
    def _calculate_btts_trend(self):
        """Calculate recent BTTS tendency"""
        # Simplified: based on recent goals for/against
        if self.last5_gf > 0 and self.last5_ga > 0:
            # Both scoring and conceding in recent games
            return 1.1  # Slightly favors BTTS
        elif self.last5_gf == 0 or self.last5_ga == 0:
            # Not scoring or not conceding
            return 0.9  # Slightly against BTTS
        return 1.0
    
    def _check_attack_crisis(self):
        """Less aggressive attack crisis detection"""
        if self.xg_pg < 1.0:
            if self.last5_gf / 5 < 0.3:  # Very low scoring
                if self.last5_losses >= 3:  # Severe losing streak
                    return True
        return False
    
    def _check_defense_crisis(self):
        """Less aggressive defense crisis detection"""
        return self.goals_against_pg > 2.0  # Only crisis if conceding >2 per game
    
    def get_clean_sheet_probability(self, opponent_xg):
        """Calculate probability of keeping clean sheet"""
        if opponent_xg > 0:
            base_prob = math.exp(-opponent_xg) * 100
            # Adjust based on recent defensive form
            if self.last5_ga == 0:
                return min(95, base_prob * 1.3)  # Recent clean sheets
            elif self.last5_ga / 5 > 2.0:
                return max(5, base_prob * 0.7)  # Leaky recently
            return base_prob
        return 0.0

class MatchPredictor:
    """Main prediction engine implementing v3.0 logic"""
    
    def __init__(self, league_name):
        self.league_config = LEAGUE_CONFIGS.get(league_name.lower())
        if not self.league_config:
            raise ValueError(f"Unknown league: {league_name}")
    
    def predict(self, home_team: TeamProfile, away_team: TeamProfile):
        """Generate predictions for a match using form-first logic"""
        
        # 1. Calculate form-based predictions (PRIMARY)
        form_pred = self._predict_from_form(home_team, away_team)
        
        # 2. Calculate expected goals with xG trends
        home_xg, away_xg = self._calculate_expected_goals(home_team, away_team)
        total_xg = home_xg + away_xg
        
        # 3. Total goals prediction
        total_pred = self._predict_total_goals(total_xg, home_team, away_team)
        
        # 4. BTTS prediction
        btts_pred = self._predict_btts(home_xg, away_xg, home_team, away_team)
        
        # 5. Quality ratings (for display only, not primary prediction)
        home_quality, away_quality = self._calculate_display_quality(home_team, away_team)
        
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
                },
                "form_notes": {
                    "home_form": f"{home_team.last5_wins}-{home_team.last5_draws}-{home_team.last5_losses}",
                    "away_form": f"{away_team.last5_wins}-{away_team.last5_draws}-{away_team.last5_losses}",
                    "home_pts": f"{home_team.last5_pts}/15",
                    "away_pts": f"{away_team.last5_pts}/15"
                }
            },
            "predictions": [form_pred, total_pred, btts_pred]
        }
    
    def _predict_from_form(self, home: TeamProfile, away: TeamProfile):
        """PRIMARY PREDICTOR: Form-based winner prediction"""
        
        # Form difference (home form - away form)
        form_diff = home.form_momentum - away.form_momentum
        
        # Get form points per game (more intuitive)
        home_ppg = home.last5_pts / 5 if home.last5_pts > 0 else 0
        away_ppg = away.last5_pts / 5 if away.last5_pts > 0 else 0
        ppg_diff = home_ppg - away_ppg
        
        # Form threshold from league config
        form_threshold = self.league_config['form_threshold']
        
        # xG trend adjustments
        xg_trend_adjust = 0
        if home.xg_trend > 1.1:  # Home underperforming xG
            xg_trend_adjust += 0.1
        if away.xg_trend < 0.9:  # Away overperforming xG
            xg_trend_adjust -= 0.1
        
        # Adjusted form difference
        adjusted_diff = ppg_diff + xg_trend_adjust
        
        # Determine winner based on form
        if adjusted_diff > form_threshold:
            selection = "Home Win"
            # Conservative confidence: base + form advantage
            base_conf = 50
            form_boost = min(25, abs(adjusted_diff) * 40)  # Max 25% boost
            confidence = base_conf + form_boost
            
        elif adjusted_diff < -form_threshold:
            selection = "Away Win"
            base_conf = 50
            form_boost = min(25, abs(adjusted_diff) * 40)
            confidence = base_conf + form_boost
            
        else:
            selection = "Draw"
            # Draw confidence based on how close the match is
            closeness = 1.0 - min(1.0, abs(adjusted_diff) / form_threshold)
            confidence = 45 + (closeness * 20)  # 45-65% range
        
        # Adjust for home advantage (reduced in v3.0)
        if "Home" in selection:
            confidence += 5
        elif "Away" in selection:
            confidence -= 5
        
        # Adjust for crises
        if home.has_attack_crisis and "Home" in selection:
            confidence -= 8
        if away.has_attack_crisis and "Away" in selection:
            confidence -= 8
        
        # Cap confidence conservatively
        confidence = max(40, min(75, confidence))
        
        return {
            "type": "Match Winner",
            "selection": selection,
            "confidence": round(confidence, 1)
        }
    
    def _calculate_expected_goals(self, home: TeamProfile, away: TeamProfile):
        """Calculate expected goals with xG trend adjustments"""
        
        # Base xG from team profiles (per game averages)
        home_base = home.xg_pg * home.xg_trend  # Adjust for xG trend
        away_base = away.xg_pg * away.xg_trend
        
        # Defense adjustment
        home_def_adj = max(0.5, 2.0 - away.goals_against_pg)
        away_def_adj = max(0.5, 2.0 - home.goals_against_pg)
        
        # Raw calculations
        home_raw = (home_base + home_def_adj) / 2
        away_raw = (away_base + away_def_adj) / 2
        
        # Apply home advantage (reduced in v3.0)
        home_adv = self.league_config['home_advantage']
        home_bonus = 0.05 if home.last5_wins >= 3 else 0
        home_final = home_raw * (1.0 + home_adv + home_bonus)
        
        # Away penalty based on form
        if away.last5_pts < 6:  # Poor away form
            away_final = away_raw * 0.85
        else:
            away_final = away_raw * 0.92  # Reduced penalty
        
        # Attack crisis adjustments (less aggressive)
        if home.has_attack_crisis:
            home_final *= 0.8  # Was 0.6
        if away.has_attack_crisis:
            away_final *= 0.8  # Was 0.6
        
        return home_final, away_final
    
    def _predict_total_goals(self, total_xg: float, home: TeamProfile, away: TeamProfile):
        """Predict total goals with form considerations"""
        
        # League adjustment
        league_avg = self.league_config['avg_goals']
        league_adjust = self.league_config['total_goals_adjust']
        
        # Recent scoring form
        recent_scoring = 0
        if home.last5_gf / 5 > 1.5 and away.last5_gf / 5 > 1.5:
            recent_scoring = 0.3  # Both scoring recently
        elif home.last5_gf / 5 < 0.8 and away.last5_gf / 5 < 0.8:
            recent_scoring = -0.3  # Both struggling to score
        
        # Adjusted total
        total = (total_xg * 0.6) + (league_avg * 0.4) + league_adjust + recent_scoring
        
        # Compare to thresholds
        over_thresh = self.league_config['over_threshold']
        under_thresh = self.league_config['under_threshold']
        
        if total > over_thresh + 0.2:  # Clear over
            selection = "Over 2.5 Goals"
            confidence = 50 + ((total - over_thresh) * 20)
        elif total < under_thresh - 0.2:  # Clear under
            selection = "Under 2.5 Goals"
            confidence = 50 + ((under_thresh - total) * 20)
        else:  # Close to threshold
            selection = "Avoid Total Goals"
            confidence = 50
        
        # Confidence cap
        confidence = max(40, min(75, confidence))
        
        return {
            "type": "Total Goals",
            "selection": selection,
            "confidence": round(confidence, 1)
        }
    
    def _predict_btts(self, home_xg: float, away_xg: float, 
                     home: TeamProfile, away: TeamProfile):
        """Predict Both Teams To Score with conservative thresholds"""
        
        # Base Poisson probability
        home_score_prob = 1 - math.exp(-home_xg)
        away_score_prob = 1 - math.exp(-away_xg)
        btts_raw = home_score_prob * away_score_prob * 100
        
        # Recent BTTS trend adjustment
        trend_adjust = (home.recent_btts_trend * away.recent_btts_trend) * 100
        btts_adjusted = btts_raw * (trend_adjust / 100)
        
        # Clean sheet risk (less aggressive)
        home_cs_prob = home.get_clean_sheet_probability(away_xg)
        away_cs_prob = away.get_clean_sheet_probability(home_xg)
        
        if home_cs_prob > 45:  # Higher threshold
            btts_adjusted *= 0.9  # Only 10% reduction
        if away_cs_prob > 45:
            btts_adjusted *= 0.9
        
        # Attack crisis adjustment
        if home.has_attack_crisis:
            btts_adjusted *= 0.85  # Was 0.7
        if away.has_attack_crisis:
            btts_adjusted *= 0.85  # Was 0.7
        
        # Compare to league baseline
        baseline = self.league_config['btts_baseline']
        
        # WIDER AVOID ZONE (40-60% instead of 48-54%)
        if btts_adjusted > 60:  # Was baseline + 8
            selection = "Yes"
            confidence = min(75, btts_adjusted)
        elif btts_adjusted < 40:  # Was baseline - 10
            selection = "No"
            confidence = min(75, 100 - btts_adjusted)
        else:
            selection = "Avoid BTTS"
            confidence = 50
        
        return {
            "type": "BTTS",
            "selection": selection,
            "confidence": round(confidence, 1)
        }
    
    def _calculate_display_quality(self, home: TeamProfile, away: TeamProfile):
        """Calculate quality ratings for display only"""
        # Attack strength
        league_factor = self.league_config['avg_goals'] / 2.7
        home_attack = min(2.5, home.xg_pg * league_factor * home.xg_trend)
        away_attack = min(2.5, away.xg_pg * league_factor * away.xg_trend)
        
        # Defense strength (less aggressive min)
        home_defense = max(0.6, 2.0 - home.goals_against_pg)  # Was 0.4
        away_defense = max(0.6, 2.0 - away.goals_against_pg)  # Was 0.4
        
        # Quality with form weighted lower (60% attack, 30% defense, 10% form)
        home_quality = (home_attack * 0.6) + (home_defense * 0.3) + (home.form_momentum * 3.33)  # Scaled
        away_quality = (away_attack * 0.6) + (away_defense * 0.3) + (away.form_momentum * 3.33)
        
        return home_quality, away_quality
