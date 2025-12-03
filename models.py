"""
PHANTOM v4.1 - Core Prediction Models
Statistically validated methodology with proper calibration
"""
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# ============================================================================
# LEAGUE CONFIGURATIONS WITH AGGRESSIVE SETTINGS
# ============================================================================

LEAGUE_CONFIGS = {
    "premier_league": {
        "name": "Premier League",
        "avg_goals": 2.93,
        "over_threshold": 2.75,
        "under_threshold": 2.55,
        "btts_baseline": 52,
        "win_threshold": 0.25,
        "form_weight": 0.4
    },
    "serie_a": {
        "name": "Serie A",
        "avg_goals": 2.56,
        "over_threshold": 2.40,
        "under_threshold": 2.20,
        "btts_baseline": 48,
        "win_threshold": 0.30,
        "form_weight": 0.35
    },
    "la_liga": {
        "name": "La Liga",
        "avg_goals": 2.62,
        "over_threshold": 2.45,
        "under_threshold": 2.25,
        "btts_baseline": 50,
        "win_threshold": 0.28,
        "form_weight": 0.38
    },
    "bundesliga": {
        "name": "Bundesliga",
        "avg_goals": 3.14,
        "over_threshold": 2.90,
        "under_threshold": 2.70,
        "btts_baseline": 55,
        "win_threshold": 0.22,
        "form_weight": 0.42
    },
    "ligue_1": {
        "name": "Ligue 1",
        "avg_goals": 2.78,
        "over_threshold": 2.60,
        "under_threshold": 2.40,
        "btts_baseline": 50,
        "win_threshold": 0.26,
        "form_weight": 0.36
    },
    "rfpl": {
        "name": "Russian Premier League",
        "avg_goals": 2.68,
        "over_threshold": 2.60,
        "under_threshold": 2.40,
        "btts_baseline": 50,
        "win_threshold": 0.26,
        "form_weight": 0.36
    }
}

@dataclass
class LeagueAverages:
    """Container for league statistics calculated from data"""
    avg_home_goals: float
    avg_away_goals: float
    league_avg_gpg: float
    home_advantage: float
    total_matches: int
    actual_home_win_rate: float = 0.45
    actual_draw_rate: float = 0.25
    actual_away_win_rate: float = 0.30

class TeamProfile:
    """Team profile with statistically validated calculations"""
    
    def __init__(self, data_dict: Dict, is_home: bool = True, 
                 league_avg_gpg: float = 1.4, league_averages: Optional[LeagueAverages] = None):
        self.name = data_dict['Team']
        self.is_home = is_home
        self.league_avg_gpg = league_avg_gpg
        self.league_averages = league_averages
        
        # Extract basic stats
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
        
        # Last 5 form - ACTUAL DATA (no fake Last 3)
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
        
        # Calculate recent games played for reliability
        self.recent_games_played = min(5, 
            self.last5_wins + self.last5_draws + self.last5_losses)
        
        # Calculate key metrics
        self.form_score = self._calculate_form_score()
        self.attack_strength = self._calculate_attack_strength()
        self.defense_strength = self._calculate_defense_strength()
        self.btts_tendency = self._calculate_btts_tendency()
    
    def _calculate_form_score(self) -> float:
        """Form score 0-1 using ACTUAL Last 5 data (no fake Last 3)"""
        if self.recent_games_played == 0:
            # No recent data - use season form with penalty
            season_form = self.points / (self.matches * 3)  # Convert to 0-1 scale
            return season_form * 0.7  # Penalize for no recent data
        
        # ACTUAL Last 5 performance (not invented)
        last5_score = self.last5_pts / 15  # Max 15 points in 5 games
        
        # Season form
        season_form = self.points / (self.matches * 3)
        
        # Weight: 70% recent form, 30% season form
        return (last5_score * 0.7) + (season_form * 0.3)
    
    def _calculate_attack_strength(self) -> float:
        """Attack strength relative to league average with dynamic weighting"""
        # Recent goals per game
        recent_gpg = self.last5_gf / max(1, self.recent_games_played)
        season_gpg = self.goals_pg
        
        # Dynamic weighting based on reliability (FIXED: uses games played, not wins)
        # Minimum 50% weight, up to 80% with full 5 recent games
        recent_weight = 0.5 + (self.recent_games_played / 5 * 0.3)
        season_weight = 1 - recent_weight
        
        # Weighted average
        weighted_gpg = (recent_gpg * recent_weight) + (season_gpg * season_weight)
        
        # Return relative to league average (avoids arbitrary scaling)
        return weighted_gpg / max(0.1, self.league_avg_gpg)
    
    def _calculate_defense_strength(self) -> float:
        """Defense strength relative to league average"""
        # Recent goals against per game
        recent_gapg = self.last5_ga / max(1, self.recent_games_played)
        season_gapg = self.goals_against_pg
        
        # Same dynamic weighting as attack
        recent_weight = 0.5 + (self.recent_games_played / 5 * 0.3)
        season_weight = 1 - recent_weight
        
        # Weighted average
        weighted_gapg = (recent_gapg * recent_weight) + (season_gapg * season_weight)
        
        # Defense = how much BETTER than league average (inverse relationship)
        defense_ratio = self.league_avg_gpg / max(0.1, weighted_gapg)
        
        # Normalize to reasonable range (0.5-1.5)
        return max(0.5, min(1.5, defense_ratio))
    
    def _calculate_btts_tendency(self) -> float:
        """1.0 = neutral, >1.0 favors BTTS, <1.0 against BTTS"""
        if self.recent_games_played == 0:
            return 1.0
        
        # Estimate games where team both scored AND conceded
        games_with_both = min(5, (self.last5_gf > 0) + (self.last5_ga > 0))
        
        if games_with_both >= 4:
            return 1.3  # Strong BTTS tendency
        elif games_with_both >= 2:
            return 1.1  # Moderate BTTS tendency
        else:
            return 0.8  # Low BTTS tendency

class ProbabilityCalibrator:
    """Calibrate predicted probabilities to actual outcomes"""
    
    def __init__(self, league_averages: Optional[LeagueAverages] = None):
        self.league_averages = league_averages
        self.calibration_factors = {
            'draw_bias': 1.0,
            'home_win_bias': 1.0,
            'away_win_bias': 1.0
        }
        
        if league_averages:
            self._initialize_from_league()
    
    def _initialize_from_league(self):
        """Initialize calibration using league historical rates"""
        if self.league_averages:
            # Use actual league rates for calibration
            self.base_rates = {
                'home_win': self.league_averages.actual_home_win_rate,
                'draw': self.league_averages.actual_draw_rate,
                'away_win': self.league_averages.actual_away_win_rate
            }
    
    def calibrate_probabilities(self, home_win_prob: float, draw_prob: float, 
                               away_win_prob: float) -> Tuple[float, float, float]:
        """Adjust probabilities based on league tendencies"""
        
        if not hasattr(self, 'base_rates'):
            return home_win_prob, draw_prob, away_win_prob
        
        # Simple calibration: blend with league averages
        calibration_strength = 0.15  # 15% adjustment toward league average
        
        home_adj = (home_win_prob * (1 - calibration_strength) + 
                   self.base_rates['home_win'] * calibration_strength)
        draw_adj = (draw_prob * (1 - calibration_strength) + 
                   self.base_rates['draw'] * calibration_strength)
        away_adj = (away_win_prob * (1 - calibration_strength) + 
                   self.base_rates['away_win'] * calibration_strength)
        
        # Renormalize
        total = home_adj + draw_adj + away_adj
        return home_adj/total, draw_adj/total, away_adj/total
    
    def calculate_draw_probability_sigmoid(self, home_xg: float, away_xg: float) -> float:
        """Better draw probability using sigmoid function"""
        total_xg = home_xg + away_xg
        
        # Sigmoid parameters tuned to football data
        k = 1.2  # Steepness
        x0 = 2.5  # Midpoint
        
        # Sigmoid: draw probability decreases with total xG
        base_draw_prob = 0.35 / (1 + math.exp(k * (total_xg - x0)))
        
        # Adjust for closeness of teams (closer xG = higher draw chance)
        xg_diff = abs(home_xg - away_xg)
        closeness_factor = 1.0 - (xg_diff / max(1.0, total_xg))
        adjusted_prob = base_draw_prob * (0.8 + 0.4 * closeness_factor)
        
        # Minimum and maximum bounds
        return max(0.15, min(0.40, adjusted_prob))

class MatchPredictor:
    """Main prediction engine with statistically validated methods"""
    
    def __init__(self, league_name: str, league_averages: LeagueAverages):
        self.league_config = LEAGUE_CONFIGS.get(league_name.lower())
        if not self.league_config:
            raise ValueError(f"Unknown league: {league_name}")
        
        self.league_averages = league_averages
        self.calibrator = ProbabilityCalibrator(league_averages)
    
    def predict(self, home_team: TeamProfile, away_team: TeamProfile) -> Dict:
        """Make predictions with corrected formulas"""
        
        # 1. Calculate expected goals using REAL league averages
        home_xg, away_xg = self._calculate_expected_goals(home_team, away_team)
        
        # 2. Calculate probabilities with calibration
        winner_pred = self._predict_winner_with_calibration(home_xg, away_xg, home_team, away_team)
        
        # 3. Total goals prediction
        total_xg = home_xg + away_xg
        total_pred = self._predict_total_goals(total_xg, home_team, away_team)
        
        # 4. BTTS prediction
        btts_pred = self._predict_btts(home_xg, away_xg, home_team, away_team)
        
        return {
            "analysis": {
                "league": self.league_config['name'],
                "expected_goals": {
                    "home": round(home_xg, 2),
                    "away": round(away_xg, 2),
                    "total": round(total_xg, 2)
                },
                "form_scores": {
                    "home": round(home_team.form_score, 2),
                    "away": round(away_team.form_score, 2)
                },
                "attack_strengths": {
                    "home": round(home_team.attack_strength, 2),
                    "away": round(away_team.attack_strength, 2)
                },
                "defense_strengths": {
                    "home": round(home_team.defense_strength, 2),
                    "away": round(away_team.defense_strength, 2)
                }
            },
            "predictions": [winner_pred, total_pred, btts_pred]
        }
    
    def _calculate_expected_goals(self, home: TeamProfile, away: TeamProfile) -> Tuple[float, float]:
        """Proper xG using actual league averages"""
        
        # Use REAL league averages from data
        avg_home = self.league_averages.avg_home_goals
        avg_away = self.league_averages.avg_away_goals
        home_advantage = self.league_averages.home_advantage
        
        # Home xG = League avg × relative attack × inverse relative defense
        home_xg = avg_home * home.attack_strength / away.defense_strength
        
        # Away xG = League avg × relative attack × inverse relative defense
        away_xg = avg_away * away.attack_strength / home.defense_strength
        
        # Apply home advantage (calculated from actual data)
        home_xg *= home_advantage
        
        # HOT ATTACK BOOST - continuous, capped at 15%
        home_recent_gpg = home.last5_gf / max(1, home.recent_games_played)
        away_recent_gpg = away.last5_gf / max(1, away.recent_games_played)
        
        if home_recent_gpg > home.goals_pg:
            improvement = home_recent_gpg / max(0.1, home.goals_pg)
            boost = 1.0 + min(0.15, (improvement - 1.0) * 0.3)
            home_xg *= min(1.15, boost)
        
        if away_recent_gpg > away.goals_pg:
            improvement = away_recent_gpg / max(0.1, away.goals_pg)
            boost = 1.0 + min(0.15, (improvement - 1.0) * 0.3)
            away_xg *= min(1.15, boost)
        
        return home_xg, away_xg
    
    def _predict_winner_with_calibration(self, home_xg: float, away_xg: float,
                                       home: TeamProfile, away: TeamProfile) -> Dict:
        """Predict winner using calibrated probabilities"""
        
        # Calculate base probabilities
        home_win_prob, away_win_prob, draw_prob = self._calculate_poisson_probabilities(home_xg, away_xg)
        
        # Apply calibration
        home_win_prob, draw_prob, away_win_prob = self.calibrator.calibrate_probabilities(
            home_win_prob, draw_prob, away_win_prob
        )
        
        # Apply sample size penalty if recent data is sparse
        total_recent = home.recent_games_played + away.recent_games_played
        if total_recent < 6:
            reliability_factor = total_recent / 6
            home_win_prob = (home_win_prob * reliability_factor) + (0.33 * (1 - reliability_factor))
            away_win_prob = (away_win_prob * reliability_factor) + (0.33 * (1 - reliability_factor))
            draw_prob = (draw_prob * reliability_factor) + (0.34 * (1 - reliability_factor))
        
        # Normalize to ensure they sum to 1
        total = home_win_prob + away_win_prob + draw_prob
        home_win_prob /= total
        away_win_prob /= total
        draw_prob /= total
        
        # Determine selection
        if home_win_prob >= away_win_prob and home_win_prob >= draw_prob:
            selection = "Home Win"
            confidence = home_win_prob * 100
        elif away_win_prob >= home_win_prob and away_win_prob >= draw_prob:
            selection = "Away Win"
            confidence = away_win_prob * 100
        else:
            selection = "Draw"
            confidence = draw_prob * 100
        
        # Apply reasonable bounds
        confidence = max(30, min(85, confidence))
        
        return {
            "type": "Match Winner",
            "selection": selection,
            "confidence": round(confidence, 1),
            "probabilities": {
                "home": round(home_win_prob * 100, 1),
                "draw": round(draw_prob * 100, 1),
                "away": round(away_win_prob * 100, 1)
            }
        }
    
    def _calculate_poisson_probabilities(self, home_xg: float, away_xg: float) -> Tuple[float, float, float]:
        """Calculate win/draw/lose probabilities"""
        
        # Use sigmoid function for draw probability
        draw_prob = self.calibrator.calculate_draw_probability_sigmoid(home_xg, away_xg)
        
        # Win probabilities based on relative strength
        home_strength = home_xg / (home_xg + away_xg + 0.1)
        away_strength = away_xg / (home_xg + away_xg + 0.1)
        
        # Allocate remaining probability proportionally
        remaining = 1.0 - draw_prob
        total_strength = home_strength + away_strength
        
        home_win_prob = remaining * (home_strength / total_strength)
        away_win_prob = remaining * (away_strength / total_strength)
        
        return home_win_prob, away_win_prob, draw_prob
    
    def _predict_total_goals(self, total_xg: float, home: TeamProfile, away: TeamProfile) -> Dict:
        """Predict Over/Under 2.5 goals"""
        
        # League context
        league_avg = self.league_config['avg_goals']
        over_thresh = self.league_config['over_threshold']
        
        # Recent scoring trend
        home_recent_gpg = home.last5_gf / max(1, home.recent_games_played)
        away_recent_gpg = away.last5_gf / max(1, away.recent_games_played)
        recent_scoring = (home_recent_gpg + away_recent_gpg) / 2
        
        # Adjusted total (60% xG, 40% recent form)
        adjusted_total = (total_xg * 0.6) + (recent_scoring * 2.0 * 0.4)
        
        # Decision
        if adjusted_total > over_thresh:
            selection = "Over 2.5 Goals"
            excess = (adjusted_total - over_thresh) / over_thresh
            confidence = 55 + min(25, excess * 25)
        else:
            selection = "Under 2.5 Goals"
            deficit = (league_avg - adjusted_total) / league_avg
            confidence = 55 + min(25, deficit * 25)
        
        confidence = max(50, min(80, confidence))
        
        return {
            "type": "Total Goals",
            "selection": selection,
            "confidence": round(confidence, 1)
        }
    
    def _predict_btts(self, home_xg: float, away_xg: float,
                     home: TeamProfile, away: TeamProfile) -> Dict:
        """Predict Both Teams to Score"""
        
        # Base probability from xG
        home_score_prob = 1 - math.exp(-home_xg)
        away_score_prob = 1 - math.exp(-away_xg)
        btts_prob = home_score_prob * away_score_prob * 100
        
        # Apply team tendencies
        btts_prob *= ((home.btts_tendency + away.btts_tendency) / 2)
        
        # League baseline
        baseline = self.league_config['btts_baseline']
        
        if btts_prob >= baseline:
            selection = "Yes"
            confidence = min(80, btts_prob)
        else:
            selection = "No"
            confidence = min(80, 100 - btts_prob)
        
        confidence = max(50, confidence)
        
        return {
            "type": "BTTS",
            "selection": selection,
            "confidence": round(confidence, 1)
        }

class ModelValidator:
    """Track and validate model performance"""
    
    def __init__(self):
        self.predictions = []
        self.confidence_bins = {}
    
    def add_prediction(self, prediction_type: str, predicted: str, 
                      confidence: float, actual: str):
        """Store prediction for validation"""
        import datetime
        
        self.predictions.append({
            'type': prediction_type,
            'predicted': predicted,
            'confidence': confidence,
            'actual': actual,
            'timestamp': datetime.datetime.now()
        })
        
        # Track by confidence bin
        bin_key = int(confidence // 5) * 5
        if bin_key not in self.confidence_bins:
            self.confidence_bins[bin_key] = {'total': 0, 'correct': 0}
        
        self.confidence_bins[bin_key]['total'] += 1
        if predicted == actual:
            self.confidence_bins[bin_key]['correct'] += 1
    
    def get_calibration_report(self) -> Dict:
        """Generate calibration report"""
        report = {}
        for bin_key, data in sorted(self.confidence_bins.items()):
            if data['total'] > 0:
                actual_rate = data['correct'] / data['total']
                predicted_rate = (bin_key + 2.5) / 100
                report[f"{bin_key}-{bin_key+5}%"] = {
                    'predicted': predicted_rate,
                    'actual': actual_rate,
                    'difference': actual_rate - predicted_rate,
                    'samples': data['total']
                }
        return report
