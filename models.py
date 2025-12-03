"""
PHANTOM v4.1 - Core Prediction Models (FIXED VERSION)
Proper home/away distinction in calculations
"""
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# ============================================================================
# LEAGUE CONFIGURATIONS WITH AGGRESSIVE SETTINGS
# ============================================================================

LEAGUE_CONFIGS = {
    "premier_league": {
        "name": "Premier League",
        "avg_goals": 2.93,  # This is total goals per MATCH (home + away)
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
    avg_home_goals: float        # Average goals by home teams
    avg_away_goals: float        # Average goals by away teams
    league_avg_gpg: float        # Average goals per team per game (for reference)
    home_advantage: float        # Home goals / Away goals
    total_matches: int
    actual_home_win_rate: float = 0.45
    actual_draw_rate: float = 0.25
    actual_away_win_rate: float = 0.30

class TeamProfile:
    """Team profile with PROPER home/away distinction"""
    
    def __init__(self, data_dict: Dict, is_home: bool = True, 
                 league_averages: Optional[LeagueAverages] = None):
        self.name = data_dict['Team']
        self.is_home = is_home
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
        
        # Last 5 form - ACTUAL DATA
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
        
        # Calculate key metrics WITH PROPER HOME/AWAY DISTINCTION
        self.form_score = self._calculate_form_score()
        self.attack_strength = self._calculate_attack_strength()
        self.defense_strength = self._calculate_defense_strength()
        self.btts_tendency = self._calculate_btts_tendency()
        
        # Debug output
        self._debug_info()
    
    def _debug_info(self):
        """Debug output to verify calculations"""
        print(f"\n🔍 DEBUG - {self.name} ({'Home' if self.is_home else 'Away'}):")
        print(f"  Goals: {self.goals_for} in {self.matches} matches = {self.goals_pg:.2f} GPG")
        print(f"  Goals Against: {self.goals_against} = {self.goals_against_pg:.2f} GAPG")
        print(f"  Form Score: {self.form_score:.2f}")
        print(f"  Attack Strength: {self.attack_strength:.2f}")
        print(f"  Defense Strength: {self.defense_strength:.2f}")
    
    def _calculate_form_score(self) -> float:
        """Form score 0-1 using ACTUAL Last 5 data"""
        if self.recent_games_played == 0:
            # No recent data - use season form with penalty
            season_form = self.points / (self.matches * 3)  # Convert to 0-1 scale
            return season_form * 0.7
        
        # ACTUAL Last 5 performance
        last5_score = self.last5_pts / 15  # Max 15 points in 5 games
        
        # Season form
        season_form = self.points / (self.matches * 3)
        
        # Weight: 70% recent form, 30% season form
        return (last5_score * 0.7) + (season_form * 0.3)
    
    def _calculate_attack_strength(self) -> float:
        """Attack strength relative to PROPER league average"""
        
        # Recent goals per game
        recent_gpg = self.last5_gf / max(1, self.recent_games_played)
        season_gpg = self.goals_pg
        
        # Dynamic weighting based on reliability
        recent_weight = 0.5 + (self.recent_games_played / 5 * 0.3)
        season_weight = 1 - recent_weight
        
        # Weighted average
        weighted_gpg = (recent_gpg * recent_weight) + (season_gpg * season_weight)
        
        # 🔥 CRITICAL FIX: Compare to PROPER league average
        if self.is_home:
            # Home teams: compare to league average HOME goals
            if self.league_averages:
                league_baseline = self.league_averages.avg_home_goals
            else:
                league_baseline = 1.65  # Premier League home average
        else:
            # Away teams: compare to league average AWAY goals
            if self.league_averages:
                league_baseline = self.league_averages.avg_away_goals
            else:
                league_baseline = 1.28  # Premier League away average
        
        print(f"    Attack Calc: {weighted_gpg:.2f} GPG vs {league_baseline:.2f} baseline")
        
        # Calculate relative strength
        attack_strength = weighted_gpg / max(0.1, league_baseline)
        
        # Reasonable bounds (no team is 3x average or 0.3x average)
        return min(2.0, max(0.5, attack_strength))
    
    def _calculate_defense_strength(self) -> float:
        """Defense strength - how much BETTER than typical opponent"""
        
        # Recent goals against per game
        recent_gapg = self.last5_ga / max(1, self.recent_games_played)
        season_gapg = self.goals_against_pg
        
        # Same dynamic weighting as attack
        recent_weight = 0.5 + (self.recent_games_played / 5 * 0.3)
        season_weight = 1 - recent_weight
        
        # Weighted average
        weighted_gapg = (recent_gapg * recent_weight) + (season_gapg * season_weight)
        
        # 🔥 CRITICAL FIX: Defense = how much BETTER than typical opponent
        # Home defense: compared to typical away team scoring
        # Away defense: compared to typical home team scoring
        if self.is_home:
            # Home team defense: opponents are away teams
            if self.league_averages:
                typical_opponent_scoring = self.league_averages.avg_away_goals
            else:
                typical_opponent_scoring = 1.28
        else:
            # Away team defense: opponents are home teams
            if self.league_averages:
                typical_opponent_scoring = self.league_averages.avg_home_goals
            else:
                typical_opponent_scoring = 1.65
        
        print(f"    Defense Calc: Opponents avg {typical_opponent_scoring:.2f}, we concede {weighted_gapg:.2f}")
        
        # Defense strength = opponent average / our conceded
        # Higher = better defense (we concede less than typical)
        defense_strength = typical_opponent_scoring / max(0.1, weighted_gapg)
        
        # Reasonable bounds
        return min(1.8, max(0.5, defense_strength))
    
    def _calculate_btts_tendency(self) -> float:
        """1.0 = neutral, >1.0 favors BTTS, <1.0 against BTTS"""
        if self.recent_games_played == 0:
            return 1.0
        
        # Both scoring AND conceding recently
        scored_in_games = min(self.recent_games_played, self.last5_gf)
        conceded_in_games = min(self.recent_games_played, self.last5_ga)
        
        # Games with both GF and GA
        if scored_in_games >= 3 and conceded_in_games >= 3:
            return 1.3  # Strong BTTS tendency
        elif scored_in_games >= 2 and conceded_in_games >= 2:
            return 1.1  # Moderate BTTS tendency
        elif scored_in_games == 0 or conceded_in_games == 0:
            return 0.8  # Low BTTS tendency
        return 1.0

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
        calibration_strength = 0.15
        
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
        
        # Sigmoid parameters
        k = 1.2
        x0 = 2.5
        
        # Sigmoid: draw probability decreases with total xG
        base_draw_prob = 0.35 / (1 + math.exp(k * (total_xg - x0)))
        
        # Adjust for closeness of teams
        xg_diff = abs(home_xg - away_xg)
        closeness_factor = 1.0 - (xg_diff / max(1.0, total_xg))
        adjusted_prob = base_draw_prob * (0.8 + 0.4 * closeness_factor)
        
        return max(0.15, min(0.40, adjusted_prob))

class MatchPredictor:
    """Main prediction engine with FIXED calculations"""
    
    def __init__(self, league_name: str, league_averages: LeagueAverages):
        self.league_config = LEAGUE_CONFIGS.get(league_name.lower())
        if not self.league_config:
            raise ValueError(f"Unknown league: {league_name}")
        
        self.league_averages = league_averages
        self.calibrator = ProbabilityCalibrator(league_averages)
        
        print(f"\n🎯 PREDICTOR INITIALIZED FOR {league_name.upper()}")
        print(f"  League Home Avg: {league_averages.avg_home_goals:.2f}")
        print(f"  League Away Avg: {league_averages.avg_away_goals:.2f}")
        print(f"  Home Advantage: {league_averages.home_advantage:.2f}x")
    
    def predict(self, home_team: TeamProfile, away_team: TeamProfile) -> Dict:
        """Make predictions with FIXED formulas"""
        
        print(f"\n🔮 PREDICTING: {home_team.name} vs {away_team.name}")
        
        # 1. Calculate expected goals with FIXED formula
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
        """Proper xG using corrected formulas"""
        
        print(f"\n📊 EXPECTED GOALS CALCULATION:")
        
        # Use REAL league averages
        avg_home = self.league_averages.avg_home_goals
        avg_away = self.league_averages.avg_away_goals
        home_advantage = self.league_averages.home_advantage
        
        print(f"  Base: Home={avg_home:.2f}, Away={avg_away:.2f}, Advantage={home_advantage:.2f}x")
        print(f"  Home: Attack={home.attack_strength:.2f}, Opp Defense={away.defense_strength:.2f}")
        print(f"  Away: Attack={away.attack_strength:.2f}, Opp Defense={home.defense_strength:.2f}")
        
        # 🔥 CORRECTED FORMULA
        # Home xG = League home avg × home attack strength / away defense strength
        home_base_xg = avg_home * home.attack_strength / max(0.5, away.defense_strength)
        
        # Away xG = League away avg × away attack strength / home defense strength
        away_base_xg = avg_away * away.attack_strength / max(0.5, home.defense_strength)
        
        # Apply home advantage (already partly in home_base_xg, but additional boost)
        home_xg = home_base_xg * (home_advantage ** 0.5)  # Square root for conservative boost
        
        print(f"  Base xG: Home={home_base_xg:.2f}, Away={away_base_xg:.2f}")
        print(f"  After Advantage: Home={home_xg:.2f}")
        
        # HOT ATTACK BOOST - conservative
        home_recent_gpg = home.last5_gf / max(1, home.recent_games_played)
        away_recent_gpg = away.last5_gf / max(1, away.recent_games_played)
        
        if home_recent_gpg > home.goals_pg * 1.2:
            improvement = min(1.5, home_recent_gpg / max(0.1, home.goals_pg))
            boost = 1.0 + (improvement - 1.0) * 0.15  # Max 7.5% boost
            home_xg *= min(1.075, boost)
            print(f"  Home Hot Boost: {improvement:.2f}x → {min(1.075, boost):.3f}")
        
        if away_recent_gpg > away.goals_pg * 1.2:
            improvement = min(1.5, away_recent_gpg / max(0.1, away.goals_pg))
            boost = 1.0 + (improvement - 1.0) * 0.15
            away_base_xg *= min(1.075, boost)
            print(f"  Away Hot Boost: {improvement:.2f}x → {min(1.075, boost):.3f}")
        
        # CAP AT REALISTIC VALUES
        home_xg = min(3.5, max(0.2, home_xg))
        away_xg = min(3.0, max(0.2, away_base_xg))  # Away teams score less
        
        print(f"  Final xG: Home={home_xg:.2f}, Away={away_xg:.2f}, Total={home_xg+away_xg:.2f}")
        
        return home_xg, away_xg
    
    def _predict_winner_with_calibration(self, home_xg: float, away_xg: float,
                                       home: TeamProfile, away: TeamProfile) -> Dict:
        """Predict winner using calibrated probabilities"""
        
        # Calculate base probabilities
        home_win_prob, away_win_prob, draw_prob = self._calculate_poisson_probabilities(home_xg, away_xg)
        
        print(f"\n🎲 WINNER PROBABILITIES:")
        print(f"  Raw: Home={home_win_prob:.2%}, Draw={draw_prob:.2%}, Away={away_win_prob:.2%}")
        
        # Apply calibration
        home_win_prob, draw_prob, away_win_prob = self.calibrator.calibrate_probabilities(
            home_win_prob, draw_prob, away_win_prob
        )
        
        print(f"  Calibrated: Home={home_win_prob:.2%}, Draw={draw_prob:.2%}, Away={away_win_prob:.2%}")
        
        # Apply sample size penalty if recent data is sparse
        total_recent = home.recent_games_played + away.recent_games_played
        if total_recent < 6:
            reliability_factor = total_recent / 6
            home_win_prob = (home_win_prob * reliability_factor) + (0.33 * (1 - reliability_factor))
            away_win_prob = (away_win_prob * reliability_factor) + (0.33 * (1 - reliability_factor))
            draw_prob = (draw_prob * reliability_factor) + (0.34 * (1 - reliability_factor))
            print(f"  Reliability Adj ({total_recent}/6 games): factor={reliability_factor:.2f}")
        
        # Normalize
        total = home_win_prob + away_win_prob + draw_prob
        home_win_prob /= total
        away_win_prob /= total
        draw_prob /= total
        
        print(f"  Normalized: Home={home_win_prob:.2%}, Draw={draw_prob:.2%}, Away={away_win_prob:.2%}")
        
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
        
        print(f"  Selection: {selection} ({confidence:.1f}%)")
        
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
        league_avg = self.league_config['avg_goals']  # Total goals per match
        over_thresh = self.league_config['over_threshold']
        
        # Recent scoring trend
        home_recent_gpg = home.last5_gf / max(1, home.recent_games_played)
        away_recent_gpg = away.last5_gf / max(1, away.recent_games_played)
        recent_scoring = (home_recent_gpg + away_recent_gpg)
        
        print(f"\n⚽ TOTAL GOALS CALCULATION:")
        print(f"  xG Total: {total_xg:.2f}")
        print(f"  Recent Scoring: {recent_scoring:.2f} GPG combined")
        print(f"  League Avg: {league_avg:.2f}, Over Threshold: {over_thresh:.2f}")
        
        # Adjusted total (60% xG, 40% recent form)
        adjusted_total = (total_xg * 0.6) + (recent_scoring * 0.4)
        
        print(f"  Adjusted Total: {adjusted_total:.2f}")
        
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
        
        print(f"  Selection: {selection} ({confidence:.1f}%)")
        
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
        
        print(f"\n🎯 BTTS CALCULATION:")
        print(f"  Home Score Prob: {home_score_prob:.2%}")
        print(f"  Away Score Prob: {away_score_prob:.2%}")
        print(f"  Base BTTS Prob: {btts_prob:.1f}%")
        
        # Apply team tendencies
        tendency_factor = (home.btts_tendency + away.btts_tendency) / 2
        btts_prob *= tendency_factor
        
        print(f"  Team Tendency Factor: {tendency_factor:.2f}")
        print(f"  Adjusted BTTS Prob: {btts_prob:.1f}%")
        
        # League baseline
        baseline = self.league_config['btts_baseline']
        
        if btts_prob >= baseline:
            selection = "Yes"
            confidence = min(80, btts_prob)
        else:
            selection = "No"
            confidence = min(80, 100 - btts_prob)
        
        confidence = max(50, confidence)
        
        print(f"  League Baseline: {baseline}%")
        print(f"  Selection: {selection} ({confidence:.1f}%)")
        
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
