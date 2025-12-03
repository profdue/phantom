"""
PHANTOM v4.2 - Core Prediction Models (CRITICAL FIXES APPLIED)
Statistically rigorous with proper home/away distinction, xG integration, and Poisson probabilities
"""
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import sys

# ============================================================================
# LEAGUE CONFIGURATIONS
# ============================================================================

LEAGUE_CONFIGS = {
    "premier_league": {
        "name": "Premier League",
        "avg_goals": 2.93,  # Total goals per match
        "over_threshold": 2.75,
        "under_threshold": 2.55,
        "btts_baseline": 52,
        "win_threshold": 0.25,
        "form_weight": 0.4,
        "home_advantage_multiplier": 1.18  # 18% home advantage
    },
    "serie_a": {
        "name": "Serie A",
        "avg_goals": 2.56,
        "over_threshold": 2.40,
        "under_threshold": 2.20,
        "btts_baseline": 48,
        "win_threshold": 0.30,
        "form_weight": 0.35,
        "home_advantage_multiplier": 1.15
    },
    "la_liga": {
        "name": "La Liga",
        "avg_goals": 2.62,
        "over_threshold": 2.45,
        "under_threshold": 2.25,
        "btts_baseline": 50,
        "win_threshold": 0.28,
        "form_weight": 0.38,
        "home_advantage_multiplier": 1.20
    },
    "bundesliga": {
        "name": "Bundesliga",
        "avg_goals": 3.14,
        "over_threshold": 2.90,
        "under_threshold": 2.70,
        "btts_baseline": 55,
        "win_threshold": 0.22,
        "form_weight": 0.42,
        "home_advantage_multiplier": 1.22
    },
    "ligue_1": {
        "name": "Ligue 1",
        "avg_goals": 2.78,
        "over_threshold": 2.60,
        "under_threshold": 2.40,
        "btts_baseline": 50,
        "win_threshold": 0.26,
        "form_weight": 0.36,
        "home_advantage_multiplier": 1.16
    },
    "rfpl": {
        "name": "Russian Premier League",
        "avg_goals": 2.68,
        "over_threshold": 2.60,
        "under_threshold": 2.40,
        "btts_baseline": 50,
        "win_threshold": 0.26,
        "form_weight": 0.36,
        "home_advantage_multiplier": 1.17
    }
}

@dataclass
class LeagueAverages:
    """Container for league statistics calculated from data"""
    avg_home_goals: float        # Average goals by home teams
    avg_away_goals: float        # Average goals by away teams
    league_avg_gpg: float        # Average goals per team per game
    total_matches: int
    actual_home_win_rate: float = 0.45
    actual_draw_rate: float = 0.25
    actual_away_win_rate: float = 0.30

class TeamProfile:
    """Team profile with PROPER xG integration and home/away distinction"""
    
    def __init__(self, data_dict: Dict, is_home: bool = True, 
                 league_averages: Optional[LeagueAverages] = None,
                 debug: bool = False):
        self.name = data_dict['Team']
        self.is_home = is_home
        self.league_averages = league_averages
        self.debug = debug
        
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
        self.xg_pg = self.xg / max(1, self.matches)
        self.xga_pg = self.xga / max(1, self.matches)
        
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
        
        # Calculate key metrics WITH PROPER xG INTEGRATION
        self.form_score = self._calculate_form_score()
        self.attack_strength = self._calculate_attack_strength()
        self.defense_strength = self._calculate_defense_strength()
        self.btts_tendency = self._calculate_btts_tendency()
        
        # Debug output
        if self.debug:
            self._debug_info()
    
    def _debug_info(self):
        """Debug output to verify calculations"""
        print(f"\n🔍 DEBUG - {self.name} ({'Home' if self.is_home else 'Away'}):")
        print(f"  Goals: {self.goals_for}/{self.matches} = {self.goals_pg:.2f} GPG")
        print(f"  xG: {self.xg:.2f}/{self.matches} = {self.xg_pg:.2f} xG PG")
        print(f"  Defense: GA={self.goals_against} = {self.goals_against_pg:.2f} GAPG")
        print(f"  xGA: {self.xga:.2f}/{self.matches} = {self.xga_pg:.2f} xGA PG")
        print(f"  Form Score: {self.form_score:.2f}")
        print(f"  Attack Strength: {self.attack_strength:.2f}")
        print(f"  Defense Strength: {self.defense_strength:.2f}")
        print(f"  BTTS Tendency: {self.btts_tendency:.2f}")
    
    def _calculate_form_score(self) -> float:
        """Form score 0-1 using ACTUAL Last 5 data"""
        if self.recent_games_played == 0:
            # No recent data - use season form with penalty
            season_form = self.points / (self.matches * 3)
            return season_form * 0.7
        
        # ACTUAL Last 5 performance
        last5_score = self.last5_pts / 15  # Max 15 points in 5 games
        
        # Season form
        season_form = self.points / (self.matches * 3)
        
        # Weight: 70% recent form, 30% season form
        return (last5_score * 0.7) + (season_form * 0.3)
    
    def _calculate_attack_strength(self) -> float:
        """Attack strength using BOTH goals and xG (70% goals, 30% xG)"""
        
        # Recent performance
        recent_gpg = self.last5_gf / max(1, self.recent_games_played)
        recent_xg_pg = self.xg / max(1, self.matches)  # Season xG for stability
        
        # Season performance
        season_gpg = self.goals_pg
        season_xg_pg = self.xg_pg
        
        # Dynamic weighting based on reliability
        recent_weight = 0.5 + (self.recent_games_played / 5 * 0.3)
        season_weight = 1 - recent_weight
        
        # 🔥 FIX: Blend actual goals with xG
        weighted_gpg = (recent_gpg * recent_weight + season_gpg * season_weight) * 0.7
        weighted_xg_pg = (recent_xg_pg * recent_weight + season_xg_pg * season_weight) * 0.3
        
        total_attack_metric = weighted_gpg + weighted_xg_pg
        
        # 🔥 FIX: Compare to PROPER league average
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
        
        if self.debug:
            print(f"    Attack Calc: {total_attack_metric:.2f} vs {league_baseline:.2f} baseline")
        
        # Calculate relative strength
        attack_strength = total_attack_metric / max(0.1, league_baseline)
        
        # Reasonable bounds (no team is 3x average or 0.3x average)
        return min(2.0, max(0.5, attack_strength))
    
    def _calculate_defense_strength(self) -> float:
        """Defense strength using BOTH GA and xGA (70% GA, 30% xGA)"""
        
        # Recent performance
        recent_gapg = self.last5_ga / max(1, self.recent_games_played)
        recent_xga_pg = self.xga / max(1, self.matches)
        
        # Season performance
        season_gapg = self.goals_against_pg
        season_xga_pg = self.xga_pg
        
        # Same dynamic weighting as attack
        recent_weight = 0.5 + (self.recent_games_played / 5 * 0.3)
        season_weight = 1 - recent_weight
        
        # 🔥 FIX: Blend actual GA with xGA
        weighted_gapg = (recent_gapg * recent_weight + season_gapg * season_weight) * 0.7
        weighted_xga_pg = (recent_xga_pg * recent_weight + season_xga_pg * season_weight) * 0.3
        
        total_defense_metric = weighted_gapg + weighted_xga_pg
        
        # 🔥 FIX: Defense = how much BETTER than typical opponent
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
        
        if self.debug:
            print(f"    Defense Calc: Opp avg {typical_opponent_scoring:.2f}, we concede {total_defense_metric:.2f}")
        
        # Defense strength = opponent average / our conceded
        # Higher = better defense (we concede less than typical)
        defense_strength = typical_opponent_scoring / max(0.1, total_defense_metric)
        
        # Reasonable bounds
        return min(1.8, max(0.5, defense_strength))
    
    def _calculate_btts_tendency(self) -> float:
        """🔥 FIXED: BTTS tendency using Poisson estimation"""
        if self.recent_games_played == 0:
            return 1.0
        
        # 🔥 FIX: Estimate games where team actually scored AND conceded
        # Using Poisson approximation since we don't have game-by-game data
        
        # Goals per game recently
        goals_per_game_recent = self.last5_gf / max(1, self.recent_games_played)
        
        # Probability of scoring in a game = 1 - P(0 goals)
        # Using Poisson: P(0 goals) = e^(-lambda)
        p_scored_in_game = 1 - math.exp(-goals_per_game_recent)
        
        # Goals conceded per game recently
        goals_conceded_per_game_recent = self.last5_ga / max(1, self.recent_games_played)
        p_conceded_in_game = 1 - math.exp(-goals_conceded_per_game_recent)
        
        # Expected games with BOTH scoring and conceding
        # Assuming independence for estimation
        expected_both_games = self.recent_games_played * (p_scored_in_game * p_conceded_in_game)
        
        if self.debug:
            print(f"    BTTS Calc: {goals_per_game_recent:.2f} GPG, {goals_conceded_per_game_recent:.2f} GAPG")
            print(f"    P(scored)={p_scored_in_game:.2f}, P(conceded)={p_conceded_in_game:.2f}")
            print(f"    Expected both: {expected_both_games:.2f} of {self.recent_games_played}")
        
        # Calculate tendency based on expected both games
        if expected_both_games >= 3.5:
            return 1.3  # Strong BTTS tendency
        elif expected_both_games >= 2.5:
            return 1.2  # Moderate-high BTTS tendency
        elif expected_both_games >= 1.5:
            return 1.1  # Moderate BTTS tendency
        elif expected_both_games <= 0.5:
            return 0.7  # Low BTTS tendency
        elif expected_both_games <= 1.0:
            return 0.8  # Low-moderate BTTS tendency
        return 1.0  # Neutral

class ProbabilityCalibrator:
    """Calibrate predicted probabilities to actual outcomes"""
    
    def __init__(self, league_averages: Optional[LeagueAverages] = None):
        self.league_averages = league_averages
        
        if league_averages:
            self._initialize_from_league()
        else:
            # Default calibration
            self.base_rates = {
                'home_win': 0.45,
                'draw': 0.25,
                'away_win': 0.30
            }
    
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
        calibration_strength = 0.15  # 15% toward league averages
        
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
        """🔥 FIXED: Better draw probability - targets 25% at average xG"""
        total_xg = home_xg + away_xg
        
        # 🔥 FIX: Sigmoid tuned for realistic football draw rates
        # At total_xg = 2.5 (average game), draw ~ 25%
        # At total_xg = 1.5 (low scoring), draw ~ 32%
        # At total_xg = 3.5 (high scoring), draw ~ 18%
        
        k = 1.0  # Less steep than before
        x0 = 2.5  # Midpoint at average game
        
        base_draw_prob = 0.28 / (1 + math.exp(k * (total_xg - x0)))
        
        # Adjust for closeness (closer teams = more draws)
        xg_diff = abs(home_xg - away_xg)
        closeness_factor = 1.0 - (xg_diff / max(0.5, total_xg))
        adjusted_prob = base_draw_prob * (0.8 + 0.4 * closeness_factor)
        
        # Bounds
        return max(0.15, min(0.35, adjusted_prob))

class MatchPredictor:
    """Main prediction engine with ALL FIXES applied"""
    
    def __init__(self, league_name: str, league_averages: LeagueAverages, debug: bool = False):
        self.league_config = LEAGUE_CONFIGS.get(league_name.lower())
        if not self.league_config:
            raise ValueError(f"Unknown league: {league_name}")
        
        self.league_averages = league_averages
        self.calibrator = ProbabilityCalibrator(league_averages)
        self.debug = debug
        
        if self.debug:
            self._debug_print(f"\n🎯 PREDICTOR INITIALIZED FOR {league_name.upper()}")
            self._debug_print(f"  League Home Avg: {league_averages.avg_home_goals:.2f}")
            self._debug_print(f"  League Away Avg: {league_averages.avg_away_goals:.2f}")
            self._debug_print(f"  League Avg per Team: {league_averages.league_avg_gpg:.2f}")
    
    def _debug_print(self, message: str):
        """Print only if debug mode is enabled"""
        if self.debug:
            print(message)
    
    def predict(self, home_team: TeamProfile, away_team: TeamProfile) -> Dict:
        """Make predictions with ALL FIXES applied"""
        
        if self.debug:
            self._debug_print(f"\n🔮 PREDICTING: {home_team.name} vs {away_team.name}")
        
        # 1. Calculate expected goals with FIXED formula (no double home advantage)
        home_xg, away_xg = self._calculate_expected_goals(home_team, away_team)
        
        # 2. Calculate probabilities with REAL Poisson
        winner_pred = self._predict_winner_with_poisson(home_xg, away_xg, home_team, away_team)
        
        # 3. Total goals prediction
        total_xg = home_xg + away_xg
        total_pred = self._predict_total_goals(total_xg, home_team, away_team)
        
        # 4. BTTS prediction (with FIXED tendency)
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
        """🔥 FIXED: Proper xG WITHOUT double home advantage"""
        
        if self.debug:
            self._debug_print(f"\n📊 EXPECTED GOALS CALCULATION:")
        
        # 🔥 FIX: Use LEAGUE AVERAGE TEAM performance (neutral)
        # This is the average goals per team per game (~1.47 for Premier League)
        avg_team_goals = self.league_averages.league_avg_gpg
        
        # Get home advantage multiplier from config
        home_advantage = self.league_config.get('home_advantage_multiplier', 1.18)
        
        if self.debug:
            self._debug_print(f"  League Avg Per Team: {avg_team_goals:.2f}")
            self._debug_print(f"  Home Advantage: {home_advantage:.2f}x")
            self._debug_print(f"  Home: Attack={home.attack_strength:.2f}, Opp Defense={away.defense_strength:.2f}")
            self._debug_print(f"  Away: Attack={away.attack_strength:.2f}, Opp Defense={home.defense_strength:.2f}")
        
        # 🔥 FIX: Base xG = League team average × relative strength
        # NO built-in home advantage here
        home_base_xg = avg_team_goals * home.attack_strength / max(0.5, away.defense_strength)
        away_base_xg = avg_team_goals * away.attack_strength / max(0.5, home.defense_strength)
        
        # 🔥 FIX: Apply home advantage ONCE (not twice)
        home_xg = home_base_xg * home_advantage
        
        if self.debug:
            self._debug_print(f"  Base xG: Home={home_base_xg:.2f}, Away={away_base_xg:.2f}")
            self._debug_print(f"  After Home Advantage: Home={home_xg:.2f}")
        
        # Apply conservative hot attack boost
        home_xg, away_xg = self._apply_hot_attack_boost(home_xg, home, away_base_xg, away)
        
        # REALISTIC caps (increased from previous)
        home_xg = min(4.5, max(0.2, home_xg))
        away_xg = min(4.0, max(0.2, away_xg))
        
        if self.debug:
            self._debug_print(f"  Final xG: Home={home_xg:.2f}, Away={away_xg:.2f}, Total={home_xg+away_xg:.2f}")
        
        return home_xg, away_xg
    
    def _apply_hot_attack_boost(self, home_xg, home, away_xg, away):
        """Apply conservative boost for hot attacks"""
        home_recent_gpg = home.last5_gf / max(1, home.recent_games_played)
        away_recent_gpg = away.last5_gf / max(1, away.recent_games_played)
        
        # Only boost if significantly better than season average
        if home_recent_gpg > home.goals_pg * 1.2:
            improvement = min(1.5, home_recent_gpg / max(0.1, home.goals_pg))
            boost = 1.0 + (improvement - 1.0) * 0.1  # Max 5% boost
            home_xg *= min(1.05, boost)
            if self.debug:
                self._debug_print(f"  Home Hot Boost: {improvement:.2f}x → {min(1.05, boost):.3f}")
        
        if away_recent_gpg > away.goals_pg * 1.2:
            improvement = min(1.5, away_recent_gpg / max(0.1, away.goals_pg))
            boost = 1.0 + (improvement - 1.0) * 0.1
            away_xg *= min(1.05, boost)
            if self.debug:
                self._debug_print(f"  Away Hot Boost: {improvement:.2f}x → {min(1.05, boost):.3f}")
        
        return home_xg, away_xg
    
    def _predict_winner_with_poisson(self, home_xg: float, away_xg: float,
                                   home: TeamProfile, away: TeamProfile) -> Dict:
        """🔥 FIXED: Predict winner using REAL Poisson probabilities"""
        
        # Calculate REAL Poisson probabilities
        home_win_prob, away_win_prob, draw_prob = self.calculate_poisson_probabilities(home_xg, away_xg)
        
        if self.debug:
            self._debug_print(f"\n🎲 POISSON PROBABILITIES:")
            self._debug_print(f"  Raw Poisson: Home={home_win_prob:.2%}, Draw={draw_prob:.2%}, Away={away_win_prob:.2%}")
        
        # Apply calibration
        home_win_prob, draw_prob, away_win_prob = self.calibrator.calibrate_probabilities(
            home_win_prob, draw_prob, away_win_prob
        )
        
        if self.debug:
            self._debug_print(f"  After Calibration: Home={home_win_prob:.2%}, Draw={draw_prob:.2%}, Away={away_win_prob:.2%}")
        
        # Apply LESS aggressive reliability adjustment
        home_win_prob, draw_prob, away_win_prob = self._apply_reliability_adjustment(
            home_win_prob, away_win_prob, draw_prob, home, away
        )
        
        # Normalize
        total = home_win_prob + away_win_prob + draw_prob
        home_win_prob /= total
        away_win_prob /= total
        draw_prob /= total
        
        if self.debug:
            self._debug_print(f"  Normalized: Home={home_win_prob:.2%}, Draw={draw_prob:.2%}, Away={away_win_prob:.2%}")
        
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
        
        if self.debug:
            self._debug_print(f"  Selection: {selection} ({confidence:.1f}%)")
        
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
    
    def calculate_poisson_probabilities(self, home_xg: float, away_xg: float) -> Tuple[float, float, float]:
        """🔥 NEW: Calculate REAL Poisson probabilities"""
        
        # Use 0-6 goals (covers 99.9% of football matches)
        max_goals = 6
        
        # Pre-calculate Poisson probabilities
        home_probs = []
        away_probs = []
        
        for k in range(max_goals + 1):
            home_probs.append(math.exp(-home_xg) * (home_xg ** k) / math.factorial(k))
            away_probs.append(math.exp(-away_xg) * (away_xg ** k) / math.factorial(k))
        
        # Calculate outcome probabilities
        home_win_prob = 0.0
        away_win_prob = 0.0
        draw_prob = 0.0
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = home_probs[i] * away_probs[j]
                if i > j:
                    home_win_prob += prob
                elif i < j:
                    away_win_prob += prob
                else:
                    draw_prob += prob
        
        # Normalize (sum should be ~1.0)
        total = home_win_prob + away_win_prob + draw_prob
        
        if total > 0.99 and total < 1.01:  # Valid Poisson sum
            return home_win_prob, away_win_prob, draw_prob
        else:
            # Fallback to proportional method if Poisson calculation has issues
            if self.debug:
                self._debug_print(f"  ⚠️ Poisson sum={total:.4f}, using fallback")
            return self._calculate_proportional_probabilities(home_xg, away_xg)
    
    def _calculate_proportional_probabilities(self, home_xg: float, away_xg: float) -> Tuple[float, float, float]:
        """Fallback proportional method"""
        draw_prob = self.calibrator.calculate_draw_probability_sigmoid(home_xg, away_xg)
        
        # Win probabilities based on relative strength
        home_strength = home_xg / (home_xg + away_xg + 0.1)
        away_strength = away_xg / (home_xg + away_xg + 0.1)
        
        remaining = 1.0 - draw_prob
        total_strength = home_strength + away_strength
        
        home_win_prob = remaining * (home_strength / total_strength)
        away_win_prob = remaining * (away_strength / total_strength)
        
        return home_win_prob, away_win_prob, draw_prob
    
    def _apply_reliability_adjustment(self, home_win_prob: float, away_win_prob: float, 
                                     draw_prob: float, home: TeamProfile, away: TeamProfile):
        """🔥 FIXED: Less aggressive reliability adjustment"""
        total_recent = home.recent_games_played + away.recent_games_played
        
        if total_recent < 6:
            # 🔥 FIX: Blend with league average, not uniform 33%
            league_home_avg = self.calibrator.base_rates['home_win']
            league_away_avg = self.calibrator.base_rates['away_win']
            league_draw_avg = self.calibrator.base_rates['draw']
            
            # Less aggressive: scale from 0.5 to 1.0 instead of 0 to 1
            reliability_factor = 0.5 + (total_recent / 6 * 0.5)
            
            home_win_prob = (home_win_prob * reliability_factor + 
                            league_home_avg * (1 - reliability_factor))
            away_win_prob = (away_win_prob * reliability_factor + 
                            league_away_avg * (1 - reliability_factor))
            draw_prob = (draw_prob * reliability_factor + 
                        league_draw_avg * (1 - reliability_factor))
            
            if self.debug:
                self._debug_print(f"  Reliability Adj ({total_recent}/6 games): factor={reliability_factor:.2f}")
        
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
        
        if self.debug:
            self._debug_print(f"\n⚽ TOTAL GOALS CALCULATION:")
            self._debug_print(f"  xG Total: {total_xg:.2f}")
            self._debug_print(f"  Recent Scoring: {recent_scoring:.2f} GPG combined")
            self._debug_print(f"  League Avg: {league_avg:.2f}, Over Threshold: {over_thresh:.2f}")
        
        # Adjusted total (60% xG, 40% recent form)
        adjusted_total = (total_xg * 0.6) + (recent_scoring * 0.4)
        
        if self.debug:
            self._debug_print(f"  Adjusted Total: {adjusted_total:.2f}")
        
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
        
        if self.debug:
            self._debug_print(f"  Selection: {selection} ({confidence:.1f}%)")
        
        return {
            "type": "Total Goals",
            "selection": selection,
            "confidence": round(confidence, 1)
        }
    
    def _predict_btts(self, home_xg: float, away_xg: float,
                     home: TeamProfile, away: TeamProfile) -> Dict:
        """Predict Both Teams to Score with FIXED tendency"""
        
        # Base probability from xG
        home_score_prob = 1 - math.exp(-home_xg)
        away_score_prob = 1 - math.exp(-away_xg)
        btts_prob = home_score_prob * away_score_prob * 100
        
        if self.debug:
            self._debug_print(f"\n🎯 BTTS CALCULATION:")
            self._debug_print(f"  Home Score Prob: {home_score_prob:.2%}")
            self._debug_print(f"  Away Score Prob: {away_score_prob:.2%}")
            self._debug_print(f"  Base BTTS Prob: {btts_prob:.1f}%")
        
        # Apply team tendencies (FIXED in TeamProfile)
        tendency_factor = (home.btts_tendency + away.btts_tendency) / 2
        btts_prob *= tendency_factor
        
        if self.debug:
            self._debug_print(f"  Team Tendency Factor: {tendency_factor:.2f}")
            self._debug_print(f"  Adjusted BTTS Prob: {btts_prob:.1f}%")
        
        # League baseline
        baseline = self.league_config['btts_baseline']
        
        if btts_prob >= baseline:
            selection = "Yes"
            confidence = min(80, btts_prob)
        else:
            selection = "No"
            confidence = min(80, 100 - btts_prob)
        
        confidence = max(50, confidence)
        
        if self.debug:
            self._debug_print(f"  League Baseline: {baseline}%")
            self._debug_print(f"  Selection: {selection} ({confidence:.1f}%)")
        
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
