"""
PHANTOM v4.4 - ALL FIXES APPLIED
Extreme form recognition, style matrices, variance-aware predictions
"""
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

# ============================================================================
# LEAGUE CONFIGURATIONS (UPDATED)
# ============================================================================

LEAGUE_CONFIGS = {
    "premier_league": {
        "name": "Premier League",
        "avg_goals": 2.93,
        "over_threshold": 2.93,  # FIXED: At average, not below
        "btts_baseline": 52,
        "home_advantage_multiplier": 1.18,
        "prior_strength": 4,  # Reduced from 10
        "btts_correlation": 0.15,
        "goal_distribution": {
            2.0: 0.70,   # 70% Under when expected goals=2.0
            2.3: 0.60,
            2.6: 0.50,   # 50/50 at expected goals=2.6
            2.9: 0.40,
            3.2: 0.30,
            3.5: 0.20
        }
    },
    "serie_a": {
        "name": "Serie A",
        "avg_goals": 2.56,
        "over_threshold": 2.56,
        "btts_baseline": 48,
        "home_advantage_multiplier": 1.15,
        "prior_strength": 4,
        "btts_correlation": 0.12,
        "goal_distribution": {
            1.8: 0.75,
            2.1: 0.65,
            2.4: 0.50,
            2.7: 0.35,
            3.0: 0.25
        }
    },
    "la_liga": {
        "name": "La Liga",
        "avg_goals": 2.62,
        "over_threshold": 2.62,
        "btts_baseline": 50,
        "home_advantage_multiplier": 1.20,
        "prior_strength": 4,
        "btts_correlation": 0.14,
        "goal_distribution": {
            1.9: 0.72,
            2.2: 0.62,
            2.5: 0.48,
            2.8: 0.38,
            3.1: 0.28
        }
    },
    "bundesliga": {
        "name": "Bundesliga",
        "avg_goals": 3.14,
        "over_threshold": 3.14,
        "btts_baseline": 55,
        "home_advantage_multiplier": 1.22,
        "prior_strength": 4,
        "btts_correlation": 0.18,
        "goal_distribution": {
            2.4: 0.65,
            2.7: 0.55,
            3.0: 0.45,
            3.3: 0.35,
            3.6: 0.25
        }
    },
    "ligue_1": {
        "name": "Ligue 1",
        "avg_goals": 2.78,
        "over_threshold": 2.78,
        "btts_baseline": 50,
        "home_advantage_multiplier": 1.16,
        "prior_strength": 4,
        "btts_correlation": 0.13,
        "goal_distribution": {
            2.1: 0.68,
            2.4: 0.58,
            2.7: 0.45,
            3.0: 0.35,
            3.3: 0.25
        }
    },
    "rfpl": {
        "name": "Russian Premier League",
        "avg_goals": 2.68,
        "over_threshold": 2.68,
        "btts_baseline": 50,
        "home_advantage_multiplier": 1.17,
        "prior_strength": 4,
        "btts_correlation": 0.11,
        "goal_distribution": {
            2.0: 0.70,
            2.3: 0.60,
            2.6: 0.48,
            2.9: 0.38,
            3.2: 0.28
        }
    }
}

# Game context multipliers
GAME_CONTEXT_FACTORS = {
    "derby": 1.15,
    "relegation_battle": 1.10,
    "title_decider": 1.05,
    "midtable_nothing": 0.95,
    "european_qualification": 1.08,
    "default": 1.0
}

@dataclass
class LeagueAverages:
    """Container for league statistics"""
    avg_home_goals: float
    avg_away_goals: float
    total_matches: int
    actual_home_win_rate: float = 0.45
    actual_draw_rate: float = 0.25
    actual_away_win_rate: float = 0.30
    
    @property
    def neutral_baseline(self) -> float:
        return (self.avg_home_goals + self.avg_away_goals) / 2

class TeamProfile:
    """Team profile with all fixes"""
    
    def __init__(self, data_dict: Dict, is_home: bool = True, 
                 league_averages: Optional[LeagueAverages] = None,
                 debug: bool = False):
        self.name = data_dict['Team']
        self.is_home = is_home
        self.league_averages = league_averages
        self.debug = debug
        
        # Basic stats
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
        
        # Last 5 form data
        self._extract_last5_data(data_dict)
        
        # Calculate recent games
        self.recent_games_played = min(5, 
            self.last5_wins + self.last5_draws + self.last5_losses)
        
        # Calculate all metrics
        self._calculate_all_metrics()
        
        if self.debug:
            self._debug_info()
    
    def _extract_last5_data(self, data_dict: Dict):
        """Extract last 5 form data"""
        if self.is_home:
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
        
        # For variance calculation
        self.recent_goals_scored = []
        self.recent_goals_conceded = []
        
        if self.recent_games_played > 0:
            avg_scored = self.last5_gf / self.recent_games_played
            avg_conceded = self.last5_ga / self.recent_games_played
            
            # Simulate distribution
            for i in range(self.recent_games_played):
                self.recent_goals_scored.append(max(0, int(round(avg_scored))))
                self.recent_goals_conceded.append(max(0, int(round(avg_conceded))))
    
    def _calculate_all_metrics(self):
        """Calculate all metrics in one place"""
        # 1. Form score with momentum weighting
        self.form_score = self._calculate_momentum_form()
        
        # 2. Attack strength with form included
        self.attack_strength = self._calculate_attack_strength()
        
        # 3. Defense strength with form included
        self.defense_strength = self._calculate_defense_strength()
        
        # 4. BTTS tendency
        self.btts_tendency = self._calculate_btts_tendency()
        
        # 5. Variance factor
        self.variance_factor = self._calculate_variance_factor()
        
        # 6. Game context (default)
        self.context_factor = GAME_CONTEXT_FACTORS['default']
    
    def _calculate_momentum_form(self) -> float:
        """Form score with exponential decay weighting"""
        if self.recent_games_played == 0:
            season_form = self.points / (self.matches * 3)
            return season_form * 0.7
        
        # Exponential decay weights
        weights = [0.35, 0.25, 0.20, 0.15, 0.05]
        
        # Calculate weighted points
        total_weighted_points = 0
        total_possible = 0
        
        # Simulate results
        results = []
        for _ in range(self.last5_wins):
            results.append(3)
        for _ in range(self.last5_draws):
            results.append(1)
        for _ in range(self.last5_losses):
            results.append(0)
        
        while len(results) < 5:
            results.append(self.points / max(1, self.matches))
        
        for i, points in enumerate(results[:5]):
            weight = weights[i] if i < len(weights) else 0.05
            total_weighted_points += points * weight
            total_possible += 3 * weight
        
        weighted_form = total_weighted_points / max(0.01, total_possible)
        
        # Blend with season form
        season_form = self.points / (self.matches * 3)
        momentum_weight = min(0.8, self.recent_games_played / 5 * 0.8)
        season_weight = 1 - momentum_weight
        
        return (weighted_form * momentum_weight) + (season_form * season_weight)
    
    def _calculate_attack_strength(self) -> float:
        """Attack strength with form included once"""
        # Base from full season
        season_attack = self.goals_pg * 0.4 + self.xg_pg * 0.6
        
        # Recent performance
        recent_gpg = self.last5_gf / max(1, self.recent_games_played)
        recent_xg_pg = self.xg / max(1, self.matches)
        
        # Form factor
        recent_total = (recent_gpg * 0.4) + (recent_xg_pg * 0.6)
        season_total = season_attack
        
        if season_total > 0:
            form_factor = recent_total / season_total
        else:
            form_factor = 1.0
        
        form_factor = min(1.4, max(0.6, form_factor))
        final_attack = season_attack * form_factor
        
        # Apply variance factor
        final_attack *= self.variance_factor
        
        # Compare to league baseline
        if self.is_home:
            baseline = self.league_averages.avg_home_goals if self.league_averages else 1.65
        else:
            baseline = self.league_averages.avg_away_goals if self.league_averages else 1.28
        
        strength = final_attack / max(0.3, baseline)
        return min(1.9, max(0.5, strength))
    
    def _calculate_defense_strength(self) -> float:
        """Defense strength with form included once"""
        # Base from full season
        season_defense = self.goals_against_pg * 0.4 + self.xga_pg * 0.6
        
        # Recent performance
        recent_gapg = self.last5_ga / max(1, self.recent_games_played)
        recent_xga_pg = self.xga / max(1, self.matches)
        
        # Form factor (inverted for defense)
        recent_total = (recent_gapg * 0.4) + (recent_xga_pg * 0.6)
        season_total = season_defense
        
        if season_total > 0:
            form_factor = recent_total / season_total
        else:
            form_factor = 1.0
        
        defense_form_factor = 1.0 / max(0.7, min(1.3, form_factor))
        final_defense = season_defense * defense_form_factor
        
        # Apply variance factor (hurts defense)
        final_defense *= self.variance_factor
        
        # What opponents typically score
        if self.is_home:
            opponent_baseline = self.league_averages.avg_away_goals if self.league_averages else 1.28
        else:
            opponent_baseline = self.league_averages.avg_home_goals if self.league_averages else 1.65
        
        strength = opponent_baseline / max(0.3, final_defense)
        return min(1.6, max(0.5, strength))
    
    def _calculate_btts_tendency(self) -> float:
        """BTTS tendency"""
        if self.recent_games_played == 0:
            return 1.0
        
        goals_per_game = self.last5_gf / max(1, self.recent_games_played)
        goals_against_per_game = self.last5_ga / max(1, self.recent_games_played)
        
        p_scored = 1 - math.exp(-goals_per_game)
        p_conceded = 1 - math.exp(-goals_against_per_game)
        
        # Expected games involved in BTTS
        expected_involved = self.recent_games_played * (p_scored + p_conceded - p_scored * p_conceded)
        involvement_rate = expected_involved / self.recent_games_played
        
        # Map to tendency factor
        if involvement_rate >= 0.85:
            return 1.25
        elif involvement_rate >= 0.75:
            return 1.15
        elif involvement_rate >= 0.65:
            return 1.05
        elif involvement_rate <= 0.45:
            return 0.75
        elif involvement_rate <= 0.55:
            return 0.85
        
        return 1.0
    
    def _calculate_variance_factor(self) -> float:
        """Calculate team variance"""
        if len(self.recent_goals_scored) < 3:
            return 1.0
        
        # Calculate variance-to-mean ratio
        goals_scored = self.recent_goals_scored
        mean_scored = sum(goals_scored) / len(goals_scored)
        
        if mean_scored > 0:
            variance_scored = sum((g - mean_scored) ** 2 for g in goals_scored) / len(goals_scored)
            ratio_scored = variance_scored / mean_scored
        else:
            ratio_scored = 1.0
        
        goals_conceded = self.recent_goals_conceded
        mean_conceded = sum(goals_conceded) / len(goals_conceded)
        
        if mean_conceded > 0:
            variance_conceded = sum((g - mean_conceded) ** 2 for g in goals_conceded) / len(goals_conceded)
            ratio_conceded = variance_conceded / mean_conceded
        else:
            ratio_conceded = 1.0
        
        avg_ratio = (ratio_scored + ratio_conceded) / 2
        
        # Map to variance factor
        if avg_ratio > 1.5:
            return 1.25
        elif avg_ratio > 1.3:
            return 1.15
        elif avg_ratio > 1.1:
            return 1.05
        elif avg_ratio < 0.6:
            return 0.8
        elif avg_ratio < 0.8:
            return 0.9
        
        return 1.0
    
    def _debug_info(self):
        """Debug output"""
        print(f"  {self.name}: Form={self.form_score:.2f}, Attack={self.attack_strength:.2f}, "
              f"Defense={self.defense_strength:.2f}, Variance={self.variance_factor:.2f}")

class PoissonCalculator:
    """Poisson probability calculator"""
    
    @staticmethod
    def calculate_poisson_probabilities(home_xg: float, away_xg: float, 
                                      home_variance: float = 1.0, 
                                      away_variance: float = 1.0,
                                      max_goals: int = 6) -> Tuple[float, float, float]:
        """Poisson probabilities with variance adjustment"""
        
        # Adjust xG based on variance
        adjusted_home_xg = home_xg * (1 + (home_variance - 1.0) * 0.1)
        adjusted_away_xg = away_xg * (1 + (away_variance - 1.0) * 0.1)
        
        # Pre-calculate Poisson probabilities
        home_probs = []
        away_probs = []
        
        for k in range(max_goals + 1):
            home_probs.append(math.exp(-adjusted_home_xg) * (adjusted_home_xg ** k) / math.factorial(k))
            away_probs.append(math.exp(-adjusted_away_xg) * (adjusted_away_xg ** k) / math.factorial(k))
        
        # Calculate outcome probabilities
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = home_probs[i] * away_probs[j]
                if i > j:
                    home_win += prob
                elif i < j:
                    away_win += prob
                else:
                    draw += prob
        
        # Normalize if needed
        total = home_win + draw + away_win
        if abs(total - 1.0) > 0.001:
            home_win /= total
            draw /= total
            away_win /= total
        
        return home_win, draw, away_win

class BayesianShrinker:
    """Bayesian shrinkage that respects extremes"""
    
    def __init__(self, base_prior_strength: int = 4):
        self.base_prior_strength = base_prior_strength
    
    def shrink_probabilities(self, home_win_prob: float, draw_prob: float,
                           away_win_prob: float, home: TeamProfile,
                           away: TeamProfile, league_priors: Dict) -> Tuple[float, float, float]:
        """Dynamic shrinkage based on extremity"""
        home_samples = home.recent_games_played
        away_samples = away.recent_games_played
        avg_samples = (home_samples + away_samples) / 2
        
        # Check for extreme cases
        home_form = home.form_score
        away_form = away.form_score
        
        # Extreme teams get less shrinkage
        if home_form > 0.8 or away_form < 0.2:
            prior_strength = 2  # Minimal smoothing
        elif home_form > 0.7 or away_form < 0.3:
            prior_strength = 3  # Reduced smoothing
        else:
            prior_strength = self.base_prior_strength
        
        # Apply shrinkage
        home_shrunk = (home_win_prob * avg_samples + league_priors['home_win'] * prior_strength) / (avg_samples + prior_strength)
        draw_shrunk = (draw_prob * avg_samples + league_priors['draw'] * prior_strength) / (avg_samples + prior_strength)
        away_shrunk = (away_win_prob * avg_samples + league_priors['away_win'] * prior_strength) / (avg_samples + prior_strength)
        
        # Renormalize
        total = home_shrunk + draw_shrunk + away_shrunk
        return home_shrunk/total, draw_shrunk/total, away_shrunk/total

class MatchPredictor:
    """Main prediction engine with ALL fixes"""
    
    def __init__(self, league_name: str, league_averages: LeagueAverages, 
                 debug: bool = False):
        self.league_config = LEAGUE_CONFIGS.get(league_name.lower())
        if not self.league_config:
            raise ValueError(f"Unknown league: {league_name}")
        
        self.league_averages = league_averages
        self.debug = debug
        
        # Initialize components
        self.poisson_calc = PoissonCalculator()
        self.bayesian_shrinker = BayesianShrinker(
            base_prior_strength=self.league_config.get('prior_strength', 4)
        )
        
        # League priors
        self.league_priors = {
            'home_win': league_averages.actual_home_win_rate,
            'draw': league_averages.actual_draw_rate,
            'away_win': league_averages.actual_away_win_rate
        }
        
        if self.debug:
            print(f"\n🎯 PREDICTOR INITIALIZED FOR {self.league_config['name'].upper()}")
    
    def predict(self, home_team: TeamProfile, away_team: TeamProfile, 
                game_context: str = "default") -> Dict:
        """Make predictions with ALL fixes"""
        
        if self.debug:
            print(f"\n🔮 PREDICTING: {home_team.name} vs {away_team.name}")
        
        # Apply game context
        context_factor = GAME_CONTEXT_FACTORS.get(game_context, 1.0)
        
        # Calculate expected goals
        home_xg, away_xg = self._calculate_expected_goals(home_team, away_team)
        home_xg *= context_factor
        away_xg *= context_factor
        
        # Make predictions
        winner_pred = self._predict_winner_poisson(home_xg, away_xg, home_team, away_team)
        total_pred = self._predict_total_goals_fixed(home_xg, away_xg, home_team, away_team)
        btts_pred = self._predict_btts_with_correlation(home_xg, away_xg, home_team, away_team)
        
        return {
            "analysis": {
                "league": self.league_config['name'],
                "expected_goals": {
                    "home": round(home_xg, 2),
                    "away": round(away_xg, 2),
                    "total": round(home_xg + away_xg, 2)
                },
                "form_scores": {
                    "home": round(home_team.form_score, 2),
                    "away": round(away_team.form_score, 2)
                }
            },
            "predictions": [winner_pred, total_pred, btts_pred]
        }
    
    def _calculate_expected_goals(self, home: TeamProfile, away: TeamProfile) -> Tuple[float, float]:
        """Calculate expected goals"""
        neutral_baseline = self.league_averages.neutral_baseline
        home_advantage = self.league_config['home_advantage_multiplier']
        
        # Base xG
        home_base_xg = neutral_baseline * home.attack_strength / max(0.6, away.defense_strength)
        away_base_xg = neutral_baseline * away.attack_strength / max(0.6, home.defense_strength)
        
        # Apply home advantage
        home_xg = home_base_xg * home_advantage
        
        # Apply momentum adjustment
        home_xg, away_xg = self._apply_momentum_adjustment(home_xg, home, away_base_xg, away)
        
        # Realistic caps
        home_xg = min(4.0, max(0.2, home_xg))
        away_xg = min(3.5, max(0.2, away_xg))
        
        return home_xg, away_xg
    
    def _apply_momentum_adjustment(self, home_xg, home, away_xg, away):
        """Apply momentum adjustment"""
        home_momentum = home.form_score
        away_momentum = away.form_score
        
        momentum_threshold = 0.7
        
        if home_momentum > 1.0 + momentum_threshold:
            boost = 1.0 + (home_momentum - 1.0) * 0.1
            home_xg *= min(1.1, boost)
        
        if away_momentum > 1.0 + momentum_threshold:
            boost = 1.0 + (away_momentum - 1.0) * 0.1
            away_xg *= min(1.1, boost)
        
        return home_xg, away_xg
    
    def _apply_extreme_form_boosts(self, home_win_prob: float, draw_prob: float, away_win_prob: float,
                                 home: TeamProfile, away: TeamProfile) -> Tuple[float, float, float]:
        """Apply extreme form multipliers"""
        home_win_rate = home.wins / max(1, home.matches) if home.is_home else 0
        away_win_rate = away.wins / max(1, away.matches) if not away.is_home else 0
        
        home_multiplier = 1.0
        
        # Rule 1: Extreme home dominance (Barcelona rule)
        if home_win_rate >= 0.85:
            home_multiplier *= 1.3
            if self.debug:
                print(f"  EXTREME HOME FORM: {home.name} ({home_win_rate:.0%} win rate) → ×1.3")
        
        # Rule 2: Extreme away weakness (Wolves rule)
        if away_win_rate <= 0.15:
            home_multiplier *= 1.2
            if self.debug:
                print(f"  EXTREME AWAY WEAKNESS: {away.name} ({away_win_rate:.0%} win rate away) → ×1.2")
        
        # Rule 3: Goal differential mismatch
        home_gd_per_game = (home.goals_for - home.goals_against) / max(1, home.matches)
        away_gd_per_game = (away.goals_for - away.goals_against) / max(1, away.matches)
        
        if home_gd_per_game > 1.0 and away_gd_per_game < -0.5:
            home_multiplier *= 1.25
            if self.debug:
                print(f"  GOAL DIFFERENTIAL MISMATCH: Home +{home_gd_per_game:.1f}/gm, Away {away_gd_per_game:.1f}/gm → ×1.25")
        
        # Apply multiplier
        if home_multiplier > 1.0:
            home_win_prob *= home_multiplier
            total = home_win_prob + draw_prob + away_win_prob
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total
        
        return home_win_prob, draw_prob, away_win_prob
    
    def _apply_league_position_adjustment(self, home_win_prob: float, draw_prob: float, away_win_prob: float,
                                        home: TeamProfile, away: TeamProfile) -> Tuple[float, float, float]:
        """Adjust for league position gaps"""
        points_per_game_home = home.points / max(1, home.matches)
        points_per_game_away = away.points / max(1, away.matches)
        
        ppg_diff = points_per_game_home - points_per_game_away
        
        if ppg_diff >= 1.8:  # Equivalent to 6+ positions
            boost = 1.18
            home_win_prob *= boost
            
            if self.debug:
                print(f"  LEAGUE POSITION GAP: {ppg_diff:.2f} PPG diff → ×{boost:.2f}")
            
            total = home_win_prob + draw_prob + away_win_prob
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total
        
        return home_win_prob, draw_prob, away_win_prob
    
    def _predict_winner_poisson(self, home_xg: float, away_xg: float,
                              home: TeamProfile, away: TeamProfile) -> Dict:
        """Fixed winner prediction with all adjustments"""
        
        # 1. Base Poisson probabilities
        home_win_prob, draw_prob, away_win_prob = self.poisson_calc.calculate_poisson_probabilities(
            home_xg, away_xg, home.variance_factor, away.variance_factor
        )
        
        if self.debug:
            print(f"\n🎲 MATCH WINNER CALCULATION:")
            print(f"  Base Poisson: Home={home_win_prob:.2%}, Draw={draw_prob:.2%}, Away={away_win_prob:.2%}")
        
        # 2. Apply extreme form boosts
        home_win_prob, draw_prob, away_win_prob = self._apply_extreme_form_boosts(
            home_win_prob, draw_prob, away_win_prob, home, away
        )
        
        # 3. Apply league position adjustment
        home_win_prob, draw_prob, away_win_prob = self._apply_league_position_adjustment(
            home_win_prob, draw_prob, away_win_prob, home, away
        )
        
        # 4. Apply Bayesian shrinkage
        home_win_prob, draw_prob, away_win_prob = self.bayesian_shrinker.shrink_probabilities(
            home_win_prob, draw_prob, away_win_prob, home, away, self.league_priors
        )
        
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
        
        # Realistic bounds
        confidence = max(15, min(85, confidence))
        
        if self.debug:
            print(f"  Final: {selection} ({confidence:.1f}%)")
        
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
    
    def _classify_team_style(self, team: TeamProfile) -> str:
        """Simple style classification"""
        league_avg_goals = self.league_averages.neutral_baseline
        
        if team.goals_pg > league_avg_goals + 0.3:
            return "ATTACKING"
        elif team.goals_against_pg < league_avg_goals - 0.3:
            return "DEFENSIVE"
        else:
            return "NEUTRAL"
    
    def _apply_style_adjustment(self, total_xg: float, home: TeamProfile, away: TeamProfile) -> float:
        """Adjust total goals based on team styles"""
        home_style = self._classify_team_style(home)
        away_style = self._classify_team_style(away)
        
        adjustment = 0.0
        
        if home_style == "ATTACKING" and away_style == "ATTACKING":
            adjustment = 0.4
            if self.debug:
                print(f"  STYLE: ATTACKING vs ATTACKING → +{adjustment:.1f} goals")
        
        elif home_style == "DEFENSIVE" and away_style == "DEFENSIVE":
            adjustment = -0.4
            if self.debug:
                print(f"  STYLE: DEFENSIVE vs DEFENSIVE → {adjustment:.1f} goals")
        
        return total_xg + adjustment
    
    def _apply_goal_flow_adjustment(self, total_xg: float, home: TeamProfile, away: TeamProfile) -> float:
        """Adjust for recent goal involvement"""
        home_involvement = (home.last5_gf + home.last5_ga) / max(1, home.recent_games_played)
        away_involvement = (away.last5_gf + away.last5_ga) / max(1, away.recent_games_played)
        
        if home_involvement > 3.0 and away_involvement > 3.0:
            adjustment = 0.3
            if self.debug:
                print(f"  GOAL FLOW: High involvement (Home {home_involvement:.1f}, Away {away_involvement:.1f}) → +{adjustment:.1f} goals")
            return total_xg + adjustment
        
        elif home_involvement < 1.5 and away_involvement < 1.5:
            adjustment = -0.3
            if self.debug:
                print(f"  GOAL FLOW: Low involvement (Home {home_involvement:.1f}, Away {away_involvement:.1f}) → {adjustment:.1f} goals")
            return total_xg + adjustment
        
        return total_xg
    
    def _get_dynamic_threshold(self, home: TeamProfile, away: TeamProfile) -> float:
        """Get dynamic Over/Under threshold"""
        base_threshold = self.league_config['over_threshold']
        avg_variance = (home.variance_factor + away.variance_factor) / 2
        
        if avg_variance > 1.2:
            adjusted_threshold = base_threshold - 0.2
            if self.debug:
                print(f"  VARIANCE ADJUSTMENT: High variance ({avg_variance:.2f}) → threshold {adjusted_threshold:.2f}")
            return adjusted_threshold
        
        elif avg_variance < 0.9:
            adjusted_threshold = base_threshold + 0.1
            if self.debug:
                print(f"  VARIANCE ADJUSTMENT: Low variance ({avg_variance:.2f}) → threshold {adjusted_threshold:.2f}")
            return adjusted_threshold
        
        return base_threshold
    
    def _interpolate_probability(self, total_xg: float, distribution: Dict) -> float:
        """Interpolate probability from distribution table"""
        if not distribution:
            return 0.5
        
        points = sorted(distribution.items())
        
        if total_xg <= points[0][0]:
            return points[0][1]
        
        if total_xg >= points[-1][0]:
            return points[-1][1]
        
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            if x1 <= total_xg <= x2:
                fraction = (total_xg - x1) / (x2 - x1)
                return y1 + fraction * (y2 - y1)
        
        return 0.5
    
    def _predict_total_goals_fixed(self, home_xg: float, away_xg: float,
                                 home: TeamProfile, away: TeamProfile) -> Dict:
        """Fixed total goals prediction with all adjustments"""
        
        total_xg = home_xg + away_xg
        
        if self.debug:
            print(f"\n⚽ TOTAL GOALS CALCULATION:")
            print(f"  Base xG Total: {total_xg:.2f}")
        
        # 1. Apply style adjustment
        total_xg = self._apply_style_adjustment(total_xg, home, away)
        
        # 2. Apply goal flow adjustment
        total_xg = self._apply_goal_flow_adjustment(total_xg, home, away)
        
        # 3. Get dynamic threshold
        threshold = self._get_dynamic_threshold(home, away)
        
        if self.debug:
            print(f"  Adjusted Total: {total_xg:.2f}")
            print(f"  Dynamic Threshold: {threshold:.2f}")
        
        # 4. Decision
        goal_dist = self.league_config.get('goal_distribution', {})
        
        if goal_dist:
            # Use empirical distribution
            p_under = self._interpolate_probability(total_xg, goal_dist)
            
            # Adjust for variance
            avg_variance = (home.variance_factor + away.variance_factor) / 2
            if avg_variance > 1.1:
                p_under = 0.5 + (p_under - 0.5) * 0.8
            
            if p_under < 0.5:
                selection = "Over 2.5 Goals"
                confidence = (1 - p_under) * 100
            else:
                selection = "Under 2.5 Goals"
                confidence = p_under * 100
                
        else:
            # Fallback to threshold
            if total_xg > threshold:
                selection = "Over 2.5 Goals"
                excess = (total_xg - threshold) / threshold
                confidence = 55 + min(20, excess * 20)
            else:
                selection = "Under 2.5 Goals"
                deficit = (threshold - total_xg) / threshold
                confidence = 55 + min(20, deficit * 20)
        
        # Apply bounds
        confidence = max(50, min(75, confidence))
        
        if self.debug:
            print(f"  Selection: {selection} ({confidence:.1f}%)")
        
        return {
            "type": "Total Goals",
            "selection": selection,
            "confidence": round(confidence, 1)
        }
    
    def _predict_btts_with_correlation(self, home_xg: float, away_xg: float,
                                     home: TeamProfile, away: TeamProfile) -> Dict:
        """Predict BTTS with correlation"""
        
        # Base probabilities
        p_home_scores = 1 - math.exp(-home_xg)
        p_away_scores = 1 - math.exp(-away_xg)
        
        # Get correlation parameter
        correlation = self.league_config.get('btts_correlation', 0.15)
        
        # Correlation-aware calculation
        p_both_score = p_home_scores * p_away_scores * (1 + correlation)
        p_home_only = p_home_scores * (1 - p_away_scores) * (1 - correlation/2)
        p_away_only = (1 - p_home_scores) * p_away_scores * (1 - correlation/2)
        
        # Total BTTS probability
        btts_prob = (p_both_score + p_home_only + p_away_only) * 100
        
        # Apply team tendencies
        tendency_factor = (home.btts_tendency + away.btts_tendency) / 2
        btts_prob *= tendency_factor
        
        # Apply variance adjustment
        avg_variance = (home.variance_factor + away.variance_factor) / 2
        if avg_variance > 1.1:
            btts_prob *= 1.05
        
        # Clean sheet streak adjustment
        if home.last5_ga == 0 and home.recent_games_played >= 2:
            btts_prob *= 0.8
        
        if away.last5_ga == 0 and away.recent_games_played >= 2:
            btts_prob *= 0.8
        
        # League baseline
        baseline = self.league_config['btts_baseline']
        
        if btts_prob >= baseline:
            selection = "Yes"
            confidence = min(75, btts_prob)
        else:
            selection = "No"
            confidence = min(75, 100 - btts_prob)
        
        confidence = max(50, confidence)
        
        return {
            "type": "BTTS",
            "selection": selection,
            "confidence": round(confidence, 1)
        }

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example team data (simplified)
    barcelona_data = {
        'Team': 'Barcelona',
        'Matches': 7,
        'Wins': 7,
        'Draws': 0,
        'Losses': 0,
        'Goals': 23,
        'Goals_Against': 4,
        'Points': 21,
        'xG': 19.50,
        'xGA': 6.68,
        'Last5_Home_Wins': 5,
        'Last5_Home_Draws': 0,
        'Last5_Home_Losses': 0,
        'Last5_Home_GF': 14,
        'Last5_Home_GA': 4,
        'Last5_Home_PTS': 15
    }
    
    atletico_data = {
        'Team': 'Atletico Madrid',
        'Matches': 6,
        'Wins': 2,
        'Draws': 3,
        'Losses': 1,
        'Goals': 7,
        'Goals_Against': 5,
        'Points': 9,
        'xG': 7.41,
        'xGA': 6.99,
        'Last5_Away_Wins': 2,
        'Last5_Away_Draws': 3,
        'Last5_Away_Losses': 0,
        'Last5_Away_GF': 6,
        'Last5_Away_GA': 3,
        'Last5_Away_PTS': 9
    }
    
    # Create league averages
    league_avgs = LeagueAverages(
        avg_home_goals=1.46,
        avg_away_goals=1.12,
        total_matches=139
    )
    
    # Create predictor
    predictor = MatchPredictor("la_liga", league_avgs, debug=True)
    
    # Create team profiles
    barcelona = TeamProfile(barcelona_data, is_home=True, 
                           league_averages=league_avgs, debug=True)
    atletico = TeamProfile(atletico_data, is_home=False, 
                          league_averages=league_avgs, debug=True)
    
    # Make prediction
    result = predictor.predict(barcelona, atletico)
    
    print(f"\n📊 FINAL PREDICTIONS:")
    for pred in result["predictions"]:
        print(f"  {pred['type']}: {pred['selection']} ({pred['confidence']}%)")