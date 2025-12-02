import numpy as np
from typing import Dict, List, Tuple
from utils import (
    calculate_btts_poisson_probability,
    calculate_btts_decision,
    get_match_context
)

class AdvancedUnderstatPredictor:
    def __init__(self):
        self.predictions_history = []
    
    def calculate_advanced_analysis(self, home_data: Dict, away_data: Dict, home_team: str, away_team: str, games_played: int = 12) -> Dict:
        """Advanced xG-based football predictor with EMPIRICALLY CALIBRATED probabilistic modeling"""
        
        # 1. PER-GAME AVERAGES (Foundation)
        home_xg_pg = home_data["xG"] / games_played
        away_xg_pg = away_data["xG"] / games_played
        home_xga_pg = home_data["xGA"] / games_played
        away_xga_pg = away_data["xGA"] / games_played
        
        # 2. DETECT OVERPERFORMANCE (Conservative: 20% impact)
        home_overperformance = (home_data["points"] - home_data["xPTS"]) / games_played
        away_overperformance = (away_data["points"] - away_data["xPTS"]) / games_played
        
        # Conservative intangible factor: 20% instead of 25%
        home_intangible = home_overperformance * 0.20
        away_intangible = away_overperformance * 0.20
        
        # 3. QUALITY ASSESSMENT (DE-BIASED)
        # Use realistic league averages instead of hardcoded 1.5
        league_avg_xga = 1.4  # Realistic average for top leagues
        
        home_def_factor = max(0.7, min(1.3, (league_avg_xga - home_xga_pg) * 0.9))
        away_def_factor = max(0.7, min(1.3, (league_avg_xga - away_xga_pg) * 0.8))  # Slightly harder for away teams
        
        home_quality = (home_xg_pg * 1.15) + (home_def_factor * 1.1) + home_intangible
        away_quality = (away_xg_pg * 1.15) + (away_def_factor * 1.1) + away_intangible
        quality_diff = home_quality - away_quality
        
        # 4. REALISTIC HOME ADVANTAGE (Reduced impact)
        if quality_diff > 0.8:  # Home team much better
            home_boost = 1.025  # 2.5% boost (was 3%)
            away_reduction = 0.975
        elif quality_diff < -0.8:  # Away team much better
            home_boost = 1.005  # 0.5% boost only
            away_reduction = 0.995
        elif abs(quality_diff) < 0.2:  # Evenly matched
            home_boost = 1.015  # 1.5% home advantage
            away_reduction = 0.985
        else:  # Slight advantage
            home_boost = 1.010  # 1.0% boost
            away_reduction = 0.990
        
        # 5. REALISTIC EXPECTED GOALS (Conservative adjustments)
        home_expected = home_xg_pg * home_boost
        away_expected = away_xg_pg * away_reduction
        
        # Matchup-specific adjustments
        home_vs_away_def = max(0.8, min(1.2, 1.0 + (away_xga_pg - league_avg_xga) * 0.15))
        away_vs_home_def = max(0.8, min(1.2, 1.0 + (home_xga_pg - league_avg_xga) * 0.15))
        
        home_expected *= home_vs_away_def
        away_expected *= away_vs_home_def
        
        # Conservative caps
        home_expected = min(2.5, max(0.4, home_expected))
        away_expected = min(2.3, max(0.4, away_expected))
        
        total_expected = home_expected + away_expected
        expected_goal_diff = home_expected - away_expected
        
        # 6. MOMENTUM CALCULATION (Gradient scale, reduced impact)
        home_momentum = 0
        away_momentum = 0
        
        # Attack momentum
        home_xg_gap = (home_data["xG"] - home_data["goals_scored"]) / games_played
        away_xg_gap = (away_data["xG"] - away_data["goals_scored"]) / games_played
        
        home_momentum += np.clip(home_xg_gap * 15, -8, 8)  # Reduced from 20
        away_momentum += np.clip(away_xg_gap * 15, -8, 8)
        
        # Defense momentum
        home_def_gap = (home_data["xGA"] - home_data["goals_conceded"]) / games_played
        away_def_gap = (away_data["xGA"] - away_data["goals_conceded"]) / games_played
        
        home_momentum += np.clip(home_def_gap * 12, -6, 6)  # Reduced from 15
        away_momentum += np.clip(away_def_gap * 12, -6, 6)
        
        momentum_diff = (home_momentum - away_momentum) * 0.08  # Reduced from 0.10
        
        # 7. GET MATCH CONTEXT
        context_factor, volatility_note = get_match_context(
            home_data, away_data, home_xg_pg, away_xg_pg, games_played
        )
        
        # 8. EMPIRICALLY CALIBRATED MATCH WINNER (BASED ON 13-MATCH ANALYSIS)
        # Total advantage with reduced weights
        total_advantage = (quality_diff * 0.6) + (momentum_diff * 0.25) + (expected_goal_diff * 0.15)
        total_advantage *= context_factor
        
        # EMPIRICAL CALIBRATION: Map advantage to realistic probabilities
        if total_advantage > 0.6:  # Strong home advantage
            base_win_prob = 68
        elif total_advantage > 0.4:  # Moderate home advantage
            base_win_prob = 62
        elif total_advantage > 0.2:  # Slight home advantage
            base_win_prob = 56
        elif total_advantage > 0.05:  # Minimal home advantage
            base_win_prob = 52
        elif total_advantage < -0.6:  # Strong away advantage
            base_win_prob = 32  # 68% away win
        elif total_advantage < -0.4:  # Moderate away advantage
            base_win_prob = 38  # 62% away win
        elif total_advantage < -0.2:  # Slight away advantage
            base_win_prob = 44  # 56% away win
        elif total_advantage < -0.05:  # Minimal away advantage
            base_win_prob = 48  # 52% away win
        else:  # Very close
            base_win_prob = 50  # Draw territory
        
        # Small home edge for truly balanced matches
        if 48 <= base_win_prob <= 52:
            if total_advantage > 0:
                base_win_prob += 3
            elif total_advantage < 0:
                base_win_prob -= 2
        
        # Determine winner
        if base_win_prob >= 52:
            winner = "Home Win"
            confidence = min(75, base_win_prob)  # CAP AT 75% MAX
        elif base_win_prob <= 48:
            winner = "Away Win"
            confidence = min(73, 100 - base_win_prob)  # CAP AT 73% MAX
        else:
            winner = "Draw"
            confidence = 48
        
        # Confidence caps based on xG difference (conservative)
        if abs(expected_goal_diff) < 0.2:
            confidence = min(65, confidence)
        elif abs(expected_goal_diff) < 0.4:
            confidence = min(70, confidence)
        elif abs(expected_goal_diff) < 0.6:
            confidence = min(73, confidence)
        else:
            confidence = min(75, confidence)
        
        # 9. TOTAL GOALS PROBABILITY (Wider decision bands, proven successful)
        if total_expected > 3.2:
            over_25_prob = 72
        elif total_expected > 2.8:
            over_25_prob = 64
        elif total_expected > 2.4:
            over_25_prob = 56
        elif total_expected > 2.0:
            over_25_prob = 48
        elif total_expected > 1.6:
            over_25_prob = 40
        else:
            over_25_prob = 32
        
        # Add volatility adjustment (reduced impact)
        volatility_boost = (abs(home_xg_gap) + abs(away_xg_gap)) * 2.5  # Reduced from 3
        over_25_prob = min(78, max(22, over_25_prob + volatility_boost))
        
        # Total Goals decision (Wider bands: 54/46 for decisive calls)
        if over_25_prob >= 54:  # Clear over signal
            goals_selection = "Over 2.5 Goals"
            goals_confidence = over_25_prob * 0.9  # Slight discount
        elif over_25_prob <= 46:  # Clear under signal
            goals_selection = "Under 2.5 Goals"
            goals_confidence = (100 - over_25_prob) * 0.9
        else:
            goals_selection = "Avoid Total Goals"
            goals_confidence = 42  # Higher than 38 to encourage caution
        
        # 10. BTTS CALCULATION (With conservative mismatch adjustment)
        btts_raw_prob = calculate_btts_poisson_probability(home_expected, away_expected)
        
        # Conservative adjustments
        if home_xga_pg < 1.0 and away_xga_pg < 1.0:
            btts_raw_prob -= 4  # Reduced from 6
        elif home_xga_pg > 1.8 or away_xga_pg > 1.8:
            btts_raw_prob += 6  # Reduced from 8
        
        btts_raw_prob = min(78, max(22, btts_raw_prob))  # Tighter bounds
        
        # Get BTTS decision with conservative thresholds
        btts_selection, btts_confidence, btts_adjust_note = calculate_btts_decision(
            btts_raw_prob, total_expected, home_quality, away_quality
        )
        
        # 11. REALISTIC ASIAN HANDICAP (WITH VARIANCE BUFFER)
        # Calculate variance factor - low scoring games have less predictable margins
        if total_expected < 1.8:
            variance_factor = 0.6
        elif total_expected < 2.2:
            variance_factor = 0.7
        elif total_expected < 2.6:
            variance_factor = 0.8
        elif total_expected < 3.0:
            variance_factor = 0.9
        else:
            variance_factor = 1.0
        
        adjusted_goal_diff = expected_goal_diff * variance_factor
        
        # Conservative handicap selection
        if "Home" in winner:
            if adjusted_goal_diff > 0.9:
                ah_selection = "Home -1.0"
                ah_confidence = min(68, 50 + adjusted_goal_diff * 10)
            elif adjusted_goal_diff > 0.6:
                ah_selection = "Home -0.75"
                ah_confidence = min(65, 48 + adjusted_goal_diff * 12)
            elif adjusted_goal_diff > 0.4:
                ah_selection = "Home -0.5"
                ah_confidence = min(62, 46 + adjusted_goal_diff * 14)
            elif adjusted_goal_diff > 0.2:
                ah_selection = "Home -0.25"
                ah_confidence = min(58, 44 + adjusted_goal_diff * 16)
            else:
                ah_selection = "Home 0.0"
                ah_confidence = 52
        elif "Away" in winner:
            if adjusted_goal_diff < -0.9:
                ah_selection = "Away -0.75"
                ah_confidence = min(66, 49 + abs(adjusted_goal_diff) * 10)
            elif adjusted_goal_diff < -0.6:
                ah_selection = "Away -0.5"
                ah_confidence = min(63, 47 + abs(adjusted_goal_diff) * 12)
            elif adjusted_goal_diff < -0.4:
                ah_selection = "Away -0.25"
                ah_confidence = min(60, 45 + abs(adjusted_goal_diff) * 14)
            elif adjusted_goal_diff < -0.2:
                ah_selection = "Away 0.0"
                ah_confidence = min(56, 43 + abs(adjusted_goal_diff) * 16)
            else:
                ah_selection = "Draw No Bet"
                ah_confidence = 52
        else:  # Draw
            ah_selection = "Draw No Bet"
            ah_confidence = 52
        
        # Apply conservative buffer for close matches
        if abs(expected_goal_diff) < 0.3:
            ah_confidence = max(25, ah_confidence - 6)
        
        ah_confidence = min(68, max(25, ah_confidence))  # Conservative caps
        
        return {
            "analysis": {
                "home_quality": home_quality,
                "away_quality": away_quality,
                "quality_diff": quality_diff,
                "home_intangible": home_intangible,
                "away_intangible": away_intangible,
                "home_momentum": home_momentum,
                "away_momentum": away_momentum,
                "momentum_diff": momentum_diff,
                "total_advantage": total_advantage,
                "expected_goal_diff": expected_goal_diff,
                "home_boost": home_boost,
                "away_reduction": away_reduction,
                "context_factor": context_factor,
                "volatility_note": volatility_note,
                "expected_goals": {
                    "home": home_expected,
                    "away": away_expected,
                    "total": total_expected
                },
                "probabilities": {
                    "over_25": over_25_prob,
                    "btts_raw": btts_raw_prob,
                    "btts_adjust_note": btts_adjust_note
                }
            },
            "predictions": [
                {"type": "Match Winner", "selection": winner, "confidence": confidence},
                {"type": "Total Goals", "selection": goals_selection, "confidence": goals_confidence},
                {"type": "Both Teams To Score", "selection": btts_selection, "confidence": btts_confidence},
                {"type": "Asian Handicap", "selection": ah_selection, "confidence": ah_confidence}
            ],
            "team_names": {
                "home": home_team,
                "away": away_team
            },
            "raw_data": {
                "home_xg_pg": home_xg_pg,
                "away_xg_pg": away_xg_pg,
                "home_xga_pg": home_xga_pg,
                "away_xga_pg": away_xga_pg,
                "home_overperformance": home_overperformance,
                "away_overperformance": away_overperformance
            }
        }