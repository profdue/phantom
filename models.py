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
        """Advanced xG-based football predictor with probabilistic modeling"""
        
        # 1. PER-GAME AVERAGES (Foundation)
        home_xg_pg = home_data["xG"] / games_played
        away_xg_pg = away_data["xG"] / games_played
        home_xga_pg = home_data["xGA"] / games_played
        away_xga_pg = away_data["xGA"] / games_played
        
        # 2. DETECT OVERPERFORMANCE (Reduced impact from 0.4 to 0.25)
        home_overperformance = (home_data["points"] - home_data["xPTS"]) / games_played
        away_overperformance = (away_data["points"] - away_data["xPTS"]) / games_played
        
        # Reduced intangible factor: 25% instead of 40%
        home_intangible = home_overperformance * 0.25
        away_intangible = away_overperformance * 0.25
        
        # 3. QUALITY ASSESSMENT
        home_quality = (home_xg_pg * 1.3) + (1.5 - home_xga_pg) * 1.2 + home_intangible
        away_quality = (away_xg_pg * 1.3) + (1.5 - away_xga_pg) * 1.2 + away_intangible
        quality_diff = home_quality - away_quality
        
        # 4. CONSERVATIVE HOME ADVANTAGE
        if quality_diff > 0.5:  # Home team much better
            home_boost = 1.03  # 3% boost
            away_reduction = 0.97
        elif quality_diff < -0.5:  # Away team much better
            home_boost = 1.01  # Only 1% boost for weak home teams
            away_reduction = 0.99
        elif abs(quality_diff) < 0.2:  # Evenly matched
            home_boost = 1.02  # 2% home advantage
            away_reduction = 0.98
        else:  # Slight advantage
            home_boost = 1.015  # 1.5% boost
            away_reduction = 0.985
        
        # 5. GRADIENT MOMENTUM CALCULATION
        home_momentum = 0
        away_momentum = 0
        
        # Attack momentum with gradient scale
        home_xg_gap = (home_data["xG"] - home_data["goals_scored"]) / games_played
        away_xg_gap = (away_data["xG"] - away_data["goals_scored"]) / games_played
        
        home_momentum += np.clip(home_xg_gap * 20, -10, 10)
        away_momentum += np.clip(away_xg_gap * 20, -10, 10)
        
        # Defense momentum with gradient scale
        home_def_gap = (home_data["xGA"] - home_data["goals_conceded"]) / games_played
        away_def_gap = (away_data["xGA"] - away_data["goals_conceded"]) / games_played
        
        home_momentum += np.clip(home_def_gap * 15, -8, 8)
        away_momentum += np.clip(away_def_gap * 15, -8, 8)
        
        momentum_diff = (home_momentum - away_momentum) * 0.10
        
        # 6. EXPECTED GOALS with home adjustment
        home_expected = home_xg_pg * home_boost
        away_expected = away_xg_pg * away_reduction
        
        # Reduce scoring expectation for weak home teams vs strong away teams
        if home_quality < 1.5 and away_quality > 2.0:  # Weak home vs strong away
            home_expected *= 0.85  # 15% reduction in scoring expectation
        
        total_expected = home_expected + away_expected
        expected_goal_diff = home_expected - away_expected
        
        # 7. GET MATCH CONTEXT
        context_factor, volatility_note = get_match_context(
            home_data, away_data, home_xg_pg, away_xg_pg, games_played
        )
        
        # 8. MATCH WINNER CALIBRATION (With home edge for balanced matches)
        # Base advantage primarily on quality difference
        total_advantage = (quality_diff * 0.7) + (momentum_diff * 0.3)
        
        # Apply context factor
        total_advantage *= context_factor
        
        # Give slight home edge for truly balanced matches
        home_edge_note = ""
        if abs(quality_diff) < 0.2 and abs(total_advantage) < 0.2:
            total_advantage += 0.15  # Small home edge for balanced matches
            home_edge_note = "Slight home edge given in balanced match"
        
        # Determine winner based on actual advantage
        if total_advantage > 0.3:  # Clear home advantage
            base_confidence = 68
            raw_confidence = min(85, base_confidence + (total_advantage - 0.3) * 20)
            winner = "Home Win"
        elif total_advantage < -0.3:  # Clear away advantage
            base_confidence = 66
            raw_confidence = min(83, base_confidence + (abs(total_advantage) - 0.3) * 19)
            winner = "Away Win"
        else:  # Closely matched
            if total_advantage > 0.1:
                winner = "Home Win"
                raw_confidence = 55 + total_advantage * 15
            elif total_advantage < -0.1:
                winner = "Away Win"
                raw_confidence = 53 + abs(total_advantage) * 15
            else:
                winner = "Draw"
                raw_confidence = 48
        
        # Apply confidence caps
        if abs(expected_goal_diff) < 0.2:
            confidence_cap = 70
        elif abs(expected_goal_diff) < 0.4:
            confidence_cap = 75
        elif abs(expected_goal_diff) < 0.6:
            confidence_cap = 80
        else:
            confidence_cap = 85
        
        confidence = min(confidence_cap, max(25, raw_confidence))
        
        # 9. TOTAL GOALS PROBABILITY (Wider decision bands)
        if total_expected > 3.5:
            over_25_prob = 78
        elif total_expected > 3.0:
            over_25_prob = 68
        elif total_expected > 2.6:
            over_25_prob = 58
        elif total_expected > 2.2:
            over_25_prob = 48
        elif total_expected > 1.8:
            over_25_prob = 38
        else:
            over_25_prob = 28
        
        # Add volatility adjustment
        home_xg_std = abs(home_xg_gap) * 2
        away_xg_std = abs(away_xg_gap) * 2
        volatility_boost = (home_xg_std + away_xg_std) * 3
        over_25_prob = min(85, max(15, over_25_prob + volatility_boost))
        
        # Total Goals decision (Wider bands: 52/48 instead of 58/42)
        if over_25_prob >= 52:  # Lowered from 58
            goals_selection = "Over 2.5 Goals"
            goals_confidence = over_25_prob * 0.88
        elif over_25_prob <= 48:  # Raised from 42
            goals_selection = "Under 2.5 Goals"
            goals_confidence = (100 - over_25_prob) * 0.88
        else:
            goals_selection = "Avoid Total Goals"
            goals_confidence = 38
        
        # 10. BTTS CALCULATION (With mismatch adjustment)
        btts_raw_prob = calculate_btts_poisson_probability(home_expected, away_expected)
        
        # Adjust based on defensive strengths
        if home_xga_pg < 1.0 and away_xga_pg < 1.0:
            btts_raw_prob -= 6
        elif home_xga_pg > 1.8 or away_xga_pg > 1.8:
            btts_raw_prob += 8
        
        btts_raw_prob = min(82, max(18, btts_raw_prob))
        
        # Get BTTS decision with dynamic thresholds and mismatch adjustment
        btts_selection, btts_confidence, btts_adjust_note = calculate_btts_decision(
            btts_raw_prob, total_expected, home_quality, away_quality
        )
        
        # 11. ASIAN HANDICAP CALIBRATION
        if "Home" in winner:
            if expected_goal_diff > 1.0:
                ah_selection = "Home -1.5"
                ah_confidence = min(80, 65 + expected_goal_diff * 12)
            elif expected_goal_diff > 0.6:
                ah_selection = "Home -1.0"
                ah_confidence = min(75, 62 + expected_goal_diff * 15)
            elif expected_goal_diff > 0.3:
                ah_selection = "Home -0.75"
                ah_confidence = min(70, 58 + expected_goal_diff * 18)
            elif expected_goal_diff > 0.1:
                ah_selection = "Home -0.5"
                ah_confidence = min(65, 55 + expected_goal_diff * 20)
            else:
                ah_selection = "Home -0.25"
                ah_confidence = min(60, 52 + expected_goal_diff * 22)
        elif "Away" in winner:
            if expected_goal_diff < -1.0:
                ah_selection = "Away -1.0"
                ah_confidence = min(78, 64 + abs(expected_goal_diff) * 12)
            elif expected_goal_diff < -0.6:
                ah_selection = "Away -0.75"
                ah_confidence = min(73, 60 + abs(expected_goal_diff) * 15)
            elif expected_goal_diff < -0.3:
                ah_selection = "Away -0.5"
                ah_confidence = min(68, 56 + abs(expected_goal_diff) * 18)
            elif expected_goal_diff < -0.1:
                ah_selection = "Away -0.25"
                ah_confidence = min(63, 52 + abs(expected_goal_diff) * 20)
            else:
                ah_selection = "Away 0.0"
                ah_confidence = min(58, 48 + abs(expected_goal_diff) * 22)
        else:  # Draw
            ah_selection = "Draw No Bet"
            ah_confidence = 52
        
        ah_confidence = min(85, max(20, ah_confidence))
        
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
                "home_edge_note": home_edge_note,
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
