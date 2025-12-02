import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, List, Optional

# Add new league adjustment function
def get_league_adjustments(league_name: str) -> Dict:
    """Get league-specific goal averages and adjustments"""
    league_data = {
        'Bundesliga': {'avg_goals': 3.14, 'over_threshold': 3.0, 'under_threshold': 2.8, 'btts_baseline': 58},
        'Premier League': {'avg_goals': 2.93, 'over_threshold': 2.8, 'under_threshold': 2.6, 'btts_baseline': 56},
        'Ligue 1': {'avg_goals': 2.96, 'over_threshold': 2.85, 'under_threshold': 2.65, 'btts_baseline': 55},
        'La Liga': {'avg_goals': 2.62, 'over_threshold': 2.5, 'under_threshold': 2.3, 'btts_baseline': 52},
        'Serie A': {'avg_goals': 2.56, 'over_threshold': 2.45, 'under_threshold': 2.25, 'btts_baseline': 51},
        'RFPL': {'avg_goals': 2.59, 'over_threshold': 2.5, 'under_threshold': 2.3, 'btts_baseline': 50}
    }
    return league_data.get(league_name, {'avg_goals': 2.7, 'over_threshold': 2.7, 'under_threshold': 2.5, 'btts_baseline': 52})

# Update BTTS decision to include team profiles
def calculate_btts_decision(btts_raw_prob: float, total_xg: float, 
                           home_offensive: float, away_offensive: float,
                           home_defensive: Optional[float] = None, 
                           away_defensive: Optional[float] = None) -> Tuple[str, float, str]:
    """Calculate BTTS decision with team profile considerations"""
    
    # Base adjustments from total xG
    if total_xg > 3.0:
        btts_boost = 6
        selection_threshold = 54
        avoid_upper = 46
    elif total_xg > 2.5:
        btts_boost = 3
        selection_threshold = 56
        avoid_upper = 44
    else:
        btts_boost = 0
        selection_threshold = 58
        avoid_upper = 42
    
    # Offensive quality adjustment
    offensive_factor = (home_offensive + away_offensive) / 3.0
    if offensive_factor > 1.1:
        btts_boost += 4
    elif offensive_factor < 0.9:
        btts_boost -= 3
    
    # Defensive quality adjustment (if provided)
    if home_defensive is not None and away_defensive is not None:
        defensive_factor = (home_defensive + away_defensive) / 2.0
        if defensive_factor > 1.2:  # Poor defenses
            btts_boost += 5
        elif defensive_factor < 0.8:  # Strong defenses
            btts_boost -= 4
    
    adjusted_prob = min(78, max(22, btts_raw_prob + btts_boost))
    
    # Determine selection
    if adjusted_prob >= selection_threshold:
        confidence = adjusted_prob * 0.88
        if offensive_factor > 1.1:
            note = f"Both teams have strong attacking quality"
        else:
            note = f"Adjusted from {btts_raw_prob:.0f}% to {adjusted_prob:.0f}%"
        return "Yes", confidence, note
    
    elif adjusted_prob <= avoid_upper:
        confidence = (100 - adjusted_prob) * 0.88
        if home_defensive is not None and home_defensive < 0.8:
            note = f"Strong defensive matchup reduces BTTS probability"
        else:
            note = f"Adjusted from {btts_raw_prob:.0f}% to {adjusted_prob:.0f}%"
        return "No", confidence, note
    
    else:
        return "Avoid BTTS", 48, f"{adjusted_prob:.0f}% probability falls in ambiguous range"

# Add team-specific adjustment calculator
def calculate_team_specific_adjustments(home_profile: Dict, away_profile: Dict) -> Dict:
    """Calculate adjustments based on team-specific profiles"""
    
    # Goal expectancy adjustment
    avg_total_goals = (home_profile.get('avg_total_goals', 2.7) + 
                      away_profile.get('avg_total_goals', 2.7)) / 2
    
    # Defensive quality impact
    home_defense = home_profile.get('goals_conceded_pg', 1.3)
    away_defense = away_profile.get('goals_conceded_pg', 1.3)
    defensive_factor = (home_defense + away_defense) / 2.6  # Normalize to 1.0
    
    # Home advantage
    home_advantage = home_profile.get('home_away_goal_diff', 0.0)
    
    return {
        'avg_total_goals': avg_total_goals,
        'defensive_factor': defensive_factor,
        'home_advantage': home_advantage,
        'expected_game_openness': 1.0 + (defensive_factor - 1.0) * 0.5,
        'btts_bias': 'positive' if defensive_factor > 1.1 else 'negative' if defensive_factor < 0.9 else 'neutral'
    }

# Keep existing functions but update formatting to include team profiles
def format_prediction_display(pred: Dict, analysis_data: Dict, team_names: Dict, team_profiles: Dict) -> Tuple[str, str, str, str]:
    """Generate formatted display with team profile insights"""
    
    confidence = pred['confidence']
    pred_type = pred['type']
    selection = pred['selection']
    
    # Color and stake determination (keep existing)
    if "Avoid" in selection:
        color = "⚪"
        stake = "❌ AVOID"
    elif confidence >= 70:
        color = "🟢"
        stake = "1.5 units"
    elif confidence >= 62:
        color = "🟢"
        stake = "1.0 units"
    elif confidence >= 55:
        color = "🟡"
        stake = "0.75 units"
    elif confidence >= 48:
        color = "🟡"
        stake = "0.5 units"
    else:
        color = "🟠"
        stake = "0.25 units"
    
    # Generate analysis with team profile insights
    if pred_type == "Total Goals":
        total_expected = analysis_data['total_expected']
        home_avg = team_profiles['home'].get('avg_total_goals', 2.7)
        away_avg = team_profiles['away'].get('avg_total_goals', 2.7)
        
        if "Over" in selection:
            if total_expected > max(home_avg, away_avg) + 0.5:
                description = f"**Analysis:** Expected goals ({total_expected:.1f}) significantly above both teams' averages ({home_avg:.1f}/{away_avg:.1f})."
            else:
                description = f"**Analysis:** Moderate expected goals ({total_expected:.1f}) favors slight edge to over."
        elif "Under" in selection:
            if total_expected < min(home_avg, away_avg) - 0.5:
                description = f"**Analysis:** Expected goals ({total_expected:.1f}) significantly below both teams' averages ({home_avg:.1f}/{away_avg:.1f})."
            else:
                description = f"**Analysis:** Below-average expected goals ({total_expected:.1f}) suggests limited scoring."
    
    elif pred_type == "Both Teams To Score":
        home_defense = team_profiles['home'].get('goals_conceded_pg', 1.3)
        away_defense = team_profiles['away'].get('goals_conceded_pg', 1.3)
        
        if selection == "Yes":
            if home_defense > 1.5 and away_defense > 1.5:
                description = f"**Analysis:** Both teams have defensive vulnerabilities ({home_defense:.1f}/{away_defense:.1f} goals conceded per game)."
            else:
                description = f"**Analysis:** Statistical models favor both teams scoring."
        elif selection == "No":
            if home_defense < 1.0 or away_defense < 1.0:
                description = f"**Analysis:** Strong defensive organization ({home_defense:.1f}/{away_defense:.1f} goals conceded) favors clean sheet."
            else:
                description = f"**Analysis:** Matchup dynamics suggest lower BTTS probability."
    
    # Generate reasoning
    reasoning = generate_enhanced_reasoning(pred, analysis_data, team_names, team_profiles)
    
    return color, stake, description, reasoning

def generate_enhanced_reasoning(pred: Dict, analysis_data: Dict, team_names: Dict, team_profiles: Dict) -> str:
    """Generate enhanced reasoning with team profile data"""
    
    pred_type = pred['type']
    selection = pred['selection']
    
    if pred_type == "Total Goals":
        total_expected = analysis_data['total_expected']
        home_profile = team_profiles['home']
        away_profile = team_profiles['away']
        
        if "Over" in selection:
            return f"**Reasoning:** Combined expected goals of {total_expected:.1f}. {team_names['home']} averages {home_profile.get('avg_total_goals', 2.7):.1f} total goals, {team_names['away']} averages {away_profile.get('avg_total_goals', 2.7):.1f}. Matchup analysis suggests higher-scoring game."
        elif "Under" in selection:
            return f"**Reasoning:** Combined expected goals of {total_expected:.1f}. Defensive analysis shows {team_names['home']} concedes {home_profile.get('goals_conceded_pg', 1.3):.1f} goals/game, {team_names['away']} concedes {away_profile.get('goals_conceded_pg', 1.3):.1f}. Likely tactical, low-scoring affair."
    
    elif pred_type == "Both Teams To Score":
        home_defense = team_profiles['home'].get('goals_conceded_pg', 1.3)
        away_defense = team_profiles['away'].get('goals_conceded_pg', 1.3)
        
        if selection == "Yes":
            return f"**Reasoning:** Both teams show defensive vulnerabilities ({home_defense:.1f} and {away_defense:.1f} goals conceded per game). Expected open match with scoring opportunities for both sides."
        elif selection == "No":
            return f"**Reasoning:** Defensive solidity ({home_defense:.1f} and {away_defense:.1f} goals conceded per game) suggests at least one team keeps a clean sheet in this matchup."
    
    # Default reasoning for other prediction types
    return f"**Reasoning:** Based on statistical models and team performance analysis."
