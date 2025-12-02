import numpy as np
import pandas as pd
import os
import streamlit as st
from typing import Dict, Tuple, List, Optional
import math

def detect_league_from_filename(filename: str) -> str:
    """Detect league from CSV filename"""
    filename_lower = filename.lower()
    
    if 'bundesliga' in filename_lower:
        return 'Bundesliga'
    elif 'premier' in filename_lower:
        return 'Premier League'
    elif 'serie_a' in filename_lower or 'serie-a' in filename_lower:
        return 'Serie A'
    elif 'la_liga' in filename_lower or 'laliga' in filename_lower:
        return 'La Liga'
    elif 'ligue_1' in filename_lower or 'ligue-1' in filename_lower:
        return 'Ligue 1'
    elif 'rfpl' in filename_lower:
        return 'RFPL'
    else:
        return 'Average'

def get_league_settings(league_name: str) -> Dict:
    """Get league-specific settings"""
    league_data = {
        'Bundesliga': {
            'avg_goals': 3.14,
            'over_threshold': 3.0,
            'under_threshold': 2.8,
            'btts_baseline': 58,
            'home_advantage': 0.4,
            'goal_variance': 'high'
        },
        'Premier League': {
            'avg_goals': 2.93,
            'over_threshold': 2.8,
            'under_threshold': 2.6,
            'btts_baseline': 56,
            'home_advantage': 0.35,
            'goal_variance': 'medium'
        },
        'Ligue 1': {
            'avg_goals': 2.96,
            'over_threshold': 2.85,
            'under_threshold': 2.65,
            'btts_baseline': 55,
            'home_advantage': 0.3,
            'goal_variance': 'medium'
        },
        'La Liga': {
            'avg_goals': 2.62,
            'over_threshold': 2.5,
            'under_threshold': 2.3,
            'btts_baseline': 52,
            'home_advantage': 0.3,
            'goal_variance': 'low'
        },
        'Serie A': {
            'avg_goals': 2.56,
            'over_threshold': 2.45,
            'under_threshold': 2.25,
            'btts_baseline': 51,
            'home_advantage': 0.25,
            'goal_variance': 'low'
        },
        'RFPL': {
            'avg_goals': 2.59,
            'over_threshold': 2.5,
            'under_threshold': 2.3,
            'btts_baseline': 50,
            'home_advantage': 0.35,
            'goal_variance': 'medium'
        }
    }
    
    return league_data.get(league_name, {
        'avg_goals': 2.7,
        'over_threshold': 2.7,
        'under_threshold': 2.5,
        'btts_baseline': 52,
        'home_advantage': 0.3,
        'goal_variance': 'medium'
    })

# Keep all existing functions but add league parameter where needed
def calculate_btts_poisson_probability(home_xg: float, away_xg: float, league: str = 'Average') -> float:
    """Calculate BTTS probability using Poisson distribution with league adjustment"""
    p_home_scores = 1 - math.exp(-home_xg)
    p_away_scores = 1 - math.exp(-away_xg)
    
    btts_prob = p_home_scores * p_away_scores * 100
    
    total_xg = home_xg + away_xg
    league_settings = get_league_settings(league)
    league_avg = league_settings['avg_goals']
    
    # League-specific adjustments
    if total_xg > league_avg + 0.5:
        btts_prob *= 1.15
    elif total_xg > league_avg + 0.2:
        btts_prob *= 1.08
    elif total_xg < league_avg - 0.3:
        btts_prob *= 0.9
    
    return min(85, max(15, btts_prob))

def calculate_btts_decision(btts_raw_prob: float, total_xg: float, 
                           home_offensive: float, away_offensive: float,
                           league: str = 'Average',
                           home_defensive: Optional[float] = None, 
                           away_defensive: Optional[float] = None) -> Tuple[str, float, str]:
    """Calculate BTTS decision with league considerations"""
    
    league_settings = get_league_settings(league)
    league_avg = league_settings['avg_goals']
    btts_baseline = league_settings['btts_baseline']
    
    # Base adjustments from total xG relative to league average
    if total_xg > league_avg + 0.5:
        btts_boost = 6
        selection_threshold = btts_baseline - 4
        avoid_upper = btts_baseline - 12
    elif total_xg > league_avg + 0.2:
        btts_boost = 3
        selection_threshold = btts_baseline - 2
        avoid_upper = btts_baseline - 8
    else:
        btts_boost = 0
        selection_threshold = btts_baseline
        avoid_upper = btts_baseline - 6
    
    # Offensive quality adjustment
    offensive_factor = (home_offensive + away_offensive) / 3.0
    if offensive_factor > 1.1:
        btts_boost += 4
    elif offensive_factor < 0.9:
        btts_boost -= 3
    
    # Defensive quality adjustment
    if home_defensive is not None and away_defensive is not None:
        defensive_factor = (home_defensive + away_defensive) / 2.0
        if defensive_factor > 1.2:
            btts_boost += 5
        elif defensive_factor < 0.8:
            btts_boost -= 4
    
    adjusted_prob = min(78, max(22, btts_raw_prob + btts_boost))
    
    # Determine selection
    if adjusted_prob >= selection_threshold:
        confidence = adjusted_prob * 0.88
        note = f"{adjusted_prob:.0f}% probability (league avg: {btts_baseline}%)"
        return "Yes", confidence, note
    
    elif adjusted_prob <= avoid_upper:
        confidence = (100 - adjusted_prob) * 0.88
        note = f"{adjusted_prob:.0f}% probability (league avg: {btts_baseline}%)"
        return "No", confidence, note
    
    else:
        return "Avoid BTTS", 48, f"{adjusted_prob:.0f}% probability falls in ambiguous range"

# Keep all other existing functions exactly as they were
def get_match_context(home_xg: float, away_xg: float, total_xg: float) -> Dict:
    """Determine match context based on xG values"""
    context = {
        "expected_pace": "Slow",
        "expected_style": "Defensive",
        "btts_tendency": "Low",
        "scoring_efficiency": "Low"
    }
    
    if total_xg > 3.0:
        context["expected_pace"] = "Fast"
    elif total_xg > 2.5:
        context["expected_pace"] = "Moderate"
    
    if abs(home_xg - away_xg) > 0.8:
        context["expected_style"] = "One-sided"
    elif abs(home_xg - away_xg) < 0.3:
        context["expected_style"] = "Balanced"
    
    btts_raw = calculate_btts_poisson_probability(home_xg, away_xg)
    if btts_raw > 60:
        context["btts_tendency"] = "High"
    elif btts_raw > 50:
        context["btts_tendency"] = "Medium"
    
    if home_xg > 1.5 or away_xg > 1.5:
        context["scoring_efficiency"] = "High"
    
    return context

def calculate_team_specific_adjustments(home_profile: Dict, away_profile: Dict) -> Dict:
    """Calculate adjustments based on team-specific profiles"""
    avg_total_goals = (home_profile.get('avg_total_goals', 2.7) + 
                      away_profile.get('avg_total_goals', 2.7)) / 2
    
    home_defense = home_profile.get('goals_conceded_pg', 1.3)
    away_defense = away_profile.get('goals_conceded_pg', 1.3)
    defensive_factor = (home_defense + away_defense) / 2.6
    
    home_advantage = home_profile.get('home_away_goal_diff', 0.0)
    
    return {
        'avg_total_goals': avg_total_goals,
        'defensive_factor': defensive_factor,
        'home_advantage': home_advantage,
        'expected_game_openness': 1.0 + (defensive_factor - 1.0) * 0.5,
        'btts_bias': 'positive' if defensive_factor > 1.1 else 'negative' if defensive_factor < 0.9 else 'neutral'
    }

def validate_input_data(home_data: Dict, away_data: Dict, games_played: int) -> Tuple[bool, str]:
    """Validate input data before processing"""
    if games_played <= 0:
        return False, "Games played must be greater than 0"
    
    required_fields = ["points", "goals_scored", "goals_conceded", "xG", "xGA", "xPTS"]
    
    for field in required_fields:
        if field not in home_data:
            return False, f"Missing {field} in home data"
        if field not in away_data:
            return False, f"Missing {field} in away data"
        
        if field in ["xG", "xGA", "xPTS"]:
            if home_data[field] < 0 or away_data[field] < 0:
                return False, f"{field} cannot be negative"
    
    return True, ""

def format_prediction_display(pred: Dict, analysis_data: Dict, team_names: Dict, team_profiles: Dict) -> Tuple[str, str, str, str]:
    """Generate formatted display with team profile insights"""
    
    confidence = pred['confidence']
    pred_type = pred['type']
    selection = pred['selection']
    
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
    
    description = ""
    if pred_type == "Total Goals":
        total_expected = analysis_data.get('total_expected', analysis_data.get('expected_goals', {}).get('total', 2.5))
        home_avg = team_profiles.get('home', {}).get('avg_total_goals', 2.7)
        away_avg = team_profiles.get('away', {}).get('avg_total_goals', 2.7)
        
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
        home_defense = team_profiles.get('home', {}).get('goals_conceded_pg', 1.3)
        away_defense = team_profiles.get('away', {}).get('goals_conceded_pg', 1.3)
        
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
    else:
        description = f"**Analysis:** Based on advanced statistical modeling and matchup analysis."
    
    reasoning = generate_enhanced_reasoning(pred, analysis_data, team_names, team_profiles)
    
    return color, stake, description, reasoning

def generate_enhanced_reasoning(pred: Dict, analysis_data: Dict, team_names: Dict, team_profiles: Dict) -> str:
    """Generate enhanced reasoning with team profile data"""
    
    pred_type = pred['type']
    selection = pred['selection']
    
    if pred_type == "Total Goals":
        total_expected = analysis_data.get('total_expected', analysis_data.get('expected_goals', {}).get('total', 2.5))
        home_profile = team_profiles.get('home', {})
        away_profile = team_profiles.get('away', {})
        
        if "Over" in selection:
            return f"**Reasoning:** Combined expected goals of {total_expected:.1f}. {team_names['home']} averages {home_profile.get('avg_total_goals', 2.7):.1f} total goals, {team_names['away']} averages {away_profile.get('avg_total_goals', 2.7):.1f}. Matchup analysis suggests higher-scoring game."
        elif "Under" in selection:
            return f"**Reasoning:** Combined expected goals of {total_expected:.1f}. Defensive analysis shows {team_names['home']} concedes {home_profile.get('goals_conceded_pg', 1.3):.1f} goals/game, {team_names['away']} concedes {away_profile.get('goals_conceded_pg', 1.3):.1f}. Likely tactical, low-scoring affair."
    
    elif pred_type == "Both Teams To Score":
        home_defense = team_profiles.get('home', {}).get('goals_conceded_pg', 1.3)
        away_defense = team_profiles.get('away', {}).get('goals_conceded_pg', 1.3)
        
        if selection == "Yes":
            return f"**Reasoning:** Both teams show defensive vulnerabilities ({home_defense:.1f} and {away_defense:.1f} goals conceded per game). Expected open match with scoring opportunities for both sides."
        elif selection == "No":
            return f"**Reasoning:** Defensive solidity ({home_defense:.1f} and {away_defense:.1f} goals conceded per game) suggests at least one team keeps a clean sheet in this matchup."
    
    return f"**Reasoning:** Based on statistical models and team performance analysis."

def load_league_data(league_name: str) -> Optional[pd.DataFrame]:
    """Load CSV data for a specific league and detect league"""
    try:
        # Detect league from filename
        detected_league = detect_league_from_filename(league_name)
        st.session_state.detected_league = detected_league
        
        # Remove .csv extension if present
        if league_name.endswith('.csv'):
            league_name = league_name[:-4]
        
        data_path = os.path.join("data", f"{league_name}.csv")
        
        if not os.path.exists(data_path):
            st.error(f"File not found: {data_path}")
            return None
        
        df = pd.read_csv(data_path)
        
        column_mapping = {}
        if 'Team_Goals_Conceded_PG' in df.columns:
            column_mapping['Team_Goals_Conceded_PG'] = 'goals_conceded_pg'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        required_columns = ['Team', 'Home_Away', 'Matches', 'Wins', 'Draws', 'Losses', 
                           'Goals', 'Goals_Against', 'xG', 'xGA', 'xPTS', 'Points']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"Error loading {league_name}: {str(e)}")
        return None

def get_available_leagues() -> List[str]:
    """Get list of available CSV files in data folder"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return []
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    return sorted(csv_files)
