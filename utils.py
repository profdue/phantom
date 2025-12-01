import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, List

def get_available_leagues(data_folder="data"):
    """Get list of available leagues from CSV files"""
    import os
    
    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"⚠️ Data folder '{data_folder}' not found. Creating it.")
        os.makedirs(data_folder, exist_ok=True)
        return []
    
    # Get all CSV files
    csv_files = []
    try:
        for file in os.listdir(data_folder):
            if file.lower().endswith('.csv'):
                csv_files.append(file)
    except Exception as e:
        print(f"Error reading data folder: {e}")
        return []
    
    if not csv_files:
        print("No CSV files found in data folder")
        return []
    
    # Process CSV filenames into league names
    leagues = []
    for file in csv_files:
        # Remove .csv extension
        name = file.replace('.csv', '')
        
        # Remove common suffixes
        name = name.replace('_home_away', '')
        name = name.replace('_homeaway', '')
        name = name.replace('_home', '')
        name = name.replace('_away', '')
        
        # Replace underscores with spaces
        name = name.replace('_', ' ')
        
        # Title case with special handling
        name = name.title()
        
        # Fix specific league names
        name = name.replace('La Liga', 'La Liga')  # Keep as is
        name = name.replace('Rfpl', 'RFPL')
        name = name.replace('Premier League', 'Premier League')
        
        leagues.append(name.strip())
    
    # Remove duplicates and sort
    return sorted(list(set(leagues)))

def load_league_data(league_name, data_folder="data"):
    """Load CSV data for selected league"""
    try:
        # Convert league name to possible filename patterns
        base_name = league_name.lower().replace(' ', '_')
        
        # Try different filename patterns
        possible_filenames = [
            f"{base_name}_home_away.csv",
            f"{base_name}.csv",
            f"{base_name}_homeaway.csv",
        ]
        
        filepath = None
        for filename in possible_filenames:
            test_path = os.path.join(data_folder, filename)
            if os.path.exists(test_path):
                filepath = test_path
                break
        
        if filepath is None:
            # Try to find any CSV file that contains the league name
            for file in os.listdir(data_folder):
                if file.endswith('.csv') and league_name.lower().replace(' ', '_') in file.lower():
                    filepath = os.path.join(data_folder, file)
                    break
        
        if filepath is None:
            print(f"No CSV file found for league: {league_name}")
            return None
        
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Clean column names (remove trailing/leading spaces)
        df.columns = df.columns.str.strip()
        
        # Ensure required columns exist
        required_columns = ['Team', 'Matches', 'Home_Away', 'Wins', 'Draws', 'Losses', 
                           'Goals', 'Goals_Against', 'Points', 'xG', 'xGA', 'xPTS']
        
        for col in required_columns:
            if col not in df.columns:
                print(f"Missing required column: {col}")
                return None
        
        # Convert numeric columns
        numeric_cols = ['Matches', 'Wins', 'Draws', 'Losses', 'Goals', 'Goals_Against', 
                       'Points', 'xG', 'xGA', 'xPTS']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"Error loading league data: {e}")
        return None

def validate_input_data(home_data: Dict, away_data: Dict, games_played: int) -> Tuple[bool, str]:
    """Validate input data for completeness and reasonableness"""
    
    required_keys = ["points", "goals_scored", "goals_conceded", "xG", "xGA", "xPTS"]
    
    for key in required_keys:
        if key not in home_data or key not in away_data:
            return False, f"Missing required key: {key}"
    
    # Check for reasonable values
    if home_data["xG"] < 0 or away_data["xG"] < 0:
        return False, "xG cannot be negative"
    
    if home_data["goals_scored"] > 100 or away_data["goals_scored"] > 100:
        return False, "Goals scored value seems unrealistic"
    
    if games_played <= 0:
        return False, "Games played must be positive"
    
    return True, "Data valid"

def calculate_poisson_probability(lambda_val: float, k: int) -> float:
    """Calculate Poisson probability P(X = k)"""
    return np.exp(-lambda_val) * (lambda_val ** k) / np.math.factorial(k)

def calculate_btts_poisson_probability(home_expected: float, away_expected: float) -> float:
    """Calculate Both Teams To Score probability using Poisson distribution"""
    
    # Adjust for opponent defense
    home_scoring_exp = home_expected * 0.85
    away_scoring_exp = away_expected * 0.85
    
    # Probability of NOT scoring
    home_no_goal = np.exp(-home_scoring_exp)
    away_no_goal = np.exp(-away_scoring_exp)
    
    # Probability both score
    btts_prob = (1 - home_no_goal) * (1 - away_no_goal)
    
    return btts_prob * 100

def get_match_context(home_data: Dict, away_data: Dict, home_xg_pg: float, away_xg_pg: float, games_played: int) -> Tuple[float, str]:
    """Get context factor for match prediction and volatility note"""
    context_factor = 1.0
    volatility_note = ""
    
    # High total xG matches are more unpredictable
    total_xg = home_xg_pg + away_xg_pg
    if total_xg > 3.0:
        context_factor *= 0.92
        volatility_note = "High expected goals match - higher variance expected"
    elif total_xg > 2.5:
        context_factor *= 0.96
        volatility_note = "Elevated expected goals - moderate variance"
    
    return context_factor, volatility_note

def get_btts_reasoning(btts_prob: float, home_xg: float, away_xg: float, 
                       home_xga: float, away_xga: float, total_xg: float, 
                       selection: str, adjusted_note: str = "") -> str:
    """Generate nuanced BTTS reasoning"""
    
    if selection == "Yes":
        if total_xg > 3.0:
            return f"**Reasoning:** High-scoring match expected ({total_xg:.1f} xG) with both teams creating quality chances. {adjusted_note}"
        elif home_xga > 1.5 and away_xga > 1.5:
            return f"**Reasoning:** Defensive vulnerabilities on both sides make BTTS likely. {adjusted_note}"
        elif btts_prob > 65:
            return f"**Reasoning:** Strong statistical probability ({btts_prob:.0f}%) that both teams will score based on attacking metrics. {adjusted_note}"
        else:
            return f"**Reasoning:** Statistical models estimate good chance both teams score. {adjusted_note}"
    
    elif selection == "No":
        if home_xg < 1.0 or away_xg < 1.0:
            return f"**Reasoning:** Limited attacking threat from one team reduces BTTS probability. {adjusted_note}"
        elif home_xga < 1.0 and away_xga < 1.0:
            return f"**Reasoning:** Strong defensive organization on both sides favors at least one clean sheet. {adjusted_note}"
        else:
            return f"**Reasoning:** Matchup dynamics or statistical models suggest lower BTTS probability. {adjusted_note}"
    
    else:  # Avoid
        if total_xg > 2.8:
            return f"**Reasoning:** Despite {btts_prob:.0f}% BTTS probability, high expected goals ({total_xg:.1f}) suggests both teams could score in an open match. {adjusted_note}"
        else:
            return f"**Reasoning:** {btts_prob:.0f}% BTTS probability falls in ambiguous range - outcome depends on finishing quality. {adjusted_note}"

def calculate_btts_decision(btts_raw_prob: float, total_xg: float, home_quality: float, away_quality: float) -> Tuple[str, float, str]:
    """Calculate BTTS decision with adjusted probabilities based on total xG"""
    
    # When total xG is high, BTTS is more likely
    if total_xg > 3.0:
        btts_boost = 8
        btts_selection_threshold = 52
        btts_avoid_upper = 48
    elif total_xg > 2.5:
        btts_boost = 4
        btts_selection_threshold = 54
        btts_avoid_upper = 46
    else:
        btts_boost = 0
        btts_selection_threshold = 55
        btts_avoid_upper = 45
    
    # Add mismatch adjustment: Very unbalanced matches reduce BTTS probability
    quality_ratio = home_quality / (away_quality + 0.001)
    mismatch_note = ""
    if quality_ratio < 0.6 or quality_ratio > 1.67:  # One team much stronger
        btts_boost -= 5
        mismatch_note = "Mismatch adjustment applied"
    
    adjusted_prob = min(85, max(15, btts_raw_prob + btts_boost))
    
    if adjusted_prob >= btts_selection_threshold:
        confidence = adjusted_prob * 0.85
        if confidence > 70:
            confidence = min(85, confidence)
        note = f"Adjusted from {btts_raw_prob:.0f}% to {adjusted_prob:.0f}%"
        if mismatch_note:
            note += f" ({mismatch_note})"
        return "Yes", confidence, note
    elif adjusted_prob <= btts_avoid_upper:
        confidence = (100 - adjusted_prob) * 0.85
        note = f"Adjusted from {btts_raw_prob:.0f}% to {adjusted_prob:.0f}%"
        if mismatch_note:
            note += f" ({mismatch_note})"
        return "No", confidence, note
    else:
        return "Avoid BTTS", 48, f"{adjusted_prob:.0f}% probability falls in ambiguous range {mismatch_note}"

def format_prediction_display(pred: Dict, analysis_data: Dict, team_names: Dict, raw_data: Dict) -> Tuple[str, str, str, str]:
    """Generate formatted display for a prediction"""
    
    confidence = pred['confidence']
    pred_type = pred['type']
    selection = pred['selection']
    
    # Determine color and stake
    if "Avoid" in selection:
        color = "⚪"
        stake = "❌ AVOID"
    elif confidence >= 70:
        color = "🟢"
        stake = "2.0 units"
    elif confidence >= 60:
        color = "🟢"
        stake = "1.5 units"
    elif confidence >= 50:
        color = "🟡" 
        stake = "1.0 units"
    else:
        color = "🟠"
        stake = "0.5 units"
    
    # Generate dynamic analysis
    if pred_type == "Match Winner":
        if "Home" in selection:
            if analysis_data['quality_diff'] > 1.0:
                advantage_source = "significant quality advantage"
            elif analysis_data['home_intangible'] > 0.2:
                advantage_source = "intangible factors and home advantage"
            elif analysis_data['home_boost'] > 1.02:
                advantage_source = "home advantage"
            else:
                advantage_source = "slight statistical edge"
            
            if confidence >= 75:
                description = f"**Analysis:** Strong statistical edge based on {advantage_source}."
            elif confidence >= 65:
                description = f"**Analysis:** Clear advantage with {advantage_source}."
            else:
                description = f"**Analysis:** Moderate edge with {advantage_source}."
                
        elif "Away" in selection:
            if abs(analysis_data['quality_diff']) > 1.0:
                advantage_source = "significant quality superiority"
            elif analysis_data['away_intangible'] > 0.2:
                advantage_source = "intangible factors overcoming home field"
            else:
                advantage_source = "quality advantage"
            
            if confidence >= 75:
                description = f"**Analysis:** Strong away advantage based on {advantage_source}."
            elif confidence >= 65:
                description = f"**Analysis:** Clear away edge with {advantage_source}."
            else:
                description = f"**Analysis:** Moderate away advantage detected."
                
        else:  # Draw
            if abs(analysis_data['quality_diff']) < 0.2:
                description = "**Analysis:** Extremely balanced match with near-identical underlying metrics."
            else:
                description = "**Analysis:** Competitive match where statistical advantages cancel out."
        
        # Add volatility note if present
        if analysis_data['volatility_note'] and confidence < 75:
            description += f" {analysis_data['volatility_note']}"
    
    elif pred_type == "Total Goals":
        total_xg = analysis_data['expected_goals']['total']
        
        if "Over" in selection:
            if total_xg > 3.5:
                description = f"**Analysis:** Very high expected goals ({total_xg:.1f} xG) indicates extremely open match."
            elif total_xg > 2.8:
                description = f"**Analysis:** Elevated expected goals ({total_xg:.1f} xG) suggests multiple scoring opportunities."
            else:
                description = f"**Analysis:** Moderate expected goals ({total_xg:.1f} xG) favors slight edge to over."
        elif "Under" in selection:
            if total_xg < 2.0:
                description = f"**Analysis:** Very low expected goals ({total_xg:.1f} xG) indicates defensive battle."
            else:
                description = f"**Analysis:** Below-average expected goals ({total_xg:.1f} xG) suggests limited scoring."
        else:  # Avoid
            description = f"**Analysis:** Expected goals total ({total_xg:.1f} xG) falls in ambiguous range."
    
    elif pred_type == "Both Teams To Score":
        if "Avoid" in selection:
            description = f"**Analysis:** BTTS probability falls in ambiguous middle range."
        elif "Yes" in selection:
            description = f"**Analysis:** Statistical models favor both teams scoring."
        else:  # No
            description = f"**Analysis:** Statistical models favor at least one clean sheet."
    
    else:  # Asian Handicap
        exp_diff = analysis_data['expected_goal_diff']
        
        if "Home" in selection:
            if exp_diff > 1.0:
                description = f"**Analysis:** Expected goal difference of {exp_diff:.1f} strongly favors home cover."
            elif exp_diff > 0.5:
                description = f"**Analysis:** Clear home advantage with {exp_diff:.1f} expected goal difference."
            else:
                description = f"**Analysis:** Moderate home edge suggests ability to cover handicap."
        elif "Away" in selection:
            if exp_diff < -1.0:
                description = f"**Analysis:** Expected goal difference of {abs(exp_diff):.1f} strongly favors away cover."
            elif exp_diff < -0.5:
                description = f"**Analysis:** Clear away advantage with {abs(exp_diff):.1f} expected goal difference."
            else:
                description = f"**Analysis:** Moderate away edge suggests ability to cover handicap."
        else:  # Draw No Bet
            description = "**Analysis:** Minimal expected goal difference suggests very close match."
    
    # Generate reasoning
    reasoning = generate_reasoning(pred, analysis_data, team_names, raw_data)
    
    return color, stake, description, reasoning

def generate_reasoning(pred: Dict, analysis_data: Dict, team_names: Dict, raw_data: Dict) -> str:
    """Generate dynamic reasoning for predictions"""
    
    pred_type = pred['type']
    selection = pred['selection']
    
    if pred_type == "Match Winner":
        if "Home" in selection:
            # Check for overperformance factor
            if raw_data['home_overperformance'] > 0.2:
                overperformance_note = f" ({team_names['home']} has been outperforming their expected points by {raw_data['home_overperformance']:.1f} per game)"
            else:
                overperformance_note = ""
                
            return f"**Reasoning:** {team_names['home']}'s underlying metrics combined with home advantage create a statistical favorite.{overperformance_note}"
        elif "Away" in selection:
            if raw_data['away_overperformance'] > 0.2:
                overperformance_note = f" ({team_names['away']} has been outperforming expected points by {raw_data['away_overperformance']:.1f} per game)"
            else:
                overperformance_note = ""
                
            return f"**Reasoning:** {team_names['away']}'s quality advantage overcomes home field disadvantage.{overperformance_note}"
        else:
            return f"**Reasoning:** Extremely balanced statistical profiles suggest neither team has a clear advantage."
    
    elif pred_type == "Total Goals":
        total_xg = analysis_data['expected_goals']['total']
        
        if "Over" in selection:
            if total_xg > 3.2:
                return f"**Reasoning:** Exceptionally high combined xG ({total_xg:.1f}) indicates an extremely open match with multiple clear scoring opportunities expected."
            elif total_xg > 2.6:
                return f"**Reasoning:** Above-average combined xG ({total_xg:.1f}) suggests an attacking game with both teams likely to create chances."
            else:
                return f"**Reasoning:** Moderate combined xG ({total_xg:.1f}) slightly favors over 2.5 goals based on both teams' attacking tendencies."
        elif "Under" in selection:
            if total_xg < 2.2:
                return f"**Reasoning:** Low combined xG ({total_xg:.1f}) indicates a tight, tactical match likely with limited clear-cut chances."
            else:
                return f"**Reasoning:** Below-par combined xG ({total_xg:.1f}) for a match at this level suggests a more controlled, defensive contest."
        else:
            return f"**Reasoning:** Expected goals total ({total_xg:.1f}) falls in the ambiguous range where historical outcomes show no clear pattern."
    
    elif pred_type == "Both Teams To Score":
        btts_raw_prob = analysis_data['probabilities']['btts_raw']
        total_xg = analysis_data['expected_goals']['total']
        home_xg = analysis_data['expected_goals']['home']
        away_xg = analysis_data['expected_goals']['away']
        
        # Use the enhanced reasoning function
        return get_btts_reasoning(
            btts_raw_prob, home_xg, away_xg, 
            raw_data['home_xga_pg'], raw_data['away_xga_pg'], total_xg,
            selection, analysis_data['probabilities']['btts_adjust_note']
        )
    
    else:  # Asian Handicap
        exp_diff = analysis_data['expected_goal_diff']
        
        if "Home" in selection:
            return f"**Reasoning:** Expected goal difference of {exp_diff:.1f} favors {team_names['home']} by enough margin to cover the handicap in the majority of simulated outcomes."
        elif "Away" in selection:
            return f"**Reasoning:** Expected goal difference of {abs(exp_diff):.1f} favors {team_names['away']} sufficiently to overcome the handicap advantage given to {team_names['home']}."
        else:
            return f"**Reasoning:** Minimal expected goal difference ({exp_diff:.1f}) suggests the match is too close to confidently predict a handicap cover for either side."
