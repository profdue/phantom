import streamlit as st
import pandas as pd
import os
from models import AdvancedUnderstatPredictor
from utils import (
    validate_input_data,
    format_prediction_display,
    load_league_data,
    get_available_leagues
)

def main():
    st.set_page_config(page_title="Advanced xG Predictor", page_icon="📊", layout="wide")
    st.title("📊 Advanced xG Football Predictor")
    st.markdown("**Probabilistic Modeling • Dynamic Analysis • Evidence-Based Predictions**")
    
    predictor = AdvancedUnderstatPredictor()
    
    # Sidebar with league selection
    st.sidebar.header("📂 Load League Data")
    
    # Get available leagues from data folder
    leagues = get_available_leagues()
    
    if not leagues:
        st.sidebar.error("No CSV files found in data folder.")
        st.sidebar.info("Please ensure CSV files are in the 'data' folder.")
        display_methodology()
        return
    
    selected_league = st.sidebar.selectbox("Select League:", leagues)
    
    if st.sidebar.button("📥 Load League Data", type="primary"):
        with st.spinner(f"Loading {selected_league} data..."):
            league_data = load_league_data(selected_league)
            if league_data is not None:
                st.session_state.league_data = league_data
                st.session_state.selected_league = selected_league
                st.sidebar.success(f"✅ Loaded {selected_league}")
                st.rerun()
            else:
                st.sidebar.error(f"❌ Failed to load {selected_league} data")
    
    st.sidebar.markdown("---")
    
    # Display current loaded league
    if 'selected_league' in st.session_state:
        st.sidebar.info(f"**Current League:** {st.session_state.selected_league}")
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Using Home/Away Data:**
    - Teams perform differently at home vs away
    - More accurate predictions
    - Realistic scoring expectations
    """)
    
    # Check if data is loaded
    if 'league_data' not in st.session_state:
        st.info("👈 Please load league data from the sidebar to get started.")
        display_methodology()
        return
    
    # Main input section with team selection
    st.subheader("📊 Select Match")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 Home Team")
        
        # Filter home teams from data (Home_Away == 'Home')
        home_teams_data = st.session_state.league_data[
            st.session_state.league_data['Home_Away'] == 'Home'
        ]
        
        if home_teams_data.empty:
            st.error("No home team data available in the loaded file.")
            return
        
        home_teams = sorted(home_teams_data['Team'].unique())
        home_team = st.selectbox("Select Home Team:", home_teams, key="home_select")
        
        # Get home team data
        home_team_data = home_teams_data[
            home_teams_data['Team'] == home_team
        ].iloc[0]
        
        # Display home team stats
        st.write(f"**Matches:** {int(home_team_data['Matches'])}")
        st.write(f"**Record:** {int(home_team_data['Wins'])}-{int(home_team_data['Draws'])}-{int(home_team_data['Losses'])}")
        st.write(f"**Goals:** {int(home_team_data['Goals'])}F, {int(home_team_data['Goals_Against'])}A")
        st.write(f"**xG:** {float(home_team_data['xG']):.2f}, **xGA:** {float(home_team_data['xGA']):.2f}")
        st.write(f"**Points:** {int(home_team_data['Points'])}, **xPTS:** {float(home_team_data['xPTS']):.2f}")
        
        # Prepare home data for model
        home_data = {
            "points": int(home_team_data['Points']),
            "goals_scored": int(home_team_data['Goals']),
            "goals_conceded": int(home_team_data['Goals_Against']),
            "xG": float(home_team_data['xG']),
            "xGA": float(home_team_data['xGA']),
            "xPTS": float(home_team_data['xPTS'])
        }
    
    with col2:
        st.markdown("### ✈️ Away Team")
        
        # Filter away teams from data (Home_Away == 'Away')
        away_teams_data = st.session_state.league_data[
            st.session_state.league_data['Home_Away'] == 'Away'
        ]
        
        if away_teams_data.empty:
            st.error("No away team data available in the loaded file.")
            return
        
        away_teams = sorted(away_teams_data['Team'].unique())
        away_team = st.selectbox("Select Away Team:", away_teams, key="away_select")
        
        # Get away team data
        away_team_data = away_teams_data[
            away_teams_data['Team'] == away_team
        ].iloc[0]
        
        # Display away team stats
        st.write(f"**Matches:** {int(away_team_data['Matches'])}")
        st.write(f"**Record:** {int(away_team_data['Wins'])}-{int(away_team_data['Draws'])}-{int(away_team_data['Losses'])}")
        st.write(f"**Goals:** {int(away_team_data['Goals'])}F, {int(away_team_data['Goals_Against'])}A")
        st.write(f"**xG:** {float(away_team_data['xG']):.2f}, **xGA:** {float(away_team_data['xGA']):.2f}")
        st.write(f"**Points:** {int(away_team_data['Points'])}, **xPTS:** {float(away_team_data['xPTS']):.2f}")
        
        # Prepare away data for model
        away_data = {
            "points": int(away_team_data['Points']),
            "goals_scored": int(away_team_data['Goals']),
            "goals_conceded": int(away_team_data['Goals_Against']),
            "xG": float(away_team_data['xG']),
            "xGA": float(away_team_data['xGA']),
            "xPTS": float(away_team_data['xPTS'])
        }
    
    # Check if teams have same number of matches (for consistency)
    games_played = min(int(home_team_data['Matches']), int(away_team_data['Matches']))
    
    # Generate analysis button
    if st.button("📈 Generate Advanced Analysis", type="primary"):
        # Validate input data
        is_valid, error_msg = validate_input_data(home_data, away_data, games_played)
        if not is_valid:
            st.error(f"❌ Data validation failed: {error_msg}")
            st.info("Please check your input values and try again.")
            return
        
        # Calculate analysis
        with st.spinner("Running advanced probabilistic models..."):
            result = predictor.calculate_advanced_analysis(
                home_data, away_data, home_team, away_team, games_played
            )
        
        # Display analysis results
        display_results(result, games_played)
    
    # Model methodology section
    display_methodology()

def display_results(result, games_played):
    """Display the analysis results in Streamlit"""
    st.subheader("📊 Advanced Analysis")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta_home = result['analysis']['home_quality'] - 2.0
        st.metric("Home Quality", f"{result['analysis']['home_quality']:.1f}", 
                 f"{delta_home:+.1f}", delta_color="normal")
        if result['raw_data']['home_overperformance'] > 0.1:
            st.caption(f"📈 +{result['raw_data']['home_overperformance']:.1f} overperformance")
    
    with col2:
        delta_away = result['analysis']['away_quality'] - 2.0
        st.metric("Away Quality", f"{result['analysis']['away_quality']:.1f}",
                 f"{delta_away:+.1f}", delta_color="normal")
        if result['raw_data']['away_overperformance'] > 0.1:
            st.caption(f"📈 +{result['raw_data']['away_overperformance']:.1f} overperformance")
    
    with col3:
        st.metric("Quality Gap", f"{result['analysis']['quality_diff']:+.1f}")
    
    with col4:
        st.metric("Total Advantage", f"{result['analysis']['total_advantage']:+.1f}")
        st.caption(f"Home boost: +{((result['analysis']['home_boost']-1)*100):.0f}%")
        st.caption(f"Based on {games_played} matches")
    
    # Expected goals row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Home xG", f"{result['analysis']['expected_goals']['home']:.2f}")
    with col2:
        st.metric("Away xG", f"{result['analysis']['expected_goals']['away']:.2f}")
    with col3:
        st.metric("Total xG", f"{result['analysis']['expected_goals']['total']:.2f}")
    with col4:
        st.metric("xG Difference", f"{result['analysis']['expected_goal_diff']:+.2f}")
    
    # Detailed metrics expander
    with st.expander("📋 Detailed Metrics", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Performance Factors**")
            st.write(f"Home Momentum: {result['analysis']['home_momentum']:.1f}")
            st.write(f"Away Momentum: {result['analysis']['away_momentum']:.1f}")
            st.write(f"Home Intangible Bonus: {result['analysis']['home_intangible']:.2f}")
            st.write(f"Away Intangible Bonus: {result['analysis']['away_intangible']:.2f}")
            
            # Check if home_edge_note exists and is not empty
            if 'home_edge_note' in result['analysis'] and result['analysis']['home_edge_note']:
                st.write(f"📊 **Match Context:** {result['analysis']['home_edge_note']}")
            
            if result['analysis']['volatility_note']:
                st.write(f"📊 **Volatility Note:** {result['analysis']['volatility_note']}")
        
        with col2:
            st.write("**Probability Analysis**")
            st.write(f"Over 2.5 Base: {result['analysis']['probabilities']['over_25']:.0f}%")
            st.write(f"BTTS Raw: {result['analysis']['probabilities']['btts_raw']:.0f}%")
            if result['analysis']['probabilities']['btts_adjust_note']:
                st.write(f"BTTS Adjustment: {result['analysis']['probabilities']['btts_adjust_note']}")
    
    # Predictions section
    st.subheader("💎 Model Predictions")
    
    for pred in result['predictions']:
        # Get formatted display
        color, stake, description, reasoning = format_prediction_display(
            pred, result['analysis'], result['team_names'], result['raw_data']
        )
        
        # Display prediction
        with st.expander(f"{color} {pred['type']}: {pred['selection']} ({pred['confidence']:.0f}%) - {stake}"):
            st.write(description)
            st.write(reasoning)
            
            # Add probability bar
            prob = pred['confidence'] / 100
            st.progress(prob, text=f"Confidence Level: {pred['confidence']:.0f}%")

def display_methodology():
    """Display model methodology section"""
    st.subheader("📖 Model Methodology")
    
    with st.expander("Learn how this model works", expanded=False):
        st.write("""
        **Calibrated xG-Based Football Predictor**
        
        **Using Home/Away Specific Data:**
        - Home team stats: Only home matches
        - Away team stats: Only away matches
        - More accurate than combined season stats
        
        **Latest Calibration Updates:**
        1. **Reduced Overperformance Impact**: Intangible bonus reduced from 40% to 25%
        2. **Wider Total Goals Bands**: More decisive predictions (52%/48% thresholds)
        3. **Mismatch Adjustment**: BTTS probability reduced for unbalanced matches
        4. **Home Edge for Balanced Matches**: Slight advantage to home teams in even contests
        
        **Core Methodology:**
        - Expected Goals (xG) as primary input
        - Poisson distribution for BTTS probabilities
        - Conservative home advantage (1-3% based on matchup)
        - Quality-driven predictions (location matters little)
        - Statistical calibration on real match results
        
        **Supported Leagues:**
        - Premier League
        - La Liga
        - Serie A
        - Bundesliga
        - Ligue 1
        - RFPL (Russian Premier League)
        
        **Model Performance:**
        - Match Winner: 60-65% accuracy
        - Total Goals: 55-60% accuracy  
        - BTTS: 50-55% accuracy
        - All predictions are probabilistic, not certainties.
        """)
    
    # Performance disclaimer
    st.info("""
    **⚡ Important Notes:**
    - This model uses separate home/away statistics for accuracy
    - Teams perform differently at home vs away
    - All predictions are probabilistic assessments
    - Latest calibration based on real match results analysis
    """)

if __name__ == "__main__":
    main()
