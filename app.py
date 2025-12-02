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
from betting_advisor import BettingAdvisor

def main():
    st.set_page_config(page_title="Advanced xG Predictor", page_icon="📊", layout="wide")
    st.title("📊 Advanced xG Football Predictor")
    st.markdown("**Probabilistic Modeling • Dynamic Analysis • Evidence-Based Predictions**")
    
    predictor = AdvancedUnderstatPredictor()
    betting_advisor = BettingAdvisor()
    
    # Sidebar with league selection
    st.sidebar.header("📂 Load League Data")
    
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
        
        home_teams_data = st.session_state.league_data[
            st.session_state.league_data['Home_Away'] == 'Home'
        ]
        
        if home_teams_data.empty:
            st.error("No home team data available in the loaded file.")
            return
        
        home_teams = sorted(home_teams_data['Team'].unique())
        home_team = st.selectbox("Select Home Team:", home_teams, key="home_select")
        
        home_team_data = home_teams_data[
            home_teams_data['Team'] == home_team
        ].iloc[0]
        
        st.write(f"**Matches:** {int(home_team_data['Matches'])}")
        st.write(f"**Record:** {int(home_team_data['Wins'])}-{int(home_team_data['Draws'])}-{int(home_team_data['Losses'])}")
        st.write(f"**Goals:** {int(home_team_data['Goals'])}F, {int(home_team_data['Goals_Against'])}A")
        st.write(f"**xG:** {float(home_team_data['xG']):.2f}, **xGA:** {float(home_team_data['xGA']):.2f}")
        st.write(f"**Points:** {int(home_team_data['Points'])}, **xPTS:** {float(home_team_data['xPTS']):.2f}")
        
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
        
        away_teams_data = st.session_state.league_data[
            st.session_state.league_data['Home_Away'] == 'Away'
        ]
        
        if away_teams_data.empty:
            st.error("No away team data available in the loaded file.")
            return
        
        away_teams = sorted(away_teams_data['Team'].unique())
        away_team = st.selectbox("Select Away Team:", away_teams, key="away_select")
        
        away_team_data = away_teams_data[
            away_teams_data['Team'] == away_team
        ].iloc[0]
        
        st.write(f"**Matches:** {int(away_team_data['Matches'])}")
        st.write(f"**Record:** {int(away_team_data['Wins'])}-{int(away_team_data['Draws'])}-{int(away_team_data['Losses'])}")
        st.write(f"**Goals:** {int(away_team_data['Goals'])}F, {int(away_team_data['Goals_Against'])}A")
        st.write(f"**xG:** {float(away_team_data['xG']):.2f}, **xGA:** {float(away_team_data['xGA']):.2f}")
        st.write(f"**Points:** {int(away_team_data['Points'])}, **xPTS:** {float(away_team_data['xPTS']):.2f}")
        
        away_data = {
            "points": int(away_team_data['Points']),
            "goals_scored": int(away_team_data['Goals']),
            "goals_conceded": int(away_team_data['Goals_Against']),
            "xG": float(away_team_data['xG']),
            "xGA": float(away_team_data['xGA']),
            "xPTS": float(away_team_data['xPTS'])
        }
    
    games_played = min(int(home_team_data['Matches']), int(away_team_data['Matches']))
    
    # Two buttons - one for analysis, one for betting recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📈 Generate Advanced Analysis", type="primary", use_container_width=True):
            run_analysis(predictor, home_data, away_data, home_team, away_team, games_played)
    
    with col2:
        if st.button("🎯 Get Betting Recommendations", type="secondary", use_container_width=True):
            run_betting_advisor(predictor, betting_advisor, home_data, away_data, home_team, away_team, games_played)
    
    # Model methodology section
    display_methodology()

def run_analysis(predictor, home_data, away_data, home_team, away_team, games_played):
    """Run the model analysis and display results"""
    is_valid, error_msg = validate_input_data(home_data, away_data, games_played)
    if not is_valid:
        st.error(f"❌ Data validation failed: {error_msg}")
        st.info("Please check your input values and try again.")
        return
    
    with st.spinner("Running advanced probabilistic models..."):
        result = predictor.calculate_advanced_analysis(
            home_data, away_data, home_team, away_team, games_played
        )
    
    display_results(result, games_played)

def run_betting_advisor(predictor, betting_advisor, home_data, away_data, home_team, away_team, games_played):
    """Run model analysis THROUGH betting advisor for recommendations"""
    is_valid, error_msg = validate_input_data(home_data, away_data, games_played)
    if not is_valid:
        st.error(f"❌ Data validation failed: {error_msg}")
        st.info("Please check your input values and try again.")
        return
    
    with st.spinner("Analyzing for profitable betting opportunities..."):
        # First get model predictions
        result = predictor.calculate_advanced_analysis(
            home_data, away_data, home_team, away_team, games_played
        )
        
        # Then analyze with betting advisor
        recommendations = betting_advisor.analyze_predictions(result)
        
        # Display both model predictions AND advisor recommendations
        st.subheader("📊 Model Predictions (Raw)")
        display_results_compact(result, games_played)
        
        st.subheader("🎯 Betting Advisor Recommendations")
        
        advisor_display = betting_advisor.display_recommendations(recommendations)
        st.markdown(advisor_display)
        
        # Show advisor performance summary (FIXED VERSION)
        with st.expander("📈 Advisor Performance Summary"):
            summary = betting_advisor.get_performance_summary()
            if summary:  # Check if summary exists
                st.write(f"**Matches Analyzed:** {summary.get('total_matches_analyzed', 0)}")
                st.write(f"**Total Recommendations:** {summary.get('total_recommendations', 0)}")
                
                # Handle both old and new field names for strong bets
                strong_bets = summary.get('strong_recommendations', 
                                         summary.get('total_strong_bets', 
                                                   summary.get('matches_with_strong_recommendations', 0)))
                st.write(f"**Strong Bets:** {strong_bets}")
                
                st.write(f"**Avg Recommendations per Match:** {summary.get('avg_recommendations_per_match', 0):.1f}")
                
                if 'avg_strong_bets_per_match' in summary:
                    st.write(f"**Avg Strong Bets per Match:** {summary['avg_strong_bets_per_match']:.1f}")
            else:
                st.write("No performance data available yet.")

def display_results(result, games_played):
    """Display the analysis results in Streamlit"""
    st.subheader("📊 Advanced Analysis")
    
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
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Home xG", f"{result['analysis']['expected_goals']['home']:.2f}")
    with col2:
        st.metric("Away xG", f"{result['analysis']['expected_goals']['away']:.2f}")
    with col3:
        st.metric("Total xG", f"{result['analysis']['expected_goals']['total']:.2f}")
    with col4:
        st.metric("xG Difference", f"{result['analysis']['expected_goal_diff']:+.2f}")
    
    with st.expander("📋 Detailed Metrics", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Performance Factors**")
            st.write(f"Home Momentum: {result['analysis']['home_momentum']:.1f}")
            st.write(f"Away Momentum: {result['analysis']['away_momentum']:.1f}")
            st.write(f"Home Intangible Bonus: {result['analysis']['home_intangible']:.2f}")
            st.write(f"Away Intangible Bonus: {result['analysis']['away_intangible']:.2f}")
            
            if result['analysis']['volatility_note']:
                st.write(f"📊 **Volatility Note:** {result['analysis']['volatility_note']}")
        
        with col2:
            st.write("**Probability Analysis**")
            st.write(f"Over 2.5 Base: {result['analysis']['probabilities']['over_25']:.0f}%")
            st.write(f"BTTS Raw: {result['analysis']['probabilities']['btts_raw']:.0f}%")
            if result['analysis']['probabilities']['btts_adjust_note']:
                st.write(f"BTTS Adjustment: {result['analysis']['probabilities']['btts_adjust_note']}")
    
    st.subheader("💎 Model Predictions")
    
    for pred in result['predictions']:
        color, stake, description, reasoning = format_prediction_display(
            pred, result['analysis'], result['team_names'], result['raw_data']
        )
        
        with st.expander(f"{color} {pred['type']}: {pred['selection']} ({pred['confidence']:.0f}%) - {stake}"):
            st.write(description)
            st.write(reasoning)
            
            prob = pred['confidence'] / 100
            st.progress(prob, text=f"Confidence Level: {pred['confidence']:.0f}%")

def display_results_compact(result, games_played):
    """Display compact version of results for betting advisor view"""
    cols = st.columns(4)
    with cols[0]:
        st.metric("Home Quality", f"{result['analysis']['home_quality']:.1f}")
    with cols[1]:
        st.metric("Away Quality", f"{result['analysis']['away_quality']:.1f}")
    with cols[2]:
        st.metric("Total xG", f"{result['analysis']['expected_goals']['total']:.2f}")
    with cols[3]:
        st.metric("xG Diff", f"{result['analysis']['expected_goal_diff']:+.2f}")
    
    for pred in result['predictions']:
        color, stake, description, reasoning = format_prediction_display(
            pred, result['analysis'], result['team_names'], result['raw_data']
        )
        st.caption(f"{color} {pred['type']}: {pred['selection']} ({pred['confidence']:.0f}%) - {stake}")

def display_methodology():
    """Display model methodology section"""
    st.subheader("📖 Model Methodology")
    
    with st.expander("Learn how this model works", expanded=False):
        st.write("""
        **EMPIRICALLY CALIBRATED xG-Based Football Predictor**
        
        **Key Improvements (Based on 13-match analysis):**
        1. **Empirical Probability Calibration**: 85% predictions reduced to max 75%
        2. **Conservative Home Advantage**: 1-2.5% boost instead of 1-3%
        3. **Variance Buffers**: Asian Handicap predictions include match variance factors
        4. **Reduced Mismatch Penalty**: BTTS less aggressively adjusted for unbalanced matches
        5. **Realistic Quality Assessment**: Removed systematic home team bias
        
        **Using Home/Away Specific Data:**
        - Home team stats: Only home matches
        - Away team stats: Only away matches
        - More accurate than combined season stats
        
        **Core Methodology:**
        - Expected Goals (xG) as primary input
        - Poisson distribution for BTTS probabilities
        - Conservative home advantage (1-2.5% based on matchup)
        - Empirical calibration based on actual match results
        
        **Betting Advisor Module:**
        - Applies strategic filters to model predictions
        - Identifies proven profitable markets (Total Goals: 69% accuracy)
        - Fades overconfident predictions (>80% confidence → 40% actual)
        - Avoids weak markets (Asian Handicap: 31% accuracy)
        
        **Expected Performance (After Fixes):**
        - Match Winner: Target 58-63% accuracy
        - Total Goals: Maintain 65-70% accuracy  
        - BTTS: Target 50-55% accuracy
        - Asian Handicap: Target 45-50% accuracy
        """)
    
    st.info("""
    **⚡ Important Notes:**
    - This model uses separate home/away statistics for accuracy
    - All predictions are probabilistic assessments
    - Betting Advisor applies empirical filters based on historical performance
    - Updated calibration based on 13-match analysis
    """)

if __name__ == "__main__":
    main()