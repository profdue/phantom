import streamlit as st
import pandas as pd
import os
import sys

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phantom.models import MatchPredictor, TeamProfile
from phantom.data_loader import DataLoader
from phantom.betting_advisor import BettingAdvisor
from phantom.utils import PredictionUtils

def main():
    st.set_page_config(
        page_title="PHANTOM Predictor v2.3",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ PHANTOM PREDICTION SYSTEM v2.3")
    st.markdown("**Intelligent Football Analytics • Evidence-Based Predictions • Risk-Managed Betting**")
    
    # Initialize components
    data_loader = DataLoader("data")
    predictor = None
    betting_advisor = BettingAdvisor()
    utils = PredictionUtils()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # League selection
        available_leagues = data_loader.available_leagues
        if not available_leagues:
            st.error("❌ No data files found in 'data' folder!")
            st.info("Please ensure CSV files are in phantom/data/")
            return
        
        selected_league_key = st.selectbox(
            "Select League:",
            list(available_leagues.keys()),
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        # Load league data
        if st.button("📥 Load League Data", type="primary"):
            with st.spinner(f"Loading {selected_league_key}..."):
                try:
                    home_df, away_df = data_loader.load_league_data(selected_league_key)
                    
                    # Store in session state
                    st.session_state.home_df = home_df
                    st.session_state.away_df = away_df
                    st.session_state.league_name = selected_league_key
                    st.session_state.league_loaded = True
                    
                    st.success(f"✅ Loaded {selected_league_key}")
                    
                    # Create predictor with loaded league
                    st.session_state.predictor = MatchPredictor(selected_league_key)
                    
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
        
        st.markdown("---")
        
        # Show loaded league info
        if 'league_loaded' in st.session_state and st.session_state.league_loaded:
            st.success(f"**Current League:** {st.session_state.league_name.replace('_', ' ').title()}")
            
            # Show data stats
            home_teams = st.session_state.home_df['Team'].nunique()
            away_teams = st.session_state.away_df['Team'].nunique()
            st.info(f"📊 {home_teams} home teams, {away_teams} away teams loaded")
        
        st.markdown("---")
        st.markdown("### 📚 About v2.3")
        st.info("""
        **Key Improvements:**
        • Attack crisis detection
        • Clean sheet probability  
        • Form momentum weighting
        • Conservative staking
        • 77.8% winner accuracy
        """)
    
    # Main content area
    if 'league_loaded' not in st.session_state or not st.session_state.league_loaded:
        st.info("👈 Please load a league from the sidebar to get started!")
        display_features()
        return
    
    # Team selection
    st.subheader("🎯 Select Match")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 Home Team")
        
        # Get home teams
        home_teams = sorted(st.session_state.home_df['Team'].unique())
        selected_home = st.selectbox(
            "Select Home Team:",
            home_teams,
            key="home_select"
        )
        
        # Display home team stats
        if selected_home:
            home_data = st.session_state.home_df[
                st.session_state.home_df['Team'] == selected_home
            ].iloc[0]
            
            display_team_stats(home_data, is_home=True)
    
    with col2:
        st.markdown("### ✈️ Away Team")
        
        # Get away teams
        away_teams = sorted(st.session_state.away_df['Team'].unique())
        selected_away = st.selectbox(
            "Select Away Team:",
            away_teams,
            key="away_select"
        )
        
        # Display away team stats
        if selected_away:
            away_data = st.session_state.away_df[
                st.session_state.away_df['Team'] == selected_away
            ].iloc[0]
            
            display_team_stats(away_data, is_home=False)
    
    # Generate prediction button
    if selected_home and selected_away:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
                generate_prediction(
                    selected_home, selected_away, 
                    st.session_state.predictor,
                    data_loader,
                    betting_advisor,
                    utils
                )
        
        with col2:
            if st.button("📈 View Methodology", type="secondary", use_container_width=True):
                display_methodology()

def display_team_stats(data, is_home=True):
    """Display team statistics in a clean format"""
    venue = "Home" if is_home else "Away"
    
    st.metric("Matches", int(data['Matches']))
    
    # Record
    record = f"{int(data['Wins'])}-{int(data['Draws'])}-{int(data['Losses'])}"
    st.metric(f"{venue} Record", record)
    
    # Goals
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Goals For", int(data['Goals']))
    with col2:
        st.metric("Goals Against", int(data['Goals_Against']))
    
    # Advanced metrics
    with st.expander(f"📊 Advanced Stats ({venue})"):
        st.write(f"**xG:** {float(data['xG']):.2f}")
        st.write(f"**xGA:** {float(data['xGA']):.2f}")
        st.write(f"**Points:** {int(data['Points'])}")
        st.write(f"**xPTS:** {float(data['xPTS']):.2f}")
        
        # Last 5 form
        if is_home:
            st.write("**Last 5 Home Form:**")
            st.write(f"W{int(data['Last5_Home_Wins'])} D{int(data['Last5_Home_Draws'])} L{int(data['Last5_Home_Losses'])}")
            st.write(f"GF: {int(data['Last5_Home_GF'])} GA: {int(data['Last5_Home_GA'])}")
            st.write(f"Pts: {int(data['Last5_Home_PTS'])}/15")
        else:
            st.write("**Last 5 Away Form:**")
            st.write(f"W{int(data['Last5_Away_Wins'])} D{int(data['Last5_Away_Draws'])} L{int(data['Last5_Away_Losses'])}")
            st.write(f"GF: {int(data['Last5_Away_GF'])} GA: {int(data['Last5_Away_GA'])}")
            st.write(f"Pts: {int(data['Last5_Away_PTS'])}/15")

def generate_prediction(home_team, away_team, predictor, data_loader, betting_advisor, utils):
    """Generate and display prediction"""
    
    with st.spinner("🔬 Running advanced analysis..."):
        try:
            # Create team profiles
            home_profile = data_loader.get_team_profile(
                home_team, 
                st.session_state.league_name, 
                is_home=True
            )
            
            away_profile = data_loader.get_team_profile(
                away_team, 
                st.session_state.league_name, 
                is_home=False
            )
            
            # Generate prediction
            result = predictor.predict(home_profile, away_profile)
            
            # Add team names
            result['analysis']['home_team'] = home_team
            result['analysis']['away_team'] = away_team
            
            # Generate betting advice
            advice = betting_advisor.generate_advice(result['predictions'])
            result['betting_advice'] = advice
            
            # Store in session
            st.session_state.last_prediction = result
            
            # Display results
            display_prediction_results(result, betting_advisor)
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

def display_prediction_results(result, betting_advisor):
    """Display prediction results"""
    
    st.subheader("📊 Analysis Results")
    
    # Quality metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        home_quality = result['analysis']['quality_ratings']['home']
        st.metric("Home Quality", f"{home_quality:.2f}")
        
    with col2:
        away_quality = result['analysis']['quality_ratings']['away']
        st.metric("Away Quality", f"{away_quality:.2f}")
        
    with col3:
        quality_diff = home_quality - away_quality
        st.metric("Quality Difference", f"{quality_diff:+.2f}")
        
    with col4:
        total_quality = home_quality + away_quality
        st.metric("Combined Quality", f"{total_quality:.2f}")
    
    # Expected goals
    st.markdown("### ⚽ Expected Goals")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        home_xg = result['analysis']['expected_goals']['home']
        st.metric("Home xG", f"{home_xg:.2f}")
        
    with col2:
        away_xg = result['analysis']['expected_goals']['away']
        st.metric("Away xG", f"{away_xg:.2f}")
        
    with col3:
        total_xg = result['analysis']['expected_goals']['total']
        st.metric("Total xG", f"{total_xg:.2f}")
        
    with col4:
        xg_diff = home_xg - away_xg
        st.metric("xG Difference", f"{xg_diff:+.2f}")
    
    # Team flags
    if result['analysis']['team_flags']:
        st.markdown("### ⚠️ Team Alerts")
        
        flags = result['analysis']['team_flags']
        alert_cols = st.columns(4)
        
        idx = 0
        for flag_name, flag_value in flags.items():
            if flag_value:
                team_type = "Home" if "home" in flag_name else "Away"
                flag_type = "Attack Crisis" if "attack" in flag_name else "Defense Crisis"
                
                with alert_cols[idx % 4]:
                    st.warning(f"**{team_type} {flag_type}**")
                idx += 1
    
    # Predictions
    st.markdown("### 🎯 Model Predictions")
    
    for pred in result['predictions']:
        pred_type = pred['type']
        selection = pred['selection']
        confidence = pred['confidence']
        
        # Get stake recommendation
        stake_info = betting_advisor.get_stake_recommendation(confidence)
        
        # Create expander for each prediction
        with st.expander(f"{stake_info['color']} **{pred_type}:** {selection} ({confidence}%)"):
            
            # Confidence bar
            st.progress(confidence/100, text=f"Confidence: {confidence}%")
            
            # Stake info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Stake", f"{stake_info['units']} units")
            with col2:
                st.metric("Risk Level", stake_info['risk'])
            with col3:
                st.metric("Color Code", stake_info['color'])
            
            # Additional context based on prediction type
            if pred_type == "Match Winner":
                if "Draw" in selection:
                    st.info("📊 Close match expected with minimal quality difference")
                elif "Home" in selection:
                    st.info("🏠 Home advantage and quality edge")
                else:
                    st.info("✈️ Away team superior form/quality")
            
            elif pred_type == "Total Goals":
                total_xg = result['analysis']['expected_goals']['total']
                league_avg = predictor.league_config['avg_goals']
                
                if total_xg > league_avg + 0.3:
                    st.info(f"⚡ High-scoring expected ({total_xg:.2f} vs league avg {league_avg})")
                elif total_xg < league_avg - 0.3:
                    st.info(f"🛡️ Defensive battle expected ({total_xg:.2f} vs league avg {league_avg})")
                else:
                    st.info(f"📈 Average scoring expected ({total_xg:.2f} vs league avg {league_avg})")
            
            elif pred_type == "BTTS":
                st.info(f"📊 BTTS probability calculated with clean sheet risk adjustment")
    
    # Betting advice
    st.markdown("### 💰 Betting Recommendations")
    
    advice = result['betting_advice']
    
    if advice['strong_plays']:
        st.success("### ✅ STRONG PLAYS")
        for play in advice['strong_plays']:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{play['market']}:** {play['selection']}")
            with col2:
                st.write(f"Confidence: {play['confidence']}%")
            with col3:
                st.write(f"{play['stake']['color']} {play['stake']['units']} units")
    
    if advice['moderate_plays']:
        st.info("### ⚖️ MODERATE PLAYS")
        for play in advice['moderate_plays']:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{play['market']}:** {play['selection']}")
            with col2:
                st.write(f"Confidence: {play['confidence']}%")
            with col3:
                st.write(f"{play['stake']['color']} {play['stake']['units']} units")
    
    if advice['avoid']:
        st.warning("### 🚫 AVOID THESE MARKETS")
        st.write(", ".join(advice['avoid']))
    
    # Expected scoreline
    st.markdown("### 📈 Expected Scoreline")
    
    home_xg = result['analysis']['expected_goals']['home']
    away_xg = result['analysis']['expected_goals']['away']
    
    # Simple score estimation
    home_est = round(home_xg)
    away_est = round(away_xg)
    
    # Adjust for minimum scoring
    if home_xg > 0.5 and home_est == 0:
        home_est = 1
    if away_xg > 0.5 and away_est == 0:
        away_est = 1
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"<h1 style='text-align: center;'>{home_est} - {away_est}</h1>", 
                   unsafe_allow_html=True)
        st.caption("Based on expected goals distribution")
    
    # Performance note
    st.markdown("---")
    st.caption(f"⚡ PHANTOM v2.3 • League: {predictor.league_config['name']} • Winner Accuracy: 77.8%")

def display_methodology():
    """Display the model methodology"""
    st.subheader("📖 PHANTOM v2.3 Methodology")
    
    with st.expander("View Complete Methodology", expanded=True):
        st.markdown("""
        ### 🔬 Core Architecture
        
        **1. League-Specific Calibration**
        Each league has unique parameters optimized through backtesting:
        - Goal scoring environments
        - Home advantage factors  
        - BTTS baselines
        - Win thresholds
        
        **2. Team Quality Calculation**
        ```
        Quality = (Attack × 70%) + (Defense × 30%) + Form Momentum
        Attack = min(2.5, xG_per_game × league_factor)
        Defense = max(0.4, 2.0 - goals_conceded_per_game)
        Form Momentum = (last_5_points / 15) × 0.3
        ```
        
        **3. Expected Goals Model**
        ```
        Home_xG = (Home_Attack + (2.0 - Away_Defense)) / 2 × Home_Advantage
        Away_xG = (Away_Attack + (2.0 - Home_Defense)) / 2 × Away_Penalty
        ```
        
        **4. Crisis Detection (NEW)**
        - **Attack Crisis:** xG < 1.0 AND last5_GF < 0.5 AND losing streak
        - **Defense Crisis:** GA > 1.8 per game
        - Triggers 40% reduction in expected goals
        
        **5. Clean Sheet Probability (NEW)**
        ```
        Clean_Sheet_Prob = e^(-opponent_xG) × 100%
        If > 35%: Reduce BTTS probability
        ```
        
        ### 🎯 Prediction Logic
        
        **Match Winner:**
        - Threshold-based on league (Premier League: 0.45, Serie A: 0.55)
        - Form momentum can override quality difference
        - 77.8% accuracy in testing
        
        **Total Goals:**
        - 60/40 blend of current form vs league average
        - Big match penalty (-15% for top-team clashes)
        - Script detection for 2-0/2-1 patterns
        
        **Both Teams To Score:**
        - Poisson probability foundation
        - Clean sheet risk adjustment
        - Attack crisis penalty (-30%)
        
        ### 💰 Stake Management
        
        **Conservative v2.3 Staking:**
        ```
        ≥ 75% → 1.0 units 🟢
        65-74% → 0.75 units 🟢  
        55-64% → 0.5 units 🟡
        45-54% → 0.25 units 🟠
        < 45% → AVOID ⚪
        ```
        
        ### 📊 Performance Tracking (9 Matches)
        
        | Market | Accuracy | ROI |
        |--------|----------|-----|
        | Match Winner | 77.8% | +55.6% |
        | Total Goals | 37.5% | -28.8% |
        | BTTS | 55.6% | +2.9% |
        
        **Key Insights:**
        1. Winner prediction is strongest
        2. Totals need conservative approach  
        3. BTTS profitable with proper staking
        4. Attack crisis detection improves accuracy
        """)

def display_features():
    """Display system features"""
    st.subheader("🌟 PHANTOM v2.3 Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Data Intelligence")
        st.write("""
        • Home/Away specific stats
        • Last 5 form momentum
        • xG advanced metrics
        • League-specific calibration
        """)
    
    with col2:
        st.markdown("### 🧠 Model Innovation")
        st.write("""
        • Attack/defense crisis detection
        • Clean sheet probability
        • Big match penalty
        • Script pattern recognition
        • Form momentum weighting
        """)
    
    with col3:
        st.markdown("### 💰 Risk Management")
        st.write("""
        • Conservative staking
        • Confidence-based units
        • Market filtering
        • Performance tracking
        • Profitability focus
        """)
    
    st.markdown("---")
    
    st.info("""
    **⚡ Quick Start:**
    1. Select league from sidebar
    2. Click "Load League Data"
    3. Choose home and away teams
    4. Click "Generate Prediction"
    5. View analysis and betting recommendations
    """)

if __name__ == "__main__":
    main()
