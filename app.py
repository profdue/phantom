"""
PHANTOM PREDICTOR v4.2 - Main Streamlit Application
Statistically Validated • Form-First Logic • Risk-Aware Staking
"""
import streamlit as st
import pandas as pd
from typing import Dict, Optional
from datetime import datetime  # Add this
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MatchPredictor, TeamProfile, ModelValidator
from utils import DataLoader, PredictionLogger
from betting_advisor import BettingAdvisor

# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

def setup_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="PHANTOM PREDICTOR v4.2",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #FF8C42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .strong-prediction {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    .moderate-prediction {
        border-left-color: #ffc107;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    }
    .light-prediction {
        border-left-color: #6c757d;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF4B4B, #FF8C42);
    }
    .data-warning {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔥 PHANTOM PREDICTOR v4.2</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Statistically Validated • Form-First Logic • xG Integration • Risk-Aware Staking</p>', unsafe_allow_html=True)

def display_welcome():
    """Display welcome screen when no league loaded"""
    st.info("👈 **Please load a league from the sidebar to get started!**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 **STATISTICAL RIGOR**")
        st.write("""
        • Real Poisson probabilities
        • xG/xGA integrated (70/30 blend)
        • Proper home/away distinction
        • League-average calibrated
        """)
    
    with col2:
        st.markdown("### 🎯 **FORM-FIRST LOGIC**")
        st.write("""
        • 70% weight to recent form
        • Dynamic reliability weighting
        • Conservative hot/cold adjustments
        • Sample-size awareness
        """)
    
    with col3:
        st.markdown("### ⚡ **RISK-AWARE**")
        st.write("""
        • Fractional Kelly staking (¼ Kelly)
        • Edge-based betting decisions
        • Bankroll management
        • Clear confidence bounds
        """)
    
    st.markdown("---")
    
    st.success("""
    **🚀 QUICK START GUIDE:**
    1. Select league from sidebar
    2. Click **"LOAD LEAGUE DATA"**
    3. Choose home and away teams
    4. Click **"GENERATE PREDICTION"**
    5. Get statistically validated predictions
    """)
    
    # Show available leagues if data loader exists
    if 'data_loader' in st.session_state:
        available = st.session_state.data_loader.available_leagues
        if available:
            st.markdown("### 📁 **AVAILABLE LEAGUES**")
            leagues_list = [l.replace('_', ' ').title() for l in available.keys()]
            st.write(", ".join(leagues_list))

def display_team_stats(data: Dict, is_home: bool = True):
    """Display team statistics in a clean format"""
    venue = "Home" if is_home else "Away"
    
    st.markdown(f"### {'🏠' if is_home else '✈️'} **{venue.upper()} STATS**")
    
    # Main metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matches Played", int(data['Matches']))
    with col2:
        record = f"{int(data['Wins'])}-{int(data['Draws'])}-{int(data['Losses'])}"
        st.metric(f"{venue} Record", record)
    with col3:
        st.metric("Points", int(data['Points']))
    
    # Goals section
    st.markdown("#### ⚽ GOALS")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Goals For", int(data['Goals']))
        avg_gf = int(data['Goals']) / max(1, int(data['Matches']))
        st.caption(f"{avg_gf:.2f} per game")
    with col2:
        st.metric("Goals Against", int(data['Goals_Against']))
        avg_ga = int(data['Goals_Against']) / max(1, int(data['Matches']))
        st.caption(f"{avg_ga:.2f} per game")
    
    # Advanced stats expander
    with st.expander(f"📈 **ADVANCED {venue.upper()} STATISTICS**"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Expected Goals (xG):** {float(data['xG']):.2f}")
            st.write(f"**xG per game:** {float(data['xG'])/max(1, int(data['Matches'])):.2f}")
        with col2:
            st.write(f"**Expected Goals Against (xGA):** {float(data['xGA']):.2f}")
            st.write(f"**xGA per game:** {float(data['xGA'])/max(1, int(data['Matches'])):.2f}")
        with col3:
            xpts = float(data.get('xPTS', 0))
            st.write(f"**Expected Points (xPTS):** {xpts:.2f}")
            st.write(f"**xPTS per game:** {xpts/max(1, int(data['Matches'])):.2f}")
        
        # Last 5 form
        if is_home:
            if 'Last5_Home_Wins' in data:
                wins = int(data['Last5_Home_Wins'])
                draws = int(data['Last5_Home_Draws'])
                losses = int(data['Last5_Home_Losses'])
                pts = int(data.get('Last5_Home_PTS', 0))
                gf = int(data.get('Last5_Home_GF', 0))
                ga = int(data.get('Last5_Home_GA', 0))
                
                st.write("**📊 LAST 5 HOME FORM:**")
                st.write(f"**W{wins} D{draws} L{losses}** ({pts}/15 pts)")
                st.write(f"**GF:** {gf} | **GA:** {ga} | **GD:** {gf-ga}")
        else:
            if 'Last5_Away_Wins' in data:
                wins = int(data['Last5_Away_Wins'])
                draws = int(data['Last5_Away_Draws'])
                losses = int(data['Last5_Away_Losses'])
                pts = int(data.get('Last5_Away_PTS', 0))
                gf = int(data.get('Last5_Away_GF', 0))
                ga = int(data.get('Last5_Away_GA', 0))
                
                st.write("**📊 LAST 5 AWAY FORM:**")
                st.write(f"**W{wins} D{draws} L{losses}** ({pts}/15 pts)")
                st.write(f"**GF:** {gf} | **GA:** {ga} | **GD:** {gf-ga}")

def display_prediction_results(result: Dict, betting_advisor: BettingAdvisor):
    """Display prediction results with analysis"""
    
    st.subheader("📊 **ANALYSIS RESULTS**")
    
    # Form scores
    st.markdown("#### 🎯 **FORM SCORES**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_form = result['analysis']['form_scores']['home']
        st.metric("Home Form Score", f"{home_form:.2f}")
        st.progress(home_form, text=f"{home_form*100:.0f}%")
        
    with col2:
        away_form = result['analysis']['form_scores']['away']
        st.metric("Away Form Score", f"{away_form:.2f}")
        st.progress(away_form, text=f"{away_form*100:.0f}%")
        
    with col3:
        form_diff = home_form - away_form
        st.metric("Form Advantage", f"{form_diff:+.2f}")
        if form_diff > 0.1:
            st.success("📈 Home form advantage")
        elif form_diff < -0.1:
            st.info("📉 Away form advantage")
        else:
            st.warning("⚖️ Even form")
    
    # Attack & Defense strengths
    st.markdown("#### ⚽ **ATTACK & DEFENSE STRENGTHS**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        home_attack = result['analysis']['attack_strengths']['home']
        st.metric("Home Attack", f"{home_attack:.2f}")
        st.caption("1.0 = league average")
        
    with col2:
        away_attack = result['analysis']['attack_strengths']['away']
        st.metric("Away Attack", f"{away_attack:.2f}")
        st.caption("1.0 = league average")
        
    with col3:
        home_defense = result['analysis']['defense_strengths']['home']
        st.metric("Home Defense", f"{home_defense:.2f}")
        st.caption("Higher = better defense")
        
    with col4:
        away_defense = result['analysis']['defense_strengths']['away']
        st.metric("Away Defense", f"{away_defense:.2f}")
        st.caption("Higher = better defense")
    
    # Expected goals
    st.markdown("#### 🎯 **EXPECTED GOALS**")
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
        from models import LEAGUE_CONFIGS
        league_key = result['analysis']['league'].lower().replace(" ", "_")
        league_avg = LEAGUE_CONFIGS.get(league_key, {}).get('avg_goals', 2.7)
        diff_vs_avg = total_xg - league_avg
        st.metric("vs League Avg", f"{diff_vs_avg:+.2f}")
    
    # Predictions
    st.markdown("---")
    st.subheader("🔥 **BOLD PREDICTIONS**")
    
    for pred in result['predictions']:
        pred_type = pred['type']
        selection = pred['selection']
        confidence = pred['confidence']
        
        # Get stake recommendation
        stake_info = betting_advisor.get_stake_recommendation(confidence, None, pred_type)
        
        # Determine card class
        if stake_info["color"] == "🟢":
            card_class = "strong-prediction"
            icon = "🔥"
        elif stake_info["color"] == "🟡":
            card_class = "moderate-prediction"
            icon = "⚡"
        elif stake_info["color"] == "🟠":
            card_class = "light-prediction"
            icon = "📊"
        else:
            card_class = "prediction-card"
            icon = "🚫"
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <h4>{icon} <strong>{pred_type}</strong></h4>
            <h3>{selection}</h3>
            <p><strong>Confidence:</strong> {confidence}%</p>
            <p><strong>Stake:</strong> {stake_info['units']} units | <strong>Risk:</strong> {stake_info['risk']} {stake_info['emoji']}</p>
            <p><small>{stake_info['reason']}</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence bar
        st.progress(confidence/100, text=f"Confidence Level: {confidence}%")
    
    # Generate betting advice
    advice = betting_advisor.generate_advice(result['predictions'])
    
    # Display advice sections
    st.markdown("---")
    st.subheader("💰 **BETTING RECOMMENDATIONS**")
    
    if advice['strong_plays']:
        st.success(f"### 🔥 STRONG PLAYS ({len(advice['strong_plays'])})")
        for play in advice['strong_plays']:
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            with col1:
                st.write(f"**{play['market']}:** {play['selection']}")
            with col2:
                st.write(f"Confidence: {play['confidence']}%")
            with col3:
                st.write(f"{play['stake']['color']} {play['stake']['units']}u")
            with col4:
                st.write(play['stake']['emoji'])
    
    if advice['moderate_plays']:
        st.info(f"### ⚡ MODERATE PLAYS ({len(advice['moderate_plays'])})")
        for play in advice['moderate_plays']:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{play['market']}:** {play['selection']}")
            with col2:
                st.write(f"Confidence: {play['confidence']}%")
            with col3:
                st.write(f"{play['stake']['color']} {play['stake']['units']}u")
    
    if advice['light_plays']:
        st.warning(f"### 📊 LIGHT PLAYS ({len(advice['light_plays'])})")
        for play in advice['light_plays']:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{play['market']}:** {play['selection']}")
            with col2:
                st.write(f"Confidence: {play['confidence']}%")
            with col3:
                st.write(f"{play['stake']['color']} {play['stake']['units']}u")
    
    # Summary
    st.markdown(f"#### 📋 **SUMMARY:** {advice['summary']}")
    
    # Expected scoreline
    st.markdown("---")
    st.subheader("📈 **EXPECTED SCORELINE**")
    
    home_xg = result['analysis']['expected_goals']['home']
    away_xg = result['analysis']['expected_goals']['away']
    
    # Convert xG to likely scoreline using Poisson
    # Simple rounding for display
    home_est = round(home_xg)
    away_est = round(away_xg)
    
    # Ensure minimum goals for reasonable xG
    if home_xg > 0.7 and home_est == 0:
        home_est = 1
    if away_xg > 0.7 and away_est == 0:
        away_est = 1
    
    # Display scoreline
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"<h1 style='text-align: center;'>{home_est} - {away_est}</h1>", 
                   unsafe_allow_html=True)
        st.caption(f"Based on xG: {home_xg:.2f} - {away_xg:.2f}")
    
    # Footer
    st.markdown("---")
    st.caption(f"⚡ PHANTOM v4.2 • League: {result['analysis']['league']} • Statistically Validated • xG Integration")

def display_methodology():
    """Display the v4.2 methodology"""
    st.subheader("📖 **PHANTOM v4.2 METHODOLOGY**")
    
    with st.expander("**View Complete Methodology**", expanded=True):
        st.markdown("""
        ### 🔬 **STATISTICAL FOUNDATION v4.2**
        
        **1. FORM-FIRST PREDICTION**
        ```
        Form Score = (Actual Last 5 Performance × 70%) + (Season Performance × 30%)
        
        Last 5 Score = Actual Points / 15 (no fake Last-3 data)
        Season Form = Total Points / (Matches × 3)
        ```
        
        **2. ATTACK & DEFENSE WITH xG INTEGRATION**
        ```
        Attack Strength = (Weighted Goals × 70% + Weighted xG × 30%) ÷ League Baseline
        
        Defense Strength = Opponent Baseline ÷ (Weighted GA × 70% + Weighted xGA × 30%)
        
        • Proper home/away baselines (home vs home avg, away vs away avg)
        • Dynamic reliability weighting (50-80% based on recent games)
        • Reasonable bounds: Attack 0.5-2.0, Defense 0.5-1.8
        ```
        
        **3. EXPECTED GOALS CALCULATION (NO DOUBLE ADVANTAGE)**
        ```
        Base xG = League Avg per Team × (Attack / Opponent Defense)
        Home xG = Base xG × Home Advantage (applied ONCE)
        
        • League average per team = ~1.47 (Premier League)
        • Home advantage = 18% (from config, applied once)
        • Conservative hot attack boost (max 5%)
        • Realistic caps: Home ≤ 4.5, Away ≤ 4.0
        ```
        
        **4. REAL POISSON PROBABILITIES**
        ```
        Calculates full Poisson convolution (0-6 goals):
        P(home_win) = Σ_i>j P(home=i) × P(away=j)
        P(draw) = Σ_i=j P(home=i) × P(away=j)
        P(away_win) = Σ_i<j P(home=i) × P(away=j)
        
        Fallback to proportional method if Poisson fails
        ```
        
        **5. PROPER BTTS CALCULATION**
        ```
        Uses Poisson estimation for games scored/conceded:
        P(scored in game) = 1 - e^(-goals_per_game)
        Expected both games = games × P(scored) × P(conceded)
        
        Tendency: 0.7 (low) to 1.3 (high) based on expected both games
        ```
        
        **6. CALIBRATION & RELIABILITY**
        ```
        • Calibration: 15% blend with league historical rates
        • Reliability: Less aggressive adjustment for sparse data
        • Sample-size aware: Blends with league averages, not uniform 33%
        ```
        
        ### 🎯 **KEY IMPROVEMENTS IN v4.2**
        
        **1. Fixed Critical Issues**
        • ✅ No double home advantage
        • ✅ BTTS uses Poisson estimation (not goals count)
        • ✅ Real Poisson probabilities (not proportional)
        • ✅ xG/xGA integrated in attack/defense
        
        **2. Statistical Rigor**
        • ✅ Proper home/away baselines
        • ✅ Realistic draw rates (25% at average xG)
        • ✅ Conservative boosts and caps
        • ✅ Dynamic reliability weighting
        
        **3. Practical Implementation**
        • ✅ Debug mode for development
        • ✅ Fallback methods for edge cases
        • ✅ Realistic value bounds
        • ✅ Transparent calculations
        ```
        """)

def display_league_stats(league_averages, league_name):
    """Display league statistics"""
    with st.expander("📊 **LEAGUE STATISTICS**", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Home Goals Avg", f"{league_averages.avg_home_goals:.2f}")
            st.caption("Goals per home game")
            
        with col2:
            st.metric("Away Goals Avg", f"{league_averages.avg_away_goals:.2f}")
            st.caption("Goals per away game")
            
        with col3:
            st.metric("League Avg per Team", f"{league_averages.league_avg_gpg:.2f}")
            st.caption("Goals per team per game")
            
        with col4:
            home_adv = league_averages.avg_home_goals / league_averages.avg_away_goals
            st.metric("Home Advantage", f"{home_adv:.2f}x")
            st.caption("Home goals / Away goals")
        
        # Outcome rates
        st.markdown("#### 📈 **HISTORICAL OUTCOME RATES**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Home Win Rate", f"{league_averages.actual_home_win_rate:.1%}")
        with col2:
            st.metric("Draw Rate", f"{league_averages.actual_draw_rate:.1%}")
        with col3:
            st.metric("Away Win Rate", f"{league_averages.actual_away_win_rate:.1%}")
        
        st.caption(f"Based on {league_averages.total_matches} matches analyzed")

def main():
    """Main Streamlit application"""
    setup_page()
    
    # Initialize session state
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
        print("✅ DataLoader initialized")
    
    if 'betting_advisor' not in st.session_state:
        st.session_state.betting_advisor = BettingAdvisor(bankroll=100.0, min_confidence=40.0)
        print("✅ BettingAdvisor initialized")
    
    if 'model_validator' not in st.session_state:
        st.session_state.model_validator = ModelValidator()
    
    if 'prediction_logger' not in st.session_state:
        st.session_state.prediction_logger = PredictionLogger()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ **CONFIGURATION**")
        
        # Debug mode toggle
        debug_mode = st.checkbox("🔧 Debug Mode", value=False, 
                                help="Show detailed calculation logs in terminal")
        
        # Bankroll setting
        bankroll = st.number_input(
            "Bankroll (units):",
            min_value=10.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            help="Total betting bankroll in units"
        )
        st.session_state.betting_advisor.update_bankroll(bankroll)
        
        # Minimum confidence threshold
        min_confidence = st.slider(
            "Minimum Confidence:",
            min_value=30,
            max_value=60,
            value=40,
            help="Minimum confidence percentage for betting"
        )
        st.session_state.betting_advisor.min_confidence = min_confidence
        
        st.markdown("---")
        
        # League selection
        available_leagues = st.session_state.data_loader.available_leagues
        if not available_leagues:
            st.error("❌ **No data files found in 'data' folder!**")
            st.info("""
            Please ensure CSV files are in the 'data' folder:
            
            data/
            ├── premier_league_home_away.csv
            ├── serie_a_home_away.csv
            ├── la_liga_home_away.csv
            ├── bundesliga_home_away.csv
            └── ligue_1_home_away.csv
            """)
            
            # Show what files were found
            if os.path.exists("data"):
                found_files = os.listdir("data")
                if found_files:
                    st.write("**Found files:**")
                    for f in found_files:
                        st.write(f"- {f}")
            
            return
        
        selected_league_key = st.selectbox(
            "Select League:",
            list(available_leagues.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
            help="Choose the league to analyze"
        )
        
        # Load league data button
        if st.button("📥 **LOAD LEAGUE DATA**", type="primary", use_container_width=True):
            with st.spinner(f"Loading {selected_league_key} data..."):
                try:
                    home_df, away_df, league_averages = st.session_state.data_loader.load_league_data(selected_league_key)
                    
                    # Store in session state
                    st.session_state.home_df = home_df
                    st.session_state.away_df = away_df
                    st.session_state.league_name = selected_league_key
                    st.session_state.league_averages = league_averages
                    st.session_state.league_loaded = True
                    
                    st.success(f"✅ **{selected_league_key.replace('_', ' ').title()} loaded successfully!**")
                    
                    # Display league statistics
                    display_league_stats(league_averages, selected_league_key)
                    
                except Exception as e:
                    st.error(f"❌ **Error loading data:** {str(e)}")
                    st.session_state.league_loaded = False
        
        st.markdown("---")
        
        # Show loaded league info
        if 'league_loaded' in st.session_state and st.session_state.league_loaded:
            st.success(f"**Current League:** {st.session_state.league_name.replace('_', ' ').title()}")
            
            # Show data stats
            home_teams = st.session_state.home_df['Team'].nunique()
            away_teams = st.session_state.away_df['Team'].nunique()
            st.info(f"📊 **{home_teams} home teams, {away_teams} away teams loaded**")
            
            # Validate data integrity
            if st.button("🔍 Validate Data", type="secondary", use_container_width=True):
                with st.spinner("Validating data integrity..."):
                    validation = st.session_state.data_loader.validate_data_integrity(
                        st.session_state.league_name
                    )
                    
                    if validation["status"] == "PASS":
                        st.success("✅ Data validation passed!")
                    elif validation["status"] == "WARNINGS":
                        st.warning("⚠️ Data validation warnings:")
                        for issue in validation["issues"]:
                            st.write(f"- {issue}")
                    else:
                        st.error(f"❌ Data validation error: {validation.get('error', 'Unknown error')}")
            
            # Show bankroll info
            risk_report = st.session_state.betting_advisor.get_risk_report()
            with st.expander("💰 **Risk Management**"):
                st.write(f"**Bankroll:** {risk_report['bankroll']:.2f} units")
                st.write(f"**Min Confidence:** {min_confidence}%")
                st.write(f"**Max Single Bet:** {risk_report['max_single_bet']:.2f} units")
                st.write(f"**Max Daily Exposure:** {risk_report['max_daily_exposure']:.2f} units")
                st.write(f"**Weekly Loss Limit:** {risk_report['weekly_loss_limit']:.2f} units")
        
        st.markdown("---")
        st.markdown("### 🎯 **v4.2 FEATURES**")
        st.info("""
        **Statistical Rigor:**
        • Real Poisson probabilities
        • xG/xGA integrated (70/30)
        • No double home advantage
        • Proper BTTS calculation
        
        **Risk-Aware:**
        • Fractional Kelly staking
        • Edge-based decisions
        • Bankroll management
        • Debug mode available
        """)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 **Reset Session**", type="secondary", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("📖 **Methodology**", type="secondary", use_container_width=True):
                st.session_state.show_methodology = True
    
    # Main content area
    if 'league_loaded' not in st.session_state or not st.session_state.league_loaded:
        display_welcome()
        return
    
    # Check for methodology display
    if 'show_methodology' in st.session_state and st.session_state.show_methodology:
        display_methodology()
        if st.button("← Back to Predictions", type="primary"):
            del st.session_state.show_methodology
            st.rerun()
        return
    
    # Team selection
    st.subheader("🎯 **SELECT MATCH**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 **HOME TEAM**")
        
        # Get home teams
        home_teams = sorted(st.session_state.home_df['Team'].unique())
        selected_home = st.selectbox(
            "Select Home Team:",
            home_teams,
            key="home_select",
            help="Choose the home team"
        )
        
        # Display home team stats
        if selected_home:
            home_data = st.session_state.home_df[
                st.session_state.home_df['Team'] == selected_home
            ].iloc[0].to_dict()
            display_team_stats(home_data, is_home=True)
    
    with col2:
        st.markdown("### ✈️ **AWAY TEAM**")
        
        # Get away teams
        away_teams = sorted(st.session_state.away_df['Team'].unique())
        selected_away = st.selectbox(
            "Select Away Team:",
            away_teams,
            key="away_select",
            help="Choose the away team"
        )
        
        # Display away team stats
        if selected_away:
            away_data = st.session_state.away_df[
                st.session_state.away_df['Team'] == selected_away
            ].iloc[0].to_dict()
            display_team_stats(away_data, is_home=False)
    
    # Generate prediction section
    if selected_home and selected_away:
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔥 **GENERATE STATISTICAL PREDICTION**", type="primary", use_container_width=True):
                with st.spinner("🔬 **Analyzing form and calculating probabilities...**"):
                    try:
                        # Create team profiles with debug mode
                        home_profile = TeamProfile(
                            data_dict=st.session_state.home_df[
                                st.session_state.home_df['Team'] == selected_home
                            ].iloc[0].to_dict(),
                            is_home=True,
                            league_averages=st.session_state.league_averages,
                            debug=debug_mode
                        )
                        
                        away_profile = TeamProfile(
                            data_dict=st.session_state.away_df[
                                st.session_state.away_df['Team'] == selected_away
                            ].iloc[0].to_dict(),
                            is_home=False,
                            league_averages=st.session_state.league_averages,
                            debug=debug_mode
                        )
                        
                        # Create predictor with debug mode
                        predictor = MatchPredictor(
                            league_name=st.session_state.league_name,
                            league_averages=st.session_state.league_averages,
                            debug=debug_mode
                        )
                        
                        # Generate prediction
                        result = predictor.predict(home_profile, away_profile)
                        
                        # Add team names
                        result['analysis']['home_team'] = selected_home
                        result['analysis']['away_team'] = selected_away
                        result['analysis']['league_stats'] = {
                            'home_goals_avg': st.session_state.league_averages.avg_home_goals,
                            'away_goals_avg': st.session_state.league_averages.avg_away_goals,
                            'league_avg_gpg': st.session_state.league_averages.league_avg_gpg
                        }
                        
                        # Store in session
                        st.session_state.last_prediction = result
                        
                        # Log prediction
                        log_data = {
                            "home_team": selected_home,
                            "away_team": selected_away,
                            "league": st.session_state.league_name,
                            "predictions": result['predictions'],
                            "analysis": {
                                "expected_goals": result['analysis']['expected_goals'],
                                "form_scores": result['analysis']['form_scores']
                            },
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.prediction_logger.log_prediction(log_data)
                        
                        # Display results
                        display_prediction_results(result, st.session_state.betting_advisor)
                        
                    except Exception as e:
                        st.error(f"❌ **Prediction error:** {str(e)}")
                        if debug_mode:
                            import traceback
                            st.code(traceback.format_exc(), language="python")
        
        with col2:
            if st.button("📊 **Methodology**", type="secondary", use_container_width=True):
                display_methodology()
        
        # Display last prediction if exists
        if 'last_prediction' in st.session_state:
            st.markdown("---")
            with st.expander("📋 **View Last Prediction Details**", expanded=False):
                st.json(st.session_state.last_prediction, expanded=False)
            
            # Data warning if unrealistic values
            total_xg = st.session_state.last_prediction['analysis']['expected_goals']['total']
            if total_xg > 5.0:
                st.markdown("""
                <div class="data-warning">
                ⚠️ **Note:** Predicted total xG ({:.1f}) is unusually high. 
                This may indicate data issues or extreme team form.
                </div>
                """.format(total_xg), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
