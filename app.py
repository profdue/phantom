import streamlit as st
import pandas as pd
import os
import sys
import math

# ============================================================================
# LEAGUE CONFIGURATIONS - AGGRESSIVE SETTINGS
# ============================================================================

LEAGUE_CONFIGS = {
    "premier_league": {
        "name": "Premier League",
        "avg_goals": 2.93,
        "over_threshold": 2.75,
        "under_threshold": 2.55,
        "home_advantage": 0.25,
        "btts_baseline": 52,
        "win_threshold": 0.25,
        "form_weight": 0.4
    },
    "serie_a": {
        "name": "Serie A",
        "avg_goals": 2.56,
        "over_threshold": 2.40,
        "under_threshold": 2.20,
        "home_advantage": 0.20,
        "btts_baseline": 48,
        "win_threshold": 0.30,
        "form_weight": 0.35
    },
    "la_liga": {
        "name": "La Liga",
        "avg_goals": 2.62,
        "over_threshold": 2.45,
        "under_threshold": 2.25,
        "home_advantage": 0.22,
        "btts_baseline": 50,
        "win_threshold": 0.28,
        "form_weight": 0.38
    },
    "bundesliga": {
        "name": "Bundesliga",
        "avg_goals": 3.14,
        "over_threshold": 2.90,
        "under_threshold": 2.70,
        "home_advantage": 0.30,
        "btts_baseline": 55,
        "win_threshold": 0.22,
        "form_weight": 0.42
    },
    "ligue_1": {
        "name": "Ligue 1",
        "avg_goals": 2.78,
        "over_threshold": 2.60,
        "under_threshold": 2.40,
        "home_advantage": 0.23,
        "btts_baseline": 50,
        "win_threshold": 0.26,
        "form_weight": 0.36
    }
}

# ============================================================================
# TEAM PROFILE - FOCUS ON WHAT MATTERS
# ============================================================================

class TeamProfile:
    def __init__(self, data_dict, is_home=True):
        self.name = data_dict['Team']
        self.is_home = is_home
        
        # Only essential stats
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
        
        # Last 5 form (THE MOST IMPORTANT)
        if is_home:
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
        
        # Calculate key metrics
        self.form_score = self._calculate_form_score()
        self.attack_strength = self._calculate_attack_strength()
        self.defense_strength = self._calculate_defense_strength()
        self.btts_tendency = self._calculate_btts_tendency()
        
    def _calculate_form_score(self):
        """Form score 0-1, heavily weighted to recent games"""
        if self.last5_pts == 0:
            return 0.3  # Default for no data
        
        # Last 3 games estimated (more weight)
        last3_est = (self.last5_pts / 5) * 1.3  # Assume recent games were better/worse
        last3_est = min(1.0, last3_est)
        
        # Last 5 games
        last5_score = self.last5_pts / 15
        
        # Weight: 70% last 3 games, 30% last 5 games
        return (last3_est * 0.7) + (last5_score * 0.3)
    
    def _calculate_attack_strength(self):
        """Attack strength 0-2 scale"""
        # Recent goals matter MORE than xG
        recent_gpg = self.last5_gf / 5 if self.last5_gf > 0 else 0.5
        season_gpg = self.goals_pg
        
        # Weight: 60% recent form, 40% season average
        gpg = (recent_gpg * 0.6) + (season_gpg * 0.4)
        
        # Cap and scale
        return min(2.0, gpg * 1.2)
    
    def _calculate_defense_strength(self):
        """Defense strength 0-2 scale (higher = better defense)"""
        # Recent goals against matter MORE
        recent_gapg = self.last5_ga / 5 if self.last5_ga > 0 else 1.0
        season_gapg = self.goals_against_pg
        
        # Weight: 70% recent form, 30% season average
        gapg = (recent_gapg * 0.7) + (season_gapg * 0.3)
        
        # Convert to defense strength (lower GA = higher strength)
        return max(0.5, 2.0 - gapg)
    
    def _calculate_btts_tendency(self):
        """1.0 = neutral, >1.0 favors BTTS, <1.0 against BTTS"""
        # Both scoring AND conceding recently = high BTTS tendency
        if self.last5_gf > 0 and self.last5_ga > 0:
            # Calculate ratio of games with both GF and GA
            # Assuming at least 2 games with both = high tendency
            if self.last5_gf >= 3 and self.last5_ga >= 3:
                return 1.3  # Strong BTTS tendency
            else:
                return 1.1  # Moderate BTTS tendency
        elif self.last5_gf == 0 or self.last5_ga == 0:
            return 0.8  # Low BTTS tendency
        return 1.0

# ============================================================================
# MATCH PREDICTOR - BOLD PREDICTIONS
# ============================================================================

class MatchPredictor:
    def __init__(self, league_name):
        self.league_config = LEAGUE_CONFIGS.get(league_name.lower())
        if not self.league_config:
            raise ValueError(f"Unknown league: {league_name}")
    
    def predict(self, home_team: TeamProfile, away_team: TeamProfile):
        """Make BOLD predictions - No Avoids allowed"""
        
        # 1. WINNER PREDICTION (ALWAYS pick winner, no Draw unless very close)
        winner_pred = self._predict_winner(home_team, away_team)
        
        # 2. EXPECTED GOALS
        home_xg, away_xg = self._calculate_expected_goals(home_team, away_team)
        total_xg = home_xg + away_xg
        
        # 3. TOTAL GOALS (ALWAYS Over or Under, no Avoid)
        total_pred = self._predict_total_goals(total_xg, home_team, away_team)
        
        # 4. BTTS (ALWAYS Yes or No, no Avoid)
        btts_pred = self._predict_btts(home_xg, away_xg, home_team, away_team)
        
        return {
            "analysis": {
                "league": self.league_config['name'],
                "form_scores": {
                    "home": round(home_team.form_score, 2),
                    "away": round(away_team.form_score, 2)
                },
                "expected_goals": {
                    "home": round(home_xg, 2),
                    "away": round(away_xg, 2),
                    "total": round(total_xg, 2)
                },
                "attack_strengths": {
                    "home": round(home_team.attack_strength, 2),
                    "away": round(away_team.attack_strength, 2)
                }
            },
            "predictions": [winner_pred, total_pred, btts_pred]
        }
    
    def _predict_winner(self, home: TeamProfile, away: TeamProfile):
        """ALWAYS predict a winner (Draw only if extremely close)"""
        
        # FORM DIFFERENCE (most important)
        form_diff = home.form_score - away.form_score
        
        # ATTACK/DEFENSE DIFFERENCE
        attack_diff = home.attack_strength - away.attack_strength
        defense_diff = home.defense_strength - away.defense_strength
        
        # COMBINED ADVANTAGE
        total_advantage = (form_diff * 0.5) + (attack_diff * 0.3) + (defense_diff * 0.2)
        
        # Apply home advantage
        total_advantage += self.league_config['home_advantage'] * 0.3
        
        # DECISION (very low threshold for draws)
        win_threshold = self.league_config['win_threshold']
        
        if total_advantage > win_threshold:
            selection = "Home Win"
            confidence = 55 + (total_advantage * 25)
            
        elif total_advantage < -win_threshold:
            selection = "Away Win"
            confidence = 55 + (abs(total_advantage) * 25)
            
        else:  # Very close match
            selection = "Draw"
            # Draw confidence based on how close
            closeness = 1.0 - (abs(total_advantage) / win_threshold)
            confidence = 50 + (closeness * 15)
        
        # CONFIDENCE ADJUSTMENTS
        # Recent goal scoring boosts confidence
        if home.last5_gf / 5 > 1.5 and "Home" in selection:
            confidence += 5
        if away.last5_gf / 5 > 1.5 and "Away" in selection:
            confidence += 5
            
        # Cap confidence
        confidence = max(45, min(80, confidence))
        
        return {
            "type": "Match Winner",
            "selection": selection,
            "confidence": round(confidence, 1)
        }
    
    def _calculate_expected_goals(self, home: TeamProfile, away: TeamProfile):
        """Calculate expected goals - SIMPLE and EFFECTIVE"""
        
        # Home xG = Home attack × (2.0 - Away defense) × Home advantage
        home_raw = home.attack_strength * (2.0 - away.defense_strength)
        home_xg = home_raw * (1.0 + self.league_config['home_advantage'])
        
        # Away xG = Away attack × (2.0 - Home defense) × Away factor
        away_raw = away.attack_strength * (2.0 - home.defense_strength)
        away_xg = away_raw * 0.9  # Standard away penalty
        
        # Recent form adjustments
        if home.last5_gf / 5 > 1.5:
            home_xg *= 1.1  # Hot attack
        if away.last5_gf / 5 > 1.5:
            away_xg *= 1.1
            
        return home_xg, away_xg
    
    def _predict_total_goals(self, total_xg: float, home: TeamProfile, away: TeamProfile):
        """ALWAYS predict Over or Under"""
        
        # League context
        league_avg = self.league_config['avg_goals']
        over_thresh = self.league_config['over_threshold']
        under_thresh = self.league_config['under_threshold']
        
        # Recent scoring trend
        home_recent_gpg = home.last5_gf / 5
        away_recent_gpg = away.last5_gf / 5
        recent_scoring = (home_recent_gpg + away_recent_gpg) / 2
        
        # Adjusted total (60% xG, 40% recent form)
        adjusted_total = (total_xg * 0.6) + (recent_scoring * 2.0 * 0.4)
        
        # DECISION - NO AVOID
        if adjusted_total > over_thresh:
            selection = "Over 2.5 Goals"
            # Confidence based on how far above threshold
            excess = (adjusted_total - over_thresh) / over_thresh
            confidence = 55 + (excess * 25)
            
        else:  # MUST be Under
            selection = "Under 2.5 Goals"
            # Confidence based on how far below average
            deficit = (league_avg - adjusted_total) / league_avg
            confidence = 55 + (deficit * 25)
        
        # Cap confidence
        confidence = max(50, min(80, confidence))
        
        return {
            "type": "Total Goals",
            "selection": selection,
            "confidence": round(confidence, 1)
        }
    
    def _predict_btts(self, home_xg: float, away_xg: float, 
                     home: TeamProfile, away: TeamProfile):
        """ALWAYS predict Yes or No"""
        
        # Base probability from xG
        home_score_prob = 1 - math.exp(-home_xg)
        away_score_prob = 1 - math.exp(-away_xg)
        btts_prob = home_score_prob * away_score_prob * 100
        
        # Apply team tendencies
        btts_prob *= ((home.btts_tendency + away.btts_tendency) / 2)
        
        # Recent BTTS trend
        home_btss_games = min(3, home.last5_gf)  # Estimate games they scored
        away_btss_games = min(3, away.last5_gf)
        recent_btss_factor = (home_btss_games + away_btss_games) / 6  # 0-1 scale
        btts_prob *= (0.8 + (recent_btss_factor * 0.4))  # 0.8-1.2 adjustment
        
        # DECISION - NO AVOID
        baseline = self.league_config['btts_baseline']
        
        if btts_prob >= baseline:
            selection = "Yes"
            confidence = min(80, btts_prob)
        else:
            selection = "No"
            confidence = min(80, 100 - btts_prob)
        
        # Minimum confidence
        confidence = max(50, confidence)
        
        return {
            "type": "BTTS",
            "selection": selection,
            "confidence": round(confidence, 1)
        }

# ============================================================================
# BETTING ADVISOR - AGGRESSIVE STAKING
# ============================================================================

class BettingAdvisor:
    """Provides betting recommendations for BOLD predictions"""
    
    @staticmethod
    def get_stake_recommendation(confidence):
        """Determine stake size - AGGRESSIVE for confident predictions"""
        if confidence >= 70:
            return {"units": 1.5, "color": "🟢", "risk": "High", "emoji": "🔥"}
        elif 65 <= confidence < 70:
            return {"units": 1.0, "color": "🟢", "risk": "Medium", "emoji": "⚡"}
        elif 60 <= confidence < 65:
            return {"units": 0.75, "color": "🟡", "risk": "Medium-Low", "emoji": "📈"}
        elif 55 <= confidence < 60:
            return {"units": 0.5, "color": "🟡", "risk": "Low", "emoji": "📊"}
        elif 50 <= confidence < 55:
            return {"units": 0.25, "color": "🟠", "risk": "Very Low", "emoji": "📉"}
        else:
            return {"units": 0, "color": "⚪", "risk": "AVOID", "emoji": "🚫"}
    
    @staticmethod
    def generate_advice(predictions):
        """Generate betting advice based on predictions"""
        advice = {
            "strong_plays": [],
            "moderate_plays": [],
            "light_plays": [],
            "summary": ""
        }
        
        for pred in predictions:
            stake_info = BettingAdvisor.get_stake_recommendation(pred['confidence'])
            
            if stake_info['units'] >= 1.0:
                advice['strong_plays'].append({
                    "market": pred['type'],
                    "selection": pred['selection'],
                    "confidence": pred['confidence'],
                    "stake": stake_info
                })
            elif stake_info['units'] >= 0.5:
                advice['moderate_plays'].append({
                    "market": pred['type'],
                    "selection": pred['selection'],
                    "confidence": pred['confidence'],
                    "stake": stake_info
                })
            elif stake_info['units'] > 0:
                advice['light_plays'].append({
                    "market": pred['type'],
                    "selection": pred['selection'],
                    "confidence": pred['confidence'],
                    "stake": stake_info
                })
        
        # Generate summary
        if advice['strong_plays']:
            advice['summary'] = f"🔥 {len(advice['strong_plays'])} STRONG betting opportunities"
        elif advice['moderate_plays']:
            advice['summary'] = f"⚡ {len(advice['moderate_plays'])} solid betting opportunities"
        elif advice['light_plays']:
            advice['summary'] = f"📊 {len(advice['light_plays'])} light betting opportunities"
        else:
            advice['summary'] = "🚫 No betting opportunities identified"
        
        return advice

# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def get_available_leagues():
    """Get list of available league CSV files"""
    data_dir = "data"
    leagues = {}
    
    if not os.path.exists(data_dir):
        return leagues
    
    for file in os.listdir(data_dir):
        if file.endswith("_home_away.csv"):
            league_name = file.replace("_home_away.csv", "")
            leagues[league_name] = os.path.join(data_dir, file)
    
    return leagues

def load_league_data(league_name):
    """Load CSV data for a specific league"""
    leagues = get_available_leagues()
    
    if league_name not in leagues:
        raise ValueError(f"League {league_name} not found. Available: {list(leagues.keys())}")
    
    file_path = leagues[league_name]
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]
    
    home_teams = df[df['Home_Away'] == 'Home']
    away_teams = df[df['Home_Away'] == 'Away']
    
    return home_teams, away_teams

# ============================================================================
# STREAMLIT APP - MAIN INTERFACE
# ============================================================================

def main():
    st.set_page_config(
        page_title="PHANTOM PREDICTOR v4.0",
        page_icon="🔥",
        layout="wide"
    )
    
    # Custom CSS for bold styling
    st.markdown("""
    <style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold !important;
    }
    .prediction-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .strong-prediction {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .moderate-prediction {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .light-prediction {
        background-color: #f8f9fa;
        border-left-color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔥 PHANTOM PREDICTOR v4.0")
    st.markdown("**BOLD PREDICTIONS • FORM-FIRST LOGIC • NO MORE AVOIDS**")
    
    # Initialize betting advisor
    betting_advisor = BettingAdvisor()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURATION")
        
        # League selection
        available_leagues = get_available_leagues()
        if not available_leagues:
            st.error("❌ No data files found in 'data' folder!")
            st.info("Please ensure CSV files are in the 'data' folder:")
            st.code("""
            data/
            ├── premier_league_home_away.csv
            ├── serie_a_home_away.csv
            ├── la_liga_home_away.csv
            ├── bundesliga_home_away.csv
            └── ligue_1_home_away.csv
            """)
            return
        
        selected_league_key = st.selectbox(
            "Select League:",
            list(available_leagues.keys()),
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        # Load league data
        if st.button("📥 LOAD LEAGUE DATA", type="primary", use_container_width=True):
            with st.spinner(f"Loading {selected_league_key}..."):
                try:
                    home_df, away_df = load_league_data(selected_league_key)
                    
                    if home_df is not None and away_df is not None:
                        # Store in session state
                        st.session_state.home_df = home_df
                        st.session_state.away_df = away_df
                        st.session_state.league_name = selected_league_key
                        st.session_state.league_loaded = True
                        
                        st.success(f"✅ Loaded {selected_league_key}")
                        
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
        st.markdown("### 🎯 v4.0 FEATURES")
        st.info("""
        **BOLD Improvements:**
        • FORM-FIRST prediction logic
        • NO MORE "Avoid" predictions
        • AGGRESSIVE staking
        • Recent form weighted 70%
        • Always clear calls
        """)
        
        st.markdown("---")
        if st.button("🔄 Reset Session", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content area
    if 'league_loaded' not in st.session_state or not st.session_state.league_loaded:
        display_welcome()
        return
    
    # Team selection
    st.subheader("🎯 SELECT MATCH")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 HOME TEAM")
        
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
        st.markdown("### ✈️ AWAY TEAM")
        
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
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔥 GENERATE BOLD PREDICTION", type="primary", use_container_width=True):
                generate_prediction(
                    selected_home, selected_away, 
                    st.session_state.league_name,
                    betting_advisor
                )
        
        with col2:
            if st.button("📖 Methodology", type="secondary", use_container_width=True):
                display_methodology()

def display_team_stats(data, is_home=True):
    """Display team statistics in a clean format"""
    venue = "Home" if is_home else "Away"
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matches", int(data['Matches']))
    with col2:
        st.metric(f"{venue} Record", f"{int(data['Wins'])}-{int(data['Draws'])}-{int(data['Losses'])}")
    with col3:
        st.metric("Points", int(data['Points']))
    
    # Goals
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Goals For", int(data['Goals']))
        st.caption(f"{int(data['Goals']) / int(data['Matches']):.2f} per game")
    with col2:
        st.metric("Goals Against", int(data['Goals_Against']))
        st.caption(f"{int(data['Goals_Against']) / int(data['Matches']):.2f} per game")
    
    # Advanced metrics expander
    with st.expander(f"📊 ADVANCED STATS ({venue})"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**xG:** {float(data['xG']):.2f}")
            st.write(f"**xGA:** {float(data['xGA']):.2f}")
        with col2:
            st.write(f"**xPTS:** {float(data.get('xPTS', 0)):.2f}")
        
        # Last 5 form
        if is_home:
            if 'Last5_Home_Wins' in data:
                st.write("**LAST 5 HOME FORM:**")
                form_str = f"W{int(data['Last5_Home_Wins'])} D{int(data['Last5_Home_Draws'])} L{int(data['Last5_Home_Losses'])}"
                pts = int(data.get('Last5_Home_PTS', 0))
                st.write(f"**{form_str}** ({pts}/15 pts)")
                st.write(f"GF: {int(data.get('Last5_Home_GF', 0))} | GA: {int(data.get('Last5_Home_GA', 0))}")
        else:
            if 'Last5_Away_Wins' in data:
                st.write("**LAST 5 AWAY FORM:**")
                form_str = f"W{int(data['Last5_Away_Wins'])} D{int(data['Last5_Away_Draws'])} L{int(data['Last5_Away_Losses'])}"
                pts = int(data.get('Last5_Away_PTS', 0))
                st.write(f"**{form_str}** ({pts}/15 pts)")
                st.write(f"GF: {int(data.get('Last5_Away_GF', 0))} | GA: {int(data.get('Last5_Away_GA', 0))}")

def generate_prediction(home_team, away_team, league_name, betting_advisor):
    """Generate and display BOLD prediction"""
    
    with st.spinner("🔥 ANALYZING FORM & GENERATING BOLD PREDICTION..."):
        try:
            # Create team profiles
            home_data = st.session_state.home_df[
                st.session_state.home_df['Team'] == home_team
            ].iloc[0].to_dict()
            
            away_data = st.session_state.away_df[
                st.session_state.away_df['Team'] == away_team
            ].iloc[0].to_dict()
            
            home_profile = TeamProfile(home_data, is_home=True)
            away_profile = TeamProfile(away_data, is_home=False)
            
            # Create predictor
            predictor = MatchPredictor(league_name)
            
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
            import traceback
            st.error(f"Debug: {traceback.format_exc()}")

def display_prediction_results(result, betting_advisor):
    """Display BOLD prediction results"""
    
    st.subheader("📊 ANALYSIS RESULTS")
    
    # Form scores (NEW - most important)
    st.markdown("#### 🎯 FORM SCORES (MOST IMPORTANT)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_form = result['analysis']['form_scores']['home']
        st.metric("Home Form", f"{home_form:.2f}")
        st.progress(home_form, text=f"{home_form*100:.0f}%")
        
    with col2:
        away_form = result['analysis']['form_scores']['away']
        st.metric("Away Form", f"{away_form:.2f}")
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
    
    # Attack strengths
    st.markdown("#### ⚽ ATTACK STRENGTHS")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_attack = result['analysis']['attack_strengths']['home']
        st.metric("Home Attack", f"{home_attack:.2f}")
        
    with col2:
        away_attack = result['analysis']['attack_strengths']['away']
        st.metric("Away Attack", f"{away_attack:.2f}")
        
    with col3:
        attack_diff = home_attack - away_attack
        st.metric("Attack Advantage", f"{attack_diff:+.2f}")
    
    # Expected goals
    st.markdown("#### 🎯 EXPECTED GOALS")
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
        league_avg = LEAGUE_CONFIGS.get(result['analysis']['league'].lower().replace(" ", "_"), {}).get('avg_goals', 2.7)
        diff_vs_avg = total_xg - league_avg
        st.metric("vs League Avg", f"{diff_vs_avg:+.2f}")
    
    # BOLD PREDICTIONS
    st.markdown("---")
    st.subheader("🔥 BOLD PREDICTIONS")
    
    for pred in result['predictions']:
        pred_type = pred['type']
        selection = pred['selection']
        confidence = pred['confidence']
        
        # Get stake recommendation
        stake_info = betting_advisor.get_stake_recommendation(confidence)
        
        # Color code based on stake
        if stake_info['units'] >= 1.0:
            box_class = "strong-prediction"
            icon = "🔥"
        elif stake_info['units'] >= 0.5:
            box_class = "moderate-prediction"
            icon = "⚡"
        else:
            box_class = "light-prediction"
            icon = "📊"
        
        # Display prediction box
        st.markdown(f"""
        <div class="prediction-box {box_class}">
            <h4>{icon} {pred_type}: <strong>{selection}</strong> ({confidence}%)</h4>
            <p>Stake: <strong>{stake_info['units']} units</strong> | Risk: {stake_info['risk']} {stake_info['emoji']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence bar
        st.progress(confidence/100, text=f"Confidence Level: {confidence}%")
    
    # Betting advice
    st.markdown("---")
    st.subheader("💰 BETTING RECOMMENDATIONS")
    
    advice = result['betting_advice']
    
    # Strong plays
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
    
    # Moderate plays
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
    
    # Light plays
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
    st.markdown(f"#### 📋 SUMMARY: {advice['summary']}")
    
    # Expected scoreline
    st.markdown("---")
    st.subheader("📈 EXPECTED SCORELINE")
    
    home_xg = result['analysis']['expected_goals']['home']
    away_xg = result['analysis']['expected_goals']['away']
    
    # Convert xG to likely scoreline
    home_est = 0
    away_est = 0
    
    if home_xg > 1.2:
        home_est = 2 if home_xg > 1.8 else 1
    elif home_xg > 0.7:
        home_est = 1
        
    if away_xg > 1.2:
        away_est = 2 if away_xg > 1.8 else 1
    elif away_xg > 0.7:
        away_est = 1
    
    # Ensure at least some goals if xG suggests
    if home_est == 0 and home_xg > 0.5:
        home_est = 1
    if away_est == 0 and away_xg > 0.5:
        away_est = 1
    
    # Display scoreline
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"<h1 style='text-align: center;'>{home_est} - {away_est}</h1>", 
                   unsafe_allow_html=True)
        st.caption(f"Based on xG: {home_xg:.2f} - {away_xg:.2f}")
    
    # Performance note
    st.markdown("---")
    st.caption(f"⚡ PHANTOM v4.0 • League: {result['analysis']['league']} • Form-First Logic • No Avoids")

def display_welcome():
    """Display welcome screen"""
    st.info("👈 Please load a league from the sidebar to get started!")
    
    st.subheader("🌟 PHANTOM v4.0 - BOLD PREDICTIONS")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 FORM-FIRST")
        st.write("""
        • Last 3 games = 70% weight
        • Recent goals > Season stats
        • Form momentum drives predictions
        • No overthinking quality
        """)
    
    with col2:
        st.markdown("### 🔥 NO AVOIDS")
        st.write("""
        • Always Home/Away/Draw
        • Always Over/Under 2.5
        • Always Yes/No BTTS
        • Clear, decisive calls
        """)
    
    with col3:
        st.markdown("### ⚡ AGGRESSIVE")
        st.write("""
        • Higher stakes for confidence
        • Lower prediction thresholds
        • Recent trends prioritized
        • Bold, actionable advice
        """)
    
    st.markdown("---")
    
    st.success("""
    **⚡ QUICK START:**
    1. Select league from sidebar
    2. Click "LOAD LEAGUE DATA"
    3. Choose home and away teams
    4. Click "GENERATE BOLD PREDICTION"
    5. Get clear betting recommendations
    """)

def display_methodology():
    """Display the v4.0 methodology"""
    st.subheader("📖 PHANTOM v4.0 METHODOLOGY")
    
    with st.expander("View Complete Methodology", expanded=True):
        st.markdown("""
        ### 🔥 BOLD PREDICTION ENGINE
        
        **1. FORM-FIRST PREDICTION**
        ```
        Form Score = (Last 3 Games × 70%) + (Last 5 Games × 30%)
        
        Last 3 Games Estimated = (Last 5 Pts / 5) × 1.3
        Recent form matters MOST
        ```
        
        **2. ATTACK & DEFENSE STRENGTHS**
        ```
        Attack Strength = (Recent GPG × 60%) + (Season GPG × 40%)
        Recent Goals Per Game > Historical Average
        
        Defense Strength = max(0.5, 2.0 - Recent GAPG)
        Recent Goals Against Per Game matters MORE
        ```
        
        **3. NO MORE "AVOID" PREDICTIONS**
        ```
        Winner: ALWAYS Home/Away/Draw
        Threshold: League-specific (Low = More decisive)
        
        Total Goals: ALWAYS Over/Under 2.5
        Decision: Adjusted Total > Over Threshold = Over
        
        BTTS: ALWAYS Yes/No
        Decision: Probability ≥ Baseline = Yes
        ```
        
        **4. EXPECTED GOALS CALCULATION**
        ```
        Home xG = Home Attack × (2.0 - Away Defense) × Home Advantage
        Away xG = Away Attack × (2.0 - Home Defense) × 0.9
        
        Recent Hot Attack: +10% boost if >1.5 GPG last 5
        ```
        
        **5. CONFIDENCE & STAKING**
        ```
        Confidence = 55 + (Total Advantage × 25)
        Total Advantage = (Form Diff × 50%) + (Attack Diff × 30%) + (Defense Diff × 20%)
        
        Staking: AGGRESSIVE
        ≥70% = 1.5 units 🔥
        65-69% = 1.0 units ⚡
        60-64% = 0.75 units 📈
        55-59% = 0.5 units 📊
        50-54% = 0.25 units 📉
        ```
        
        ### 🎯 WHY THIS WORKS
        
        **Based on 9-match analysis:**
        1. **Form predicts winners** better than quality ratings
        2. **Recent performance** matters more than season averages
        3. **Clear predictions** beat "Avoid" for betting profitability
        4. **Football has variance** - accept it and be bold
        
        **Target Accuracy:** 55-60% winners, 50-55% totals, 50-55% BTTS
        **Key:** Clear, actionable predictions for betting success
        """)

if __name__ == "__main__":
    main()