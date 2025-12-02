"""
League configuration parameters v2.3
"""

LEAGUE_CONFIGS = {
    "premier_league": {
        "name": "Premier League",
        "avg_goals": 2.93,
        "over_threshold": 2.8,
        "under_threshold": 2.6,
        "home_advantage": 0.35,
        "btts_baseline": 56,
        "win_threshold": 0.45,
        "momentum_weight": 0.3
    },
    "serie_a": {
        "name": "Serie A",
        "avg_goals": 2.56,
        "over_threshold": 2.45,
        "under_threshold": 2.25,
        "home_advantage": 0.25,
        "btts_baseline": 51,
        "win_threshold": 0.55,
        "momentum_weight": 0.25
    },
    "la_liga": {
        "name": "La Liga",
        "avg_goals": 2.62,
        "over_threshold": 2.5,
        "under_threshold": 2.3,
        "home_advantage": 0.30,
        "btts_baseline": 56,
        "win_threshold": 0.55,
        "momentum_weight": 0.28
    },
    "bundesliga": {
        "name": "Bundesliga",
        "avg_goals": 3.14,
        "over_threshold": 3.0,
        "under_threshold": 2.8,
        "home_advantage": 0.40,
        "btts_baseline": 58,
        "win_threshold": 0.40,
        "momentum_weight": 0.35
    },
    "ligue_1": {
        "name": "Ligue 1",
        "avg_goals": 2.78,
        "over_threshold": 2.7,
        "under_threshold": 2.5,
        "home_advantage": 0.32,
        "btts_baseline": 54,
        "win_threshold": 0.48,
        "momentum_weight": 0.30
    },
    "rfpl": {
        "name": "Russian Premier League",
        "avg_goals": 2.68,
        "over_threshold": 2.6,
        "under_threshold": 2.4,
        "home_advantage": 0.28,
        "btts_baseline": 53,
        "win_threshold": 0.50,
        "momentum_weight": 0.26
    }
}

MODEL_PARAMS = {
    "attack_weight": 0.7,
    "defense_weight": 0.3,
    "league_blend_current": 0.6,
    "league_blend_league": 0.4,
    "mismatch_blend_current": 0.7,
    "mismatch_blend_league": 0.3,
    "away_form_penalty_threshold": 6,
    "attack_crisis_threshold": 1.0,
    "defense_crisis_threshold": 1.8,
    "big_match_quality_threshold": 1.5,
    "clean_sheet_risk_threshold": 35
}
