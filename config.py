"""
Configuration for league-specific parameters and thresholds
"""

LEAGUE_SETTINGS = {
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
    'Ligue 1': {
        'avg_goals': 2.96,
        'over_threshold': 2.85,
        'under_threshold': 2.65,
        'btts_baseline': 55,
        'home_advantage': 0.3,
        'goal_variance': 'medium'
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

# Team profile weightings
PROFILE_WEIGHTS = {
    'matchup_analysis': 0.5,      # Current form and xG matchup
    'team_history': 0.3,          # Historical team averages
    'league_baseline': 0.2        # League averages
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'very_strong': 70,
    'strong': 62,
    'moderate': 55,
    'weak': 48,
    'avoid': 45
}

# Stake sizes (units)
STAKE_SIZES = {
    'very_strong': 1.5,
    'strong': 1.0,
    'moderate': 0.75,
    'weak': 0.5,
    'speculative': 0.25
}
