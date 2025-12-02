from typing import Dict, List, Tuple
import json
from datetime import datetime

class BettingAdvisor:
    def __init__(self):
        self.recommendations_history = []
        
    def analyze_predictions(self, model_predictions: Dict) -> List[Dict]:
        """
        FIXED: Better mismatch detection and threshold adjustments
        """
        recommendations = []
        
        # Extract key data
        match_winner_pred = next(p for p in model_predictions['predictions'] if p['type'] == 'Match Winner')
        total_goals_pred = next(p for p in model_predictions['predictions'] if p['type'] == 'Total Goals')
        btts_pred = next(p for p in model_predictions['predictions'] if p['type'] == 'Both Teams To Score')
        
        analysis = model_predictions['analysis']
        total_xg = analysis['expected_goals']['total']
        xg_diff = analysis['expected_goal_diff']
        home_quality = analysis['home_quality']
        away_quality = analysis['away_quality']
        quality_diff = analysis['quality_diff']
        
        home_profile = model_predictions['raw_data']['home_profile']
        away_profile = model_predictions['raw_data']['away_profile']
        
        # DETECT EXTREME MISMATCHES
        is_extreme_mismatch = False
        mismatch_reason = ""
        
        # Offensive mismatch (one team scores <0.5, other >1.5)
        if home_profile['offensive_rating'] < 0.5 and away_profile['offensive_rating'] > 1.5:
            is_extreme_mismatch = True
            mismatch_reason = f"{model_predictions['team_names']['away']} offensive mismatch"
        elif away_profile['offensive_rating'] < 0.5 and home_profile['offensive_rating'] > 1.5:
            is_extreme_mismatch = True
            mismatch_reason = f"{model_predictions['team_names']['home']} offensive mismatch"
        
        # Defensive mismatch (one team concedes >1.8, other <0.8)
        if home_profile['goals_conceded_pg'] > 1.8 and away_profile['goals_conceded_pg'] < 0.8:
            is_extreme_mismatch = True
            mismatch_reason = f"{model_predictions['team_names']['away']} defensive mismatch"
        elif away_profile['goals_conceded_pg'] > 1.8 and home_profile['goals_conceded_pg'] < 0.8:
            is_extreme_mismatch = True
            mismatch_reason = f"{model_predictions['team_names']['home']} defensive mismatch"
        
        # 1. TOTAL GOALS STRATEGY - LOWERED THRESHOLDS
        if total_goals_pred['confidence'] >= 58:  # Lowered from 62%
            if 'Over' in total_goals_pred['selection']:
                if total_xg > 3.0 or is_extreme_mismatch:
                    recommendations.append({
                        'market': 'Total Goals - Over 2.5',
                        'selection': 'OVER',
                        'confidence': total_goals_pred['confidence'],
                        'stake': '1.5 units',
                        'reason': f"STRONG: {total_xg:.1f} xG. {mismatch_reason}" if is_extreme_mismatch else f"STRONG: {total_goals_pred['confidence']:.0f}% confidence with {total_xg:.1f} xG.",
                        'edge_type': 'STRONG',
                        'filter_applied': 'High xG or Mismatch Filter'
                    })
                elif total_xg > 2.7:
                    recommendations.append({
                        'market': 'Total Goals - Over 2.5',
                        'selection': 'OVER',
                        'confidence': total_goals_pred['confidence'],
                        'stake': '1.0 units',
                        'reason': f"MODERATE: {total_goals_pred['confidence']:.0f}% confidence with {total_xg:.1f} xG.",
                        'edge_type': 'MODERATE',
                        'filter_applied': 'Moderate xG Filter'
                    })
            elif 'Under' in total_goals_pred['selection']:
                if total_xg < 2.0 or (home_profile['offensive_rating'] < 0.5 and away_profile['offensive_rating'] < 0.5):
                    recommendations.append({
                        'market': 'Total Goals - Under 2.5',
                        'selection': 'UNDER',
                        'confidence': total_goals_pred['confidence'],
                        'stake': '1.0 units',
                        'reason': f"STRONG: {total_goals_pred['confidence']:.0f}% confidence with low {total_xg:.1f} xG.",
                        'edge_type': 'STRONG',
                        'filter_applied': 'Low xG Filter'
                    })
        
        # 2. MATCH WINNER BETS - IMPROVED MISMATCH DETECTION
        if is_extreme_mismatch and match_winner_pred['confidence'] >= 65:
            # Extreme mismatch = stronger bet
            recommendations.append({
                'market': 'Match Winner',
                'selection': match_winner_pred['selection'],
                'confidence': match_winner_pred['confidence'],
                'stake': '1.5 units' if match_winner_pred['confidence'] >= 70 else '1.0 units',
                'reason': f"EXTREME MISMATCH: {mismatch_reason}. {abs(quality_diff):.1f} quality difference.",
                'edge_type': 'STRONG',
                'filter_applied': 'Extreme Mismatch Filter'
            })
        elif match_winner_pred['confidence'] >= 68 and abs(quality_diff) > 0.8:  # Lowered threshold
            recommendations.append({
                'market': 'Match Winner',
                'selection': match_winner_pred['selection'],
                'confidence': match_winner_pred['confidence'],
                'stake': '1.0 units',
                'reason': f"Clear favorite: {abs(quality_diff):.1f} quality difference.",
                'edge_type': 'MODERATE',
                'filter_applied': 'Clear Favorite Filter'
            })
        
        # 3. FADE MODERATE FAVORITES (Only if NOT extreme mismatch)
        if not is_extreme_mismatch and 62 <= match_winner_pred['confidence'] <= 70 and abs(quality_diff) < 0.6:
            if 'Home' in match_winner_pred['selection']:
                fade_selection = f"{model_predictions['team_names']['away']} Double Chance"
            else:
                fade_selection = f"{model_predictions['team_names']['home']} Double Chance"
            
            recommendations.append({
                'market': 'Double Chance (Fade Favorite)',
                'selection': fade_selection,
                'confidence': 62,
                'stake': '0.75 units',
                'reason': f"Fading {match_winner_pred['confidence']:.0f}% favorite with only {abs(quality_diff):.1f} quality difference.",
                'edge_type': 'CONTRARIAN',
                'filter_applied': 'Moderate Favorite Fade'
            })
        
        # 4. BTTS VALUE DETECTION
        if btts_pred['confidence'] >= 58 and btts_pred['selection'] != 'Avoid BTTS':  # Lowered threshold
            if total_xg > 2.8 and abs(xg_diff) < 1.2:
                recommendations.append({
                    'market': 'Both Teams to Score',
                    'selection': btts_pred['selection'].upper(),  # FIXED
                    'confidence': btts_pred['confidence'],
                    'stake': '0.75 units',
                    'reason': f"High-scoring ({total_xg:.1f} xG) game favors BTTS.",
                    'edge_type': 'MODERATE',
                    'filter_applied': 'High xG BTTS'
                })
            elif is_extreme_mismatch and 'defensive' in mismatch_reason.lower():
                # Defensive mismatch = likely BTTS
                recommendations.append({
                    'market': 'Both Teams to Score',
                    'selection': btts_pred['selection'].upper(),  # FIXED
                    'confidence': btts_pred['confidence'],
                    'stake': '0.5 units',
                    'reason': f"Defensive mismatch favors BTTS: {mismatch_reason}",
                    'edge_type': 'MODERATE',
                    'filter_applied': 'Defensive Mismatch BTTS'
                })
        
        # Record this analysis
        self.record_recommendations(
            model_predictions['team_names'],
            recommendations,
            total_xg,
            match_winner_pred['confidence'],
            xg_diff,
            is_extreme_mismatch
        )
        
        return recommendations
    
    def record_recommendations(self, team_names: Dict, recommendations: List[Dict], total_xg: float, 
                              mw_confidence: float, xg_diff: float, is_extreme_mismatch: bool = False):
        """Record recommendations for performance tracking"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'match': f"{team_names['home']} vs {team_names['away']}",
            'total_xg': total_xg,
            'mw_confidence': mw_confidence,
            'xg_diff': xg_diff,
            'is_extreme_mismatch': is_extreme_mismatch,
            'recommendations': recommendations,
            'total_bets': len(recommendations),
            'strong_bets': len([r for r in recommendations if r['edge_type'] in ['STRONG']]),
            'moderate_bets': len([r for r in recommendations if r['edge_type'] == 'MODERATE'])
        }
        self.recommendations_history.append(record)
    
    def get_performance_summary(self) -> Dict:
        """Get summary of advisor performance"""
        if not self.recommendations_history:
            return {}
        
        total_matches = len(self.recommendations_history)
        total_recommendations = sum(r['total_bets'] for r in self.recommendations_history)
        total_strong_bets = sum(r['strong_bets'] for r in self.recommendations_history)
        total_moderate_bets = sum(r['moderate_bets'] for r in self.recommendations_history)
        
        return {
            'total_matches_analyzed': total_matches,
            'total_recommendations': total_recommendations,
            'total_strong_bets': total_strong_bets,
            'total_moderate_bets': total_moderate_bets,
            'avg_recommendations_per_match': total_recommendations / max(1, total_matches),
            'avg_strong_bets_per_match': total_strong_bets / max(1, total_matches),
            'mismatch_matches': len([r for r in self.recommendations_history if r['is_extreme_mismatch']])
        }
    
    def display_recommendations(self, recommendations: List[Dict]) -> str:
        """Format recommendations for display"""
        if not recommendations:
            return "✅ **NO STRONG BETS** - Advisor avoiding weak opportunities.\n\n*Better to miss a bet than make a bad one.*"
        
        output = "🎯 **BETTING ADVISOR RECOMMENDATIONS** 🎯\n\n"
        
        # Group by edge type
        strong_recs = [r for r in recommendations if r['edge_type'] == 'STRONG']
        moderate_recs = [r for r in recommendations if r['edge_type'] == 'MODERATE']
        contrarian_recs = [r for r in recommendations if r['edge_type'] == 'CONTRARIAN']
        
        if strong_recs:
            output += "🔥 **STRONG BETS (1.0-1.5 UNITS)** 🔥\n"
            for rec in strong_recs:
                output += f"• **{rec['market']}** ({rec['selection']}) - {rec['stake']}\n"
                output += f"  Confidence: {rec['confidence']:.0f}% | {rec['reason']}\n\n"
        
        if moderate_recs:
            output += "📈 **MODERATE BETS (0.5-0.75 UNITS)** 📈\n"
            for rec in moderate_recs:
                output += f"• **{rec['market']}** ({rec['selection']}) - {rec['stake']}\n"
                output += f"  Confidence: {rec['confidence']:.0f}% | {rec['reason']}\n\n"
        
        if contrarian_recs:
            output += "🔄 **CONTRARIAN BETS (0.75 UNITS)** 🔄\n"
            for rec in contrarian_recs:
                output += f"• **{rec['market']}** ({rec['selection']})\n"
                output += f"  Confidence: {rec['confidence']:.0f}% | {rec['reason']}\n\n"
        
        # Add summary
        total_bets = len(strong_recs) + len(moderate_recs) + len(contrarian_recs)
        output += f"📊 **Summary**: {len(strong_recs)} strong, {len(moderate_recs)} moderate, {total_bets} total betting opportunities.\n"
        
        return output
