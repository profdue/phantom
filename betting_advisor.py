"""
OPTIMIZED BETTING ADVISOR MODULE
Updated thresholds for CALIBRATED model predictions
"""
from typing import Dict, List, Tuple
import json
from datetime import datetime

class BettingAdvisor:
    def __init__(self):
        self.recommendations_history = []
        self.performance_tracker = {}
        
    def analyze_predictions(self, model_predictions: Dict) -> List[Dict]:
        """
        Apply strategic filters to CALIBRATED model predictions
        """
        recommendations = []
        
        # Extract key data
        match_winner_pred = next(p for p in model_predictions['predictions'] if p['type'] == 'Match Winner')
        total_goals_pred = next(p for p in model_predictions['predictions'] if p['type'] == 'Total Goals')
        btts_pred = next(p for p in model_predictions['predictions'] if p['type'] == 'Both Teams To Score')
        ah_pred = next(p for p in model_predictions['predictions'] if p['type'] == 'Asian Handicap')
        
        analysis = model_predictions['analysis']
        total_xg = analysis['expected_goals']['total']
        xg_diff = analysis['expected_goal_diff']
        
        # 1. TOTAL GOALS STRATEGY (OPTIMIZED FOR CALIBRATION)
        if total_goals_pred['confidence'] >= 58:  # LOWERED from 62%
            if 'Over' in total_goals_pred['selection'] and total_xg > 2.8:
                recommendations.append({
                    'market': 'Total Goals - Over 2.5',
                    'selection': 'OVER',
                    'confidence': total_goals_pred['confidence'],
                    'stake': '1.0 units',
                    'reason': f"Strong Over signal ({total_goals_pred['confidence']:.0f}%) with {total_xg:.1f} total xG.",
                    'edge_type': 'STRONG',
                    'filter_applied': 'Total Goals Over Filter'
                })
            elif 'Under' in total_goals_pred['selection'] and total_xg < 2.2:
                recommendations.append({
                    'market': 'Total Goals - Under 2.5',
                    'selection': 'UNDER',
                    'confidence': total_goals_pred['confidence'],
                    'stake': '1.0 units',
                    'reason': f"Clear Under signal ({total_goals_pred['confidence']:.0f}%) with low {total_xg:.1f} total xG.",
                    'edge_type': 'STRONG',
                    'filter_applied': 'Total Goals Under Filter'
                })
        
        # 2. POSITIVE MATCH WINNER BETS (NEW FOR CALIBRATED MODEL)
        if match_winner_pred['confidence'] >= 65:
            # GOOD VALUE BETS in calibrated range
            if abs(xg_diff) > 0.8:  # Significant xG advantage
                recommendations.append({
                    'market': 'Match Winner',
                    'selection': match_winner_pred['selection'],
                    'confidence': match_winner_pred['confidence'],
                    'stake': '1.0 units',
                    'reason': f"Calibrated confidence ({match_winner_pred['confidence']:.0f}%) with significant xG advantage ({xg_diff:+.1f}).",
                    'edge_type': 'STRONG',
                    'filter_applied': 'Calibrated Value Bet Filter'
                })
            elif match_winner_pred['confidence'] >= 60:
                recommendations.append({
                    'market': 'Match Winner',
                    'selection': match_winner_pred['selection'],
                    'confidence': match_winner_pred['confidence'],
                    'stake': '0.75 units',
                    'reason': f"Moderate value bet ({match_winner_pred['confidence']:.0f}% confidence).",
                    'edge_type': 'MODERATE',
                    'filter_applied': 'Moderate Value Filter'
                })
        
        # 3. FADE OVERCONFIDENT PREDICTIONS (ADJUSTED THRESHOLD)
        if match_winner_pred['confidence'] >= 75:  # RAISED from 72%
            # Only fade truly overconfident predictions
            fade_market = 'Draw or Underdog'
            if 'Home' in match_winner_pred['selection']:
                fade_selection = f"{model_predictions['team_names']['away']} Double Chance"
            else:
                fade_selection = f"{model_predictions['team_names']['home']} Double Chance"
            
            recommendations.append({
                'market': fade_market,
                'selection': fade_selection,
                'confidence': 62,
                'stake': '0.5 units',
                'reason': f"High confidence ({match_winner_pred['confidence']:.0f}%) prediction - contrarian value.",
                'edge_type': 'CONTRARIAN',
                'filter_applied': 'Overconfidence Fade Filter'
            })
        
        # 4. BTTS VALUE DETECTION
        if btts_pred['confidence'] >= 60 and btts_pred['selection'] != 'Avoid BTTS':
            # Good BTTS signal
            recommendations.append({
                'market': 'Both Teams to Score',
                'selection': btts_pred['selection'].upper(),
                'confidence': btts_pred['confidence'],
                'stake': '0.75 units',
                'reason': f"Clear BTTS signal ({btts_pred['confidence']:.0f}% confidence).",
                'edge_type': 'MODERATE',
                'filter_applied': 'BTTS Value Filter'
            })
        
        # 5. ASIAN HANDICAP WARNING (31% ACCURACY - CAUTION)
        if ah_pred['confidence'] >= 65:
            recommendations.append({
                'market': 'CAUTION: Asian Handicap',
                'selection': 'REDUCED STAKE',
                'confidence': ah_pred['confidence'],
                'stake': '0.5 units (max)',
                'reason': f"Model historically 31% accurate on handicaps. Reduced stake recommended.",
                'edge_type': 'HIGH RISK',
                'filter_applied': 'Asian Handicap Caution Filter'
            })
        
        # 6. PARLAY OPPORTUNITIES
        strong_recommendations = [r for r in recommendations if r['edge_type'] == 'STRONG']
        if len(strong_recommendations) >= 2:
            parlay_markets = [r['market'] for r in strong_recommendations[:2]]
            parlay_confidence = min(70, strong_recommendations[0]['confidence'] * strong_recommendations[1]['confidence'] / 120)
            
            recommendations.append({
                'market': 'Parlay Opportunity',
                'selection': ' + '.join(parlay_markets),
                'confidence': parlay_confidence,
                'stake': '0.5 units',
                'reason': f"Two strong recommendations. Combined probability: {parlay_confidence:.0f}%",
                'edge_type': 'PARLAY',
                'filter_applied': 'Strong Signals Parlay Filter'
            })
        
        # Record this analysis
        self.record_recommendations(
            model_predictions['team_names'],
            recommendations,
            total_xg,
            match_winner_pred['confidence']
        )
        
        return recommendations
    
    def record_recommendations(self, team_names: Dict, recommendations: List[Dict], total_xg: float, mw_confidence: float):
        """Record recommendations for performance tracking"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'match': f"{team_names['home']} vs {team_names['away']}",
            'total_xg': total_xg,
            'mw_confidence': mw_confidence,
            'recommendations': recommendations
        }
        self.recommendations_history.append(record)
    
    def get_performance_summary(self) -> Dict:
        """Get summary of advisor performance"""
        total_matches = len(self.recommendations_history)
        total_recommendations = sum(len(r['recommendations']) for r in self.recommendations_history)
        strong_recommendations = sum(
            1 for r in self.recommendations_history 
            for rec in r['recommendations'] 
            if rec['edge_type'] == 'STRONG'
        )
        
        return {
            'total_matches_analyzed': total_matches,
            'total_recommendations': total_recommendations,
            'strong_recommendations': strong_recommendations,
            'avg_recommendations_per_match': total_recommendations / max(1, total_matches)
        }
    
    def display_recommendations(self, recommendations: List[Dict]) -> str:
        """Format recommendations for display"""
        if not recommendations:
            return "⚠️ No strong betting opportunities detected for this match."
        
        output = "🎯 **BETTING ADVISOR RECOMMENDATIONS** 🎯\n\n"
        
        # Group by edge type
        strong_recs = [r for r in recommendations if r['edge_type'] == 'STRONG']
        moderate_recs = [r for r in recommendations if r['edge_type'] == 'MODERATE']
        contrarian_recs = [r for r in recommendations if r['edge_type'] == 'CONTRARIAN']
        risk_recs = [r for r in recommendations if r['edge_type'] == 'HIGH RISK']
        parlays = [r for r in recommendations if r['edge_type'] == 'PARLAY']
        
        if strong_recs:
            output += "🔥 **STRONG BETS** 🔥\n"
            for rec in strong_recs:
                output += f"• **{rec['market']}** ({rec['selection']}) - {rec['stake']}\n"
                output += f"  Confidence: {rec['confidence']:.0f}% | {rec['reason']}\n\n"
        
        if moderate_recs:
            output += "📈 **MODERATE BETS** 📈\n"
            for rec in moderate_recs:
                output += f"• **{rec['market']}** ({rec['selection']}) - {rec['stake']}\n"
                output += f"  Confidence: {rec['confidence']:.0f}% | {rec['reason']}\n\n"
        
        if contrarian_recs:
            output += "🔄 **CONTRARIAN BETS** 🔄\n"
            for rec in contrarian_recs:
                output += f"• **{rec['market']}** ({rec['selection']}) - {rec['stake']}\n"
                output += f"  Confidence: {rec['confidence']:.0f}% | {rec['reason']}\n\n"
        
        if risk_recs:
            output += "⚠️ **HIGH RISK BETS (CAUTION)** ⚠️\n"
            for rec in risk_recs:
                output += f"• **{rec['market']}** - {rec['stake']}\n"
                output += f"  Confidence: {rec['confidence']:.0f}% | {rec['reason']}\n\n"
        
        if parlays:
            output += "🎲 **PARLAY OPPORTUNITIES** 🎲\n"
            for rec in parlays:
                output += f"• **{rec['market']}** - {rec['stake']}\n"
                output += f"  Combined Confidence: {rec['confidence']:.0f}% | {rec['reason']}\n\n"
        
        # Add summary
        output += f"📊 **Summary**: {len(strong_recs)} strong, {len(moderate_recs)} moderate, {len(contrarian_recs)} contrarian bets detected.\n"
        
        return output