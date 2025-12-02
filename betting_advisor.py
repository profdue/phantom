"""
OPTIMIZED BETTING ADVISOR MODULE v2.0
Based on new 4-match performance data
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
        Apply strategic filters based on NEW performance data
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
        
        # 1. TOTAL GOALS STRATEGY (PROVEN 100% ACCURACY IN LAST 4 MATCHES!)
        # LOWER thresholds based on proven performance
        if total_goals_pred['confidence'] >= 55:  # LOWERED from 58%
            if 'Over' in total_goals_pred['selection']:
                if total_xg > 2.8:
                    # STRONG OVER (high xG + good confidence)
                    recommendations.append({
                        'market': 'Total Goals - Over 2.5',
                        'selection': 'OVER',
                        'confidence': total_goals_pred['confidence'],
                        'stake': '1.5 units',
                        'reason': f"STRONG Over signal ({total_goals_pred['confidence']:.0f}%) with {total_xg:.1f} total xG. Model 100% accurate on Over predictions in last 4 matches.",
                        'edge_type': 'VERY STRONG',
                        'filter_applied': 'Total Goals Over Filter'
                    })
                elif total_xg > 2.5:
                    # MODERATE OVER
                    recommendations.append({
                        'market': 'Total Goals - Over 2.5',
                        'selection': 'OVER',
                        'confidence': total_goals_pred['confidence'],
                        'stake': '1.0 units',
                        'reason': f"Good Over value ({total_goals_pred['confidence']:.0f}%) with {total_xg:.1f} total xG.",
                        'edge_type': 'STRONG',
                        'filter_applied': 'Total Goals Over Filter'
                    })
            elif 'Under' in total_goals_pred['selection']:
                if total_xg < 2.2:
                    recommendations.append({
                        'market': 'Total Goals - Under 2.5',
                        'selection': 'UNDER',
                        'confidence': total_goals_pred['confidence'],
                        'stake': '1.0 units',
                        'reason': f"Clear Under signal ({total_goals_pred['confidence']:.0f}%) with low {total_xg:.1f} total xG.",
                        'edge_type': 'STRONG',
                        'filter_applied': 'Total Goals Under Filter'
                    })
        
        # 2. MATCH WINNER BETS (CONSERVATIVE - MODEL STILL UNRELIABLE)
        # Only bet on CLEAR favorites with significant xG advantage
        if match_winner_pred['confidence'] >= 65 and abs(xg_diff) > 1.0:
            # Only strong favorites with big xG advantage
            recommendations.append({
                'market': 'Match Winner',
                'selection': match_winner_pred['selection'],
                'confidence': match_winner_pred['confidence'],
                'stake': '0.75 units',  # Conservative stake
                'reason': f"Significant xG advantage ({xg_diff:+.1f}) supports favorite.",
                'edge_type': 'MODERATE',
                'filter_applied': 'Strong Favorite Filter'
            })
        
        # 3. FADE OVERCONFIDENT MATCH WINNER PREDICTIONS
        if match_winner_pred['confidence'] >= 70 and abs(xg_diff) < 0.8:
            # High confidence but small xG advantage = FADE
            if 'Home' in match_winner_pred['selection']:
                fade_selection = f"{model_predictions['team_names']['away']} Double Chance"
            else:
                fade_selection = f"{model_predictions['team_names']['home']} Double Chance"
            
            recommendations.append({
                'market': 'Double Chance (Fade)',
                'selection': fade_selection,
                'confidence': 65,
                'stake': '0.5 units',
                'reason': f"High confidence ({match_winner_pred['confidence']:.0f}%) but only {abs(xg_diff):.1f} xG advantage. Contrarian value.",
                'edge_type': 'CONTRARIAN',
                'filter_applied': 'Overconfidence Fade Filter'
            })
        
        # 4. BTTS VALUE DETECTION
        if btts_pred['confidence'] >= 58 and btts_pred['selection'] != 'Avoid BTTS':
            if total_xg > 2.8:  # High-scoring games favor BTTS
                recommendations.append({
                    'market': 'Both Teams to Score',
                    'selection': btts_pred['selection'].upper(),
                    'confidence': btts_pred['confidence'],
                    'stake': '0.75 units',
                    'reason': f"BTTS signal ({btts_pred['confidence']:.0f}%) in high-scoring game ({total_xg:.1f} xG).",
                    'edge_type': 'MODERATE',
                    'filter_applied': 'BTTS High Scoring Filter'
                })
        
        # 5. ASIAN HANDICAP CAUTION (HISTORICALLY 31% ACCURATE)
        if ah_pred['confidence'] >= 60:
            recommendations.append({
                'market': 'ASIAN HANDICAP - CAUTION',
                'selection': 'REDUCE STAKE',
                'confidence': ah_pred['confidence'],
                'stake': '0.25 units (MAX)',
                'reason': f"⚠️ HIGH RISK: Model only 31% accurate on handicaps. Max 0.25 units.",
                'edge_type': 'VERY HIGH RISK',
                'filter_applied': 'Asian Handicap Warning Filter'
            })
        
        # 6. PARLAY OPPORTUNITIES (TOTAL GOALS + STRONG FAVORITE)
        strong_total_goals = [r for r in recommendations if r['market'] == 'Total Goals - Over 2.5' and r['edge_type'] in ['VERY STRONG', 'STRONG']]
        strong_match_winner = [r for r in recommendations if r['market'] == 'Match Winner' and r['edge_type'] == 'MODERATE']
        
        if strong_total_goals and strong_match_winner:
            parlay_confidence = min(75, strong_total_goals[0]['confidence'] * strong_match_winner[0]['confidence'] / 100)
            
            recommendations.append({
                'market': 'PARLAY: Match Winner + Over 2.5',
                'selection': f"{strong_match_winner[0]['selection']} & OVER",
                'confidence': parlay_confidence,
                'stake': '0.5 units',
                'reason': f"Combined strong signals. Estimated probability: {parlay_confidence:.0f}%",
                'edge_type': 'PARLAY',
                'filter_applied': 'Parlay Builder Filter'
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
            'recommendations': recommendations,
            'total_bets': len(recommendations),
            'strong_bets': len([r for r in recommendations if r['edge_type'] in ['VERY STRONG', 'STRONG']])
        }
        self.recommendations_history.append(record)
    
    def get_performance_summary(self) -> Dict:
        """Get summary of advisor performance"""
        if not self.recommendations_history:
            return {}
        
        total_matches = len(self.recommendations_history)
        total_recommendations = sum(r['total_bets'] for r in self.recommendations_history)
        total_strong_bets = sum(r['strong_bets'] for r in self.recommendations_history)
        
        return {
            'total_matches_analyzed': total_matches,
            'total_recommendations': total_recommendations,
            'total_strong_bets': total_strong_bets,
            'avg_recommendations_per_match': total_recommendations / max(1, total_matches),
            'avg_strong_bets_per_match': total_strong_bets / max(1, total_matches)
        }
    
    def display_recommendations(self, recommendations: List[Dict]) -> str:
        """Format recommendations for display"""
        if not recommendations:
            return "⚠️ No strong betting opportunities detected for this match."
        
        output = "🎯 **BETTING ADVISOR RECOMMENDATIONS** 🎯\n\n"
        
        # Group by edge type
        very_strong_recs = [r for r in recommendations if r['edge_type'] == 'VERY STRONG']
        strong_recs = [r for r in recommendations if r['edge_type'] == 'STRONG']
        moderate_recs = [r for r in recommendations if r['edge_type'] == 'MODERATE']
        contrarian_recs = [r for r in recommendations if r['edge_type'] == 'CONTRARIAN']
        risk_recs = [r for r in recommendations if r['edge_type'] == 'VERY HIGH RISK']
        parlays = [r for r in recommendations if r['edge_type'] == 'PARLAY']
        
        if very_strong_recs:
            output += "🔥 **VERY STRONG BETS** 🔥\n"
            for rec in very_strong_recs:
                output += f"• **{rec['market']}** ({rec['selection']}) - {rec['stake']}\n"
                output += f"  Confidence: {rec['confidence']:.0f}% | {rec['reason']}\n\n"
        
        if strong_recs:
            output += "✅ **STRONG BETS** ✅\n"
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
            output += "🚨 **VERY HIGH RISK BETS (MAX 0.25 UNITS)** 🚨\n"
            for rec in risk_recs:
                output += f"• **{rec['market']}** - {rec['stake']}\n"
                output += f"  Confidence: {rec['confidence']:.0f}% | {rec['reason']}\n\n"
        
        if parlays:
            output += "🎲 **PARLAY OPPORTUNITIES** 🎲\n"
            for rec in parlays:
                output += f"• **{rec['market']}** - {rec['stake']}\n"
                output += f"  Combined Confidence: {rec['confidence']:.0f}% | {rec['reason']}\n\n"
        
        # Add summary
        output += f"📊 **Summary**: {len(very_strong_recs)} very strong, {len(strong_recs)} strong, {len(moderate_recs)} moderate bets detected.\n"
        
        return output