"""
Betting advice and stake management for v3.0
"""

class BettingAdvisor:
    """Provides betting recommendations based on confidence"""
    
    @staticmethod
    def get_stake_recommendation(confidence):
        """Determine stake size based on confidence (v3.0 conservative)"""
        if confidence >= 70:
            return {"units": 0.75, "color": "🟢", "risk": "Medium"}  # Was 1.0
        elif 62 <= confidence < 70:
            return {"units": 0.5, "color": "🟢", "risk": "Low-Medium"}  # Was 0.75
        elif 55 <= confidence < 62:
            return {"units": 0.25, "color": "🟡", "risk": "Low"}  # Was 0.5
        elif 48 <= confidence < 55:
            return {"units": 0.1, "color": "🟠", "risk": "Very Low"}  # Was 0.25
        else:
            return {"units": 0, "color": "⚪", "risk": "AVOID"}
    
    @staticmethod
    def generate_advice(predictions):
        """Generate betting advice based on predictions"""
        advice = {
            "strong_plays": [],
            "moderate_plays": [],
            "avoid": [],
            "summary": ""
        }
        
        for pred in predictions:
            stake_info = BettingAdvisor.get_stake_recommendation(pred['confidence'])
            
            if stake_info['units'] >= 0.5:
                advice['strong_plays'].append({
                    "market": pred['type'],
                    "selection": pred['selection'],
                    "confidence": pred['confidence'],
                    "stake": stake_info
                })
            elif stake_info['units'] >= 0.25:
                advice['moderate_plays'].append({
                    "market": pred['type'],
                    "selection": pred['selection'],
                    "confidence": pred['confidence'],
                    "stake": stake_info
                })
            elif stake_info['units'] > 0:
                advice['moderate_plays'].append({
                    "market": pred['type'],
                    "selection": pred['selection'],
                    "confidence": pred['confidence'],
                    "stake": stake_info
                })
            else:
                advice['avoid'].append(pred['type'])
        
        # Generate summary
        if advice['strong_plays']:
            advice['summary'] = f"Strong betting opportunities found ({len(advice['strong_plays'])} markets)"
        elif advice['moderate_plays']:
            advice['summary'] = f"Moderate betting opportunities ({len(advice['moderate_plays'])} markets)"
        else:
            advice['summary'] = "No strong betting opportunities - consider avoiding"
        
        return advice
