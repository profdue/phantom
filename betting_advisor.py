"""
Betting advice and stake management
"""

class BettingAdvisor:
    """Provides betting recommendations based on confidence"""
    
    @staticmethod
    def get_stake_recommendation(confidence):
        """Determine stake size based on confidence (v2.3 conservative)"""
        if confidence >= 75:
            return {"units": 1.0, "color": "🟢", "risk": "Medium"}
        elif 65 <= confidence < 75:
            return {"units": 0.75, "color": "🟢", "risk": "Medium"}
        elif 55 <= confidence < 65:
            return {"units": 0.5, "color": "🟡", "risk": "Low-Medium"}
        elif 45 <= confidence < 55:
            return {"units": 0.25, "color": "🟠", "risk": "Low"}
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
            
            if stake_info['units'] >= 0.75:
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
    
    @staticmethod
    def calculate_expected_value(confidence, decimal_odds):
        """Calculate expected value of a bet"""
        probability = confidence / 100
        ev = (probability * (decimal_odds - 1)) - ((1 - probability) * 1)
        return round(ev * 100, 2)  # Return as percentage
