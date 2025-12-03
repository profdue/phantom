"""
PHANTOM v4.1 - Betting Advisor with Fractional Kelly Staking
"""
from typing import Dict, Optional, List

class BettingAdvisor:
    """Provides betting recommendations using fractional Kelly staking"""
    
    def __init__(self, bankroll: float = 100.0, min_confidence: float = 50.0):
        self.bankroll = bankroll
        self.min_confidence = min_confidence
        
    def get_stake_recommendation(self, confidence: float, 
                                market_odds: Optional[float] = None,
                                prediction_type: str = "generic") -> Dict:
        """
        Calculate stake recommendation using fractional Kelly
        
        Args:
            confidence: Model confidence (0-100)
            market_odds: Decimal odds from bookmaker (optional)
            prediction_type: Type of prediction (winner/total/btts)
            
        Returns:
            Dictionary with stake recommendation and details
        """
        
        # No bet if below minimum confidence
        if confidence < self.min_confidence:
            return {
                "units": 0,
                "reason": f"Below minimum confidence ({confidence}% < {self.min_confidence}%)",
                "method": "reject",
                "color": "⚪",
                "risk": "AVOID",
                "emoji": "🚫"
            }
        
        # If no market odds, use conservative confidence-based staking
        if market_odds is None or market_odds <= 1.0:
            return self._confidence_based_stake(confidence)
        
        # WITH MARKET ODDS: Use fractional Kelly
        return self._kelly_based_stake(confidence, market_odds, prediction_type)
    
    def _kelly_based_stake(self, confidence: float, market_odds: float, 
                          prediction_type: str) -> Dict:
        """Calculate stake using fractional Kelly formula"""
        
        your_prob = confidence / 100.0
        implied_prob = 1.0 / market_odds
        
        # No edge - no bet
        if your_prob <= implied_prob:
            return {
                "units": 0,
                "reason": f"No edge (your {your_prob:.1%} ≤ market {implied_prob:.1%})",
                "method": "no_edge",
                "color": "⚪",
                "risk": "AVOID",
                "emoji": "🚫"
            }
        
        # Kelly formula: f* = (bp - q) / b
        b = market_odds - 1.0  # Decimal odds minus 1
        p = your_prob          # Your probability
        q = 1.0 - p           # Probability of losing
        
        # Calculate Kelly fraction
        full_kelly = (b * p - q) / b
        full_kelly = max(0, full_kelly)  # Ensure non-negative
        
        # Use 1/4 Kelly for safety (quarter Kelly)
        fractional_kelly = full_kelly * 0.25
        
        # Convert to percentage of bankroll
        stake_percent = fractional_kelly * 100
        
        # Apply type-specific caps
        if prediction_type.lower() == "match winner":
            max_percent = 2.0  # 2% max for match winner
        elif prediction_type.lower() == "total goals":
            max_percent = 1.5  # 1.5% max for totals
        else:  # BTTS or others
            max_percent = 1.0  # 1% max for others
        
        # Calculate stake units (1 unit = 1% of bankroll)
        stake_units = (stake_percent / 100) * self.bankroll
        max_stake_units = (max_percent / 100) * self.bankroll
        
        # Apply caps
        stake_units = min(stake_units, max_stake_units)
        
        # Minimum stake (0.5% of bankroll)
        min_stake_units = (0.5 / 100) * self.bankroll
        if stake_units < min_stake_units:
            stake_units = 0  # Too small to bother
        
        # Determine risk level and styling
        if stake_units >= (1.0 / 100) * self.bankroll:
            color = "🟢"
            risk = "High"
            emoji = "🔥"
        elif stake_units >= (0.5 / 100) * self.bankroll:
            color = "🟡"
            risk = "Medium"
            emoji = "⚡"
        elif stake_units > 0:
            color = "🟠"
            risk = "Low"
            emoji = "📊"
        else:
            color = "⚪"
            risk = "Very Low"
            emoji = "📉"
        
        return {
            "units": round(stake_units, 2),
            "reason": f"Edge: {(your_prob - implied_prob)*100:.1f}%, Kelly: {fractional_kelly:.2%}",
            "method": "fractional_kelly",
            "edge": round((your_prob - implied_prob) * 100, 1),
            "color": color,
            "risk": risk,
            "emoji": emoji
        }
    
    def _confidence_based_stake(self, confidence: float) -> Dict:
        """Fallback stake calculation when no market odds available"""
        
        # Conservative staking based on confidence only
        if confidence >= 70:
            stake_pct = 1.5
            color = "🟢"
            risk = "High"
            emoji = "🔥"
        elif confidence >= 65:
            stake_pct = 1.0
            color = "🟢"
            risk = "Medium"
            emoji = "⚡"
        elif confidence >= 60:
            stake_pct = 0.75
            color = "🟡"
            risk = "Medium-Low"
            emoji = "📈"
        elif confidence >= 55:
            stake_pct = 0.5
            color = "🟡"
            risk = "Low"
            emoji = "📊"
        elif confidence >= 50:
            stake_pct = 0.25
            color = "🟠"
            risk = "Very Low"
            emoji = "📉"
        else:
            stake_pct = 0
            color = "⚪"
            risk = "AVOID"
            emoji = "🚫"
        
        stake_units = (stake_pct / 100) * self.bankroll
        
        return {
            "units": round(stake_units, 2),
            "reason": f"Confidence-based: {confidence}%",
            "method": "confidence_based",
            "color": color,
            "risk": risk,
            "emoji": emoji
        }
    
    def generate_advice(self, predictions: List[Dict]) -> Dict:
        """Generate betting advice based on predictions"""
        advice = {
            "strong_plays": [],
            "moderate_plays": [],
            "light_plays": [],
            "avoid": [],
            "summary": "",
            "total_units": 0
        }
        
        for pred in predictions:
            # Get stake recommendation
            market_odds = pred.get('market_odds')  # Could be added to prediction dict
            stake_info = self.get_stake_recommendation(
                pred['confidence'],
                market_odds,
                pred['type']
            )
            
            # Update total units
            advice["total_units"] += stake_info["units"]
            
            # Categorize plays
            play_info = {
                "market": pred['type'],
                "selection": pred['selection'],
                "confidence": pred['confidence'],
                "stake": stake_info
            }
            
            if stake_info["units"] >= (1.0 / 100) * self.bankroll:
                advice["strong_plays"].append(play_info)
            elif stake_info["units"] >= (0.5 / 100) * self.bankroll:
                advice["moderate_plays"].append(play_info)
            elif stake_info["units"] > 0:
                advice["light_plays"].append(play_info)
            else:
                advice["avoid"].append(pred['type'])
        
        # Generate summary
        if advice["strong_plays"]:
            advice["summary"] = f"🔥 {len(advice['strong_plays'])} STRONG betting opportunities ({advice['total_units']:.2f} total units)"
        elif advice["moderate_plays"]:
            advice["summary"] = f"⚡ {len(advice['moderate_plays'])} solid betting opportunities ({advice['total_units']:.2f} total units)"
        elif advice["light_plays"]:
            advice["summary"] = f"📊 {len(advice['light_plays'])} light betting opportunities ({advice['total_units']:.2f} total units)"
        else:
            advice["summary"] = "🚫 No betting opportunities identified"
        
        return advice
    
    def update_bankroll(self, new_bankroll: float):
        """Update the current bankroll amount"""
        self.bankroll = max(0, new_bankroll)
    
    def get_risk_report(self) -> Dict:
        """Generate risk management report"""
        return {
            "bankroll": self.bankroll,
            "min_confidence": self.min_confidence,
            "max_single_bet": 0.02 * self.bankroll,  # 2% max
            "max_daily_exposure": 0.05 * self.bankroll,  # 5% max
            "weekly_loss_limit": 0.10 * self.bankroll  # 10% stop loss
        }
