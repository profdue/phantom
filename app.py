"""
Main application entry point
"""
import argparse
from data_loader import DataLoader
from models import MatchPredictor, TeamProfile
from betting_advisor import BettingAdvisor
from utils import PredictionUtils

class PhantomPredictor:
    """Main application class"""
    
    def __init__(self, data_dir="data"):
        self.data_loader = DataLoader(data_dir)
        self.utils = PredictionUtils()
        self.advisor = BettingAdvisor()
    
    def predict_match(self, league_name, home_team_name, away_team_name):
        """Make prediction for a specific match"""
        try:
            # Load team profiles
            home_team = self.data_loader.get_team_profile(home_team_name, league_name, is_home=True)
            away_team = self.data_loader.get_team_profile(away_team_name, league_name, is_home=False)
            
            # Create predictor
            predictor = MatchPredictor(league_name)
            
            # Generate prediction
            result = predictor.predict(home_team, away_team)
            
            # Add team names to analysis
            result['analysis']['home_team'] = home_team_name
            result['analysis']['away_team'] = away_team_name
            
            # Generate betting advice
            advice = self.advisor.generate_advice(result['predictions'])
            result['betting_advice'] = advice
            
            # Format output
            formatted = self.utils.format_prediction_output(result)
            
            return formatted
            
        except Exception as e:
            return {"error": str(e)}
    
    def list_available_leagues(self):
        """List all available leagues"""
        return list(self.data_loader.available_leagues.keys())
    
    def list_teams_in_league(self, league_name):
        """List all teams in a league"""
        return self.data_loader.get_all_teams(league_name)
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("⚽ PHANTOM PREDICTOR v2.3 ⚽")
        print("=" * 40)
        
        # List leagues
        leagues = self.list_available_leagues()
        print("\nAvailable Leagues:")
        for i, league in enumerate(leagues, 1):
            print(f"{i}. {league}")
        
        # Select league
        league_choice = int(input("\nSelect league number: ")) - 1
        league_name = leagues[league_choice]
        
        # List teams
        teams = self.list_teams_in_league(league_name)
        print(f"\nTeams in {league_name}:")
        for i, team in enumerate(teams, 1):
            print(f"{i}. {team}")
        
        # Select teams
        home_idx = int(input("\nSelect home team number: ")) - 1
        away_idx = int(input("Select away team number: ")) - 1
        
        home_team = teams[home_idx]
        away_team = teams[away_idx]
        
        print(f"\n🔮 Predicting: {home_team} vs {away_team}")
        print("=" * 40)
        
        # Make prediction
        result = self.predict_match(league_name, home_team, away_team)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        # Display results
        self._display_prediction(result)
    
    def _display_prediction(self, result):
        """Display prediction results nicely"""
        analysis = result['analysis']
        predictions = result['predictions']
        advice = result['betting_advice']
        
        print(f"\n📊 ANALYSIS ({analysis['league']})")
        print(f"Quality Ratings: {analysis['quality_ratings']['home']} vs {analysis['quality_ratings']['away']}")
        print(f"Expected Goals: {analysis['expected_goals']['home']} - {analysis['expected_goals']['away']} (Total: {analysis['expected_goals']['total']})")
        
        print(f"\n🎯 PREDICTIONS")
        for pred in predictions:
            stake = BettingAdvisor.get_stake_recommendation(pred['confidence'])
            print(f"{pred['type']}: {pred['selection']} ({pred['confidence']}%) {stake['color']} {stake['units']} units")
        
        print(f"\n💰 BETTING ADVICE")
        print(advice['summary'])
        
        if advice['strong_plays']:
            print("\nStrong Plays:")
            for play in advice['strong_plays']:
                print(f"  • {play['market']}: {play['selection']}")
        
        if advice['moderate_plays']:
            print("\nModerate Plays:")
            for play in advice['moderate_plays']:
                print(f"  • {play['market']}: {play['selection']}")
        
        if advice['avoid']:
            print(f"\nAvoid: {', '.join(advice['avoid'])}")
        
        print(f"\n📈 Expected Scoreline: Based on xG, expect a {self._estimate_scoreline(analysis['expected_goals'])} result")
    
    def _estimate_scoreline(self, xg_data):
        """Estimate most likely scoreline from xG"""
        home_xg = xg_data['home']
        away_xg = xg_data['away']
        
        # Simple rounding for display
        home_est = round(home_xg)
        away_est = round(away_xg)
        
        # Ensure at least 1 goal if xG > 0.5
        if home_xg > 0.5 and home_est == 0:
            home_est = 1
        if away_xg > 0.5 and away_est == 0:
            away_est = 1
        
        return f"{home_est}-{away_est}"


def main():
    parser = argparse.ArgumentParser(description="Phantom Prediction System v2.3")
    parser.add_argument("--league", help="League name")
    parser.add_argument("--home", help="Home team name")
    parser.add_argument("--away", help="Away team name")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    predictor = PhantomPredictor()
    
    if args.interactive:
        predictor.interactive_mode()
    elif args.league and args.home and args.away:
        result = predictor.predict_match(args.league, args.home, args.away)
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            predictor._display_prediction(result)
    else:
        print("Please use --interactive or provide --league, --home, and --away arguments")
        print("Example: python app.py --league premier_league --home \"Man City\" --away \"Liverpool\"")

if __name__ == "__main__":
    main()
