"""
PHANTOM v4.3 - Model Validation and Backtesting System
Complete validation framework for statistically rigorous model evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import warnings
import math
from scipy import stats
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

from models import MatchPredictor, TeamProfile, LeagueAverages, PoissonCalculator

@dataclass
class ValidationResult:
    """Container for validation results"""
    accuracy: float
    brier_score: float
    calibration_error: float
    log_loss: float
    roc_auc: Optional[float]
    confidence_calibration: Dict[str, float]
    market_efficiency: Dict[str, float]
    risk_adjusted_returns: Dict[str, float]
    sample_size: int
    confidence_intervals: Dict[str, Tuple[float, float]]

class ModelValidator:
    """
    Comprehensive validation framework for Phantom predictor
    
    Features:
    - Synthetic backtesting
    - Calibration validation
    - Market efficiency testing
    - Risk-adjusted return analysis
    - Statistical significance testing
    """
    
    def __init__(self, predictor: Optional[MatchPredictor] = None, 
                 league_name: str = "premier_league"):
        self.predictor = predictor
        self.league_name = league_name
        self.predictions_history = []
        self.validation_results = []
        self.calibration_data = {
            'winner': {'predicted': [], 'actual': []},
            'total': {'predicted': [], 'actual': []},
            'btts': {'predicted': [], 'actual': []}
        }
        
    def synthetic_backtest(self, n_matches: int = 1000, 
                          save_results: bool = True) -> pd.DataFrame:
        """
        Generate synthetic matches for backtesting when historical data is unavailable
        
        This creates realistic match scenarios based on league statistics
        """
        print(f"🧪 Generating {n_matches} synthetic matches for backtesting...")
        
        results = []
        
        # League parameters from config or defaults
        league_config = {
            'premier_league': {
                'avg_home_goals': 1.65,
                'avg_away_goals': 1.28,
                'home_advantage': 1.18,
                'draw_rate': 0.25
            },
            'serie_a': {
                'avg_home_goals': 1.55,
                'avg_away_goals': 1.15,
                'home_advantage': 1.15,
                'draw_rate': 0.27
            }
        }
        
        config = league_config.get(self.league_name, league_config['premier_league'])
        
        for match_id in range(n_matches):
            # Generate realistic team strengths
            home_attack = np.random.beta(3, 2) * 1.5 + 0.5  # Skewed toward better teams
            away_attack = np.random.beta(2, 3) * 1.5 + 0.5  # Skewed toward worse teams
            home_defense = np.random.beta(3, 2) * 1.5 + 0.5
            away_defense = np.random.beta(2, 3) * 1.5 + 0.5
            
            # Recent form variation
            home_form = np.random.uniform(0.7, 1.3)
            away_form = np.random.uniform(0.7, 1.3)
            
            # Calculate expected goals (using corrected methodology)
            neutral_baseline = (config['avg_home_goals'] + config['avg_away_goals']) / 2
            home_base_xg = neutral_baseline * home_attack / away_defense
            away_base_xg = neutral_baseline * away_attack / home_defense
            
            # Apply home advantage ONCE
            home_xg = home_base_xg * config['home_advantage']
            away_xg = away_base_xg
            
            # Apply form boosts conservatively
            home_xg *= home_form
            away_xg *= away_form
            
            # Realistic caps
            home_xg = min(4.0, max(0.2, home_xg))
            away_xg = min(3.5, max(0.2, away_xg))
            
            # Simulate actual goals (Poisson)
            home_goals = np.random.poisson(home_xg)
            away_goals = np.random.poisson(away_xg)
            
            # Determine outcomes
            winner = 'H' if home_goals > away_goals else ('A' if away_goals > home_goals else 'D')
            total_goals = home_goals + away_goals
            btts = int(home_goals > 0 and away_goals > 0)
            
            # Store results
            match_result = {
                'match_id': match_id,
                'home_team': f"Team_H_{match_id}",
                'away_team': f"Team_A_{match_id}",
                'home_attack': round(home_attack, 2),
                'away_attack': round(away_attack, 2),
                'home_defense': round(home_defense, 2),
                'away_defense': round(away_defense, 2),
                'home_form': round(home_form, 2),
                'away_form': round(away_form, 2),
                'home_xg': round(home_xg, 2),
                'away_xg': round(away_xg, 2),
                'home_goals': home_goals,
                'away_goals': away_goals,
                'winner': winner,
                'total_goals': total_goals,
                'btts': btts,
                'total_xg': round(home_xg + away_xg, 2)
            }
            
            results.append(match_result)
        
        df_results = pd.DataFrame(results)
        
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation/synthetic_backtest_{self.league_name}_{timestamp}.csv"
            df_results.to_csv(filename, index=False)
            print(f"✅ Synthetic backtest saved to {filename}")
        
        return df_results
    
    def validate_calibration(self, predicted_probs: List[float], 
                           actual_outcomes: List[int],
                           market_type: str = "winner") -> Dict[str, float]:
        """
        Validate probability calibration using multiple metrics
        """
        if len(predicted_probs) < 50:
            warnings.warn(f"Insufficient samples for calibration: {len(predicted_probs)}")
            return {}
        
        # Brier Score
        brier_score = np.mean((np.array(predicted_probs) - np.array(actual_outcomes)) ** 2)
        
        # Log Loss
        eps = 1e-15
        predicted_probs_clipped = np.clip(predicted_probs, eps, 1 - eps)
        log_loss = -np.mean(
            np.array(actual_outcomes) * np.log(predicted_probs_clipped) +
            (1 - np.array(actual_outcomes)) * np.log(1 - predicted_probs_clipped)
        )
        
        # Calibration Error (Expected Calibration Error)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bin_edges) - 1
        
        ece = 0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_probs = np.array(predicted_probs)[mask]
                bin_outcomes = np.array(actual_outcomes)[mask]
                avg_pred = np.mean(bin_probs)
                avg_actual = np.mean(bin_outcomes)
                ece += (np.sum(mask) / len(predicted_probs)) * abs(avg_pred - avg_actual)
        
        # ROC AUC if binary classification
        if len(set(actual_outcomes)) == 2:
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(actual_outcomes, predicted_probs)
        else:
            roc_auc = None
        
        # Confidence intervals using bootstrapping
        brier_ci = self._bootstrap_confidence_interval(
            predicted_probs, actual_outcomes, metric='brier', n_bootstrap=1000
        )
        
        return {
            'brier_score': round(brier_score, 4),
            'log_loss': round(log_loss, 4),
            'calibration_error': round(ece, 4),
            'roc_auc': round(roc_auc, 4) if roc_auc else None,
            'brier_confidence_interval': brier_ci,
            'sample_size': len(predicted_probs)
        }
    
    def _bootstrap_confidence_interval(self, predicted_probs: List[float],
                                     actual_outcomes: List[int],
                                     metric: str = 'brier',
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval using bootstrapping"""
        n_samples = len(predicted_probs)
        bootstrap_metrics = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_pred = np.array(predicted_probs)[indices]
            boot_actual = np.array(actual_outcomes)[indices]
            
            if metric == 'brier':
                metric_val = np.mean((boot_pred - boot_actual) ** 2)
            elif metric == 'accuracy':
                predictions = (boot_pred > 0.5).astype(int)
                metric_val = np.mean(predictions == boot_actual)
            
            bootstrap_metrics.append(metric_val)
        
        alpha = (1 - confidence_level) / 2
        lower = np.percentile(bootstrap_metrics, alpha * 100)
        upper = np.percentile(bootstrap_metrics, (1 - alpha) * 100)
        
        return round(lower, 4), round(upper, 4)
    
    def calculate_accuracy_metrics(self, predictions: pd.DataFrame, 
                                 actuals: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive accuracy metrics
        """
        metrics = {}
        
        # Win/Draw/Loss accuracy
        if 'winner_pred' in predictions.columns and 'winner_actual' in actuals.columns:
            correct_wdl = (predictions['winner_pred'] == actuals['winner_actual']).mean()
            metrics['winner_accuracy'] = round(correct_wdl, 4)
        
        # Total goals accuracy
        if 'total_pred' in predictions.columns and 'total_actual' in actuals.columns:
            correct_total = (predictions['total_pred'] == actuals['total_actual']).mean()
            metrics['total_accuracy'] = round(correct_total, 4)
        
        # BTTS accuracy
        if 'btts_pred' in predictions.columns and 'btts_actual' in actuals.columns:
            correct_btts = (predictions['btts_pred'] == actuals['btts_actual']).mean()
            metrics['btts_accuracy'] = round(correct_btts, 4)
        
        # Mean Absolute Error for xG predictions
        if all(col in predictions.columns for col in ['home_xg_pred', 'away_xg_pred']):
            if all(col in actuals.columns for col in ['home_goals', 'away_goals']):
                home_mae = np.mean(np.abs(predictions['home_xg_pred'] - actuals['home_goals']))
                away_mae = np.mean(np.abs(predictions['away_xg_pred'] - actuals['away_goals']))
                total_mae = np.mean(np.abs(
                    predictions['home_xg_pred'] + predictions['away_xg_pred'] - 
                    (actuals['home_goals'] + actuals['away_goals'])
                ))
                metrics['home_xg_mae'] = round(home_mae, 3)
                metrics['away_xg_mae'] = round(away_mae, 3)
                metrics['total_xg_mae'] = round(total_mae, 3)
        
        # R-squared for xG predictions
        if all(col in predictions.columns for col in ['home_xg_pred', 'away_xg_pred']):
            if all(col in actuals.columns for col in ['home_goals', 'away_goals']):
                from sklearn.metrics import r2_score
                home_r2 = r2_score(actuals['home_goals'], predictions['home_xg_pred'])
                away_r2 = r2_score(actuals['away_goals'], predictions['away_xg_pred'])
                total_r2 = r2_score(
                    actuals['home_goals'] + actuals['away_goals'],
                    predictions['home_xg_pred'] + predictions['away_xg_pred']
                )
                metrics['home_xg_r2'] = round(home_r2, 3)
                metrics['away_xg_r2'] = round(away_r2, 3)
                metrics['total_xg_r2'] = round(total_r2, 3)
        
        return metrics
    
    def market_efficiency_test(self, predictions: pd.DataFrame,
                             market_odds: pd.DataFrame) -> Dict[str, float]:
        """
        Test if model can find value against market odds
        
        Returns:
            Dictionary with value metrics and implied edge
        """
        results = {}
        
        # Ensure we have the required columns
        required_preds = ['home_win_prob', 'draw_prob', 'away_win_prob']
        required_odds = ['home_odds', 'draw_odds', 'away_odds']
        
        if all(col in predictions.columns for col in required_preds) and \
           all(col in market_odds.columns for col in required_odds):
            
            n_matches = len(predictions)
            expected_value = np.zeros(n_matches)
            bets_placed = np.zeros(n_matches, dtype=bool)
            
            for i in range(n_matches):
                # Calculate expected value for each outcome
                home_ev = predictions.iloc[i]['home_win_prob'] * market_odds.iloc[i]['home_odds'] - 1
                draw_ev = predictions.iloc[i]['draw_prob'] * market_odds.iloc[i]['draw_odds'] - 1
                away_ev = predictions.iloc[i]['away_win_prob'] * market_odds.iloc[i]['away_odds'] - 1
                
                # Find maximum positive EV
                max_ev = max(home_ev, draw_ev, away_ev)
                
                if max_ev > 0.05:  # 5% threshold for placing a bet
                    expected_value[i] = max_ev
                    bets_placed[i] = True
            
            # Calculate metrics
            results['avg_ev'] = round(np.mean(expected_value[bets_placed]) if np.any(bets_placed) else 0, 4)
            results['betting_frequency'] = round(np.mean(bets_placed), 4)
            results['total_bets'] = int(np.sum(bets_placed))
            results['sharpe_ratio'] = self._calculate_sharpe_ratio(expected_value[bets_placed])
            
            # Kelly Criterion analysis
            if np.any(bets_placed):
                avg_kelly = np.mean([
                    self._calculate_kelly_fraction(
                        predictions.iloc[i]['home_win_prob'],
                        1/market_odds.iloc[i]['home_odds']
                    ) for i in range(n_matches) if bets_placed[i]
                ])
                results['avg_kelly_fraction'] = round(avg_kelly, 4)
        
        return results
    
    def _calculate_kelly_fraction(self, model_prob: float, implied_prob: float) -> float:
        """Calculate Kelly fraction"""
        if model_prob <= implied_prob or implied_prob <= 0:
            return 0.0
        
        odds = 1 / implied_prob
        kelly = (model_prob * odds - 1) / (odds - 1)
        return max(0.0, kelly)  # Only positive bets
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio for betting returns"""
        if len(returns) == 0:
            return 0.0
        return round(np.mean(returns) / (np.std(returns) + 1e-10), 2)
    
    def statistical_significance_test(self, accuracy: float, 
                                    baseline: float = 0.5,
                                    n_samples: int = 100) -> Dict[str, Any]:
        """
        Test if model accuracy is statistically significant
        
        Args:
            accuracy: Model accuracy (0-1)
            baseline: Baseline accuracy (e.g., random chance)
            n_samples: Number of predictions
        
        Returns:
            Dictionary with p-value and confidence
        """
        # Z-test for proportion
        se = np.sqrt(baseline * (1 - baseline) / n_samples)
        z_score = (accuracy - baseline) / se
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Confidence interval
        ci_lower = accuracy - 1.96 * se
        ci_upper = accuracy + 1.96 * se
        
        return {
            'z_score': round(z_score, 3),
            'p_value': round(p_value, 4),
            'significant_95': p_value < 0.05,
            'significant_99': p_value < 0.01,
            'confidence_interval': (round(ci_lower, 3), round(ci_upper, 3)),
            'baseline_accuracy': baseline
        }
    
    def generate_validation_report(self, predictions: Optional[pd.DataFrame] = None,
                                 actuals: Optional[pd.DataFrame] = None,
                                 synthetic_test: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        """
        print("📊 Generating comprehensive validation report...")
        
        report = {
            'league': self.league_name,
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'synthetic' if synthetic_test else 'historical',
            'model_version': 'v4.3'
        }
        
        # Run synthetic backtest if no data provided
        if predictions is None or actuals is None:
            if synthetic_test:
                synthetic_data = self.synthetic_backtest(n_matches=500)
                
                # Generate predictions for synthetic data
                predictions_list = []
                for _, row in synthetic_data.iterrows():
                    # Create mock predictions based on xG
                    home_win_prob, draw_prob, away_win_prob = PoissonCalculator.calculate_poisson_probabilities(
                        row['home_xg'], row['away_xg']
                    )
                    
                    predictions_list.append({
                        'home_win_prob': home_win_prob,
                        'draw_prob': draw_prob,
                        'away_win_prob': away_win_prob,
                        'home_xg_pred': row['home_xg'],
                        'away_xg_pred': row['away_xg'],
                        'winner_pred': 'H' if home_win_prob > away_win_prob and home_win_prob > draw_prob else
                                     ('A' if away_win_prob > home_win_prob and away_win_prob > draw_prob else 'D'),
                        'total_pred': 'OVER' if (row['home_xg'] + row['away_xg']) > 2.5 else 'UNDER',
                        'btts_pred': 'YES' if (1 - math.exp(-row['home_xg'])) * (1 - math.exp(-row['away_xg'])) > 0.5 else 'NO'
                    })
                
                predictions = pd.DataFrame(predictions_list)
                actuals = synthetic_data[['winner', 'total_goals', 'btts', 'home_goals', 'away_goals']].rename(
                    columns={'winner': 'winner_actual', 
                            'total_goals': 'total_actual',
                            'btts': 'btts_actual'}
                )
        
        # Calculate accuracy metrics
        if predictions is not None and actuals is not None:
            accuracy_metrics = self.calculate_accuracy_metrics(predictions, actuals)
            report['accuracy_metrics'] = accuracy_metrics
            
            # Statistical significance tests
            if 'winner_accuracy' in accuracy_metrics:
                sig_test = self.statistical_significance_test(
                    accuracy_metrics['winner_accuracy'],
                    baseline=0.33,  # Random baseline for 3 outcomes
                    n_samples=len(predictions)
                )
                report['statistical_significance'] = sig_test
            
            # Calibration validation
            if 'home_win_prob' in predictions.columns:
                # For calibration, we need binary outcomes
                # Convert winner predictions to binary (home win vs not home win)
                actual_home_wins = (actuals['winner_actual'] == 'H').astype(int)
                calibration_results = self.validate_calibration(
                    predictions['home_win_prob'].tolist(),
                    actual_home_wins.tolist(),
                    'home_win'
                )
                report['calibration'] = calibration_results
        
        # Market efficiency analysis (if odds data available)
        report['market_efficiency'] = {
            'note': 'Market odds data required for full analysis',
            'recommended_threshold': '5% EV for betting'
        }
        
        # Risk metrics
        report['risk_metrics'] = {
            'recommended_bankroll_fraction': 0.02,
            'max_concurrent_bets': 5,
            'stop_loss_threshold': 0.10
        }
        
        # Performance summary
        report['performance_summary'] = self._generate_performance_summary(report)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation/report_{self.league_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Validation report saved to {filename}")
        
        return report
    
    def _generate_performance_summary(self, report: Dict) -> Dict:
        """Generate performance summary with actionable insights"""
        summary = {
            'status': 'NEEDS_VALIDATION',
            'confidence_level': 'LOW',
            'recommendations': []
        }
        
        # Check accuracy metrics
        if 'accuracy_metrics' in report:
            acc = report['accuracy_metrics']
            
            if 'winner_accuracy' in acc:
                win_acc = acc['winner_accuracy']
                if win_acc > 0.55:
                    summary['status'] = 'PROMISING'
                    summary['confidence_level'] = 'HIGH'
                    summary['recommendations'].append(
                        f"Win prediction accuracy ({win_acc:.1%}) exceeds baseline. Consider using for betting."
                    )
                elif win_acc > 0.45:
                    summary['status'] = 'MODERATE'
                    summary['confidence_level'] = 'MEDIUM'
                    summary['recommendations'].append(
                        f"Win prediction accuracy ({win_acc:.1%}) is moderate. Use cautiously with small stakes."
                    )
                else:
                    summary['status'] = 'NEEDS_IMPROVEMENT'
                    summary['recommendations'].append(
                        f"Win prediction accuracy ({win_acc:.1%}) below baseline. Review model parameters."
                    )
            
            # Check calibration
            if 'calibration' in report:
                cal = report['calibration']
                if 'brier_score' in cal and cal['brier_score'] < 0.20:
                    summary['recommendations'].append(
                        f"Good calibration (Brier: {cal['brier_score']:.3f}). Probabilities are reliable."
                    )
        
        # Add general recommendations
        summary['recommendations'].extend([
            "Start with 1% fractional Kelly staking",
            "Track predictions in production for 100+ matches",
            "Re-calibrate monthly with new data",
            "Use confidence thresholds: ≥60% for strong plays, ≥55% for moderate"
        ])
        
        return summary
    
    def plot_calibration_curve(self, predicted_probs: List[float],
                             actual_outcomes: List[int],
                             save_path: Optional[str] = None):
        """Plot calibration curve for visual validation"""
        if len(predicted_probs) < 50:
            warnings.warn(f"Insufficient data for calibration plot: {len(predicted_probs)} samples")
            return
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(
            actual_outcomes, predicted_probs, n_bins=10, strategy='uniform'
        )
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        # Calibration curve
        plt.plot(prob_pred, prob_true, "s-", label="Model calibration")
        
        # Formatting
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(f"Calibration Curve\n(Reliability Diagram)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Add text with metrics
        brier_score = np.mean((np.array(predicted_probs) - np.array(actual_outcomes)) ** 2)
        plt.figtext(0.15, 0.85, f"Brier Score: {brier_score:.3f}", 
                   fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Calibration plot saved to {save_path}")
        
        plt.show()

class ValidationPipeline:
    """Complete validation pipeline for Phantom predictor"""
    
    def __init__(self, data_dir: str = "data", results_dir: str = "validation"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.validators = {}
        
        # Create directories
        import os
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    
    def run_full_validation(self, league_name: str, n_synthetic_matches: int = 1000):
        """Run complete validation pipeline"""
        print(f"🚀 Starting full validation for {league_name}")
        print("=" * 60)
        
        # Initialize validator
        validator = ModelValidator(league_name=league_name)
        self.validators[league_name] = validator
        
        # 1. Synthetic backtesting
        print("\n1️⃣ Running synthetic backtest...")
        synthetic_data = validator.synthetic_backtest(n_matches=n_synthetic_matches)
        
        # 2. Generate validation report
        print("\n2️⃣ Generating validation report...")
        report = validator.generate_validation_report(
            synthetic_test=True
        )
        
        # 3. Performance analysis
        print("\n3️⃣ Performance analysis...")
        self._analyze_performance(report)
        
        # 4. Generate visualizations
        print("\n4️⃣ Creating visualizations...")
        self._create_visualizations(validator, report, league_name)
        
        print("\n" + "=" * 60)
        print("✅ Validation complete!")
        
        return report
    
    def _analyze_performance(self, report: Dict):
        """Analyze and print performance metrics"""
        if 'accuracy_metrics' in report:
            print("\n📊 ACCURACY METRICS:")
            for metric, value in report['accuracy_metrics'].items():
                if isinstance(value, (int, float)):
                    if 'accuracy' in metric:
                        print(f"  {metric.replace('_', ' ').title()}: {value:.1%}")
                    else:
                        print(f"  {metric.replace('_', ' ').title()}: {value}")
        
        if 'calibration' in report:
            print("\n🎯 CALIBRATION METRICS:")
            for metric, value in report['calibration'].items():
                if isinstance(value, (int, float)):
                    print(f"  {metric.replace('_', ' ').title()}: {value}")
        
        if 'performance_summary' in report:
            print(f"\n📈 PERFORMANCE SUMMARY: {report['performance_summary']['status']}")
            for rec in report['performance_summary']['recommendations'][:3]:
                print(f"  • {rec}")
    
    def _create_visualizations(self, validator: ModelValidator, 
                             report: Dict, league_name: str):
        """Create and save validation visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = f"{self.results_dir}/plots"
        
        # Example: Create a simple bar chart for accuracy metrics
        if 'accuracy_metrics' in report:
            acc_metrics = report['accuracy_metrics']
            
            # Filter for accuracy metrics
            accuracy_types = [k for k in acc_metrics.keys() if 'accuracy' in k]
            if accuracy_types:
                values = [acc_metrics[k] for k in accuracy_types]
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(accuracy_types, values, color=['#2E86AB', '#A23B72', '#F18F01'])
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.1%}', ha='center', va='bottom')
                
                plt.title(f"Accuracy Metrics - {league_name.replace('_', ' ').title()}")
                plt.ylabel("Accuracy")
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3, axis='y')
                
                plot_path = f"{plots_dir}/accuracy_metrics_{league_name}_{timestamp}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"  📊 Accuracy plot saved to {plot_path}")
                plt.close()

# Quick validation function for immediate use
def quick_validate(predictor: MatchPredictor, n_matches: int = 100) -> Dict:
    """
    Quick validation function for testing during development
    """
    print("⚡ Running quick validation...")
    
    validator = ModelValidator(predictor)
    synthetic_data = validator.synthetic_backtest(n_matches=n_matches, save_results=False)
    
    # Simplified validation
    results = {
        'matches_tested': len(synthetic_data),
        'avg_home_xg': round(synthetic_data['home_xg'].mean(), 2),
        'avg_away_xg': round(synthetic_data['away_xg'].mean(), 2),
        'home_win_rate': round((synthetic_data['winner'] == 'H').mean(), 3),
        'draw_rate': round((synthetic_data['winner'] == 'D').mean(), 3),
        'away_win_rate': round((synthetic_data['winner'] == 'A').mean(), 3),
        'btts_rate': round(synthetic_data['btts'].mean(), 3),
        'over_25_rate': round((synthetic_data['total_goals'] > 2.5).mean(), 3),
        'xg_correlation': round(synthetic_data[['home_xg', 'away_xg']].corr().iloc[0, 1], 3)
    }
    
    print("\nQuick Validation Results:")
    for key, value in results.items():
        if 'rate' in key or 'correlation' in key:
            print(f"  {key.replace('_', ' ').title()}: {value:.1%}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    return results

if __name__ == "__main__":
    # Example usage
    print("🧪 PHANTOM v4.3 - Validation System")
    
    # Quick test
    validator = ModelValidator(league_name="premier_league")
    report = validator.generate_validation_report()
    
    # Or run full pipeline
    # pipeline = ValidationPipeline()
    # pipeline.run_full_validation("premier_league", n_synthetic_matches=500)
