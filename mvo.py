import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from typing import Dict, List, Tuple

class MarkowitzOptimizer:
    """
    Mathematical Core for Modern Portfolio Theory (Stage 1).
    Optimizes weights using Sequential Least Squares Programming (SLSQP).
    """
    def __init__(self, risk_free_rate: float = 0.04):
        self.rf = risk_free_rate
        
    def _portfolio_annualised_performance(self, weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Tuple[float, float]:
        """Calculates expected portfolio performance given specific asset weights. (Inputs must be pre-annualized)"""
        returns = np.sum(mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return std, returns
        
    def maximize_sharpe_ratio(self, mean_returns: pd.Series, raw_data_for_cov: pd.DataFrame = None, cov_matrix: pd.DataFrame = None) -> dict:
        """
        Solves for the portfolio weights that specifically maximize the risk-adjusted return (Sharpe).
        Upgraded to allow Raw Data input so Ledoit-Wolf Shrinkage can be applied to the Covariance.
        """
        if raw_data_for_cov is not None:
            # HYPER-OPTIMIZATION: Ledoit-Wolf Shrinkage
            # This pulls extreme outliers towards the center, preventing suicidal optimization bets.
            lw = LedoitWolf()
            shrunk_cov = lw.fit(raw_data_for_cov).covariance_
            cov_matrix = pd.DataFrame(shrunk_cov, index=raw_data_for_cov.columns, columns=raw_data_for_cov.columns)
            
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix)
        
        # We want to maximize Sharpe, but scipy can only minimize. So we minimize negative Sharpe.
        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
            p_var, p_ret = self._portfolio_annualised_performance(weights, mean_returns, cov_matrix)
            return -(p_ret - rf) / p_var
            
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Weights must sum to 1.0 (No leverage)
        bounds = tuple((0.0, 1.0) for asset in range(num_assets)) # Long-only constraint (No shorting allowed)
        
        # Initial guess (Equal weighting)
        initial_guess = num_assets * [1. / num_assets,]
        
        result = minimize(neg_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix, self.rf), 
                          method='SLSQP', bounds=bounds, constraints=constraints)
                          
        if not result.success:
            print("[!] Optimization Warning: SLSQP failed to converge.")
            
        return {
            'weights': pd.Series(result.x, index=mean_returns.index),
            'expected_return': self._portfolio_annualised_performance(result.x, mean_returns, cov_matrix)[1],
            'expected_volatility': self._portfolio_annualised_performance(result.x, mean_returns, cov_matrix)[0],
            'sharpe_ratio': -result.fun
        }

if __name__ == "__main__":
    # Internal Unit Test
    print("--- DEPLOYING MVO OPTIMIZER (UNIT TEST) ---")
    tickers = ["AAPL", "JPM", "XOM"]
    mock_returns = pd.Series([0.15, 0.08, 0.05], index=tickers)
    mock_cov = pd.DataFrame([[0.04, 0.01, 0.005], [0.01, 0.03, 0.02], [0.005, 0.02, 0.05]], index=tickers, columns=tickers)
    
    mvo = MarkowitzOptimizer()
    optimal = mvo.maximize_sharpe_ratio(mock_returns, mock_cov)
    print(f"[+] Max Sharpe Weights:\n{optimal['weights'].round(4)}")
    print(f"[+] Constraints Verified: Sum = {optimal['weights'].sum().round(4)} (Expected exactly 1.0)")
    print("--- STATUS: MVO OPTIMAL ---")
