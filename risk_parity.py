import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict

class RiskParityModel:
    """
    Mathematical Core for Risk Parity (Stage 3).
    Ignores expected returns entirely. Optimizes weights so that every asset 
    contributes the exact same amount of risk (volatility) to the overall portfolio.
    This creates an 'All-Weather' defensive structure.
    """
    def _calculate_portfolio_variance(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    def _calculate_risk_contribution(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Risk Contribution = Weight * Marginal Risk
        RC_i = w_i * (Cov * W)_i / sqrt(W^T * Cov * W)
        """
        portfolio_var = self._calculate_portfolio_variance(weights, cov_matrix)
        marginal_risk = np.dot(cov_matrix, weights)
        # Adding epsilon (1e-8) to prevent ZeroDivisionError during edge-case flat volatility
        risk_contribution = np.multiply(weights, marginal_risk) / (np.sqrt(portfolio_var) + 1e-8)
        return risk_contribution

    def _risk_budget_objective(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        """
        Objective function: sum of squared differences between asset risk contribution 
        and the target risk contribution (which is equal for all assets: 1/N).
        Minimizing this forces all RC_i to equal 1/N.
        """
        num_assets = len(weights)
        target_risk_contribution = 1.0 / num_assets
        
        # Calculate actual risk contributions (normalized as a percentage of total risk)
        portfolio_vol = np.sqrt(self._calculate_portfolio_variance(weights, cov_matrix)) + 1e-8
        risk_contribution = self._calculate_risk_contribution(weights, cov_matrix)
        risk_contribution_pct = risk_contribution / portfolio_vol
        
        # SSE (Sum of Squared Errors) penalty
        error = np.sum(np.square(risk_contribution_pct - target_risk_contribution))
        return error

    def generate_all_weather_weights(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Solves for Equal Risk Contribution (ERC).
        """
        num_assets = len(cov_matrix)
        
        # Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
        # Bounds: Long-only, no leverage
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        # Initial guess: Equal weighting
        initial_guess = np.array(num_assets * [1.0 / num_assets])
        
        # Optimization
        result = minimize(self._risk_budget_objective, initial_guess, args=(cov_matrix,), 
                          method='SLSQP', bounds=bounds, constraints=constraints)
                          
        if not result.success:
            print("[!] Warning: Risk Parity optimization failed to converge exactly.")
            
        return pd.Series(result.x, index=cov_matrix.index)

if __name__ == "__main__":
    # Internal Unit Test
    print("--- DEPLOYING RISK PARITY OPTIMIZER (UNIT TEST) ---")
    tickers = ["BOND", "EQUITY", "GOLD"]
    # Simulated covariance. Equities have high variance, Bonds very low variance.
    mock_cov = pd.DataFrame([[0.01, 0.002, 0.001], 
                             [0.002, 0.15, 0.03], 
                             [0.001, 0.03, 0.08]], index=tickers, columns=tickers)
                             
    rp = RiskParityModel()
    weights = rp.generate_all_weather_weights(mock_cov)
    
    print(f"\n[+] Ideal 'All Weather' Weights Allocation:")
    print(weights.round(4))
    
    # Mathematical Proof: Check the risk contributions
    rc = rp._calculate_risk_contribution(weights.values, mock_cov)
    vol = np.sqrt(rp._calculate_portfolio_variance(weights.values, mock_cov))
    rc_pct = rc / vol
    
    print("\n[+] Risk Contribution Verification (Should all be ~33.3%):")
    for i, t in enumerate(tickers):
        print(f"{t}: {rc_pct[i]*100:.2f}% of Total Risk")
        
    print("\n--- STATUS: RISK PARITY SHIELD OPERATIONAL ---")
