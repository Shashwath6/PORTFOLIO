import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from numpy.linalg import inv

class BlackLittermanEngine:
    """
    Mathematical Core for Black-Litterman (Stage 2).
    Integrates the AI's predictions (Q) securely as Tactical Views into the baseline equilibrium.
    """
    def __init__(self, tau: float = 0.05, risk_aversion: float = 2.5):
        self.tau = tau # Weight of the prior (market baseline vs your views)
        self.delta = risk_aversion
        
    def calculate_posterior_returns(self, 
                                    market_weights: pd.Series, 
                                    cov_matrix: pd.DataFrame, 
                                    views_dict: Dict[str, float], 
                                    confidences: Dict[str, float] = None) -> pd.Series:
        """
        Executes the complex Black Litterman master equation to blend baseline and AI views.
        Formula: E[R] = [ (tau*Cov)^-1 + P^T * Omega^-1 * P ]^-1 * [ (tau*Cov)^-1 * Pi + P^T * Omega^-1 * Q ]
        """
        # Step 1: The Prior (Implied Market Equilibrium Returns)
        # Pi = risk_aversion * Cov * Market_Weights
        Pi = self.delta * cov_matrix.dot(market_weights)
        
        tickers = list(cov_matrix.columns)
        num_assets = len(tickers)
        num_views = len(views_dict)
        
        # If the Deep Learning model hasn't generated any views yet, return the baseline.
        if num_views == 0:
            return Pi
            
        # Step 2: The Views Matrix (P) and AI Predictions (Q)
        valid_views = {k: v for k, v in views_dict.items() if k in tickers}
        num_views = len(valid_views)
        
        if num_views == 0:
            return Pi
            
        P = np.zeros((num_views, num_assets))
        Q = np.zeros(num_views)
        view_tickers = []
        
        for i, (ticker, view_return) in enumerate(valid_views.items()):
            asset_idx = tickers.index(ticker)
            P[i, asset_idx] = 1.0 # This indicates absolute view (not relative "A > B")
            Q[i] = view_return
            view_tickers.append(ticker)
            
        # Step 3: The Uncertainty Matrix (Omega)
        # Instead of guessing confidence, we construct Omega proportional to the variance of the underlying assets.
        # Omega = diag(P * (tau * Cov) * P^T). If specific confidences are passed, we scale them.
        Omega = np.zeros((num_views, num_views))
        tauV = self.tau * cov_matrix.values
        
        for i in range(num_views):
            variance = np.dot(np.dot(P[i], tauV), P[i].T)
            # Higher confidence lowers uncertainty (Omega)
            # FIXED: Map tracking explicitly to view_tickers, not the blind asset ticker list.
            conf_scale = 1.0 if not confidences else (1.0 / confidences.get(view_tickers[i], 1.0))
            Omega[i, i] = variance * conf_scale
            
        # Step 4: Master Bayesian Blending (Matrix Inversions)
        # We split the big equation into Left Hand Side (LHS) and Right Hand Side (RHS) for cleaner calculus
        tauV_inv = inv(tauV)
        Omega_inv = inv(Omega)
        
        LHS = inv(tauV_inv + np.dot(np.dot(P.T, Omega_inv), P))
        RHS = np.dot(tauV_inv, Pi.values) + np.dot(np.dot(P.T, Omega_inv), Q)
        
        posterior_expected_returns = np.dot(LHS, RHS)
        
        return pd.Series(posterior_expected_returns, index=tickers)

if __name__ == "__main__":
    # Internal Unit Test
    print("--- DEPLOYING BLACK-LITTERMAN BAYESIAN ENGINE (UNIT TEST) ---")
    tickers = ["AAPL", "MSFT"]
    weights = pd.Series([0.6, 0.4], index=tickers)
    cov = pd.DataFrame([[0.04, 0.02], [0.02, 0.05]], index=tickers, columns=tickers)
    
    # Simulating AI picking Apple to jump 10%
    ai_predictions = {"AAPL": 0.10}
    ai_confidence = {"AAPL": 0.9} # High confidence
    
    bl = BlackLittermanEngine()
    posterior = bl.calculate_posterior_returns(weights, cov, ai_predictions, ai_confidence)
    
    print(f"[+] Equilibrium Shift Successfully Calculated:\n{posterior.round(4)}")
    print("--- STATUS: BAYESIAN ENGINE OPERATIONAL ---")
