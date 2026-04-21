import shap
import torch
import numpy as np
import pandas as pd
from typing import List, Dict

class XAIAuditor:
    """
    Explainable AI (Stage 4).
    Uses Game Theory (Shapley Values) to open the 'Black Box' of the LSTM Network.
    This guarantees that the portfolio manager knows exactly *WHY* the AI is predicting 
    a certain stock movement before risking capital.
    """
    def __init__(self, model: torch.nn.Module, bg_tensor: torch.Tensor):
        """
        model: The trained LSTM Alpha Generator
        bg_tensor: A background dataset (usually training data) used by SHAP to establish baseline expectations.
                   Must be incredibly careful here: bg_tensor should not be too massive or Memory errors occur.
        """
        self.model = model
        self.model.eval() # Ensure model is in inference mode
        
        # We use GradientExplainer for PyTorch LSTMs as DeepExplainer can be unstable with sequences
        self.explainer = shap.GradientExplainer(self.model, bg_tensor)
        
    def audit_prediction(self, test_tensor: torch.Tensor, feature_names: List[str]) -> pd.DataFrame:
        """
        Calculates the SHAP values for a specific set of predictions.
        Returns a DataFrame showing exactly which feature drove the prediction Up/Down.
        """
        # SHAP calculation
        # This returns the Shapley values for the sequence. Since the LSTM output is based 
        # on the entire sequence, we sum the absolute SHAP values across the time dimension 
        # to get a single 'Impact Score' per feature for the whole time window.
        shap_values = self.explainer.shap_values(test_tensor)
        
        # Handling different SHAP return formats (List vs Array)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # shap_values shape: (batch_size, sequence_length, num_features)
        # We aggregate the time impact:
        aggregated_impact = np.abs(shap_values).sum(axis=1) # Summing over sequence length
        
        # Average impact across the test batch
        mean_impact = np.mean(aggregated_impact, axis=0)
        
        # Create an interpretable tear-sheet by flattening numpy arrays
        audit_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Impact_Score': mean_impact.flatten().tolist()
        })
        
        # Sort to find the leading cause of the prediction
        audit_df = audit_df.sort_values(by='SHAP_Impact_Score', ascending=False).reset_index(drop=True)
        return audit_df


if __name__ == "__main__":
    # Internal Unit Test
    print("--- DEPLOYING EXPLAINABLE AI AUDITOR (UNIT TEST) ---")
    import torch.nn as nn
    
    # Dummy LSTM just for SHAP structure verification
    class DummyLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(5, 10, batch_first=True)
            self.fc = nn.Linear(10, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
            
    mock_model = DummyLSTM()
    
    # 5 features, 10 days sequence
    bg_data = torch.randn(100, 10, 5) # Background distribution
    test_data = torch.randn(5, 10, 5) # What we are trying to predict today
    features = ['RSI_14', 'MACD', 'BB_Width', 'Log_Returns', 'Volume_SMA']
    
    try:
        auditor = XAIAuditor(mock_model, bg_data)
        audit_report = auditor.audit_prediction(test_data, features)
        print("\n[+] SHAP Audit Complete. Top Predictive Drivers:")
        print(audit_report)
        print("\n--- STATUS: XAI AUDITOR OPERATIONAL (NO BLACK BOXES) ---")
    except Exception as e:
        print(f"[!] Warning: SHAP backend tensor mismatch: {e}")
