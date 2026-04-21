import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform

class HierarchicalRiskParity:
    """
    Mathematical Core for HRP (Stage 4).
    Uses Unsupervised Machine Learning (Agglomerative Clustering) to allocate capital.
    Solves the 'Invertibility Problem' of standard covariance matrices by grouping assets hierarchically.
    """
    def _get_distance_matrix(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """Converts correlations into mathematical distances (0 = identical, 1 = perfectly inverse)."""
        # Clip to handle floating point imprecision
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        dist_matrix_vals = np.sqrt(0.5 * (1 - corr_matrix)).values.copy()
        
        # EXACT MATHEMATICAL CONSTRAINT: 
        # Scipy's squareform strictly rejects matrices where the diagonal is not EXACTLY 0.0.
        # Floating point imprecision (like 1e-16) causes catastrophic pipeline failure.
        np.fill_diagonal(dist_matrix_vals, 0.0)
        
        dist_matrix = pd.DataFrame(dist_matrix_vals, index=corr_matrix.index, columns=corr_matrix.columns)
        return dist_matrix

    def _get_quasi_diag(self, link: np.ndarray) -> list:
        """Sorts the clustered items. Leaves belong to the same cluster."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # Make space
            df0 = sort_ix[sort_ix >= num_items]  # Find clusters
            i = df0.index
            j = df0.values - num_items
            sort_ix.loc[i] = link[j, 0]  # Item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])  # Item 2
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def _get_cluster_var(self, cov_matrix: pd.DataFrame, cluster_items: list) -> float:
        """Calculates variance of a cluster as if it were a single asset."""
        cov_slice = cov_matrix.iloc[cluster_items, cluster_items]
        # Inverse-variance risk parity weight within the cluster
        ivp = 1.0 / np.diag(cov_slice)
        ivp /= ivp.sum()
        return np.dot(ivp.T, np.dot(cov_slice, ivp))

    def _get_rec_bipart(self, cov_matrix: pd.DataFrame, sort_ix: list) -> pd.Series:
        """Recursive Bisection to distribute risk top-down through the tree."""
        weights = pd.Series(1.0, index=sort_ix)
        clusters = [sort_ix]
        while len(clusters) > 0:
            clusters = [i[j:k] for i in clusters for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(clusters), 2):
                cluster_1 = clusters[i]
                cluster_2 = clusters[i + 1]
                var_1 = self._get_cluster_var(cov_matrix, cluster_1)
                var_2 = self._get_cluster_var(cov_matrix, cluster_2)
                alpha = 1.0 - var_1 / (var_1 + var_2)  # Capital allocation factor
                weights[cluster_1] *= alpha
                weights[cluster_2] *= (1.0 - alpha)
        return weights

    def generate_hrp_weights(self, cov_matrix: pd.DataFrame, corr_matrix: pd.DataFrame = None) -> pd.Series:
        """The Master Execution method."""
        if corr_matrix is None:
            # Reconstruct correlation from covariance
            vols = np.sqrt(np.diag(cov_matrix))
            corr_matrix = cov_matrix / np.outer(vols, vols)
            
        dist_matrix = self._get_distance_matrix(corr_matrix)
        # 1. Tree Clustering (Unsupervised ML)
        condensed_dist = squareform(dist_matrix, checks=False)
        link = sch.linkage(condensed_dist, method='single')
        # 2. Quasi-Diagonalization
        sort_ix = self._get_quasi_diag(link)
        # 3. Recursive Bisection
        weights = self._get_rec_bipart(cov_matrix, sort_ix)
        
        # Map indexed weights back to specific Ticker strings
        hrp_weights = weights.sort_index()
        hrp_weights.index = cov_matrix.index
        return hrp_weights

if __name__ == "__main__":
    # Internal Unit Test
    print("--- DEPLOYING MACHINE LEARNING HRP OPTIMIZER (UNIT TEST) ---")
    tickers = ["AAPL", "MSFT", "GOOG", "TSLA"]
    raw_data = np.abs(np.random.randn(4, 4))
    raw_data = (raw_data + raw_data.T) / 2 # Ensure symmetry
    np.fill_diagonal(raw_data, np.abs(np.random.randn(4)))
    mock_cov = pd.DataFrame(raw_data, index=tickers, columns=tickers)
    
    hrp = HierarchicalRiskParity()
    weights = hrp.generate_hrp_weights(mock_cov)
    
    print("\n[+] Unsupervised Machine Learning Allocation via Recursive Bisection:")
    print(weights.round(4))
    print(f"\n[+] Stability Check: Sum of Weights = {weights.sum():.6f}")
    print("--- STATUS: HRP MACHINE ENGINE OPERATIONAL ---")
