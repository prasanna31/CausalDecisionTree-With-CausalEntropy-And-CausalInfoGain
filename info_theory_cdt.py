from base_model import BaseCausalDecisionTree
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression

class InformationTheoryCDT(BaseCausalDecisionTree):
    """
    Concrete implementation of CDT using Causal Information Theory 
    metrics: Causal Entropy and Causal Information Gain.
    """

    def __init__(self, max_height=5, min_samples=10, alpha=0.05, mode='PS', sps_groups=5, min_ig=1e-4):
        super().__init__(max_height, min_samples)
        self.alpha = alpha
        self.mode = mode
        self.sps_groups = sps_groups
        self.min_information_gain = min_ig

    def _find_best_split(self, current_data, available_attrs):
        # 1. Filter Relevant Attributes via Correlation (Algorithm 1, Line 5)
        relevant_attrs = []
        for attr in available_attrs:
            if current_data[attr].nunique() > 1 and current_data[self.target].nunique() > 1:
                try:
                    _, p_value = pearsonr(current_data[attr], current_data[self.target])
                    if p_value <= self.alpha:
                        relevant_attrs.append(attr)
                except:
                    continue
        
        if not relevant_attrs:
            return None, {}

        # 2. Calculate Parent Entropy H(Y) (Standard Shannon Entropy)
        # Definition 5: Ic = H(Y) - Hc 
        current_entropy = self._calculate_shannon_entropy(current_data)
        
        best_attr = None
        best_information_gain = -float('inf') # Initialize low as Ic can be negative [cite: 1163]
        best_meta = {}

        for X_i in available_attrs: 
            # Identify covariates Z \ {Xi}
            stratifying_cols = [col for col in relevant_attrs if col != X_i]
            
            # 3. Perform Stratification
            # We must group the data before calculating causal probabilities
            try:
                if self.mode == 'PS':
                    strata = self._perfect_stratification(current_data, stratifying_cols)
                else: 
                    strata = self._propensity_stratification(current_data, X_i, stratifying_cols)
            except:
                continue 

            # 4. Calculate Causal Entropy Hc(Y | do(X~X')) 
            causal_entropy = self._calculate_causal_entropy(strata, X_i, current_data)
            
            # 5. Calculate Causal Information Gain Ic 
            information_gain = current_entropy - causal_entropy
            
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_attr = X_i
                best_meta = {
                    'Causal_IG': round(information_gain, 4),
                    'H(Y)': round(current_entropy, 4),
                    'Hc': round(causal_entropy, 4)
                }

        # Threshold check
        if best_attr is None or best_information_gain < self.min_information_gain:
            return None, {}

        return best_attr, best_meta                                                                                                          

    def _calculate_shannon_entropy(self, data):
        """Calculates standard H(Y) [cite: 1009]"""
        if len(data) == 0: return 0.0
        p1 = data[self.target].mean()
        if p1 == 0 or p1 == 1: return 0.0
        return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

    def _calculate_causal_entropy(self, strata, X_i, current_data):
        """
        Calculates Hc(Y | do(X~X')) = E[ H(Y | do(X=x)) ] 
        """
        # 1. Calculate P(Y=1 | do(X=1)) and P(Y=1 | do(X=0))
        do_probs = self._calculate_causal_probabilities(strata, X_i)
        
        causal_entropy = 0.0
        
        # 2. Iterate through intervention values x in {0, 1}
        for x_val in [0, 1]:
            # Get P(Y=1 | do(X=x_val))
            p_y_do_x = do_probs[x_val]
            
            # Compute Entropy H(Y | do(X=x_val))
            if p_y_do_x <= 0 or p_y_do_x >= 1:
                h_do_x = 0.0
            else:
                h_do_x = -p_y_do_x * np.log2(p_y_do_x) - (1 - p_y_do_x) * np.log2(1 - p_y_do_x)
            
            p_prime_x = len(current_data[current_data[X_i] == x_val]) / len(current_data)
            
            causal_entropy += p_prime_x * h_do_x
            
        return causal_entropy

    def _calculate_causal_probabilities(self, strata, X_i):
        """
        Standard Adjustment Formula to get P(Y | do(X))
        """
        total_samples = sum(len(s) for s in strata)
        if total_samples == 0: return {1: 0.0, 0: 0.0}

        prob_do_1 = 0.0
        prob_do_0 = 0.0

        for stratum in strata:
            n_k = len(stratum)
            if n_k == 0: continue
            w_k = n_k / total_samples
            
            # E[Y | X=1, S=s]
            treated = stratum[stratum[X_i] == 1]
            p_y1_given_x1 = treated[self.target].mean() if len(treated) > 0 else 0.0
            
            # E[Y | X=0, S=s]
            control = stratum[stratum[X_i] == 0]
            p_y1_given_x0 = control[self.target].mean() if len(control) > 0 else 0.0

            prob_do_1 += w_k * p_y1_given_x1
            prob_do_0 += w_k * p_y1_given_x0

        return {1: prob_do_1, 0: prob_do_0}

    def _perfect_stratification(self, data, stratifying_cols):
        if not stratifying_cols: return [data]
        return [group for _, group in data.groupby(stratifying_cols)]

    def _propensity_stratification(self, data, treatment_col, covariates):
        if not covariates or data[treatment_col].nunique() < 2: return [data]
        try:
            model = LogisticRegression(solver='liblinear', max_iter=200)
            model.fit(data[covariates], data[treatment_col])
            scores = model.predict_proba(data[covariates])[:, 1]
            temp = data.copy()
            temp['stratum'] = pd.qcut(scores, self.sps_groups, duplicates='drop')
            return [group for _, group in temp.groupby('stratum')]
        except:
            return [data]
