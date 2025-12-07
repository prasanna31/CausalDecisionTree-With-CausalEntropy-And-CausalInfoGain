from base_model import BaseCausalDecisionTree
import pandas as pd
from scipy.stats import chi2, pearsonr
from sklearn.linear_model import LogisticRegression

class MantelHaenszelCDT(BaseCausalDecisionTree):
    """
    Concrete implementation of CDT using the Mantel-Haenszel Test
    """

    def __init__(self, max_height=5, min_samples=10, alpha=0.05, mode='PS', sps_groups=5):
        super().__init__(max_height, min_samples)
        self.alpha = alpha
        self.mode = mode
        self.sps_groups = sps_groups
        self.chi2_threshold = chi2.ppf(1 - alpha, df=1)

    def _find_best_split(self, current_data, available_attrs):
        """
        Implements the logic from Algorithm 1:
        1. Filter by Correlation.
        2. Stratify.
        3. Calculate MH Statistic.
        4. Check Significance.
        """
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

        best_mh_stat = -1
        best_attr = None
        best_strata = None

        for X_i in available_attrs: 
            stratifying_cols = [col for col in relevant_attrs if col != X_i]
            
            try:
                if self.mode == 'PS':
                    strata = self._perfect_stratification(current_data, stratifying_cols)
                else: 
                    strata = self._propensity_stratification(current_data, X_i, stratifying_cols)
            except:
                continue 

            pamh_score = self._calculate_mh_statistic(strata, X_i)

            if pamh_score > best_mh_stat:
                best_mh_stat = pamh_score
                best_attr = X_i
                best_strata = strata

        if best_attr is None or best_mh_stat < self.chi2_threshold:
            return None, {}

        # Calculate Causal Probabilities
        do_probs = self._calculate_causal_probabilities(best_strata, best_attr)
        ace = do_probs[1] - do_probs[0]

        meta = {
            'MH_Stat': round(best_mh_stat, 2),
            'ACE': round(ace, 3),
            'P(do(X=1))': round(do_probs[1], 3),
            'P(do(X=0))': round(do_probs[0], 3)
        }
        
        return best_attr, meta


    def _calculate_mh_statistic(self, strata, X_i):
        numerator_sum = 0
        denominator_sum = 0
        
        for stratum in strata:
            if len(stratum) < 2: continue
            
            n11 = len(stratum[(stratum[X_i] == 1) & (stratum[self.target] == 1)])
            n12 = len(stratum[(stratum[X_i] == 1) & (stratum[self.target] == 0)])
            n21 = len(stratum[(stratum[X_i] == 0) & (stratum[self.target] == 1)])
            n22 = len(stratum[(stratum[X_i] == 0) & (stratum[self.target] == 0)])
            
            n_k = n11 + n12 + n21 + n22
            if n_k <= 1: continue
            num_term = (n11 * n22 - n21 * n12) / n_k
            denom_term = ((n11 + n12) * (n21 + n22) * (n11 + n21) * (n12 + n22)) / ((n_k**2) * (n_k - 1))
            
            numerator_sum += num_term
            denominator_sum += denom_term

        if denominator_sum == 0: return 0
        return (abs(numerator_sum) - 0.5)**2 / denominator_sum

    def _calculate_causal_probabilities(self, strata, X_i):
        total_samples = sum(len(s) for s in strata)
        if total_samples == 0: return {1: 0.0, 0: 0.0}

        prob_do_1 = 0.0
        prob_do_0 = 0.0

        for stratum in strata:
            n_k = len(stratum)
            if n_k == 0: continue
            w_k = n_k / total_samples
            
            treated = stratum[stratum[X_i] == 1]
            p_y1_given_x1 = treated[self.target].mean() if len(treated) > 0 else 0.0
            
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
