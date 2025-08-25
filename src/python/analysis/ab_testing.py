"""
A/B Testing and Experimental Design Framework
Statistical testing for marketing experiments
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestPower
from statsmodels.stats.proportion import proportions_ztest
import hashlib


class ABTestingFramework:
    """A/B testing with proper experimental design"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def assign_variant(self, customer_id: int, test_name: str = 'default') -> str:
        """Deterministic variant assignment using hashing"""
        hash_input = f"{test_name}_{customer_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        return 'control' if hash_value % 2 == 0 else 'treatment'
    
    def calculate_sample_size(self, baseline_rate: float, mde: float = 0.05, 
                            power: float = 0.8) -> int:
        """Calculate required sample size for experiment"""
        effect_size = mde / np.sqrt(baseline_rate * (1 - baseline_rate))
        analysis = TTestPower()
        sample_size = analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=self.alpha
        )
        return int(np.ceil(sample_size))
    
    def run_proportion_test(self, control_conversions: int, control_total: int,
                           treatment_conversions: int, treatment_total: int):
        """Run statistical test for conversion rate difference"""
        successes = [control_conversions, treatment_conversions]
        totals = [control_total, treatment_total]
        
        stat, pvalue = proportions_ztest(successes, totals)
        
        control_rate = control_conversions / control_total
        treatment_rate = treatment_conversions / treatment_total
        lift = (treatment_rate - control_rate) / control_rate
        
        # Confidence interval for lift
        se = np.sqrt(control_rate*(1-control_rate)/control_total + 
                    treatment_rate*(1-treatment_rate)/treatment_total)
        ci_lower = (treatment_rate - control_rate) - 1.96 * se
        ci_upper = (treatment_rate - control_rate) + 1.96 * se
        
        return {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'lift': lift,
            'p_value': pvalue,
            'significant': pvalue < self.alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'z_statistic': stat
        }
    
    def simulate_email_experiment(self, df: pd.DataFrame):
        """Simulate an A/B test on email channel"""
        # Assign variants
        df['variant'] = df['customer_id'].apply(
            lambda x: self.assign_variant(x, 'email_subject_test')
        )
        
        # Simulate different performance (treatment gets 20% lift)
        df['simulated_conversion'] = df.apply(
            lambda row: np.random.binomial(1, 0.05 * 1.2) 
            if row['variant'] == 'treatment' and row['channel'] == 'email'
            else np.random.binomial(1, 0.05),
            axis=1
        )
        
        # Get results
        control = df[df['variant'] == 'control']
        treatment = df[df['variant'] == 'treatment']
        
        results = self.run_proportion_test(
            control['simulated_conversion'].sum(),
            len(control),
            treatment['simulated_conversion'].sum(),
            len(treatment)
        )
        
        return results
    
    def sequential_testing(self, df: pd.DataFrame, check_points: int = 5):
        """Implement sequential testing with early stopping"""
        results = []
        n = len(df)
        
        for i in range(1, check_points + 1):
            sample_size = int(n * i / check_points)
            sample = df.iloc[:sample_size]
            
            control = sample[sample['variant'] == 'control']
            treatment = sample[sample['variant'] == 'treatment']
            
            if len(control) > 30 and len(treatment) > 30:
                test_result = self.run_proportion_test(
                    control['simulated_conversion'].sum(),
                    len(control),
                    treatment['simulated_conversion'].sum(),
                    len(treatment)
                )
                
                results.append({
                    'sample_size': sample_size,
                    'p_value': test_result['p_value'],
                    'lift': test_result['lift'],
                    'significant': test_result['significant']
                })
                
                # Early stopping if highly significant
                if test_result['p_value'] < 0.001:
                    print(f"Early stopping at {sample_size} samples (p < 0.001)")
                    break
        
        return results


def main():
    """Run A/B testing experiments"""
    print("="*60)
    print("A/B TESTING & EXPERIMENTAL DESIGN")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/raw/touchpoints.csv')
    
    # Initialize framework
    ab_test = ABTestingFramework(confidence_level=0.95)
    
    # Sample size calculation
    print("\n1. SAMPLE SIZE CALCULATION")
    baseline_conversion = 0.05
    sample_size = ab_test.calculate_sample_size(
        baseline_rate=baseline_conversion,
        mde=0.02,  # 2% minimum detectable effect
        power=0.8
    )
    print(f"Required sample size per variant: {sample_size:,}")
    print(f"Total samples needed: {sample_size * 2:,}")
    
    # Run simulated experiment
    print("\n2. EMAIL SUBJECT LINE A/B TEST")
    results = ab_test.simulate_email_experiment(df)
    print(f"Control conversion rate: {results['control_rate']:.2%}")
    print(f"Treatment conversion rate: {results['treatment_rate']:.2%}")
    print(f"Lift: {results['lift']:.2%}")
    print(f"P-value: {results['p_value']:.4f}")
    print(f"Statistically significant: {results['significant']}")
    print(f"95% CI for difference: ({results['confidence_interval'][0]:.4f}, "
          f"{results['confidence_interval'][1]:.4f})")
    
    # Sequential testing
    print("\n3. SEQUENTIAL TESTING WITH EARLY STOPPING")
    df['variant'] = df['customer_id'].apply(
        lambda x: ab_test.assign_variant(x, 'sequential_test')
    )
    df['simulated_conversion'] = np.random.binomial(1, 0.05, len(df))
    
    sequential_results = ab_test.sequential_testing(df)
    for result in sequential_results:
        print(f"At {result['sample_size']:4} samples: "
              f"p={result['p_value']:.4f}, lift={result['lift']:.2%}")
    
    print("\n✅ A/B testing analysis complete!")


if __name__ == '__main__':
    main()
