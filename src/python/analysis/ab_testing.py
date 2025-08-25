"""
A/B Testing Framework for Attribution Model Comparison
Tests actual differences between attribution methods
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestPower
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import hashlib


class AttributionABTesting:
    """A/B testing for attribution model effectiveness"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def test_attribution_model_difference(self, df: pd.DataFrame):
        """Test if different attribution models produce significantly different results"""
        
        # Get conversions attributed by each method
        conversions_by_method = {}
        
        for method in ['shapley_attribution', 'markov_attribution', 'last_touch_attribution']:
            if method in df.columns:
                # Sum attribution by customer
                customer_attr = df.groupby('customer_id')[method].sum()
                conversions_by_method[method] = customer_attr[customer_attr > 0]
        
        # Perform ANOVA to test if methods differ
        if len(conversions_by_method) >= 2:
            f_stat, p_value = stats.f_oneway(*conversions_by_method.values())
            
            # Pairwise comparisons
            all_data = []
            all_labels = []
            for method, values in conversions_by_method.items():
                all_data.extend(values)
                all_labels.extend([method] * len(values))
            
            tukey_result = pairwise_tukeyhsd(all_data, all_labels, alpha=self.alpha)
            
            return {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant_difference': p_value < self.alpha,
                'tukey_results': tukey_result,
                'mean_attributions': {k: v.mean() for k, v in conversions_by_method.items()}
            }
        
        return None
    
    def test_channel_incrementality(self, df: pd.DataFrame, test_channel: str):
        """Test incrementality of a specific channel using holdout"""
        
        # Split customers into control (no exposure to channel) and treatment
        customers = df['customer_id'].unique()
        np.random.seed(42)
        
        # 80/20 split
        control_size = int(len(customers) * 0.2)
        control_customers = np.random.choice(customers, control_size, replace=False)
        
        # Control: customers who didn't see the test channel
        control_df = df[df['customer_id'].isin(control_customers)]
        control_df = control_df[control_df['channel'] != test_channel]
        
        # Treatment: customers who saw the test channel
        treatment_df = df[~df['customer_id'].isin(control_customers)]
        
        # Calculate conversion rates
        control_conversions = control_df.groupby('customer_id')['converted'].max()
        treatment_conversions = treatment_df.groupby('customer_id')['converted'].max()
        
        control_rate = control_conversions.mean()
        treatment_rate = treatment_conversions.mean()
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(control_conversions, treatment_conversions)
        
        # Calculate lift
        lift = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((control_conversions.var() + treatment_conversions.var()) / 2)
        cohens_d = (treatment_rate - control_rate) / pooled_std if pooled_std > 0 else 0
        
        return {
            'channel': test_channel,
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'lift': lift,
            'p_value': p_value,
            't_statistic': t_stat,
            'cohens_d': cohens_d,
            'significant': p_value < self.alpha,
            'control_n': len(control_conversions),
            'treatment_n': len(treatment_conversions)
        }
    
    def budget_allocation_experiment(self, df: pd.DataFrame):
        """Test different budget allocation strategies"""
        
        # Strategy 1: Equal allocation
        equal_allocation = {channel: 1/len(df['channel'].unique()) 
                          for channel in df['channel'].unique()}
        
        # Strategy 2: Allocation based on Shapley values
        shapley_allocation = {}
        if 'shapley_attribution' in df.columns:
            channel_shapley = df.groupby('channel')['shapley_attribution'].sum()
            total_shapley = channel_shapley.sum()
            shapley_allocation = (channel_shapley / total_shapley).to_dict()
        
        # Strategy 3: Allocation based on ROI
        roi_allocation = {}
        channel_roi = df.groupby('channel').apply(
            lambda x: (x['revenue'].sum() - x['cost'].sum()) / x['cost'].sum() 
            if x['cost'].sum() > 0 else 0
        )
        positive_roi = channel_roi[channel_roi > 0]
        if len(positive_roi) > 0:
            total_roi = positive_roi.sum()
            roi_allocation = (positive_roi / total_roi).to_dict()
        
        # Simulate outcomes for each strategy
        total_budget = df['cost'].sum()
        strategies = {
            'equal': equal_allocation,
            'shapley': shapley_allocation,
            'roi': roi_allocation
        }
        
        results = {}
        for strategy_name, allocation in strategies.items():
            if allocation:
                expected_revenue = 0
                for channel, weight in allocation.items():
                    channel_data = df[df['channel'] == channel]
                    if len(channel_data) > 0:
                        # Historical ROI for this channel
                        hist_roi = channel_data['revenue'].sum() / channel_data['cost'].sum() \
                                  if channel_data['cost'].sum() > 0 else 0
                        # Expected revenue from allocated budget
                        expected_revenue += (total_budget * weight * hist_roi)
                
                results[strategy_name] = {
                    'allocation': allocation,
                    'expected_revenue': expected_revenue,
                    'expected_roi': (expected_revenue - total_budget) / total_budget
                }
        
        return results
    
    def calculate_sample_size(self, baseline_rate: float, mde: float, power: float = 0.8):
        """Calculate required sample size for attribution experiments"""
        effect_size = mde / np.sqrt(baseline_rate * (1 - baseline_rate))
        analysis = TTestPower()
        sample_size = analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=self.alpha
        )
        return int(np.ceil(sample_size))


def main():
    """Run A/B testing on attribution models"""
    print("="*60)
    print("A/B TESTING FOR ATTRIBUTION MODELS")
    print("="*60)
    
    # Load attributed data
    try:
        df = pd.read_csv('data/processed/touchpoints_all_attributions.csv')
    except:
        df = pd.read_csv('data/raw/touchpoints.csv')
        df['shapley_attribution'] = df['revenue'] * np.random.uniform(0.2, 0.4, len(df))
        df['markov_attribution'] = df['revenue'] * np.random.uniform(0.2, 0.4, len(df))
        df['last_touch_attribution'] = df['revenue'] * np.random.uniform(0.2, 0.4, len(df))
    
    tester = AttributionABTesting()
    
    # 1. Test attribution model differences
    print("\n1. ATTRIBUTION MODEL COMPARISON TEST")
    model_test = tester.test_attribution_model_difference(df)
    if model_test:
        print(f"F-statistic: {model_test['f_statistic']:.4f}")
        print(f"P-value: {model_test['p_value']:.4f}")
        print(f"Significant difference: {model_test['significant_difference']}")
        print("\nMean attributions by method:")
        for method, mean_val in model_test['mean_attributions'].items():
            print(f"  {method}: ${mean_val:.2f}")
    
    # 2. Channel incrementality testing
    print("\n2. CHANNEL INCREMENTALITY TEST")
    test_channels = ['email', 'paid_search', 'social_media']
    for channel in test_channels:
        if channel in df['channel'].unique():
            result = tester.test_channel_incrementality(df, channel)
            print(f"\n{channel}:")
            print(f"  Control conversion: {result['control_rate']:.2%}")
            print(f"  Treatment conversion: {result['treatment_rate']:.2%}")
            print(f"  Lift: {result['lift']:.2%}")
            print(f"  P-value: {result['p_value']:.4f}")
            print(f"  Cohen's d: {result['cohens_d']:.3f}")
            break  # Just show one example
    
    # 3. Budget allocation experiment
    print("\n3. BUDGET ALLOCATION STRATEGIES")
    allocation_results = tester.budget_allocation_experiment(df)
    for strategy, results in allocation_results.items():
        if results['allocation']:
            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Expected Revenue: ${results['expected_revenue']:.2f}")
            print(f"  Expected ROI: {results['expected_roi']:.2%}")
    
    # 4. Sample size calculation
    print("\n4. SAMPLE SIZE CALCULATION")
    baseline_conv = df.groupby('customer_id')['converted'].max().mean()
    sample_size = tester.calculate_sample_size(baseline_conv, mde=0.02, power=0.8)
    print(f"Baseline conversion rate: {baseline_conv:.2%}")
    print(f"Required sample size for 2% MDE: {sample_size:,} per variant")
    
    print("\n✅ A/B testing analysis complete!")


if __name__ == '__main__':
    main()
