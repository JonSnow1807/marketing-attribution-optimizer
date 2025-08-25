"""
Multi-Touch Attribution Comparison
Compares different attribution models and provides unified insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Handle both module and direct execution imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from .shapley import ShapleyAttributionModel
    from .markov import MarkovChainAttribution
except ImportError:
    from shapley import ShapleyAttributionModel
    from markov import MarkovChainAttribution


class MultiTouchAttribution:
    """
    Unified multi-touch attribution analysis
    Compares Shapley, Markov, and traditional attribution methods
    """
    
    def __init__(self):
        """Initialize multi-touch attribution analyzer"""
        self.models = {
            'shapley': ShapleyAttributionModel(model_type='random_forest'),
            'markov': MarkovChainAttribution(order=1)
        }
        self.results = {}
        self.comparison_df = None
        
    def calculate_traditional_attributions(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate traditional attribution methods
        
        Args:
            df: Touchpoint data
            
        Returns:
            Dictionary of attribution DataFrames
        """
        df = df.copy()
        results = {}
        
        # Last Touch Attribution
        df['last_touch_attribution'] = 0.0
        for customer_id, journey in df.groupby('customer_id'):
            if journey['converted'].max():
                revenue = journey['revenue'].max()
                last_idx = journey['touchpoint_number'].idxmax()
                df.loc[last_idx, 'last_touch_attribution'] = revenue
        
        # First Touch Attribution
        df['first_touch_attribution'] = 0.0
        for customer_id, journey in df.groupby('customer_id'):
            if journey['converted'].max():
                revenue = journey['revenue'].max()
                first_idx = journey['touchpoint_number'].idxmin()
                df.loc[first_idx, 'first_touch_attribution'] = revenue
        
        # Linear Attribution (Equal Credit)
        df['linear_attribution'] = 0.0
        for customer_id, journey in df.groupby('customer_id'):
            if journey['converted'].max():
                revenue = journey['revenue'].max()
                n_touches = len(journey)
                attribution_per_touch = revenue / n_touches if n_touches > 0 else 0
                df.loc[journey.index, 'linear_attribution'] = attribution_per_touch
        
        # Time Decay Attribution
        df['time_decay_attribution'] = 0.0
        decay_rate = 0.7  # 30% decay per touchpoint backwards
        
        for customer_id, journey in df.groupby('customer_id'):
            if journey['converted'].max():
                revenue = journey['revenue'].max()
                journey_sorted = journey.sort_values('touchpoint_number')
                n_touches = len(journey_sorted)
                
                # Calculate decay weights
                weights = [decay_rate ** (n_touches - i - 1) for i in range(n_touches)]
                weights = np.array(weights) / sum(weights)  # Normalize
                
                # Apply weights
                attributions = weights * revenue
                df.loc[journey_sorted.index, 'time_decay_attribution'] = attributions
        
        # Position-Based Attribution (U-shaped)
        df['position_based_attribution'] = 0.0
        first_touch_weight = 0.4
        last_touch_weight = 0.4
        middle_touch_weight = 0.2
        
        for customer_id, journey in df.groupby('customer_id'):
            if journey['converted'].max():
                revenue = journey['revenue'].max()
                n_touches = len(journey)
                
                if n_touches == 1:
                    df.loc[journey.index[0], 'position_based_attribution'] = revenue
                elif n_touches == 2:
                    df.loc[journey.index[0], 'position_based_attribution'] = revenue * 0.5
                    df.loc[journey.index[-1], 'position_based_attribution'] = revenue * 0.5
                else:
                    # First touch
                    first_idx = journey['touchpoint_number'].idxmin()
                    df.loc[first_idx, 'position_based_attribution'] = revenue * first_touch_weight
                    
                    # Last touch
                    last_idx = journey['touchpoint_number'].idxmax()
                    df.loc[last_idx, 'position_based_attribution'] = revenue * last_touch_weight
                    
                    # Middle touches
                    middle_indices = journey.index[
                        (journey['touchpoint_number'] != journey['touchpoint_number'].min()) & 
                        (journey['touchpoint_number'] != journey['touchpoint_number'].max())
                    ]
                    if len(middle_indices) > 0:
                        middle_attribution = (revenue * middle_touch_weight) / len(middle_indices)
                        df.loc[middle_indices, 'position_based_attribution'] = middle_attribution
        
        return df
    
    def fit_all_models(self, df: pd.DataFrame):
        """
        Fit all attribution models
        
        Args:
            df: Touchpoint data
        """
        print("="*60)
        print("MULTI-TOUCH ATTRIBUTION ANALYSIS")
        print("="*60)
        
        # Fit advanced models
        print("\n1. Fitting Shapley Value Model...")
        self.models['shapley'].fit(df)
        df = self.models['shapley'].attribute_conversions(df)
        
        print("\n2. Fitting Markov Chain Model...")
        self.models['markov'].fit(df)
        df = self.models['markov'].attribute_conversions(df)
        
        print("\n3. Calculating Traditional Attribution Methods...")
        df = self.calculate_traditional_attributions(df)
        
        self.results['attributed_df'] = df
        
        # Create comparison DataFrame
        self._create_comparison_df(df)
        
    def _create_comparison_df(self, df: pd.DataFrame):
        """
        Create comparison DataFrame across all attribution methods
        
        Args:
            df: DataFrame with all attributions
        """
        attribution_columns = [
            ('Shapley Value', 'shapley_attribution'),
            ('Markov Chain', 'markov_attribution'),
            ('Last Touch', 'last_touch_attribution'),
            ('First Touch', 'first_touch_attribution'),
            ('Linear', 'linear_attribution'),
            ('Time Decay', 'time_decay_attribution'),
            ('Position Based', 'position_based_attribution')
        ]
        
        comparison_data = []
        
        for method_name, col_name in attribution_columns:
            channel_attribution = df.groupby('channel')[col_name].sum()
            
            for channel, attributed_value in channel_attribution.items():
                comparison_data.append({
                    'Channel': channel,
                    'Attribution Method': method_name,
                    'Attributed Revenue': attributed_value
                })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        # Add channel costs for ROI calculation
        channel_costs = df.groupby('channel')['cost'].sum().reset_index()
        channel_costs.columns = ['Channel', 'Channel Cost']
        
        self.comparison_df = self.comparison_df.merge(
            channel_costs,
            on='Channel',
            how='left'
        )
        self.comparison_df['ROI'] = (
            (self.comparison_df['Attributed Revenue'] - self.comparison_df['Channel Cost']) / 
            self.comparison_df['Channel Cost']
        ).replace([np.inf, -np.inf], 0)
    
    def get_attribution_summary(self) -> pd.DataFrame:
        """
        Get summary of all attribution methods by channel
        
        Returns:
            Summary DataFrame
        """
        if self.comparison_df is None:
            raise ValueError("Models must be fitted first")
        
        pivot_table = self.comparison_df.pivot_table(
            values='Attributed Revenue',
            index='Channel',
            columns='Attribution Method',
            aggfunc='sum'
        ).fillna(0)
        
        # Add variance across methods
        pivot_table['Variance'] = pivot_table.var(axis=1)
        pivot_table['Std Dev'] = pivot_table.std(axis=1)
        pivot_table['CV'] = pivot_table['Std Dev'] / pivot_table.mean(axis=1)
        
        # Sort by average attribution
        pivot_table['Average'] = pivot_table.iloc[:, :-3].mean(axis=1)
        pivot_table = pivot_table.sort_values('Average', ascending=False)
        
        return pivot_table
    
    def plot_attribution_comparison(self, figsize: tuple = (15, 10)):
        """
        Create comprehensive visualization of attribution methods
        
        Args:
            figsize: Figure size
        """
        if self.comparison_df is None:
            raise ValueError("Models must be fitted first")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Attribution by Method and Channel
        pivot_for_plot = self.comparison_df.pivot(
            index='Channel',
            columns='Attribution Method',
            values='Attributed Revenue'
        ).fillna(0)
        
        pivot_for_plot.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Attribution Comparison by Channel')
        axes[0, 0].set_xlabel('Channel')
        axes[0, 0].set_ylabel('Attributed Revenue')
        axes[0, 0].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Heatmap of Attribution Methods
        sns.heatmap(pivot_for_plot.T, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0, 1])
        axes[0, 1].set_title('Attribution Heatmap')
        axes[0, 1].set_xlabel('Channel')
        axes[0, 1].set_ylabel('Attribution Method')
        
        # 3. Attribution Variance by Channel
        summary = self.get_attribution_summary()
        summary['CV'].plot(kind='barh', ax=axes[1, 0])
        axes[1, 0].set_title('Attribution Method Agreement (Lower CV = More Agreement)')
        axes[1, 0].set_xlabel('Coefficient of Variation')
        axes[1, 0].set_ylabel('Channel')
        
        # 4. ROI Comparison
        roi_pivot = self.comparison_df[self.comparison_df['ROI'].notna()].pivot(
            index='Channel',
            columns='Attribution Method',
            values='ROI'
        ).fillna(0)
        
        # Select top methods for ROI comparison
        top_methods = ['Shapley Value', 'Markov Chain', 'Linear']
        if all(m in roi_pivot.columns for m in top_methods):
            roi_pivot[top_methods].plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('ROI by Attribution Method')
            axes[1, 1].set_xlabel('Channel')
            axes[1, 1].set_ylabel('ROI')
            axes[1, 1].legend(title='Method')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def get_recommendations(self) -> Dict[str, any]:
        """
        Get actionable recommendations based on attribution analysis
        
        Returns:
            Dictionary of recommendations
        """
        if self.results.get('attributed_df') is None:
            raise ValueError("Models must be fitted first")
        
        df = self.results['attributed_df']
        recommendations = {}
        
        # 1. Identify undervalued channels (high Shapley/Markov, low last-touch)
        channel_comparison = df.groupby('channel').agg({
            'shapley_attribution': 'sum',
            'markov_attribution': 'sum',
            'last_touch_attribution': 'sum',
            'cost': 'sum'
        })
        
        channel_comparison['advanced_avg'] = (
            channel_comparison['shapley_attribution'] + 
            channel_comparison['markov_attribution']
        ) / 2
        
        channel_comparison['undervaluation'] = (
            channel_comparison['advanced_avg'] - 
            channel_comparison['last_touch_attribution']
        )
        
        undervalued = channel_comparison.nlargest(3, 'undervaluation')
        overvalued = channel_comparison.nsmallest(3, 'undervaluation')
        
        recommendations['undervalued_channels'] = undervalued.index.tolist()
        recommendations['overvalued_channels'] = overvalued.index.tolist()
        
        # 2. Budget reallocation suggestions
        total_budget = df['cost'].sum()
        optimal_allocation = (
            channel_comparison['advanced_avg'] / 
            channel_comparison['advanced_avg'].sum() * total_budget
        )
        
        reallocation = optimal_allocation - channel_comparison['cost']
        recommendations['budget_reallocation'] = reallocation.to_dict()
        
        # 3. ROI optimization opportunities
        channel_comparison['roi'] = (
            (channel_comparison['advanced_avg'] - channel_comparison['cost']) / 
            channel_comparison['cost']
        ).replace([np.inf, -np.inf], 0)
        
        high_roi = channel_comparison.nlargest(3, 'roi')
        recommendations['high_roi_channels'] = high_roi.index.tolist()
        
        return recommendations


def main():
    """Run complete multi-touch attribution analysis"""
    # Load data
    print("Loading sample data...")
    df = pd.read_csv('data/raw/touchpoints.csv')
    
    # Initialize analyzer
    analyzer = MultiTouchAttribution()
    
    # Fit all models
    analyzer.fit_all_models(df)
    
    # Get summary
    print("\n" + "="*60)
    print("ATTRIBUTION SUMMARY BY CHANNEL")
    print("="*60)
    summary = analyzer.get_attribution_summary()
    print(summary)
    
    # Get recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    recommendations = analyzer.get_recommendations()
    
    print("\n📈 Undervalued Channels (Increase Investment):")
    for channel in recommendations['undervalued_channels']:
        print(f"  • {channel}")
    
    print("\n📉 Overvalued Channels (Review Investment):")
    for channel in recommendations['overvalued_channels']:
        print(f"  • {channel}")
    
    print("\n💰 High ROI Channels:")
    for channel in recommendations['high_roi_channels']:
        print(f"  • {channel}")
    
    # Save results
    analyzer.results['attributed_df'].to_csv(
        'data/processed/touchpoints_all_attributions.csv', 
        index=False
    )
    summary.to_csv('data/processed/attribution_summary.csv')
    
    # Create visualization
    print("\n📊 Creating attribution comparison visualization...")
    fig = analyzer.plot_attribution_comparison()
    fig.savefig('data/processed/attribution_comparison.png', dpi=150, bbox_inches='tight')
    
    print("\n✅ Multi-touch attribution analysis complete!")
    print("Results saved to data/processed/")
    
    return analyzer


if __name__ == '__main__':
    analyzer = main()
