"""
Shapley Value Attribution Model
Uses game theory to fairly distribute conversion credit across touchpoints
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from itertools import combinations, permutations
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

class ShapleyAttributionModel:
    """
    Shapley Value Attribution using sklearn models
    Implements both data-driven and channel-based Shapley values
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize Shapley Attribution Model
        
        Args:
            model_type: Type of sklearn model ('random_forest', 'gradient_boost', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.channel_encoder = LabelEncoder()
        self.feature_importance = {}
        self.channel_shapley_values = {}
        
    def _initialize_model(self):
        """Initialize the sklearn model based on model_type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:  # logistic regression
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            )
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model training
        
        Args:
            df: Raw touchpoint data
            
        Returns:
            Feature matrix for modeling
        """
        # Aggregate by customer journey
        journey_features = []
        
        for customer_id, journey in df.groupby('customer_id'):
            # Channel presence and frequency
            channel_counts = journey['channel'].value_counts().to_dict()
            
            # Journey characteristics
            features = {
                'customer_id': customer_id,
                'journey_length': len(journey),
                'total_cost': journey['cost'].sum(),
                'total_time_on_site': journey['time_on_site'].sum(),
                'total_pages_viewed': journey['pages_viewed'].sum(),
                'unique_channels': journey['channel'].nunique(),
                'unique_devices': journey['device'].nunique(),
                'converted': journey['converted'].max(),
                'revenue': journey['revenue'].sum(),
                
                # Time features
                'journey_duration_hours': (
                    pd.to_datetime(journey['timestamp'].max()) - 
                    pd.to_datetime(journey['timestamp'].min())
                ).total_seconds() / 3600,
                
                # Channel-specific features
                **{f'channel_{ch}_count': channel_counts.get(ch, 0) 
                   for ch in df['channel'].unique()},
                
                # First and last touch channels
                'first_channel': journey.iloc[0]['channel'],
                'last_channel': journey.iloc[-1]['channel'],
                
                # Device journey
                'mobile_touches': (journey['device'] == 'mobile').sum(),
                'desktop_touches': (journey['device'] == 'desktop').sum(),
            }
            
            journey_features.append(features)
        
        return pd.DataFrame(journey_features)
    
    def calculate_channel_shapley(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Shapley values for each channel
        
        Args:
            df: Touchpoint data
            
        Returns:
            Dictionary of channel Shapley values
        """
        # Get all unique channels
        channels = df['channel'].unique()
        n_channels = len(channels)
        
        # Initialize Shapley values
        shapley_values = {channel: 0.0 for channel in channels}
        
        # Get conversions by channel combination
        conversions_by_coalition = self._get_conversion_rates_by_coalition(df)
        
        # Calculate Shapley value for each channel
        for channel in channels:
            shapley_value = 0.0
            
            # Iterate through all possible coalitions
            for r in range(n_channels):
                for coalition in combinations([ch for ch in channels if ch != channel], r):
                    coalition_with = tuple(sorted(coalition + (channel,)))
                    coalition_without = tuple(sorted(coalition))
                    
                    # Get conversion rates
                    v_with = conversions_by_coalition.get(coalition_with, 0)
                    v_without = conversions_by_coalition.get(coalition_without, 0)
                    
                    # Calculate marginal contribution
                    marginal_contribution = v_with - v_without
                    
                    # Weight by coalition size
                    weight = (np.math.factorial(len(coalition)) * 
                             np.math.factorial(n_channels - len(coalition) - 1) / 
                             np.math.factorial(n_channels))
                    
                    shapley_value += weight * marginal_contribution
            
            shapley_values[channel] = shapley_value
        
        # Normalize to sum to 1
        total = sum(shapley_values.values())
        if total > 0:
            shapley_values = {k: v/total for k, v in shapley_values.items()}
        
        self.channel_shapley_values = shapley_values
        return shapley_values
    
    def _get_conversion_rates_by_coalition(self, df: pd.DataFrame) -> Dict[Tuple, float]:
        """
        Calculate conversion rates for each channel coalition
        
        Args:
            df: Touchpoint data
            
        Returns:
            Dictionary mapping channel coalitions to conversion rates
        """
        conversion_rates = {}
        
        # Group by customer
        for customer_id, journey in df.groupby('customer_id'):
            channels_in_journey = tuple(sorted(journey['channel'].unique()))
            converted = journey['converted'].max()
            
            # Add to all subsets of channels in journey
            for r in range(1, len(channels_in_journey) + 1):
                for coalition in combinations(channels_in_journey, r):
                    coalition = tuple(sorted(coalition))
                    if coalition not in conversion_rates:
                        conversion_rates[coalition] = {'conversions': 0, 'total': 0}
                    
                    conversion_rates[coalition]['total'] += 1
                    if converted:
                        conversion_rates[coalition]['conversions'] += 1
        
        # Calculate rates
        return {
            coalition: stats['conversions'] / stats['total'] if stats['total'] > 0 else 0
            for coalition, stats in conversion_rates.items()
        }
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the attribution model
        
        Args:
            df: Touchpoint data
        """
        print("Preparing features...")
        features_df = self.prepare_features(df)
        
        # Prepare features for modeling
        feature_cols = [col for col in features_df.columns 
                       if col not in ['customer_id', 'converted', 'revenue', 
                                     'first_channel', 'last_channel']]
        
        # Encode categorical features - FIX: fit on all unique channels
        all_channels = pd.concat([
            features_df['first_channel'],
            features_df['last_channel']
        ]).unique()
        self.channel_encoder.fit(all_channels)
        
        features_df['first_channel_encoded'] = self.channel_encoder.transform(
            features_df['first_channel']
        )
        features_df['last_channel_encoded'] = self.channel_encoder.transform(
            features_df['last_channel']
        )
        
        feature_cols.extend(['first_channel_encoded', 'last_channel_encoded'])
        
        X = features_df[feature_cols].values
        y = features_df['converted'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training {self.model_type} model...")
        self._initialize_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"Model AUC Score: {auc_score:.4f}")
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        
        # Calculate channel Shapley values
        print("Calculating channel Shapley values...")
        self.calculate_channel_shapley(df)
        
    def attribute_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attribute conversions to touchpoints using Shapley values
        
        Args:
            df: Touchpoint data
            
        Returns:
            DataFrame with attribution scores
        """
        df = df.copy()
        df['shapley_attribution'] = 0.0
        
        # Apply Shapley values to each journey
        for customer_id, journey in df.groupby('customer_id'):
            if journey['converted'].max():
                # Get channels in journey
                journey_channels = journey['channel'].values
                n_touches = len(journey_channels)
                
                # Calculate attribution for each touchpoint
                attributions = []
                for channel in journey_channels:
                    # Base attribution from channel Shapley value
                    base_attribution = self.channel_shapley_values.get(channel, 0)
                    
                    # Adjust for position in journey (optional enhancement)
                    attributions.append(base_attribution)
                
                # Normalize to sum to 1 for the journey
                total_attribution = sum(attributions)
                if total_attribution > 0:
                    attributions = [a / total_attribution for a in attributions]
                
                # Apply revenue if available
                revenue = journey['revenue'].max()
                if revenue > 0:
                    attributions = [a * revenue for a in attributions]
                
                # Assign attributions
                df.loc[journey.index, 'shapley_attribution'] = attributions
        
        return df
    
    def get_channel_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get channel performance metrics based on Shapley attribution
        
        Args:
            df: Touchpoint data with attributions
            
        Returns:
            Channel performance summary
        """
        if 'shapley_attribution' not in df.columns:
            df = self.attribute_conversions(df)
        
        channel_metrics = df.groupby('channel').agg({
            'shapley_attribution': 'sum',
            'cost': 'sum',
            'customer_id': 'nunique',
            'converted': lambda x: x[df.loc[x.index, 'touchpoint_number'] == 
                                   df.loc[x.index, 'total_touchpoints']].sum()
        }).rename(columns={
            'shapley_attribution': 'attributed_revenue',
            'customer_id': 'unique_customers',
            'converted': 'last_touch_conversions'
        })
        
        # Calculate ROI
        channel_metrics['roi'] = (
            (channel_metrics['attributed_revenue'] - channel_metrics['cost']) / 
            channel_metrics['cost']
        ).replace([np.inf, -np.inf], 0)
        
        # Add Shapley values
        channel_metrics['shapley_value'] = pd.Series(self.channel_shapley_values)
        
        # Sort by attributed revenue
        channel_metrics = channel_metrics.sort_values('attributed_revenue', ascending=False)
        
        return channel_metrics
    
    def plot_attribution_comparison(self, df: pd.DataFrame):
        """
        Plot comparison of different attribution methods
        """
        import matplotlib.pyplot as plt
        
        # Calculate different attribution methods
        last_touch = df[df['converted'] == 1].groupby('channel')['revenue'].sum()
        first_touch = df[df['touchpoint_number'] == 1].groupby('channel')['revenue'].sum()
        
        if 'shapley_attribution' not in df.columns:
            df = self.attribute_conversions(df)
        
        shapley = df.groupby('channel')['shapley_attribution'].sum()
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Last Touch': last_touch,
            'First Touch': first_touch,
            'Shapley Value': shapley
        }).fillna(0)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Attribution comparison
        comparison.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Attribution Model Comparison')
        axes[0].set_xlabel('Channel')
        axes[0].set_ylabel('Attributed Revenue')
        axes[0].legend(title='Attribution Model')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Shapley values
        shapley_df = pd.Series(self.channel_shapley_values).sort_values(ascending=True)
        shapley_df.plot(kind='barh', ax=axes[1])
        axes[1].set_title('Channel Shapley Values')
        axes[1].set_xlabel('Shapley Value')
        axes[1].set_ylabel('Channel')
        
        plt.tight_layout()
        return fig


def main():
    """Test Shapley attribution with sample data"""
    # Load sample data
    print("Loading sample data...")
    df = pd.read_csv('data/raw/touchpoints.csv')
    
    # Initialize and fit model
    print("\nInitializing Shapley Attribution Model...")
    model = ShapleyAttributionModel(model_type='random_forest')
    model.fit(df)
    
    # Attribute conversions
    print("\nAttributing conversions...")
    df_attributed = model.attribute_conversions(df)
    
    # Get channel performance
    print("\nChannel Performance Metrics:")
    print("="*60)
    performance = model.get_channel_performance(df_attributed)
    print(performance)
    
    # Save results
    df_attributed.to_csv('data/processed/touchpoints_shapley_attributed.csv', index=False)
    performance.to_csv('data/processed/channel_performance_shapley.csv')
    
    print("\n✅ Shapley attribution complete!")
    print(f"Results saved to data/processed/")
    
    return df_attributed, performance

if __name__ == '__main__':
    main()
