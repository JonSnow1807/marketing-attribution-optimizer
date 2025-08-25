"""
Test suite for attribution models
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from attribution.shapley import ShapleyAttributionModel
from attribution.markov import MarkovChainAttribution
from attribution.multi_touch import MultiTouchAttribution


class TestShapleyAttribution:
    """Test cases for Shapley Value Attribution Model"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample touchpoint data for testing"""
        data = {
            'customer_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
            'channel': ['paid_search', 'email', 'direct', 
                       'organic_search', 'direct',
                       'social_media', 'email', 'paid_search', 'direct'],
            'touchpoint_number': [1, 2, 3, 1, 2, 1, 2, 3, 4],
            'total_touchpoints': [3, 3, 3, 2, 2, 4, 4, 4, 4],
            'timestamp': pd.date_range('2024-01-01', periods=9, freq='D').astype(str),
            'cost': [2.5, 0.1, 0, 0, 0, 1.5, 0.1, 2.5, 0],
            'converted': [0, 0, 1, 0, 1, 0, 0, 0, 1],
            'revenue': [0, 0, 100, 0, 50, 0, 0, 0, 150],
            'time_on_site': np.random.uniform(30, 300, 9),
            'pages_viewed': np.random.randint(1, 10, 9),
            'device': ['mobile', 'desktop', 'mobile', 'desktop', 'desktop',
                      'mobile', 'tablet', 'desktop', 'mobile'],
            'customer_segment': ['new', 'new', 'new', 'returning', 'returning',
                                'new', 'new', 'new', 'new']
        }
        return pd.DataFrame(data)
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = ShapleyAttributionModel(model_type='random_forest')
        assert model.model_type == 'random_forest'
        assert model.model is None
        assert model.channel_shapley_values == {}
    
    def test_prepare_features(self, sample_data):
        """Test feature preparation"""
        model = ShapleyAttributionModel()
        features_df = model.prepare_features(sample_data)
        
        # Check that features are created
        assert 'journey_length' in features_df.columns
        assert 'total_cost' in features_df.columns
        assert 'unique_channels' in features_df.columns
        assert len(features_df) == len(sample_data['customer_id'].unique())
    
    def test_shapley_calculation(self, sample_data):
        """Test Shapley value calculation"""
        model = ShapleyAttributionModel()
        shapley_values = model.calculate_channel_shapley(sample_data)
        
        # Check that all channels have Shapley values
        channels = sample_data['channel'].unique()
        assert all(ch in shapley_values for ch in channels)
        
        # Check that Shapley values sum to approximately 1
        assert abs(sum(shapley_values.values()) - 1.0) < 0.01
    
    def test_model_fitting(self, sample_data):
        """Test model fitting process"""
        model = ShapleyAttributionModel(model_type='logistic')
        model.fit(sample_data)
        
        assert model.model is not None
        assert len(model.channel_shapley_values) > 0
    
    def test_attribution_assignment(self, sample_data):
        """Test attribution assignment to touchpoints"""
        model = ShapleyAttributionModel()
        model.fit(sample_data)
        df_attributed = model.attribute_conversions(sample_data)
        
        # Check that attribution column is added
        assert 'shapley_attribution' in df_attributed.columns
        
        # Check that only converted journeys have attribution
        converted_customers = sample_data[sample_data['converted'] == 1]['customer_id'].unique()
        for customer_id in converted_customers:
            customer_data = df_attributed[df_attributed['customer_id'] == customer_id]
            assert customer_data['shapley_attribution'].sum() > 0


class TestMarkovChainAttribution:
    """Test cases for Markov Chain Attribution Model"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample touchpoint data for testing"""
        return TestShapleyAttribution().sample_data()
    
    def test_journey_sequence_creation(self, sample_data):
        """Test journey sequence creation"""
        model = MarkovChainAttribution()
        sequences = model._create_journey_sequences(sample_data)
        
        # Check that sequences are created
        assert len(sequences) == len(sample_data['customer_id'].unique())
        
        # Check that sequences start with 'start' and end with 'conversion' or 'null'
        for seq in sequences:
            assert seq[0] == 'start'
            assert seq[-1] in ['conversion', 'null']
    
    def test_transition_matrix(self, sample_data):
        """Test transition matrix creation"""
        model = MarkovChainAttribution()
        sequences = model._create_journey_sequences(sample_data)
        transition_matrix = model._build_transition_matrix(sequences)
        
        # Check that transition matrix is created
        assert not transition_matrix.empty
        
        # Check that probabilities sum to 1 for each state
        for state in transition_matrix.index:
            row_sum = transition_matrix.loc[state].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 0.01
    
    def test_removal_effect(self, sample_data):
        """Test removal effect calculation"""
        model = MarkovChainAttribution()
        sequences = model._create_journey_sequences(sample_data)
        
        # Test removal effect for a specific channel
        removal_effect = model._calculate_removal_effect(sequences, 'email')
        
        # Removal effect should be between 0 and 1
        assert 0 <= removal_effect <= 1
    
    def test_markov_attribution(self, sample_data):
        """Test Markov chain attribution"""
        model = MarkovChainAttribution()
        model.fit(sample_data)
        df_attributed = model.attribute_conversions(sample_data)
        
        # Check that attribution column is added
        assert 'markov_attribution' in df_attributed.columns
        
        # Check that attributions are non-negative
        assert all(df_attributed['markov_attribution'] >= 0)


class TestMultiTouchAttribution:
    """Test cases for Multi-Touch Attribution Comparison"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample touchpoint data for testing"""
        return TestShapleyAttribution().sample_data()
    
    def test_traditional_attribution_methods(self, sample_data):
        """Test traditional attribution calculations"""
        analyzer = MultiTouchAttribution()
        df_attributed = analyzer.calculate_traditional_attributions(sample_data)
        
        # Check that all traditional methods are calculated
        traditional_methods = ['last_touch_attribution', 'first_touch_attribution',
                              'linear_attribution', 'time_decay_attribution',
                              'position_based_attribution']
        
        for method in traditional_methods:
            assert method in df_attributed.columns
            assert all(df_attributed[method] >= 0)
    
    def test_last_touch_attribution(self, sample_data):
        """Test last touch attribution logic"""
        analyzer = MultiTouchAttribution()
        df_attributed = analyzer.calculate_traditional_attributions(sample_data)
        
        # For each converted customer, only last touchpoint should have attribution
        for customer_id in sample_data[sample_data['converted'] == 1]['customer_id'].unique():
            customer_data = df_attributed[df_attributed['customer_id'] == customer_id]
            last_touch_attr = customer_data['last_touch_attribution'].values
            
            # Only one touchpoint should have non-zero attribution
            assert sum(last_touch_attr > 0) == 1
            
            # It should be the last touchpoint
            last_idx = customer_data['touchpoint_number'].idxmax()
            assert customer_data.loc[last_idx, 'last_touch_attribution'] > 0
    
    def test_linear_attribution(self, sample_data):
        """Test linear (equal) attribution logic"""
        analyzer = MultiTouchAttribution()
        df_attributed = analyzer.calculate_traditional_attributions(sample_data)
        
        # For each converted customer, attribution should be equal across touchpoints
        for customer_id in sample_data[sample_data['converted'] == 1]['customer_id'].unique():
            customer_data = df_attributed[df_attributed['customer_id'] == customer_id]
            linear_attr = customer_data['linear_attribution'].values
            
            # All touchpoints should have equal attribution
            if len(linear_attr) > 1:
                assert np.allclose(linear_attr[0], linear_attr[1:])
    
    def test_comparison_dataframe_creation(self, sample_data):
        """Test comparison DataFrame creation"""
        analyzer = MultiTouchAttribution()
        analyzer.fit_all_models(sample_data)
        
        assert analyzer.comparison_df is not None
        assert 'Channel' in analyzer.comparison_df.columns
        assert 'Attribution Method' in analyzer.comparison_df.columns
        assert 'Attributed Revenue' in analyzer.comparison_df.columns
    
    def test_recommendations_generation(self, sample_data):
        """Test recommendation generation"""
        analyzer = MultiTouchAttribution()
        analyzer.fit_all_models(sample_data)
        recommendations = analyzer.get_recommendations()
        
        assert 'undervalued_channels' in recommendations
        assert 'overvalued_channels' in recommendations
        assert 'high_roi_channels' in recommendations
        assert 'budget_reallocation' in recommendations


@pytest.fixture(scope="session")
def test_data_generator():
    """Generate larger test dataset"""
    from data.sample_generator import MarketingDataGenerator
    
    generator = MarketingDataGenerator(seed=42)
    df = generator.generate_dataset(num_customers=50)
    return df


def test_end_to_end_pipeline(test_data_generator):
    """Test complete attribution pipeline"""
    df = test_data_generator
    
    # Test Shapley attribution
    shapley_model = ShapleyAttributionModel()
    shapley_model.fit(df)
    df_shapley = shapley_model.attribute_conversions(df)
    assert 'shapley_attribution' in df_shapley.columns
    
    # Test Markov attribution
    markov_model = MarkovChainAttribution()
    markov_model.fit(df)
    df_markov = markov_model.attribute_conversions(df)
    assert 'markov_attribution' in df_markov.columns
    
    # Test multi-touch comparison
    analyzer = MultiTouchAttribution()
    analyzer.fit_all_models(df)
    summary = analyzer.get_attribution_summary()
    assert not summary.empty
    
    print("✅ End-to-end pipeline test passed!")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
