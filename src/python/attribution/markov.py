"""
Markov Chain Attribution Model
Models customer journeys as state transitions to understand channel interactions
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')


class MarkovChainAttribution:
    """
    Markov Chain Attribution Model
    Implements removal effect and transition probability analysis
    """
    
    def __init__(self, order: int = 1):
        """
        Initialize Markov Chain Attribution
        
        Args:
            order: Order of Markov chain (1 for first-order, 2 for second-order)
        """
        self.order = order
        self.transition_matrix = None
        self.removal_effects = {}
        self.channel_attributions = {}
        self.transition_graph = None
        
    def _create_journey_sequences(self, df: pd.DataFrame) -> List[List[str]]:
        """
        Create journey sequences from touchpoint data
        
        Args:
            df: Touchpoint data
            
        Returns:
            List of channel sequences
        """
        sequences = []
        
        for customer_id, journey in df.groupby('customer_id'):
            # Get channel sequence
            channels = journey.sort_values('touchpoint_number')['channel'].tolist()
            
            # Add start state
            sequence = ['start'] + channels
            
            # Add end state (conversion or null)
            if journey['converted'].max():
                sequence.append('conversion')
            else:
                sequence.append('null')
                
            sequences.append(sequence)
            
        return sequences
    
    def _build_transition_matrix(self, sequences: List[List[str]]) -> pd.DataFrame:
        """
        Build transition probability matrix
        
        Args:
            sequences: List of channel sequences
            
        Returns:
            Transition probability matrix
        """
        transitions = defaultdict(lambda: defaultdict(int))
        
        for sequence in sequences:
            if self.order == 1:
                # First-order Markov chain
                for i in range(len(sequence) - 1):
                    current_state = sequence[i]
                    next_state = sequence[i + 1]
                    transitions[current_state][next_state] += 1
            else:
                # Higher-order Markov chain
                for i in range(len(sequence) - self.order):
                    current_state = tuple(sequence[i:i+self.order])
                    next_state = sequence[i + self.order]
                    transitions[current_state][next_state] += 1
        
        # Convert to probability matrix
        states = list(transitions.keys())
        matrix_dict = {}
        
        for state in states:
            state_transitions = transitions[state]
            total = sum(state_transitions.values())
            
            if total > 0:
                matrix_dict[state] = {
                    next_state: count / total 
                    for next_state, count in state_transitions.items()
                }
            else:
                matrix_dict[state] = {}
        
        # Create DataFrame
        all_states = set(states)
        for state_dict in matrix_dict.values():
            all_states.update(state_dict.keys())
        
        matrix = pd.DataFrame(0.0, index=states, columns=list(all_states))
        
        for state, transitions in matrix_dict.items():
            for next_state, prob in transitions.items():
                matrix.loc[state, next_state] = prob
                
        return matrix
    
    def _calculate_removal_effect(self, sequences: List[List[str]], 
                                 channel: str) -> float:
        """
        Calculate removal effect for a channel
        
        Args:
            sequences: Original journey sequences
            channel: Channel to remove
            
        Returns:
            Removal effect (reduction in conversion probability)
        """
        # Original conversion rate
        original_conversions = sum(
            1 for seq in sequences if seq[-1] == 'conversion'
        )
        original_rate = original_conversions / len(sequences) if sequences else 0
        
        # Remove channel and recalculate
        modified_sequences = []
        for sequence in sequences:
            # Remove all instances of the channel
            modified_seq = [s for s in sequence if s != channel]
            
            # If sequence becomes empty (except start/end), it fails
            if len(modified_seq) <= 2:  # Only start and end states
                modified_seq = ['start', 'null']
                
            modified_sequences.append(modified_seq)
        
        # Calculate new conversion rate
        modified_conversions = sum(
            1 for seq in modified_sequences if seq[-1] == 'conversion'
        )
        modified_rate = modified_conversions / len(modified_sequences) if modified_sequences else 0
        
        # Removal effect
        removal_effect = (original_rate - modified_rate) / original_rate if original_rate > 0 else 0
        
        return max(0, removal_effect)  # Ensure non-negative
    
    def _create_transition_graph(self, transition_matrix: pd.DataFrame) -> nx.DiGraph:
        """
        Create directed graph from transition matrix
        
        Args:
            transition_matrix: Transition probability matrix
            
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        for source in transition_matrix.index:
            for target in transition_matrix.columns:
                prob = transition_matrix.loc[source, target]
                if prob > 0:
                    G.add_edge(source, target, weight=prob)
                    
        return G
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the Markov chain attribution model
        
        Args:
            df: Touchpoint data
        """
        print("Creating journey sequences...")
        sequences = self._create_journey_sequences(df)
        
        print("Building transition matrix...")
        self.transition_matrix = self._build_transition_matrix(sequences)
        
        print("Creating transition graph...")
        self.transition_graph = self._create_transition_graph(self.transition_matrix)
        
        print("Calculating removal effects...")
        channels = df['channel'].unique()
        
        for channel in channels:
            removal_effect = self._calculate_removal_effect(sequences, channel)
            self.removal_effects[channel] = removal_effect
        
        # Normalize removal effects to get attribution
        total_effect = sum(self.removal_effects.values())
        if total_effect > 0:
            self.channel_attributions = {
                channel: effect / total_effect
                for channel, effect in self.removal_effects.items()
            }
        else:
            # Equal attribution if no effects
            self.channel_attributions = {
                channel: 1.0 / len(channels)
                for channel in channels
            }
    
    def attribute_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attribute conversions using Markov chain model
        
        Args:
            df: Touchpoint data
            
        Returns:
            DataFrame with Markov attribution
        """
        df = df.copy()
        df['markov_attribution'] = 0.0
        
        for customer_id, journey in df.groupby('customer_id'):
            if journey['converted'].max():
                # Get channels in journey
                journey_channels = journey['channel'].values
                
                # Calculate attribution for each touchpoint
                attributions = []
                for channel in journey_channels:
                    attribution = self.channel_attributions.get(channel, 0)
                    attributions.append(attribution)
                
                # Normalize to sum to 1
                total = sum(attributions)
                if total > 0:
                    attributions = [a / total for a in attributions]
                
                # Apply revenue
                revenue = journey['revenue'].max()
                if revenue > 0:
                    attributions = [a * revenue for a in attributions]
                
                # Assign attributions
                df.loc[journey.index, 'markov_attribution'] = attributions
                
        return df
    
    def get_channel_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get channel performance based on Markov attribution
        
        Args:
            df: DataFrame with attributions
            
        Returns:
            Channel performance metrics
        """
        if 'markov_attribution' not in df.columns:
            df = self.attribute_conversions(df)
        
        performance = df.groupby('channel').agg({
            'markov_attribution': 'sum',
            'cost': 'sum',
            'customer_id': 'nunique',
        }).rename(columns={
            'markov_attribution': 'attributed_revenue',
            'customer_id': 'unique_customers'
        })
        
        # Add removal effects and attribution percentages
        performance['removal_effect'] = pd.Series(self.removal_effects)
        performance['attribution_share'] = pd.Series(self.channel_attributions)
        
        # Calculate ROI
        performance['roi'] = (
            (performance['attributed_revenue'] - performance['cost']) / 
            performance['cost']
        ).replace([np.inf, -np.inf], 0)
        
        # Sort by attributed revenue
        performance = performance.sort_values('attributed_revenue', ascending=False)
        
        return performance
    
    def plot_transition_graph(self, top_n: int = 20):
        """
        Visualize the transition graph
        
        Args:
            top_n: Number of top transitions to show
        """
        import matplotlib.pyplot as plt
        
        if self.transition_graph is None:
            raise ValueError("Model must be fitted first")
        
        # Get top transitions by weight
        edges = self.transition_graph.edges(data=True)
        sorted_edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)[:top_n]
        
        # Create subgraph with top transitions
        G_sub = nx.DiGraph()
        for source, target, data in sorted_edges:
            G_sub.add_edge(source, target, weight=data['weight'])
        
        # Plot
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G_sub, k=2, iterations=50)
        
        # Draw nodes
        node_colors = []
        for node in G_sub.nodes():
            if node == 'start':
                node_colors.append('green')
            elif node == 'conversion':
                node_colors.append('gold')
            elif node == 'null':
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, 
                              node_size=1000, alpha=0.8)
        
        # Draw edges with weights
        edge_weights = [G_sub[u][v]['weight'] for u, v in G_sub.edges()]
        nx.draw_networkx_edges(G_sub, pos, width=[w*5 for w in edge_weights],
                              alpha=0.6, edge_color='gray', arrows=True,
                              arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G_sub, pos, font_size=10)
        
        # Add edge labels with probabilities
        edge_labels = {(u, v): f"{data['weight']:.2f}" 
                      for u, v, data in G_sub.edges(data=True)}
        nx.draw_networkx_edge_labels(G_sub, pos, edge_labels, font_size=8)
        
        plt.title(f"Customer Journey Transition Graph (Top {top_n} Transitions)")
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()
    
    def get_transition_probabilities(self, from_channel: str) -> pd.Series:
        """
        Get transition probabilities from a specific channel
        
        Args:
            from_channel: Source channel
            
        Returns:
            Series of transition probabilities
        """
        if self.transition_matrix is None:
            raise ValueError("Model must be fitted first")
            
        if from_channel in self.transition_matrix.index:
            return self.transition_matrix.loc[from_channel].sort_values(ascending=False)
        else:
            return pd.Series()


def main():
    """Test Markov chain attribution"""
    # Load data
    print("Loading sample data...")
    df = pd.read_csv('data/raw/touchpoints.csv')
    
    # Initialize and fit model
    print("\nInitializing Markov Chain Attribution Model...")
    model = MarkovChainAttribution(order=1)
    model.fit(df)
    
    # Attribute conversions
    print("\nAttributing conversions...")
    df_attributed = model.attribute_conversions(df)
    
    # Get performance metrics
    print("\nChannel Performance (Markov Attribution):")
    print("="*60)
    performance = model.get_channel_performance(df_attributed)
    print(performance)
    
    print("\nChannel Removal Effects:")
    print("-"*40)
    for channel, effect in sorted(model.removal_effects.items(), 
                                 key=lambda x: x[1], reverse=True):
        print(f"{channel:15} {effect:.2%}")
    
    # Save results
    df_attributed.to_csv('data/processed/touchpoints_markov_attributed.csv', index=False)
    performance.to_csv('data/processed/channel_performance_markov.csv')
    
    print("\n✅ Markov chain attribution complete!")
    print(f"Results saved to data/processed/")
    
    return df_attributed, performance


if __name__ == '__main__':
    main()
