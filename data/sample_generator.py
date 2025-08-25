#!/usr/bin/env python3
"""
Sample Data Generator for Marketing Attribution System
Generates realistic multi-channel marketing data with customer journeys
"""

import os
import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import click
from tqdm import tqdm

# Marketing channels with their characteristics
CHANNELS = {
    'organic_search': {'cost_per_interaction': 0, 'conversion_rate': 0.03, 'avg_position': 1},
    'paid_search': {'cost_per_interaction': 2.5, 'conversion_rate': 0.05, 'avg_position': 2},
    'social_media': {'cost_per_interaction': 1.5, 'conversion_rate': 0.02, 'avg_position': 3},
    'email': {'cost_per_interaction': 0.1, 'conversion_rate': 0.08, 'avg_position': 4},
    'display': {'cost_per_interaction': 0.8, 'conversion_rate': 0.01, 'avg_position': 5},
    'affiliate': {'cost_per_interaction': 5.0, 'conversion_rate': 0.06, 'avg_position': 6},
    'direct': {'cost_per_interaction': 0, 'conversion_rate': 0.10, 'avg_position': 7},
    'referral': {'cost_per_interaction': 0, 'conversion_rate': 0.04, 'avg_position': 8},
    'video': {'cost_per_interaction': 3.0, 'conversion_rate': 0.025, 'avg_position': 9},
    'retargeting': {'cost_per_interaction': 4.0, 'conversion_rate': 0.07, 'avg_position': 10}
}

# Device types
DEVICES = ['desktop', 'mobile', 'tablet']

# Customer segments
SEGMENTS = ['new', 'returning', 'loyal', 'at_risk', 'churned']

# Campaign types
CAMPAIGN_TYPES = ['brand', 'performance', 'awareness', 'retargeting', 'seasonal']

class MarketingDataGenerator:
    """Generate realistic marketing attribution data"""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        
    def generate_customer_journey(self, customer_id: int) -> List[Dict]:
        """Generate a single customer's journey through multiple touchpoints"""
        journey = []
        current_date = self._random_date()
        
        # Determine journey length (1-15 touchpoints)
        journey_length = np.random.poisson(3) + 1
        journey_length = min(journey_length, 15)
        
        # Customer characteristics
        segment = random.choice(SEGMENTS)
        device_preference = random.choice(DEVICES)
        
        # Will this journey convert?
        base_conversion_prob = 0.05 if segment == 'new' else 0.08 if segment == 'returning' else 0.12
        will_convert = random.random() < base_conversion_prob
        
        # Generate touchpoints
        for i in range(journey_length):
            touchpoint = self._generate_touchpoint(
                customer_id=customer_id,
                timestamp=current_date,
                sequence=i + 1,
                total_touchpoints=journey_length,
                segment=segment,
                device_preference=device_preference,
                is_last=(i == journey_length - 1) and will_convert
            )
            journey.append(touchpoint)
            
            # Move to next touchpoint time (0-7 days later)
            hours_to_next = np.random.exponential(24) * (1 + random.random() * 6)
            current_date += timedelta(hours=hours_to_next)
            
        return journey
    
    def _generate_touchpoint(self, customer_id: int, timestamp: datetime, 
                            sequence: int, total_touchpoints: int,
                            segment: str, device_preference: str, 
                            is_last: bool) -> Dict:
        """Generate a single touchpoint"""
        
        # Channel selection logic (some channels more likely at different stages)
        if sequence == 1:
            # First touch - often discovery channels
            channel_weights = {
                'organic_search': 0.3,
                'paid_search': 0.25,
                'social_media': 0.2,
                'display': 0.15,
                'direct': 0.05,
                'referral': 0.05
            }
        elif is_last:
            # Last touch before conversion
            channel_weights = {
                'direct': 0.25,
                'email': 0.2,
                'paid_search': 0.2,
                'retargeting': 0.15,
                'organic_search': 0.1,
                'affiliate': 0.1
            }
        else:
            # Middle touches
            channel_weights = {
                'email': 0.2,
                'social_media': 0.15,
                'retargeting': 0.15,
                'display': 0.15,
                'video': 0.1,
                'paid_search': 0.1,
                'organic_search': 0.1,
                'affiliate': 0.05
            }
        
        # Add missing channels with small weight
        for ch in CHANNELS:
            if ch not in channel_weights:
                channel_weights[ch] = 0.01
                
        # Normalize weights
        total_weight = sum(channel_weights.values())
        channel_weights = {k: v/total_weight for k, v in channel_weights.items()}
        
        # Select channel
        channel = np.random.choice(
            list(channel_weights.keys()),
            p=list(channel_weights.values())
        )
        
        # Device (with some consistency to preference)
        if random.random() < 0.7:
            device = device_preference
        else:
            device = random.choice(DEVICES)
        
        # Campaign
        campaign_type = random.choice(CAMPAIGN_TYPES)
        campaign_id = f"{campaign_type}_{random.randint(1, 20)}"
        
        # Cost calculation
        base_cost = CHANNELS[channel]['cost_per_interaction']
        cost_multiplier = 1.0
        if device == 'mobile':
            cost_multiplier *= 0.8
        if segment == 'loyal':
            cost_multiplier *= 0.6
        
        cost = round(base_cost * cost_multiplier * (0.5 + random.random()), 2)
        
        # Revenue (only if converting)
        revenue = 0
        if is_last:
            # Revenue based on segment and journey length
            base_revenue = 50 if segment == 'new' else 100 if segment == 'returning' else 150
            revenue = round(base_revenue * (1 + random.random() * 2), 2)
        
        return {
            'customer_id': customer_id,
            'session_id': f"s_{customer_id}_{sequence}_{random.randint(1000, 9999)}",
            'timestamp': timestamp.isoformat(),
            'channel': channel,
            'campaign_id': campaign_id,
            'campaign_type': campaign_type,
            'device': device,
            'touchpoint_number': sequence,
            'total_touchpoints': total_touchpoints,
            'time_on_site': round(np.random.exponential(180), 2),  # seconds
            'pages_viewed': np.random.poisson(3) + 1,
            'cost': cost,
            'converted': is_last,
            'revenue': revenue,
            'customer_segment': segment,
            'interaction_type': 'click' if random.random() < 0.8 else 'impression',
            'attribution_credit': 0.0  # To be calculated by attribution models
        }
    
    def _random_date(self) -> datetime:
        """Generate random date within range"""
        time_between = self.end_date - self.start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        return self.start_date + timedelta(days=random_days)
    
    def generate_dataset(self, num_customers: int, conversion_rate: float = 0.05) -> pd.DataFrame:
        """Generate complete dataset"""
        all_touchpoints = []
        
        print(f"Generating data for {num_customers} customers...")
        for customer_id in tqdm(range(1, num_customers + 1)):
            journey = self.generate_customer_journey(customer_id)
            all_touchpoints.extend(journey)
        
        df = pd.DataFrame(all_touchpoints)
        
        # Add derived features
        df['hour_of_day'] = pd.to_datetime(df['timestamp'], format='ISO8601').dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp'], format='ISO8601').dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp'], format='ISO8601').dt.month
        df['quarter'] = pd.to_datetime(df['timestamp'], format='ISO8601').dt.quarter
        
        # Calculate journey-level metrics
        journey_metrics = df.groupby('customer_id').agg({
            'converted': 'max',
            'revenue': 'sum',
            'cost': 'sum',
            'touchpoint_number': 'max'
        }).rename(columns={'touchpoint_number': 'journey_length'})
        
        # Merge back
        df = df.merge(
            journey_metrics[['journey_length']], 
            left_on='customer_id', 
            right_index=True,
            suffixes=('', '_journey')
        )
        
        return df

@click.command()
@click.option('--num-customers', default=10000, help='Number of customers to generate')
@click.option('--output-dir', default='data/raw', help='Output directory for generated data')
@click.option('--seed', default=42, help='Random seed for reproducibility')
def main(num_customers: int, output_dir: str, seed: int):
    """Generate sample marketing attribution data"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = MarketingDataGenerator(seed=seed)
    
    # Generate main dataset
    df = generator.generate_dataset(num_customers)
    
    # Save to multiple formats
    output_path_csv = os.path.join(output_dir, 'touchpoints.csv')
    output_path_parquet = os.path.join(output_dir, 'touchpoints.parquet')
    
    df.to_csv(output_path_csv, index=False)
    df.to_parquet(output_path_parquet, index=False)
    
    # Generate summary statistics
    summary = {
        'total_customers': num_customers,
        'total_touchpoints': len(df),
        'total_conversions': df.groupby('customer_id')['converted'].max().sum(),
        'conversion_rate': df.groupby('customer_id')['converted'].max().mean(),
        'avg_journey_length': df.groupby('customer_id').size().mean(),
        'total_revenue': df['revenue'].sum(),
        'total_cost': df['cost'].sum(),
        'roas': df['revenue'].sum() / df['cost'].sum() if df['cost'].sum() > 0 else 0,
        'channels': list(df['channel'].unique()),
        'date_range': {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max()
        }
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, 'data_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*50)
    print("DATA GENERATION COMPLETE")
    print("="*50)
    print(f"Total Customers: {summary['total_customers']:,}")
    print(f"Total Touchpoints: {summary['total_touchpoints']:,}")
    print(f"Total Conversions: {summary['total_conversions']:,}")
    print(f"Conversion Rate: {summary['conversion_rate']:.2%}")
    print(f"Average Journey Length: {summary['avg_journey_length']:.2f} touchpoints")
    print(f"Total Revenue: ${summary['total_revenue']:,.2f}")
    print(f"Total Cost: ${summary['total_cost']:,.2f}")
    print(f"ROAS: {summary['roas']:.2f}x")
    print(f"\nData saved to:")
    print(f"  - {output_path_csv}")
    print(f"  - {output_path_parquet}")
    print(f"  - {summary_path}")

if __name__ == '__main__':
    main()
