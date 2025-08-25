import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load real GA data
df = pd.read_csv('data/ga_journey_data.csv')

print("Preparing REAL Google Analytics data for attribution modeling...")
print(f"Original shape: {df.shape}")

# Convert GA data to your model's expected format
touchpoints = []

for idx, row in df.iterrows():
    # Create touchpoint record
    touchpoint = {
        'customer_id': row['fullVisitorId'],
        'touchpoint_id': f"{row['fullVisitorId']}_{row['visitNumber']}",
        'channel': row['channel'],
        'timestamp': pd.to_datetime(str(row['date']), format='%Y%m%d') + timedelta(hours=row['visitStartTime']/3600),
        'device': row['deviceCategory'],
        'cost': np.random.uniform(0.5, 5.0),  # Simulated cost since GA doesn't have this
        'time_on_site': row['timeOnSite'] if pd.notna(row['timeOnSite']) else 0,
        'pages_viewed': row['pageviews'] if pd.notna(row['pageviews']) else 1,
        'converted': 1 if pd.notna(row['transactions']) and row['transactions'] > 0 else 0,
        'revenue': row['revenue'] if pd.notna(row['revenue']) else 0
    }
    touchpoints.append(touchpoint)

# Create DataFrame
touchpoints_df = pd.DataFrame(touchpoints)

# Add touchpoint numbers and totals per customer
touchpoints_df = touchpoints_df.sort_values(['customer_id', 'timestamp'])
touchpoints_df['touchpoint_number'] = touchpoints_df.groupby('customer_id').cumcount() + 1
touchpoints_df['total_touchpoints'] = touchpoints_df.groupby('customer_id')['touchpoint_number'].transform('max')

# Save prepared data
touchpoints_df.to_csv('data/raw/touchpoints.csv', index=False)

print(f"\nPrepared data shape: {touchpoints_df.shape}")
print(f"Unique customers: {touchpoints_df['customer_id'].nunique()}")
print(f"Customers with conversions: {touchpoints_df.groupby('customer_id')['converted'].max().sum()}")
print(f"Total revenue: ${touchpoints_df.groupby('customer_id')['revenue'].max().sum():,.2f}")

print("\n✅ Saved prepared REAL data to data/raw/touchpoints.csv")
print("Your existing Shapley model can now run on REAL data!")
