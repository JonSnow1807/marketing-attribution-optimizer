import pandas as pd
import json

# Load the real GA data
df = pd.read_csv('data/ga_journey_data.csv')

print("=== REAL Google Analytics Data ===")
print(f"Total rows: {len(df)}")
print(f"Unique visitors: {df['fullVisitorId'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Check for conversions
conversions = df[df['transactions'] > 0]
print(f"\nRows with conversions: {len(conversions)}")
print(f"Total transactions: {df['transactions'].sum():.0f}")
print(f"Total revenue: ${df['revenue'].sum():,.2f}")

print("\n=== Top 10 Channels ===")
print(df['channel'].value_counts().head(10))

print("\n=== Sample Converting Customer Journey ===")
if len(conversions) > 0:
    sample_customer = conversions['fullVisitorId'].iloc[0]
    journey = df[df['fullVisitorId'] == sample_customer][['visitNumber', 'date', 'channel', 'transactions', 'revenue']]
    print(f"Customer ID: {sample_customer}")
    print(journey)

# Save summary for documentation
summary = {
    "data_source": "Google Analytics Sample - BigQuery Public Dataset",
    "total_rows": len(df),
    "unique_visitors": df['fullVisitorId'].nunique(),
    "date_range": f"{df['date'].min()} to {df['date'].max()}",
    "total_transactions": int(df['transactions'].sum()),
    "total_revenue": float(df['revenue'].sum()),
    "data_type": "REAL e-commerce data from Google Merchandise Store"
}

with open('data/real_data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\n✅ Saved summary to data/real_data_summary.json")
