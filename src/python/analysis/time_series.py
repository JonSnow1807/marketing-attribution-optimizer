"""
Time Series Analysis for Marketing Attribution
Implements ARIMA forecasting and trend analysis
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalysis:
    """Time series forecasting for marketing metrics"""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        
    def analyze_revenue_trends(self, df: pd.DataFrame):
        """Analyze and forecast revenue trends"""
        # Aggregate daily revenue
        df['date'] = pd.to_datetime(df['timestamp'], format='ISO8601').dt.date
        daily_revenue = df.groupby('date')['revenue'].sum().sort_index()
        
        # Test for stationarity
        adf_test = adfuller(daily_revenue)
        print(f"ADF Statistic: {adf_test[0]:.4f}")
        print(f"p-value: {adf_test[1]:.4f}")
        
        # Fit ARIMA model
        model = ARIMA(daily_revenue, order=(1,1,1))
        self.models['revenue'] = model.fit()
        
        # Generate forecasts
        forecast_steps = 30
        self.forecasts['revenue'] = self.models['revenue'].forecast(steps=forecast_steps)
        
        print(f"\n30-day Revenue Forecast:")
        print(f"Total Expected: ${self.forecasts['revenue'].sum():.2f}")
        print(f"Daily Average: ${self.forecasts['revenue'].mean():.2f}")
        
        return self.forecasts['revenue']
    
    def channel_seasonality_analysis(self, df: pd.DataFrame):
        """Analyze seasonality patterns by channel"""
        results = {}
        
        for channel in df['channel'].unique():
            channel_df = df[df['channel'] == channel].copy()
            channel_df['date'] = pd.to_datetime(channel_df['timestamp'], format='ISO8601')
            channel_df = channel_df.set_index('date')
            
            # Hourly patterns
            hourly_pattern = channel_df.groupby(channel_df.index.hour).size()
            
            # Day of week patterns
            dow_pattern = channel_df.groupby(channel_df.index.dayofweek).size()
            
            results[channel] = {
                'peak_hour': hourly_pattern.idxmax(),
                'peak_day': dow_pattern.idxmax(),
                'hourly_variance': hourly_pattern.var(),
                'weekly_variance': dow_pattern.var()
            }
        
        return results
    
    def forecast_conversions(self, df: pd.DataFrame, horizon: int = 7):
        """Forecast conversion rates using time series"""
        daily_conversions = df.groupby(pd.to_datetime(df['timestamp'], format='ISO8601').dt.date).agg({
            'converted': 'sum',
            'customer_id': 'nunique'
        })
        daily_conversions['conversion_rate'] = (
            daily_conversions['converted'] / daily_conversions['customer_id']
        )
        
        # ARIMA for conversion rate
        model = ARIMA(daily_conversions['conversion_rate'], order=(2,1,2))
        fitted_model = model.fit()
        
        forecast = fitted_model.forecast(steps=horizon)
        confidence_intervals = fitted_model.forecast(steps=horizon, alpha=0.05)
        
        return {
            'forecast': forecast,
            'model_aic': fitted_model.aic,
            'model_bic': fitted_model.bic
        }


def main():
    """Run time series analysis"""
    print("="*60)
    print("TIME SERIES ANALYSIS")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/raw/touchpoints.csv')
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalysis()
    
    # Revenue forecasting
    print("\n1. REVENUE FORECASTING (ARIMA)")
    revenue_forecast = analyzer.analyze_revenue_trends(df)
    
    # Seasonality analysis
    print("\n2. CHANNEL SEASONALITY PATTERNS")
    seasonality = analyzer.channel_seasonality_analysis(df)
    for channel, patterns in list(seasonality.items())[:3]:
        print(f"\n{channel}:")
        print(f"  Peak Hour: {patterns['peak_hour']}:00")
        print(f"  Peak Day: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][patterns['peak_day']]}")
    
    # Conversion forecasting
    print("\n3. CONVERSION RATE FORECASTING")
    conv_forecast = analyzer.forecast_conversions(df)
    print(f"Next 7 days avg conversion rate: {conv_forecast['forecast'].mean():.2%}")
    print(f"Model AIC: {conv_forecast['model_aic']:.2f}")
    
    print("\n✅ Time series analysis complete!")


if __name__ == '__main__':
    main()
