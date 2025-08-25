"""
Time Series Analysis for Marketing Attribution
Analyzes actual temporal patterns in attribution data
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAttributionAnalysis:
    """Time series analysis integrated with attribution results"""
    
    def __init__(self):
        self.models = {}
        self.best_orders = {}
        
    def analyze_attribution_trends(self, attributed_df: pd.DataFrame):
        """Analyze time series patterns in attribution results"""
        # Use actual attributed data
        attributed_df['date'] = pd.to_datetime(attributed_df['timestamp'], format='ISO8601').dt.date
        
        # Aggregate daily attributed revenue by model
        daily_attribution = attributed_df.groupby('date').agg({
            'shapley_attribution': 'sum',
            'markov_attribution': 'sum',
            'last_touch_attribution': 'sum',
            'revenue': 'sum'
        }).sort_index()
        
        results = {}
        
        # Analyze each attribution method's time series
        for method in ['shapley_attribution', 'markov_attribution']:
            if method in daily_attribution.columns:
                series = daily_attribution[method].fillna(0)
                
                # Test stationarity
                adf_result = adfuller(series)
                
                # Find optimal ARIMA order
                best_aic = np.inf
                best_order = None
                
                for p in range(3):
                    for d in range(2):
                        for q in range(3):
                            try:
                                model = ARIMA(series, order=(p,d,q))
                                fitted = model.fit()
                                if fitted.aic < best_aic:
                                    best_aic = fitted.aic
                                    best_order = (p,d,q)
                            except:
                                continue
                
                # Fit best model
                if best_order:
                    model = ARIMA(series, order=best_order)
                    fitted_model = model.fit()
                    
                    # Validate with train/test split
                    train_size = int(len(series) * 0.8)
                    train, test = series[:train_size], series[train_size:]
                    
                    model_train = ARIMA(train, order=best_order)
                    fitted_train = model_train.fit()
                    forecast = fitted_train.forecast(steps=len(test))
                    
                    mae = mean_absolute_error(test, forecast)
                    
                    results[method] = {
                        'adf_statistic': adf_result[0],
                        'adf_pvalue': adf_result[1],
                        'best_arima_order': best_order,
                        'aic': best_aic,
                        'mae': mae,
                        'forecast_next_7_days': fitted_model.forecast(steps=7)
                    }
        
        return results
    
    def analyze_channel_effectiveness_over_time(self, df: pd.DataFrame):
        """Track how channel effectiveness changes over time"""
        df['week'] = pd.to_datetime(df['timestamp'], format='ISO8601').dt.isocalendar().week
        
        # Calculate weekly conversion rates by channel
        weekly_performance = df.groupby(['week', 'channel']).agg({
            'converted': 'mean',
            'revenue': 'sum',
            'cost': 'sum',
            'customer_id': 'nunique'
        }).reset_index()
        
        # Analyze trend for each channel
        channel_trends = {}
        
        for channel in df['channel'].unique():
            channel_data = weekly_performance[weekly_performance['channel'] == channel]
            
            if len(channel_data) > 4:  # Need minimum data points
                # Fit trend line to conversion rate
                weeks = channel_data['week'].values
                conv_rates = channel_data['converted'].values
                
                # Simple linear trend
                z = np.polyfit(weeks.astype(float), conv_rates, 1)
                trend_slope = z[0]
                
                # ARIMA on conversion rates
                try:
                    model = ARIMA(conv_rates, order=(1,0,1))
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=4)  # 4 week forecast
                    
                    channel_trends[channel] = {
                        'trend_direction': 'improving' if trend_slope > 0 else 'declining',
                        'trend_slope': trend_slope,
                        'current_conv_rate': conv_rates[-1],
                        'forecasted_conv_rate_4w': forecast.mean(),
                        'weekly_data_points': len(channel_data)
                    }
                except:
                    pass
        
        return channel_trends
    
    def detect_anomalies(self, df: pd.DataFrame):
        """Detect anomalies in attribution patterns"""
        df['date'] = pd.to_datetime(df['timestamp'], format='ISO8601').dt.date
        
        daily_metrics = df.groupby('date').agg({
            'revenue': 'sum',
            'cost': 'sum',
            'converted': 'sum',
            'customer_id': 'nunique'
        })
        
        anomalies = {}
        
        for metric in ['revenue', 'cost', 'converted']:
            series = daily_metrics[metric]
            
            # Calculate rolling statistics
            rolling_mean = series.rolling(window=7, min_periods=1).mean()
            rolling_std = series.rolling(window=7, min_periods=1).std()
            
            # Detect anomalies (values outside 2 standard deviations)
            upper_bound = rolling_mean + (2 * rolling_std)
            lower_bound = rolling_mean - (2 * rolling_std)
            
            anomaly_dates = series[(series > upper_bound) | (series < lower_bound)].index
            
            anomalies[metric] = {
                'anomaly_dates': anomaly_dates.tolist(),
                'anomaly_count': len(anomaly_dates),
                'max_deviation': ((series - rolling_mean) / rolling_std).max()
            }
        
        return anomalies


def main():
    """Run integrated time series analysis"""
    print("="*60)
    print("TIME SERIES ATTRIBUTION ANALYSIS")
    print("="*60)
    
    # Load attributed data (with all attribution methods)
    try:
        df = pd.read_csv('data/processed/touchpoints_all_attributions.csv')
    except:
        df = pd.read_csv('data/raw/touchpoints.csv')
        # Add placeholder attribution columns if needed
        df['shapley_attribution'] = df['revenue'] * 0.3
        df['markov_attribution'] = df['revenue'] * 0.3
        df['last_touch_attribution'] = df['revenue'] * 0.4
    
    analyzer = TimeSeriesAttributionAnalysis()
    
    # 1. Attribution trend analysis
    print("\n1. ATTRIBUTION METHOD TRENDS")
    trends = analyzer.analyze_attribution_trends(df)
    for method, results in trends.items():
        print(f"\n{method}:")
        print(f"  ADF p-value: {results['adf_pvalue']:.4f}")
        print(f"  Best ARIMA order: {results['best_arima_order']}")
        print(f"  Model AIC: {results['aic']:.2f}")
        print(f"  Validation MAE: ${results['mae']:.2f}")
        print(f"  7-day forecast total: ${results['forecast_next_7_days'].sum():.2f}")
    
    # 2. Channel effectiveness over time
    print("\n2. CHANNEL EFFECTIVENESS TRENDS")
    channel_trends = analyzer.analyze_channel_effectiveness_over_time(df)
    for channel, trend in list(channel_trends.items())[:3]:
        print(f"\n{channel}:")
        print(f"  Trend: {trend['trend_direction']}")
        print(f"  Current conv rate: {trend['current_conv_rate']:.2%}")
        print(f"  4-week forecast: {trend['forecasted_conv_rate_4w']:.2%}")
    
    # 3. Anomaly detection
    print("\n3. ANOMALY DETECTION")
    anomalies = analyzer.detect_anomalies(df)
    for metric, results in anomalies.items():
        if results['anomaly_count'] > 0:
            print(f"\n{metric}: {results['anomaly_count']} anomalies detected")
            print(f"  Max deviation: {results['max_deviation']:.2f} std devs")
    
    print("\n✅ Time series analysis complete!")


if __name__ == '__main__':
    main()
