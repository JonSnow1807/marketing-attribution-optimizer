# API Documentation

## Python Attribution Models

### ShapleyAttributionModel

#### Class: `ShapleyAttributionModel`
Game theory-based attribution using Shapley values to fairly distribute conversion credit.

```python
from attribution.shapley import ShapleyAttributionModel

# Initialize model
model = ShapleyAttributionModel(model_type='random_forest')
# model_type options: 'random_forest', 'gradient_boost', 'logistic'

# Fit the model
model.fit(df)

# Attribute conversions
attributed_df = model.attribute_conversions(df)

# Get channel performance metrics
performance = model.get_channel_performance(attributed_df)
```

**Parameters:**
- `model_type` (str): Type of sklearn model to use for conversion prediction

**Methods:**
- `fit(df)`: Train the attribution model on touchpoint data
- `attribute_conversions(df)`: Calculate attribution scores for each touchpoint
- `get_channel_performance(df)`: Generate channel-level performance metrics
- `plot_attribution_comparison(df)`: Visualize attribution method comparisons

**Returns:**
- DataFrame with `shapley_attribution` column containing attribution scores

---

### MarkovChainAttribution

#### Class: `MarkovChainAttribution`
Models customer journeys as state transitions to understand channel interactions.

```python
from attribution.markov import MarkovChainAttribution

# Initialize model
model = MarkovChainAttribution(order=1)
# order: 1 for first-order, 2 for second-order Markov chains

# Fit the model
model.fit(df)

# Attribute conversions
attributed_df = model.attribute_conversions(df)

# Get transition probabilities from a channel
transitions = model.get_transition_probabilities('paid_search')

# Visualize transition graph
fig = model.plot_transition_graph(top_n=20)
```

**Parameters:**
- `order` (int): Order of the Markov chain (default: 1)

**Methods:**
- `fit(df)`: Build transition matrix and calculate removal effects
- `attribute_conversions(df)`: Apply Markov attribution to touchpoints
- `get_channel_performance(df)`: Calculate channel metrics with removal effects
- `get_transition_probabilities(channel)`: Get transition probabilities from a channel
- `plot_transition_graph(top_n)`: Visualize customer journey transitions

**Returns:**
- DataFrame with `markov_attribution` column
- Removal effects dictionary
- Transition probability matrix

---

### MultiTouchAttribution

#### Class: `MultiTouchAttribution`
Unified framework comparing multiple attribution methods.

```python
from attribution.multi_touch import MultiTouchAttribution

# Initialize analyzer
analyzer = MultiTouchAttribution()

# Fit all models
analyzer.fit_all_models(df)

# Get attribution summary
summary = analyzer.get_attribution_summary()

# Get recommendations
recommendations = analyzer.get_recommendations()

# Create visualization
fig = analyzer.plot_attribution_comparison(figsize=(15, 10))
```

**Methods:**
- `fit_all_models(df)`: Fit Shapley, Markov, and traditional attribution models
- `get_attribution_summary()`: Get comparison table of all methods
- `get_recommendations()`: Generate actionable budget reallocation suggestions
- `plot_attribution_comparison()`: Create comprehensive visualization

**Returns:**
- Attribution summary DataFrame with variance analysis
- Recommendations dictionary with:
  - `undervalued_channels`: Channels to increase investment
  - `overvalued_channels`: Channels to review investment
  - `high_roi_channels`: Top performing channels
  - `budget_reallocation`: Specific budget change recommendations

---

## Spark Pipeline

### DataProcessor

Process large-scale touchpoint data using Apache Spark.

```bash
# Run data processor
spark-submit --class com.attribution.DataProcessor \
  --master local[*] \
  src/scala/target/scala-2.12/attribution-pipeline.jar \
  data/raw/touchpoints.csv \
  data/processed/spark_output
```

**Input Parameters:**
1. Input path (CSV file)
2. Output path (directory for Parquet files)

**Output:**
- Customer journeys with enriched features
- Channel performance metrics
- Journey patterns analysis
- Conversion paths
- Time-based patterns

### AttributionPipeline

ML pipeline using Spark MLlib for scalable attribution modeling.

```bash
# Run attribution pipeline
spark-submit --class com.attribution.AttributionPipeline \
  --master local[*] \
  src/scala/target/scala-2.12/attribution-pipeline.jar \
  data/raw/touchpoints.csv \
  models/spark_attribution_model
```

**Features:**
- Random Forest classification
- Feature importance analysis
- Model evaluation metrics
- Attribution score generation

---

## R Statistical Validation

### Statistical Validation Script

```r
# Source the validation script
source("src/r/statistical_validation.R")

# Run complete validation
results <- main()

# Access specific results
results$tests           # Hypothesis test results
results$confidence_intervals  # CI for channels
results$model          # Logistic regression model
results$interactions   # Channel interaction effects
results$power          # Statistical power analysis
```

**Functions:**
- `test_attribution_methods(df)`: ANOVA and Tukey HSD tests
- `calculate_confidence_intervals(df)`: Bootstrap CIs for channels
- `test_conversion_model(df)`: Logistic regression validation
- `test_channel_interactions(df)`: Interaction effects analysis
- `calculate_statistical_power(df)`: Effect size and power analysis

### Media Mix Modeling (MMM)

```r
# Source MMM script
source("src/r/mmm_analysis.R")

# Run MMM analysis
results <- run_mmm_analysis("data/raw/touchpoints.csv")

# Access results
results$model          # Ridge regression model
results$contributions  # Channel contributions
results$optimization   # Budget optimization recommendations
```

**Features:**
- Adstock transformation for lag effects
- Hill saturation curves for diminishing returns
- Ridge regression for robust estimation
- Budget optimization algorithm
- ROI projections

---

## Dashboard API

### Streamlit Application

```bash
# Run dashboard
streamlit run dashboard/app.py

# With custom port
streamlit run dashboard/app.py --server.port 8080
```

**Pages:**
1. **Overview**: Executive summary with KPIs
2. **Attribution Models**: Model comparison and analysis
3. **Channel Analysis**: Deep dive into channel performance
4. **Customer Journeys**: Journey patterns and Sankey diagrams
5. **ROI Optimization**: Budget recommendations
6. **Statistical Validation**: Test results and confidence intervals

**Features:**
- Interactive filters and date ranges
- Real-time model selection
- Export functionality for charts
- Responsive design

---

## Data Formats

### Input Data Schema

**touchpoints.csv**
```
customer_id: int
session_id: str
timestamp: datetime
channel: str
campaign_id: str
device: str
touchpoint_number: int
total_touchpoints: int
cost: float
converted: int (0/1)
revenue: float
time_on_site: float
pages_viewed: int
```

### Output Data Formats

**Attribution Results**
```
All input columns plus:
- shapley_attribution: float
- markov_attribution: float
- last_touch_attribution: float
- first_touch_attribution: float
- linear_attribution: float
- time_decay_attribution: float
- position_based_attribution: float
```

**Channel Performance**
```
channel: str
attributed_revenue: float
cost: float
unique_customers: int
roi: float
removal_effect: float
shapley_value: float
```

---

## Error Handling

All models include comprehensive error handling:

```python
try:
    model = ShapleyAttributionModel()
    model.fit(df)
except ValueError as e:
    print(f"Data validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

Common errors:
- `ValueError`: Invalid data format or missing columns
- `KeyError`: Required column not found
- `RuntimeError`: Model fitting failed
- `MemoryError`: Dataset too large for available memory

---

## Performance Considerations

### Optimization Tips
1. **Data Sampling**: For initial testing, use sample data
2. **Parallel Processing**: Leverage Spark for large datasets
3. **Caching**: Cache intermediate results in production
4. **Batch Processing**: Process attributions in batches
5. **Index Optimization**: Add indexes on customer_id and timestamp

### Benchmarks
- 10K customers: ~30 seconds
- 100K customers: ~5 minutes
- 1M customers: ~45 minutes (single machine)
- 1M customers: ~5 minutes (Spark cluster)
