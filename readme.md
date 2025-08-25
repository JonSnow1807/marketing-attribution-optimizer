# Marketing Attribution Optimizer

Real-world marketing attribution system using Google Analytics data from BigQuery public datasets.

## 🎯 Project Overview

Built to demonstrate production ML engineering skills for marketing attribution after receiving feedback about needing Scala/Spark and attribution modeling experience.

## 📊 Real Results (Not Simulated)

- **Data Source**: Google Analytics Sample (`bigquery-public-data.google_analytics_sample`)
- **Records Processed**: 9,617 real e-commerce touchpoints
- **Unique Visitors**: 8,076
- **Actual Conversions**: 138 transactions ($14,950 revenue)
- **Conversion Rate**: 1.43% (realistic e-commerce rate)
- **Processing Platform**: Databricks Spark 4.0.0

## ✅ Technologies Demonstrated

### Spark/Scala
- Compiled Scala attribution pipeline with sbt
- Executed PySpark on Databricks cloud platform
- Processed real GA data on distributed Spark cluster
- [View Databricks execution proof](docs/databricks_screenshots/)

### Python (scikit-learn)
- Shapley value attribution with Monte Carlo approximation
- Random Forest: 0.98 AUC (likely overfit on small dataset)
- Markov chain attribution modeling

### R Statistical Validation
- ANOVA hypothesis testing (p<2e-16)
- Statistical significance validation

## 🚀 Key Features

- **Multi-Touch Attribution**: Shapley values, Markov chains, position-based
- **Real Data Pipeline**: BigQuery → Spark → Attribution Models
- **Cloud Deployment**: Executed on Databricks Community Edition
- **ML Models**: RandomForestClassifier for conversion prediction

## 📈 Attribution Performance

| Channel | Attributed Revenue | Touchpoints | Conv Rate |
|---------|-------------------|-------------|-----------|
| Direct | $7,848 | 2,784 | 2.59% |
| Google Organic | $2,265 | 5,009 | 1.00% |
| Google CPC | $444 | 249 | 2.81% |

## 🛠️ Setup & Execution

See documentation for:
- [Databricks execution screenshots](docs/databricks_screenshots/)
- [Attribution model details](src/python/attribution/)
- [Statistical validation](src/r/)

## 📝 Note

This project demonstrates the ability to quickly learn and implement required technologies. While not claiming years of production experience, it shows competence in the exact skills requested: Scala/Spark processing, attribution modeling, and statistical validation using R.

---
Built by Chinmay Shrivastava | [GitHub](https://github.com/JonSnow1807)
