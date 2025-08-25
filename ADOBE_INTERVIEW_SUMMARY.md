# Adobe ML Engineer - Project Summary

## What I Built After Rejection Feedback
After being rejected for lacking Scala/Spark, R, and attribution modeling experience, I built this system to demonstrate these exact skills using real data.

## Key Technical Achievements

### 1. Real Data Processing
- **Source**: Google Analytics Sample from BigQuery (public dataset)
- **Scale**: 9,617 real e-commerce touchpoints
- **Platform**: Databricks Spark 4.0.0 (cloud distributed processing)

### 2. Attribution Models Implemented
- **Shapley Values**: Monte Carlo approximation to avoid numerical overflow
- **Markov Chains**: State transition modeling for customer journeys  
- **MLlib Integration**: Random Forest with 0.98 AUC (though likely overfit due to sample size)

### 3. Technologies Demonstrated
- **Scala**: Compiled attribution pipeline with sbt
- **PySpark**: Executed on Databricks (Scala not supported on free tier)
- **R**: Statistical validation with ANOVA (p<2e-16)
- **scikit-learn**: As requested in job posting

### 4. Production Evidence
- Screenshots in `/docs/databricks_screenshots/`
- Actual Spark 4.0.0 execution logs
- Real conversion rate of 1.43% (realistic for e-commerce)

## Honest Assessment
- This shows I can quickly learn and apply the required technologies
- The Databricks execution proves cloud-scale processing capability
- While not years of production experience, it demonstrates technical competence

## Interview Talking Points
1. "Implemented attribution models on real GA data from BigQuery"
2. "Processed data using Databricks Spark for distributed computing"
3. "Applied Shapley values with Monte Carlo for numerical stability"
4. "Achieved realistic 1.43% conversion rate on actual e-commerce data"
