# Install required R packages for attribution analysis
# Run this script first to set up R environment

packages <- c(
  "tidyverse",      # Data manipulation and visualization
  "ChannelAttribution",  # Attribution modeling
  "markovchain",    # Markov chain analysis
  "ggplot2",        # Advanced plotting
  "corrplot",       # Correlation plots
  "forecast",       # Time series analysis
  "broom",          # Tidy statistical output
  "car",            # Regression diagnostics
  "lmtest",         # Linear model testing
  "sandwich",       # Robust standard errors
  "boot",           # Bootstrap methods
  "pROC",           # ROC analysis
  "caret",          # Machine learning
  "glmnet",         # Regularized regression
  "jsonlite",       # JSON handling
  "data.table"      # Fast data manipulation
)

# Function to install packages if not already installed
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Install all packages
sapply(packages, install_if_missing)

cat("✅ All R packages installed successfully!\n")
