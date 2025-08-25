#!/usr/bin/env Rscript

# Media Mix Modeling (MMM) Analysis
# Implements adstock transformation, saturation curves, and budget optimization

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(forecast)
  library(glmnet)
  library(ggplot2)
})

cat("========================================\n")
cat("MEDIA MIX MODELING (MMM) ANALYSIS\n")
cat("========================================\n\n")

# Adstock transformation function
apply_adstock <- function(x, rate = 0.5, max_lag = 8) {
  # Geometric adstock transformation
  n <- length(x)
  adstocked <- numeric(n)
  
  for (i in 1:n) {
    for (j in 0:min(i-1, max_lag)) {
      adstocked[i] <- adstocked[i] + x[i-j] * (rate^j)
    }
  }
  
  return(adstocked)
}

# Hill saturation transformation
apply_saturation <- function(x, alpha = 1, gamma = 1) {
  # Hill transformation for diminishing returns
  x_scaled <- x / max(x, na.rm = TRUE)
  return((x_scaled^alpha) / (x_scaled^alpha + gamma^alpha))
}

# Main MMM function
run_mmm_analysis <- function(data_path = "data/raw/touchpoints.csv") {
  
  # Load and prepare data
  cat("1. Loading and preparing data...\n")
  df <- read_csv(data_path, show_col_types = FALSE)
  
  # Aggregate by date and channel
  daily_data <- df %>%
    mutate(date = as.Date(timestamp)) %>%
    group_by(date, channel) %>%
    summarise(
      spend = sum(cost, na.rm = TRUE),
      impressions = n(),
      conversions = sum(converted, na.rm = TRUE),
      revenue = sum(revenue, na.rm = TRUE),
      .groups = 'drop'
    )
  
  # Pivot wider for modeling
  model_data <- daily_data %>%
    select(date, channel, spend, revenue) %>%
    pivot_wider(
      names_from = channel,
      values_from = c(spend, revenue),
      values_fill = 0
    )
  
  # Calculate total daily revenue (target variable)
  revenue_cols <- grep("revenue_", names(model_data), value = TRUE)
  model_data$total_revenue <- rowSums(model_data[revenue_cols])
  
  # Apply transformations
  cat("\n2. Applying adstock and saturation transformations...\n")
  
  spend_cols <- grep("spend_", names(model_data), value = TRUE)
  
  # Apply adstock to each channel
  for (col in spend_cols) {
    adstock_col <- gsub("spend_", "adstock_", col)
    model_data[[adstock_col]] <- apply_adstock(model_data[[col]], rate = 0.7)
  }
  
  # Apply saturation to adstocked values
  adstock_cols <- grep("adstock_", names(model_data), value = TRUE)
  
  for (col in adstock_cols) {
    saturation_col <- gsub("adstock_", "saturated_", col)
    model_data[[saturation_col]] <- apply_saturation(model_data[[col]])
  }
  
  # Build MMM model
  cat("\n3. Building Media Mix Model...\n")
  
  # Prepare features and target
  feature_cols <- grep("saturated_", names(model_data), value = TRUE)
  X <- as.matrix(model_data[feature_cols])
  y <- model_data$total_revenue
  
  # Add time trend and seasonality
  model_data$trend <- 1:nrow(model_data)
  model_data$day_of_week <- wday(model_data$date)
  model_data$month <- month(model_data$date)
  
  # Ridge regression for MMM
  cv_model <- cv.glmnet(X, y, alpha = 0)  # alpha=0 for ridge
  best_lambda <- cv_model$lambda.min
  
  final_model <- glmnet(X, y, alpha = 0, lambda = best_lambda)
  
  # Extract coefficients
  coef_values <- coef(final_model)[,1]
  channel_coefs <- data.frame(
    channel = gsub("saturated_", "", names(coef_values)[-1]),
    coefficient = as.numeric(coef_values[-1])
  ) %>%
    filter(coefficient != 0) %>%
    arrange(desc(abs(coefficient)))
  
  cat("\nChannel Coefficients:\n")
  print(channel_coefs, row.names = FALSE)
  
  # Calculate contribution
  cat("\n4. Calculating channel contributions...\n")
  
  contributions <- data.frame(channel = character(), 
                              contribution = numeric(),
                              roi = numeric())
  
  for (i in 1:length(feature_cols)) {
    channel_name <- gsub("saturated_|spend_", "", feature_cols[i])
    contribution <- sum(X[,i] * coef_values[i+1])
    spend_col <- paste0("spend_", channel_name)
    total_spend <- sum(model_data[[spend_col]], na.rm = TRUE)
    
    contributions <- rbind(contributions, data.frame(
      channel = channel_name,
      contribution = contribution,
      spend = total_spend,
      roi = ifelse(total_spend > 0, contribution / total_spend, 0)
    ))
  }
  
  contributions <- contributions %>%
    mutate(
      contribution_pct = contribution / sum(contribution) * 100
    ) %>%
    arrange(desc(contribution))
  
  cat("\nChannel Contributions:\n")
  print(contributions, row.names = FALSE)
  
  # Budget optimization
  cat("\n5. Budget Optimization Recommendations...\n")
  
  total_budget <- sum(contributions$spend)
  
  # Allocate budget proportional to ROI
  contributions <- contributions %>%
    mutate(
      optimal_spend = total_budget * (roi / sum(roi)),
      spend_change = optimal_spend - spend,
      spend_change_pct = (spend_change / spend) * 100
    )
  
  optimization_results <- contributions %>%
    select(channel, current_spend = spend, optimal_spend, 
           change_pct = spend_change_pct, roi) %>%
    arrange(desc(change_pct))
  
  cat("\nBudget Optimization Recommendations:\n")
  print(optimization_results, row.names = FALSE)
  
  # Create visualizations
  cat("\n6. Creating visualizations...\n")
  
  # ROI by channel
  p1 <- ggplot(contributions, aes(x = reorder(channel, roi), y = roi)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "ROI by Marketing Channel",
         x = "Channel", y = "ROI") +
    theme_minimal()
  
  # Contribution by channel
  p2 <- ggplot(contributions, aes(x = reorder(channel, contribution), y = contribution)) +
    geom_bar(stat = "identity", fill = "darkgreen") +
    coord_flip() +
    labs(title = "Revenue Contribution by Channel",
         x = "Channel", y = "Contribution") +
    theme_minimal()
  
  # Save plots
  output_dir <- "data/processed/mmm_results"
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  ggsave(file.path(output_dir, "roi_by_channel.png"), p1, width = 10, height = 6)
  ggsave(file.path(output_dir, "contribution_by_channel.png"), p2, width = 10, height = 6)
  
  # Save results
  write_csv(contributions, file.path(output_dir, "channel_contributions.csv"))
  write_csv(optimization_results, file.path(output_dir, "budget_optimization.csv"))
  
  cat(sprintf("\n✅ MMM analysis complete! Results saved to %s\n", output_dir))
  
  return(list(
    model = final_model,
    contributions = contributions,
    optimization = optimization_results
  ))
}

# Run if executed directly
if (!interactive()) {
  results <- run_mmm_analysis()
}
