#!/usr/bin/env Rscript

# Statistical Validation for Marketing Attribution Models
# Performs hypothesis testing, confidence intervals, and statistical significance analysis

# Load libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(broom)
  library(car)
  library(lmtest)
  library(boot)
  library(pROC)
})

# Set working directory to project root
if (!exists("PROJECT_ROOT")) {
  PROJECT_ROOT <- here::here()
}

cat("========================================\n")
cat("STATISTICAL VALIDATION - R ANALYSIS\n")
cat("========================================\n\n")

# Function to read attribution data
read_attribution_data <- function(file_path = "data/processed/touchpoints_all_attributions.csv") {
  if (!file.exists(file_path)) {
    file_path <- file.path(PROJECT_ROOT, file_path)
  }
  
  df <- read_csv(file_path, show_col_types = FALSE)
  cat(sprintf("✓ Loaded %d touchpoints from %d customers\n", 
              nrow(df), length(unique(df$customer_id))))
  return(df)
}

# Function to perform statistical tests on attribution methods
test_attribution_methods <- function(df) {
  cat("\n1. COMPARING ATTRIBUTION METHODS\n")
  cat(paste(rep("=", 40), collapse=""), "\n")
  
  # Prepare data for comparison
  attribution_cols <- c("shapley_attribution", "markov_attribution", 
                       "last_touch_attribution", "linear_attribution")
  
  # Filter to only converted touchpoints for fair comparison
  converted_df <- df %>%
    filter(converted == 1)
  
  # Perform ANOVA to test if attribution methods differ significantly
  attribution_long <- converted_df %>%
    select(all_of(attribution_cols)) %>%
    pivot_longer(everything(), names_to = "method", values_to = "attribution")
  
  anova_result <- aov(attribution ~ method, data = attribution_long)
  cat("\nANOVA Test Results:\n")
  print(summary(anova_result))
  
  # Tukey HSD for pairwise comparisons
  tukey_result <- TukeyHSD(anova_result)
  cat("\nPairwise Comparisons (Tukey HSD):\n")
  print(tukey_result)
  
  # Return test statistics
  return(list(
    anova = anova_result,
    tukey = tukey_result
  ))
}

# Function to calculate confidence intervals
calculate_confidence_intervals <- function(df) {
  cat("\n2. CONFIDENCE INTERVALS FOR CHANNEL PERFORMANCE\n")
  cat(paste(rep("=", 40), collapse=""), "\n")
  
  # Bootstrap confidence intervals for channel attribution
  channels <- unique(df$channel)
  ci_results <- data.frame()
  
  for (ch in channels) {
    channel_data <- df %>% filter(channel == ch)
    
    # Bootstrap mean attribution
    boot_mean <- function(data, indices) {
      mean(data[indices, "shapley_attribution"], na.rm = TRUE)
    }
    
    boot_result <- boot(
      data = channel_data,
      statistic = boot_mean,
      R = 1000
    )
    
    ci <- boot.ci(boot_result, type = "perc")
    
    ci_results <- rbind(ci_results, data.frame(
      channel = ch,
      mean_attribution = mean(channel_data$shapley_attribution, na.rm = TRUE),
      ci_lower = ci$percent[4],
      ci_upper = ci$percent[5],
      std_error = sd(boot_result$t)
    ))
  }
  
  ci_results <- ci_results %>% arrange(desc(mean_attribution))
  cat("\n95% Confidence Intervals for Channel Attribution:\n")
  print(ci_results, row.names = FALSE)
  
  return(ci_results)
}

# Function to test conversion prediction model
test_conversion_model <- function(df) {
  cat("\n3. CONVERSION PREDICTION MODEL VALIDATION\n")
  cat(paste(rep("=", 40), collapse=""), "\n")
  
  # Prepare features for logistic regression
  model_data <- df %>%
    group_by(customer_id) %>%
    summarise(
      converted = max(converted),
      journey_length = n(),
      unique_channels = n_distinct(channel),
      total_cost = sum(cost),
      total_time = sum(time_on_site),
      has_paid_search = as.integer("paid_search" %in% channel),
      has_email = as.integer("email" %in% channel),
      has_social = as.integer("social_media" %in% channel),
      .groups = 'drop'
    )
  
  # Split data
  set.seed(42)
  train_idx <- sample(nrow(model_data), 0.8 * nrow(model_data))
  train_data <- model_data[train_idx, ]
  test_data <- model_data[-train_idx, ]
  
  # Fit logistic regression model
  model <- glm(converted ~ journey_length + unique_channels + total_cost + 
               total_time + has_paid_search + has_email + has_social,
               data = train_data, family = binomial)
  
  cat("\nLogistic Regression Model Summary:\n")
  print(summary(model))
  
  # Model diagnostics
  cat("\n\nVariance Inflation Factors (Multicollinearity Check):\n")
  print(vif(model))
  
  # Predictions on test set
  test_data$pred_prob <- predict(model, test_data, type = "response")
  test_data$pred_class <- ifelse(test_data$pred_prob > 0.5, 1, 0)
  
  # Calculate accuracy
  accuracy <- mean(test_data$pred_class == test_data$converted)
  cat(sprintf("\nModel Accuracy: %.2f%%\n", accuracy * 100))
  
  # ROC Analysis
  roc_obj <- roc(test_data$converted, test_data$pred_prob)
  auc_value <- auc(roc_obj)
  cat(sprintf("AUC Score: %.4f\n", auc_value))
  
  return(list(
    model = model,
    accuracy = accuracy,
    auc = auc_value,
    roc = roc_obj
  ))
}

# Function to test channel interaction effects
test_channel_interactions <- function(df) {
  cat("\n4. CHANNEL INTERACTION EFFECTS\n")
  cat(paste(rep("=", 40), collapse=""), "\n")
  
  # Create channel combination features
  journey_data <- df %>%
    group_by(customer_id) %>%
    summarise(
      converted = max(converted),
      revenue = max(revenue),
      has_search = as.integer(any(channel %in% c("paid_search", "organic_search"))),
      has_social = as.integer(any(channel == "social_media")),
      has_email = as.integer(any(channel == "email")),
      has_display = as.integer(any(channel == "display")),
      .groups = 'drop'
    ) %>%
    mutate(
      search_social = has_search * has_social,
      search_email = has_search * has_email,
      social_email = has_social * has_email
    )
  
  # Test interaction effects
  model_no_interaction <- glm(converted ~ has_search + has_social + has_email + has_display,
                              data = journey_data, family = binomial)
  
  model_with_interaction <- glm(converted ~ has_search + has_social + has_email + has_display +
                                search_social + search_email + social_email,
                                data = journey_data, family = binomial)
  
  # Likelihood ratio test
  lr_test <- lrtest(model_no_interaction, model_with_interaction)
  cat("\nLikelihood Ratio Test for Interaction Effects:\n")
  print(lr_test)
  
  # Check significance of interaction terms
  cat("\nInteraction Terms Significance:\n")
  interaction_summary <- summary(model_with_interaction)$coefficients[
    c("search_social", "search_email", "social_email"), 
  ]
  print(interaction_summary)
  
  return(list(
    lr_test = lr_test,
    interaction_model = model_with_interaction
  ))
}

# Function to calculate statistical power
calculate_statistical_power <- function(df) {
  cat("\n5. STATISTICAL POWER ANALYSIS\n")
  cat(paste(rep("=", 40), collapse=""), "\n")
  
  # Calculate effect sizes for different attribution methods
  converted_df <- df %>% filter(converted == 1)
  
  # Cohen's d for effect size
  calculate_cohens_d <- function(x, y) {
    nx <- length(x)
    ny <- length(y)
    mx <- mean(x, na.rm = TRUE)
    my <- mean(y, na.rm = TRUE)
    sx <- sd(x, na.rm = TRUE)
    sy <- sd(y, na.rm = TRUE)
    s_pooled <- sqrt(((nx - 1) * sx^2 + (ny - 1) * sy^2) / (nx + ny - 2))
    d <- (mx - my) / s_pooled
    return(d)
  }
  
  # Compare Shapley vs Last Touch
  shapley_values <- converted_df$shapley_attribution
  last_touch_values <- converted_df$last_touch_attribution
  
  effect_size <- calculate_cohens_d(shapley_values, last_touch_values)
  
  cat(sprintf("\nEffect Size (Cohen's d) - Shapley vs Last Touch: %.3f\n", effect_size))
  
  # Interpret effect size
  if (abs(effect_size) < 0.2) {
    interpretation <- "Small"
  } else if (abs(effect_size) < 0.5) {
    interpretation <- "Medium"
  } else {
    interpretation <- "Large"
  }
  
  cat(sprintf("Effect Size Interpretation: %s\n", interpretation))
  
  # Sample size calculation for future studies
  library(pwr)
  power_calc <- pwr.t.test(d = effect_size, sig.level = 0.05, power = 0.8)
  cat(sprintf("\nRequired sample size for 80%% power: %.0f per group\n", ceiling(power_calc$n)))
  
  return(list(
    effect_size = effect_size,
    interpretation = interpretation,
    required_n = ceiling(power_calc$n)
  ))
}

# Main execution
main <- function() {
  # Load data
  df <- read_attribution_data()
  
  # Run statistical tests
  test_results <- test_attribution_methods(df)
  ci_results <- calculate_confidence_intervals(df)
  model_results <- test_conversion_model(df)
  interaction_results <- test_channel_interactions(df)
  power_results <- calculate_statistical_power(df)
  
  # Save results
  cat("\n\nSAVING RESULTS\n")
  cat(paste(rep("=", 40), collapse=""), "\n")
  
  # Create output directory if it doesn't exist
  output_dir <- file.path(PROJECT_ROOT, "data", "processed", "r_validation")
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Save confidence intervals
  write_csv(ci_results, file.path(output_dir, "confidence_intervals.csv"))
  
  # Save model diagnostics
  model_summary <- tidy(model_results$model)
  write_csv(model_summary, file.path(output_dir, "model_coefficients.csv"))
  
  # Create summary report
  summary_report <- list(
    test_date = Sys.Date(),
    n_customers = length(unique(df$customer_id)),
    n_touchpoints = nrow(df),
    model_accuracy = model_results$accuracy,
    model_auc = model_results$auc,
    effect_size = power_results$effect_size,
    required_sample_size = power_results$required_n
  )
  
  write(toJSON(summary_report, pretty = TRUE), 
        file.path(output_dir, "statistical_summary.json"))
  
  cat(sprintf("\n✅ Results saved to %s\n", output_dir))
  
  return(list(
    tests = test_results,
    confidence_intervals = ci_results,
    model = model_results,
    interactions = interaction_results,
    power = power_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  results <- main()
}
