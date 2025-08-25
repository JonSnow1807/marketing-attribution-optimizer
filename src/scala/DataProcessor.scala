package com.attribution

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._

/**
 * Marketing Attribution Data Processor
 * Handles large-scale data processing for attribution analysis
 */
object DataProcessor {
  
  def main(args: Array[String]): Unit = {
    // Initialize Spark session
    val spark = SparkSession.builder()
      .appName("Marketing Attribution Data Processor")
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .getOrCreate()
    
    import spark.implicits._
    
    // Set log level
    spark.sparkContext.setLogLevel("WARN")
    
    println("=" * 60)
    println("MARKETING ATTRIBUTION DATA PROCESSOR")
    println("Powered by Apache Spark")
    println("=" * 60)
    
    // Read input data
    val inputPath = if (args.length > 0) args(0) else "data/raw/touchpoints.csv"
    val outputPath = if (args.length > 1) args(1) else "data/processed/spark_output"
    
    println(s"\nReading data from: $inputPath")
    val rawData = readTouchpointData(spark, inputPath)
    
    // Show data statistics
    println(s"\nData Statistics:")
    println(s"Total Records: ${rawData.count()}")
    println(s"Unique Customers: ${rawData.select("customer_id").distinct().count()}")
    println(s"Unique Channels: ${rawData.select("channel").distinct().count()}")
    
    // Process customer journeys
    println("\nProcessing customer journeys...")
    val journeys = processCustomerJourneys(rawData)
    
    // Calculate channel statistics
    println("\nCalculating channel performance metrics...")
    val channelMetrics = calculateChannelMetrics(rawData)
    channelMetrics.show(20, false)
    
    // Calculate journey patterns
    println("\nAnalyzing journey patterns...")
    val journeyPatterns = analyzeJourneyPatterns(journeys)
    journeyPatterns.show(10, false)
    
    // Calculate conversion paths
    println("\nIdentifying top conversion paths...")
    val conversionPaths = identifyConversionPaths(rawData)
    conversionPaths.show(10, false)
    
    // Calculate time-based patterns
    println("\nAnalyzing time-based patterns...")
    val timePatterns = analyzeTimePatterns(rawData)
    timePatterns.show(10, false)
    
    // Save processed data
    println(s"\nSaving processed data to: $outputPath")
    saveProcessedData(journeys, channelMetrics, journeyPatterns, conversionPaths, outputPath)
    
    println("\n✅ Spark processing complete!")
    
    spark.stop()
  }
  
  /**
   * Read touchpoint data from CSV
   */
  def readTouchpointData(spark: SparkSession, path: String): DataFrame = {
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)
  }
  
  /**
   * Process customer journeys
   */
  def processCustomerJourneys(df: DataFrame): DataFrame = {
    import df.sparkSession.implicits._
    
    // Define window for journey analysis
    val customerWindow = Window.partitionBy("customer_id").orderBy("touchpoint_number")
    
    df.withColumn("prev_channel", lag("channel", 1).over(customerWindow))
      .withColumn("next_channel", lead("channel", 1).over(customerWindow))
      .withColumn("time_since_prev", 
        when(lag("timestamp", 1).over(customerWindow).isNotNull,
          unix_timestamp(col("timestamp")) - unix_timestamp(lag("timestamp", 1).over(customerWindow))
        ).otherwise(0)
      )
      .withColumn("is_first_touch", col("touchpoint_number") === 1)
      .withColumn("is_last_touch", col("touchpoint_number") === col("total_touchpoints"))
  }
  
  /**
   * Calculate channel-level metrics
   */
  def calculateChannelMetrics(df: DataFrame): DataFrame = {
    df.groupBy("channel")
      .agg(
        count("*").alias("total_touches"),
        countDistinct("customer_id").alias("unique_customers"),
        sum("cost").alias("total_cost"),
        sum("revenue").alias("total_revenue"),
        avg("time_on_site").alias("avg_time_on_site"),
        avg("pages_viewed").alias("avg_pages_viewed"),
        sum(when(col("converted") === 1, 1).otherwise(0)).alias("conversions"),
        sum(when(col("touchpoint_number") === 1, 1).otherwise(0)).alias("first_touches"),
        sum(when(col("touchpoint_number") === col("total_touchpoints"), 1).otherwise(0)).alias("last_touches")
      )
      .withColumn("conversion_rate", col("conversions") / col("total_touches"))
      .withColumn("roi", (col("total_revenue") - col("total_cost")) / col("total_cost"))
      .withColumn("revenue_per_customer", col("total_revenue") / col("unique_customers"))
      .orderBy(desc("total_revenue"))
  }
  
  /**
   * Analyze journey patterns
   */
  def analyzeJourneyPatterns(df: DataFrame): DataFrame = {
    import df.sparkSession.implicits._
    
    df.groupBy("customer_id")
      .agg(
        collect_list("channel").alias("journey_path"),
        max("converted").alias("converted"),
        max("revenue").alias("revenue"),
        sum("cost").alias("total_cost"),
        count("*").alias("journey_length"),
        countDistinct("channel").alias("unique_channels"),
        countDistinct("device").alias("unique_devices")
      )
      .withColumn("journey_path_str", concat_ws(" -> ", col("journey_path")))
      .groupBy("journey_path_str", "converted")
      .agg(
        count("*").alias("frequency"),
        avg("revenue").alias("avg_revenue"),
        avg("journey_length").alias("avg_length")
      )
      .orderBy(desc("frequency"))
  }
  
  /**
   * Identify top conversion paths
   */
  def identifyConversionPaths(df: DataFrame): DataFrame = {
    import df.sparkSession.implicits._
    
    // Filter for converting journeys only
    val convertingJourneys = df.filter(col("converted") === 1)
    
    convertingJourneys
      .groupBy("customer_id")
      .agg(
        collect_list("channel").alias("conversion_path"),
        max("revenue").alias("revenue")
      )
      .withColumn("path_str", concat_ws(" -> ", col("conversion_path")))
      .groupBy("path_str")
      .agg(
        count("*").alias("conversions"),
        sum("revenue").alias("total_revenue"),
        avg("revenue").alias("avg_revenue")
      )
      .orderBy(desc("conversions"))
  }
  
  /**
   * Analyze time-based patterns
   */
  def analyzeTimePatterns(df: DataFrame): DataFrame = {
    df.withColumn("hour", hour(col("timestamp")))
      .withColumn("day_of_week", dayofweek(col("timestamp")))
      .withColumn("month", month(col("timestamp")))
      .groupBy("hour", "day_of_week")
      .agg(
        count("*").alias("touches"),
        sum(when(col("converted") === 1, 1).otherwise(0)).alias("conversions"),
        avg("time_on_site").alias("avg_engagement")
      )
      .withColumn("conversion_rate", col("conversions") / col("touches"))
      .orderBy("hour", "day_of_week")
  }
  
  /**
   * Save processed data
   */
  def saveProcessedData(
    journeys: DataFrame,
    channelMetrics: DataFrame,
    journeyPatterns: DataFrame,
    conversionPaths: DataFrame,
    outputPath: String
  ): Unit = {
    // Save as Parquet for efficient storage
    journeys.coalesce(1).write.mode("overwrite").parquet(s"$outputPath/journeys")
    channelMetrics.coalesce(1).write.mode("overwrite").parquet(s"$outputPath/channel_metrics")
    journeyPatterns.coalesce(1).write.mode("overwrite").parquet(s"$outputPath/journey_patterns")
    conversionPaths.coalesce(1).write.mode("overwrite").parquet(s"$outputPath/conversion_paths")
    
    // Also save channel metrics as CSV for easy viewing
    channelMetrics.coalesce(1).write
      .mode("overwrite")
      .option("header", "true")
      .csv(s"$outputPath/channel_metrics_csv")
  }
}
