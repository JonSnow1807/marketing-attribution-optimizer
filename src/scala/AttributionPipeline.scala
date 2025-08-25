package com.attribution

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}

/**
 * Attribution Pipeline using Spark MLlib
 * Implements scalable attribution modeling
 */
object AttributionPipeline {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Attribution Pipeline")
      .config("spark.sql.adaptive.enabled", "true")
      .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    println("=" * 60)
    println("ATTRIBUTION PIPELINE - SPARK MLLIB")
    println("=" * 60)
    
    // Read processed data
    val inputPath = if (args.length > 0) args(0) else "data/raw/touchpoints.csv"
    val modelPath = if (args.length > 1) args(1) else "models/spark_attribution_model"
    
    val data = DataProcessor.readTouchpointData(spark, inputPath)
    
    // Prepare features for ML
    println("\nPreparing features for ML pipeline...")
    val featuredData = prepareFeatures(data)
    
    // Split data
    val Array(trainData, testData) = featuredData.randomSplit(Array(0.8, 0.2), seed = 42)
    
    // Build and train model
    println("\nTraining Random Forest model...")
    val model = trainAttributionModel(trainData)
    
    // Evaluate model
    println("\nEvaluating model performance...")
    evaluateModel(model, testData)
    
    // Save model
    println(s"\nSaving model to: $modelPath")
    model.write.overwrite().save(modelPath)
    
    // Generate attribution scores
    println("\nGenerating attribution scores...")
    val attributions = generateAttributions(model, data)
    attributions.show(20, false)
    
    spark.stop()
  }
  
  /**
   * Prepare features for ML pipeline
   */
  def prepareFeatures(df: DataFrame): DataFrame = {
    import df.sparkSession.implicits._
    
    // Create customer-level features
    val customerFeatures = df.groupBy("customer_id")
      .agg(
        max("converted").alias("label"),
        count("*").alias("journey_length"),
        countDistinct("channel").alias("unique_channels"),
        countDistinct("device").alias("unique_devices"),
        sum("cost").alias("total_cost"),
        sum("time_on_site").alias("total_time_on_site"),
        sum("pages_viewed").alias("total_pages_viewed"),
        avg("time_on_site").alias("avg_time_on_site"),
        // Channel-specific features
        sum(when(col("channel") === "paid_search", 1).otherwise(0)).alias("paid_search_touches"),
        sum(when(col("channel") === "organic_search", 1).otherwise(0)).alias("organic_search_touches"),
        sum(when(col("channel") === "social_media", 1).otherwise(0)).alias("social_media_touches"),
        sum(when(col("channel") === "email", 1).otherwise(0)).alias("email_touches"),
        sum(when(col("channel") === "direct", 1).otherwise(0)).alias("direct_touches"),
        sum(when(col("channel") === "display", 1).otherwise(0)).alias("display_touches"),
        // Device features
        sum(when(col("device") === "mobile", 1).otherwise(0)).alias("mobile_touches"),
        sum(when(col("device") === "desktop", 1).otherwise(0)).alias("desktop_touches")
      )
    
    customerFeatures
  }
  
  /**
   * Train attribution model using Random Forest
   */
  def trainAttributionModel(trainData: DataFrame): PipelineModel = {
    // Select feature columns
    val featureCols = Array(
      "journey_length", "unique_channels", "unique_devices",
      "total_cost", "total_time_on_site", "total_pages_viewed", "avg_time_on_site",
      "paid_search_touches", "organic_search_touches", "social_media_touches",
      "email_touches", "direct_touches", "display_touches",
      "mobile_touches", "desktop_touches"
    )
    
    // Create feature vector
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features_raw")
    
    // Scale features
    val scaler = new StandardScaler()
      .setInputCol("features_raw")
      .setOutputCol("features")
    
    // Random Forest classifier
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(100)
      .setMaxDepth(10)
      .setFeatureSubsetStrategy("sqrt")
    
    // Create pipeline
    val pipeline = new Pipeline()
      .setStages(Array(assembler, scaler, rf))
    
    // Train model
    pipeline.fit(trainData)
  }
  
  /**
   * Evaluate model performance
   */
  def evaluateModel(model: PipelineModel, testData: DataFrame): Unit = {
    val predictions = model.transform(testData)
    
    // Binary classification evaluator
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("probability")
      .setMetricName("areaUnderROC")
    
    val auc = evaluator.evaluate(predictions)
    println(f"Model AUC: $auc%.4f")
    
    // Calculate accuracy
    val correctPredictions = predictions
      .select(col("label"), col("prediction"))
      .filter(col("label") === col("prediction"))
      .count()
    
    val totalPredictions = predictions.count()
    val accuracy = correctPredictions.toDouble / totalPredictions
    println(f"Model Accuracy: $accuracy%.4f")
    
    // Feature importance (for Random Forest)
    val rf = model.stages.last.asInstanceOf[org.apache.spark.ml.classification.RandomForestClassificationModel]
    println("\nFeature Importance:")
    val featureImportances = rf.featureImportances.toArray
    val featureCols = model.stages(0).asInstanceOf[VectorAssembler].getInputCols
    featureCols.zip(featureImportances).sortBy(-_._2).take(10).foreach { case (feature, importance) =>
      println(f"  $feature%-25s: $importance%.4f")
    }
  }
  
  /**
   * Generate attribution scores
   */
  def generateAttributions(model: PipelineModel, data: DataFrame): DataFrame = {
    val features = prepareFeatures(data)
    val predictions = model.transform(features)
    
    // Calculate attribution scores based on model predictions
    predictions.select(
      col("customer_id"),
      col("label").alias("actual_conversion"),
      col("prediction").alias("predicted_conversion"),
      col("probability").getItem(1).alias("conversion_probability"),
      col("journey_length"),
      col("unique_channels"),
      col("total_cost")
    ).withColumn("attribution_score", col("conversion_probability") * 100)
      .orderBy(desc("attribution_score"))
  }
}
