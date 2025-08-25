name := "marketing-attribution-pipeline"

version := "1.0"

scalaVersion := "2.12.17"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.4.1",
  "org.apache.spark" %% "spark-sql" % "3.4.1",
  "org.apache.spark" %% "spark-mllib" % "3.4.1",
  "com.typesafe" % "config" % "1.4.2",
  "org.scalatest" %% "scalatest" % "3.2.15" % Test
)

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}
