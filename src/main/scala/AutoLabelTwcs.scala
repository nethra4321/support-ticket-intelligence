import org.apache.spark.sql.{SparkSession, Column}
import org.apache.spark.sql.functions._

object AutoLabelTwcs {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("TWCS Auto Label")
      .getOrCreate()

    // ---- input/output ----
    val inputPath  = if (args.length > 0) args(0) else "data/twcs.csv"
    val outputPath = if (args.length > 1) args(1) else "data/twitter_train_spark"

    // ---- Load CSV ----
    val df0 = spark.read
      .option("header", "true")
      .option("multiLine", "true")
      .option("escape", "\"")
      .option("quote", "\"")
      .csv(inputPath)

    // inbound in this dataset is often "True"/"False" or "true"/"false"
    val df = df0
      .withColumn("inbound_str", lower(col("inbound").cast("string")))
      .filter(col("inbound_str") === "true")
      .withColumn("text_lc", lower(col("text")))

    // ---- Rules (priority order matters) ----
    // Note: rlike uses Java regex. Escape backslashes as needed.

    def r(p: String): Column = col("text_lc").rlike(p)

    val labelCol =
      when(
        r("installation.*failed|install.*error|unable to install|setup.*failed|setup.*error|can't install|cannot install|installing.*stuck|installation.*stuck"),
        lit("installation_issue")
      ).when(
        r("not working|down|unavailable|outage|offline|server.*down|service.*down"),
        lit("service_outage")
      ).when(
        r("app.*not working|app.*crash|app.*stuck|keeps crashing|won't open"),
        lit("app_issue")
      ).when(
        r("website.*not working|site.*down|page.*not loading|error.*page|404|500 error"),
        lit("website_issue")
      ).when(
        r("can't log in|cannot log in|login failed|unable to login|sign.*in.*issue|password.*incorrect"),
        lit("login_issue")
      ).when(
        r("delivery.*late|delayed|still waiting|not delivered yet|where.*delivery"),
        lit("delivery_delay")
      ).when(
        r("order.*not received|haven't received|never arrived|missing order|where.*my order"),
        lit("order_not_received")
      ).when(
        r("wrong item|incorrect item|missing item|item missing|sent wrong"),
        lit("wrong_or_missing_item")
      ).when(
        r("payment.*failed|transaction failed|card.*declined|payment.*error"),
        lit("payment_issue")
      ).when(
        r("charged twice|double charged|charged two times"),
        lit("charged_twice")
      ).when(
        r("refund.*not received|waiting for refund|refund pending|no refund yet"),
        lit("refund_not_received")
      ).when(
        r("defective|broken product|damaged|doesn't work|faulty"),
        lit("product_defective")
      ).when(
        r("poor quality|bad quality|cheap quality|not as expected"),
        lit("product_quality_issue")
      ).when(
        r("no response|no reply|support.*not responding|customer service.*bad|no help"),
        lit("support_issue")
      ).when(
        r("still not resolved|issue persists|problem not fixed|no solution"),
        lit("issue_not_resolved")
      ).when(
        r("disappointed|unhappy|terrible|worst|very bad|angry|frustrated"),
        lit("general_complaint")
      ).when(
        r("\\bhow\\b|\\bwhen\\b|\\bwhat\\b|\\bwhere\\b|can i|does|is it"),
        lit("query")
      ).when(
        r("thanks|thank you|great|awesome|love|excellent"),
        lit("praise")
      ).otherwise(lit("other"))

    val out = df
      .withColumn("label", labelCol)
      .select(col("text"), col("label"))

    // Optional: see distribution
    out.groupBy("label").count().orderBy(desc("count")).show(50, truncate = false)

    // ---- Write output ----
    // Spark writes a folder, not a single file.
    out
      .repartition(1) // makes a single part file (slower but convenient)
      .write.mode("overwrite")
      .option("header", "true")
      .csv(outputPath)

    spark.stop()
  }
}
