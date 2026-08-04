from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, when, lit

def main():
    spark = (
        SparkSession.builder
        .appName("TWCS Auto Label (Fast)")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()

    )

    input_path = "../data/twcs.csv"
    output_path = "../data/twitter_train_parquet"

    df = (
        spark.read
        .option("header", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .csv(input_path)
    )

    df = df.withColumn("inbound_lc", lower(col("inbound").cast("string"))).filter(col("inbound_lc") == "true")
    df = df.withColumn("text_lc", lower(col("text")))

    txt = col("text_lc")

    label_expr = (
        when(txt.rlike("installation.*failed|install.*error|unable to install|setup.*failed|setup.*error|can't install|cannot install|installing.*stuck|installation.*stuck"), lit("installation_issue"))
        .when(txt.rlike("not working|down|unavailable|outage|offline|server.*down|service.*down"), lit("service_outage"))
        .when(txt.rlike("app.*not working|app.*crash|app.*stuck|keeps crashing|won't open"), lit("app_issue"))
        .when(txt.rlike("website.*not working|site.*down|page.*not loading|error.*page|404|500 error"), lit("website_issue"))
        .when(txt.rlike("can't log in|cannot log in|login failed|unable to login|sign.*in.*issue|password.*incorrect"), lit("login_issue"))
        .when(txt.rlike("delivery.*late|delayed|still waiting|not delivered yet|where.*delivery"), lit("delivery_delay"))
        .when(txt.rlike("order.*not received|haven't received|never arrived|missing order|where.*my order"), lit("order_not_received"))
        .when(txt.rlike("wrong item|incorrect item|missing item|item missing|sent wrong"), lit("wrong_or_missing_item"))
        .when(txt.rlike("payment.*failed|transaction failed|card.*declined|payment.*error"), lit("payment_issue"))
        .when(txt.rlike("charged twice|double charged|charged two times"), lit("charged_twice"))
        .when(txt.rlike("refund.*not received|waiting for refund|refund pending|no refund yet"), lit("refund_not_received"))
        .when(txt.rlike("defective|broken product|damaged|doesn't work|faulty"), lit("product_defective"))
        .when(txt.rlike("poor quality|bad quality|cheap quality|not as expected"), lit("product_quality_issue"))
        .when(txt.rlike("no response|no reply|support.*not responding|customer service.*bad|no help"), lit("support_issue"))
        .when(txt.rlike("still not resolved|issue persists|problem not fixed|no solution"), lit("issue_not_resolved"))
        .when(txt.rlike("disappointed|unhappy|terrible|worst|very bad|angry|frustrated"), lit("general_complaint"))
        .when(txt.rlike("\\bhow\\b|\\bwhen\\b|\\bwhat\\b|\\bwhere\\b|can i|does|is it"), lit("query"))
        .when(txt.rlike("thanks|thank you|great|awesome|love|excellent"), lit("praise"))
        .otherwise(lit("other"))
    )

    out = df.withColumn("label", label_expr).select(col("text").alias("text"), col("label"))

    out.groupBy("label").count().orderBy(col("count").desc()).show(50, truncate=False)

    out.write.mode("overwrite").parquet(output_path)

    print(f"Wrote parquet to: {output_path}")

    
    spark.stop()

if __name__ == "__main__":
    main()
