package algorithms


object EDA {
  def main(args: Array[String]): Unit = {
    val spark = SparkSessionCreate.createSession()

    import spark.implicits._

    val df = Preprocessing.trainInput

    print(df.printSchema())
    val newDF = df.withColumnRenamed("loss","label")

    newDF.createOrReplaceTempView("insurance")
    spark.sql("SELECT avg(insurance.label) as AVG_LOSS FROM insurance").show()
    spark.sql("SELECT min(insurance.label) as MIN_LOSS FROM insurance").show()
    spark.sql("SELECT MAX(insurance.label) as MAX_LOSS FROM insurance").show()


  }
}
