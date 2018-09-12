from pyspark.sql import SparkSession


def get_spark(app_name):
    # init spark session
    spark = SparkSession \
        .builder \
        .appName(app_name) \
        .master("local") \
        .getOrCreate()

    log4jLogger = spark.sparkContext._jvm.org.apache.log4j
    log4jLogger.LogManager.getLogger('org').setLevel(log4jLogger.Level.ERROR)
    log4jLogger.LogManager.getLogger('akka').setLevel(log4jLogger.Level.ERROR)
    log4jLogger.LogManager.getRootLogger().setLevel(log4jLogger.Level.ERROR)
    return spark


def get_logger(spark, name):
    log4jLogger = spark.sparkContext._jvm.org.apache.log4j
    log4jLogger.LogManager.getLogger(name).setLevel(log4jLogger.Level.INFO)
    return log4jLogger.LogManager.getLogger(name)
