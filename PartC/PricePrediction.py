from pyspark import SparkConf, SparkContext, SQLContext
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import pyspark as spark
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from pyspark.sql.types import *
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.model_selection import KFold
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import DateType
from sklearn.ensemble import RandomForestRegressor
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import from_unixtime
from pyspark.sql.functions import mean
from pyspark.sql.functions import lag, col
from pyspark.sql.window import Window
from pyspark.ml import Pipeline


sc = SparkContext()
sql = spark.SQLContext(sc)

# load data into dataframe
bit_df = sql.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('input/bitstamp.csv')

# drop null values
bit_df = bit_df.na.drop()

# rename some columns
bit_df = bit_df.withColumnRenamed('Volume_(BTC)', 'Volume_BTC').withColumnRenamed('Volume_(Currency)', 'Volume_Currency')

# group all rows by date and calculate daily averages for each column
bit_df = bit_df.groupBy(from_unixtime("Timestamp", "yyyy-MM-dd").alias("Date")).agg(mean('Open').alias('Open'),
    mean('High').alias('High'), mean('Low').alias('Low'), mean('Close').alias('Close'),
    mean('Volume_BTC').alias('Volume_BTC'), mean('Volume_Currency').alias('Volume_Currency'), mean("Weighted_Price").alias("Weighted_Price"))

# set a window to order by Date
w = Window().partitionBy().orderBy(col("Date"))

# Copy the value of the Close field into the Price_After_Month field, 30 lines above
bit_df = bit_df.withColumn("Price_After_Month", lag("Close", -30, 'NaN').over(w))

# drop null rows
bit_df = bit_df.na.drop()

# vectorize the data
vectorAssembler = VectorAssembler(inputCols=['Open', 'High', 'Low', 'Close', 'Volume_BTC', 'Volume_Currency', 'Weighted_Price'], outputCol='features')
# vectorAssembler = VectorAssembler(inputCols=['DayMeanPrice'], outputCol='features')
vbit_df = vectorAssembler.transform(bit_df)

# split into train and test set. 70% train and 30% test
splits = vbit_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]

# train the Linear Regression Model
lr = LinearRegression(featuresCol='features', labelCol='Price_After_Month', maxIter=1000, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

train_df.describe().show()

# perform prediction on test data
lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction", "Price_After_Month", "features").show(5)
lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="Price_After_Month", metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

test_result = lr_model.evaluate(test_df)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)

# perform prediction on test data
predictions = lr_model.transform(test_df)
predictions = predictions.select("Date", "prediction", "Price_After_Month")

predictions.write.option("header", "true").csv('out/predictions')








