# Databricks notebook source
# MAGIC %md # Data cleaning and transformation for 1950 to 2010

# COMMAND ----------

# MAGIC %md ##### Notes:
# MAGIC 
# MAGIC 1) This script is preparing for data in **inferential model** and also the training data set for **prediction model**
# MAGIC 
# MAGIC 2) I tried to convert the **target** variable to binary variable with position = 2 is 1 and other position is 0, but the model results returned super awful with auc = 0.5, meaning 0 accurancy in prediction, and negative r-squared. Therefore, I stay with the seemly continuous position variable as target. However, the code used to converted target variable to binary is still in my script, in command #47, only being commentted. 
# MAGIC 
# MAGIC 3) About **categorical** variables. Converting nominal variables to one hot columns in pyspark leaves the dataframe unable to be written in csv file. Therefore, I cast the spark dataframe to pandas dataframe and convert them back to save to csv. Still, I leave the spark code to transform the nominal variables to categorical at bottom of this script, being commentted to avoid execution. 

# COMMAND ----------

# import neccessary libraries
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DateType
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from sklearn.preprocessing import OneHotEncoder

# COMMAND ----------

# see what's in the raw bucket 
display(dbutils.fs.ls("/mnt/ne-gr5069/raw"))

# COMMAND ----------

# code for inspecting through these csv files. 
#df_tmp = spark.read.csv('dbfs:/mnt/ne-gr5069/raw/qualifying.csv', header = True)
#print(df_tmp.count())
#display(df_tmp)

# COMMAND ----------

# results.csv is the main data frame here, with target variable "positionOrder"
# read in this csv file, and use it to combine with others to get enough information. 
df_results = spark.read.csv('dbfs:/mnt/ne-gr5069/raw/results.csv', header = True)
df_results.count()

# COMMAND ----------

#display(df_results)

# COMMAND ----------

# select useful columns to avoid redundance
df_results = df_results.select(['resultId', 'raceId', 'driverId', 'constructorId', 'positionOrder'])

# COMMAND ----------

# read in races.csv for races' date information. 
df_races = spark.read.csv('dbfs:/mnt/ne-gr5069/raw/races.csv', header = True)
df_races.count()

# COMMAND ----------

#display(df_races)

# COMMAND ----------

# select useful columns
df_races = df_races.select(['raceId', 'circuitId', 'date'])
# cast date type
df_races = df_races.withColumn('date', df_races.date.cast(DateType()))

# COMMAND ----------

# read in drivers.csv to get drivers' info for age and nationality.
df_drivers = spark.read.csv('dbfs:/mnt/ne-gr5069/raw/drivers.csv', header = True)
df_drivers.count()

# COMMAND ----------

#display(df_drivers)

# COMMAND ----------

# select useful columns
df_drivers = df_drivers.select(['driverId', 'dob', 'nationality'])
# cast date of birth to date type, preparing for calculating age. 
df_drivers = df_drivers.withColumn('dob', df_drivers.dob.cast(DateType()))

# COMMAND ----------

# combine info from results.csv, races.csv, and drivers.csv
# to calculate age, to filter out year from 1950 to 2010.
df_combine = df_results.join(df_races, on = 'raceId')
df_combine = df_combine.join(df_drivers, on = 'driverId')

# COMMAND ----------

# calculate drivers' age at the race year. 
df_combine = df_combine.withColumn('age', F.datediff(df_combine.date, df_combine.dob)/365, )
df_combine = df_combine.withColumn('age', df_combine.age.cast(IntegerType()))
# drop column no longer useful. 
df_combine = df_combine.drop('dob')

# COMMAND ----------

# filter out the results year between 1950 and 2010. 
df_combine = df_combine.filter(df_combine.date.between('1950-01-01', '2010-12-31'))

# COMMAND ----------

# check how many records in df_results is filtered out. 
print(df_combine.count())
df_results.count()

# COMMAND ----------

#display(df_combine)

# COMMAND ----------

# read in lap_times.csv, to get drivers' performance in each race. 
df_laptimes = spark.read.csv('dbfs:/mnt/ne-gr5069/raw/lap_times.csv', header = True)
df_laptimes.count()

# COMMAND ----------

#display(df_laptimes)

# COMMAND ----------

# select useful info, cast column data type, preparing for average calculation.
df_laptimes = df_laptimes.select(['raceId', 'driverId', 'milliseconds'])
df_laptimes = df_laptimes.withColumn('milliseconds', df_laptimes.milliseconds.cast(IntegerType()))

# COMMAND ----------

# calculate the average time spent for one lap for each driver used in each race. 
df_avg_laptimes = df_laptimes.groupBy('raceId', 'driverId').avg('milliseconds')
# change column name to be more meaningful
df_avg_laptimes = df_avg_laptimes.withColumnRenamed('avg(milliseconds)', 'avg_laptime')

# COMMAND ----------

# join the avg_laptime to the major dataframe.
df_combine = df_combine.join(df_avg_laptimes, on = ['raceId', 'driverId'])

# COMMAND ----------

#display(df_combine)

# COMMAND ----------

# get drivers' average time spent in pit stop in each race
# but the left join below failed to work
# so this piece of information is not adopted in the later model. 
df_pitstops = spark.read.csv('dbfs:/mnt/ne-gr5069/raw/pit_stops.csv', header = True)
df_pitstops.count()

# COMMAND ----------

#display(df_pitstops)

# COMMAND ----------

df_pitstops = df_pitstops.select(['raceId', 'driverId', 'milliseconds'])
df_pitstops = df_pitstops.withColumn('milliseconds', df_pitstops.milliseconds.cast(IntegerType()))

# COMMAND ----------

# calculate average time spent in pit stop for each driver in each race. 
df_avg_pitstops = df_pitstops.groupBy('raceId', 'driverId').avg('milliseconds')
df_avg_pitstops = df_avg_pitstops.withColumnRenamed('avg(milliseconds)', 'avg_pitstop')

# COMMAND ----------

#display(df_avg_pitstops)

# COMMAND ----------

# causing problem in df_combine, droped.
#df_combine = df_combine.join(df_avg_pitstops, on = ['raceId', 'driverId'])

# COMMAND ----------

#display(df_combine)

# COMMAND ----------

# get construtors' info from constructors.csv
df_constructors = spark.read.csv('dbfs:/mnt/ne-gr5069/raw/constructors.csv', header = True)
df_constructors.count()

# COMMAND ----------

#display(df_constructors)

# COMMAND ----------

df_constructors = df_constructors.select(['constructorId', 'constructorRef', 'nationality'])
# rename the column so that no conflict to drivers' nationality
df_constructors = df_constructors.withColumnRenamed('nationality', 'constructor_nationality')

# COMMAND ----------

# join information to the major dataframe
df_combine = df_combine.join(df_constructors, on = 'constructorId')

# COMMAND ----------

#display(df_combine)

# COMMAND ----------

# get circuits' info like which country the race happened from circuits.csv
df_circuits = spark.read.csv('dbfs:/mnt/ne-gr5069/raw/circuits.csv', header = True)
df_circuits.count()

# COMMAND ----------

#display(df_circuits)

# COMMAND ----------

# see how many counties has already thrown f1
#df_circuits.select('country').dropDuplicates().count()

# COMMAND ----------

df_circuits = df_circuits.select(['circuitId', 'circuitRef', 'country'])

# COMMAND ----------

# join circuits' info to major dataframe
df_combine = df_combine.join(df_circuits, on = 'circuitId')

# COMMAND ----------

#display(df_combine)

# COMMAND ----------

# alter target variable: 
#    change "positionOrder" column to binary variable "is the 2nd position or not"
#df_combine = df_combine.withColumn('positionOrder', 
#                                   F.when(F.col('positionOrder') == '2', 1).otherwise(0))
df_combine = df_combine.withColumnRenamed('positionOrder', 'target') 

# COMMAND ----------

# convert spark df to pandas df, for categorical variables transformation.
df_combine_pandas = df_combine.toPandas()

# COMMAND ----------

# get a copy to combine with converted columns
df_combine_cat = df_combine_pandas
# which columns are nominal variables.
categoricalColumns = ["nationality", "constructorRef", "constructor_nationality", "country"]
# loop over these columns to convert them to one hot columns.
for category in categoricalColumns:
  ohc = OneHotEncoder()
  # transformation
  ohe = ohc.fit_transform(df_combine_pandas[category].values.reshape(-1,1)).toarray()
  # transform array to pandas df
  df_onehot = pd.DataFrame(ohe, columns = [category + '_' + 
                                          ohc.categories_[0][i]
                                          for i in range(len(ohc.categories_[0]))])
  # combined with major dataframe
  df_combine_cat = pd.concat([df_combine_cat, df_onehot], axis = 1)

# COMMAND ----------

# check the columns
#df_combine_cat.columns

# COMMAND ----------

# convert back to spark df, to save to csv to bucket
df_combine = spark.createDataFrame(df_combine_cat)

# COMMAND ----------

# drop out no longer useful columns, make it easier to modelling.
df_combine = df_combine.drop('date', *categoricalColumns, 'circuitRef')

# COMMAND ----------

# write to csv to "processed" bucket in AWS s3.
df_combine.write.csv('/mnt/cw3210-gr5069/processed/f1_1950_2010.csv', 
                     header = True, mode = 'overwrite')

# COMMAND ----------

# MAGIC %md #### the end

# COMMAND ----------



# COMMAND ----------

# change nominal columns to one hot columns in spark data frame
# failed to save the returned df to csv because of the returned data type. 

#encoder = OneHotEncoderEstimator(inputCols=['nationality', 'constructorRef', 
#                                            'constructor_nationality', 'country'],
#                                 outputCols=["nationalityVec", "constructorVec", 
#                                            "constructor_nationalityVec", "countryVec"])

# COMMAND ----------

#categoricalColumns = ["nationality", "constructorRef", "constructor_nationality", "country"]
#stages = [] # stages in our Pipeline
#for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
#    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
#    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    # Add stages.  These are not run here, but will run all at once later on.
#    stages += [stringIndexer, encoder]

# COMMAND ----------

#partialPipeline = Pipeline().setStages(stages)
#pipelineModel = partialPipeline.fit(df_combine)

# COMMAND ----------

#preppedDataDF = pipelineModel.transform(df_combine)

# COMMAND ----------

#display(preppedDataDF)

# COMMAND ----------



# COMMAND ----------


