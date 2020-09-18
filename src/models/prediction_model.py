# Databricks notebook source
# MAGIC %md # Prediction Model

# COMMAND ----------

# MAGIC %md ### Notes:
# MAGIC 
# MAGIC The features I feed into the random forest model are :
# MAGIC 
# MAGIC 1) drivers' age and nationality
# MAGIC 
# MAGIC 2) drivers' performance in each race: average lap time
# MAGIC 
# MAGIC 3) Constructors' reference/name, where they are from (nationality)
# MAGIC 
# MAGIC 4) Where the circuits happened, countries. 

# COMMAND ----------

# install mlflow package.
dbutils.library.installPyPI("mlflow", "1.8.0")

# COMMAND ----------

# install packages
import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, auc, r2_score
from matplotlib import pyplot as plt
import seaborn as sns

import tempfile

# COMMAND ----------

# read in the data for 1950 to 2010, convert to pandas df
df_1950 = spark.read.csv('dbfs:/mnt/cw3210-gr5069/processed/f1_1950_2010.csv', header = True)
df_1950.count()
df_1950 = df_1950.toPandas()

# COMMAND ----------

# read in the data for 2011 to 2017, convert to pandas df
df_2011 = spark.read.csv('dbfs:/mnt/cw3210-gr5069/processed/f1_2011_2017.csv', header = True)
df_2011.count()
df_2011 = df_2011.toPandas()

# COMMAND ----------

# check and find out the number of columns are different in two dataset
# cause by the conversion from nominal variable to one hot columns. 
print(str(len(df_1950.columns)) + ' vs ' + str(len(df_2011.columns)))

# COMMAND ----------

# as the columns are different in two dataframes, 
# therefore, I have to concatenate them first, 
# filling 0 in the columns either of them don't have data.
X = pd.concat([df_1950.loc[:, "age":], df_2011.loc[:, "age":]]).fillna(0)

# COMMAND ----------

df_1950.shape

# COMMAND ----------

df_2011.shape

# COMMAND ----------

# get train data from 1950-2010 data set. 
X_train = X.iloc[0:5233, :].astype(float)
y_train = df_1950.loc[:, "target"].astype('int')

# COMMAND ----------

# get test data from 2011-2017 data set. 
X_test = X.iloc[5233:, :].astype(float)
y_test = df_2011.loc[:, "target"].astype('int')

# COMMAND ----------

with mlflow.start_run(run_name="Basic RF Experiment") as run:
  # Create model, train it, and create predictions
  rf = RandomForestRegressor()
  rf.fit(X_train, y_train)
  y_pred = rf.predict(X_test)
  
  # Log model
  mlflow.sklearn.log_model(rf, "random-forest-model")
  
  # Create metrics
  mse = mean_squared_error(y_test, y_pred)
  print("  mse: {}".format(mse))
  
  # Log metrics
  mlflow.log_metric("mse", mse)
  
  runID = run.info.run_uuid
  experimentID = run.info.experiment_id
  
  print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

# COMMAND ----------

def log_rf(experimentID, run_name, params, X_train, X_test, y_train, y_test):
  with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(**params) # 2 asterisks means unpack the values in the dictionary into functions, not the keys.
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-model")

    # Log params
    [mlflow.log_param(param, value) for param, value in params.items()]

    # Create metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("  mse: {}".format(mse))
    print("  R2: {}".format(r2))

    # Log metrics
    mlflow.log_metric("mse", mse) 
    mlflow.log_metric("r2", r2)  
    
    # Create feature importance
    importance = pd.DataFrame(list(zip(X.columns, rf.feature_importances_)), 
                                columns=["Feature", "Importance"]
                              ).sort_values("Importance", ascending=False)
    
    # Log importances using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
    temp_name = temp.name
    try:
      importance.to_csv(temp_name, index=False)
      mlflow.log_artifact(temp_name, "feature-importance.csv")
    finally:
      temp.close() # Delete the temp file
    
    # Create plot
    fig, ax = plt.subplots()

    sns.residplot(y_pred, y_test, lowess=True)
    plt.xlabel("Predicted values for Price ($)")
    plt.ylabel("Residual")
    plt.title("Residual Plot")

    # Log residuals using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="residuals-", suffix=".png")
    temp_name = temp.name
    try:
      fig.savefig(temp_name)
      mlflow.log_artifact(temp_name, "residuals.png")
    finally:
      temp.close() # Delete the temp file
      
    display(fig)
    return run.info.run_uuid

# COMMAND ----------

params = {
  "n_estimators": 100,
  "max_depth": 5,
  "random_state": 42
}

log_rf(experimentID, "First Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

params = {
  "n_estimators": 1000,
  "max_depth": 5,
  "random_state": 42
}

log_rf(experimentID, "Second Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

params = {
  "n_estimators": 100,
  "max_depth": 10,
  "random_state": 42
}

log_rf(experimentID, "Third Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

params = {
  "n_estimators": 1000,
  "max_depth": 10,
  "random_state": 42
}

log_rf(experimentID, "Fourth Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

params = {
  "n_estimators": 100,
  "max_depth": 15,
  "random_state": 42
}

log_rf(experimentID, "Fifth Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

params = {
  "n_estimators": 1000,
  "max_depth": 15,
  "random_state": 42
}

log_rf(experimentID, "Sixth Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------


