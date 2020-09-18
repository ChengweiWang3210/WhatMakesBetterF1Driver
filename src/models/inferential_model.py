# Databricks notebook source
# MAGIC %md # Inferential Model

# COMMAND ----------

# MAGIC %md ### Notes:
# MAGIC I will explore the relationship between whether or not 2nd position and three parts of variables: 
# MAGIC 
# MAGIC 1) drivers' personal experience and expertise : their age, nationality
# MAGIC 
# MAGIC 2) their performance at the specific race : average time to complete one lap in one race
# MAGIC 
# MAGIC therefore, I have to gather columns from several csv files, including results, races, drivers, lap, pitstop, constructors.

# COMMAND ----------

# install mlflow package.
dbutils.library.installPyPI("mlflow", "1.8.0")

# COMMAND ----------

# install package
import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, roc_curve, auc, r2_score
from matplotlib import pyplot as plt
import seaborn as sns
import tempfile

# COMMAND ----------

# read in data
df_combine = spark.read.csv('dbfs:/mnt/cw3210-gr5069/processed/f1_1950_2010.csv', header = True)
df_combine.count()

# COMMAND ----------

# convert spark df to pandas df
df_combine = df_combine.toPandas()

# COMMAND ----------

# train, test data split
X = df_combine.loc[:, "age":"nationality_Swiss"].astype(float)
y = df_combine.loc[:, "target"].astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# COMMAND ----------

# the first 6 columns are Ids and target. 
#X.columns
#X.head()

# COMMAND ----------

# basic run to get experiment ID
with mlflow.start_run(run_name="Basic RF Experiment") as run:
  # Create model, train it, and create predictions
  ridge = Ridge()
  ridge.fit(X_train, y_train)
  y_pred = ridge.predict(X_test)
  
  # Log model
  mlflow.sklearn.log_model(ridge, "ridge-regression-model")
  
  # Create metrics
  mse = mean_squared_error(y_test, predictions)
  print("  mse: {}".format(mse))
#  fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#  auc_lg = auc(fpr, tpr)
#  print("  auc: {}".format(auc_lg))
  
  # Log metrics
  mlflow.log_metric("mse", mse)
  
  runID = run.info.run_uuid
  experimentID = run.info.experiment_id
  
  print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

# COMMAND ----------

# function to run the regression and record key information about the model
def log_ridge(experimentID, run_name, params, X_train, X_test, y_train, y_test):
  
  with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
    # Create model, train it, and create predictions
    ridge = Ridge(**params) # 2 asterisks means unpack the values in the dictionary into functions, not the keys.
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(ridge, "ridge-model")

    # Log params
    [mlflow.log_param(param, value) for param, value in params.items()]

    # Create metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # auc
    #fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    #auc_lg = auc(fpr, tpr)
    print("  mse: {}".format(mse))
    print("  R2: {}".format(r2))

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)  
    #mlflow.log_metric("auc", auc_lg) 
    
    # Create feature importance
    coefficients = pd.DataFrame(list(zip(X.columns, ridge.coef_)), 
                                columns=["Feature", "Coefficient"]
                              ).sort_values("Coefficient", ascending=False)
    
    # Log importances using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="feature-coefficient-", suffix=".csv")
    temp_name = temp.name
    try:
      coefficients.to_csv(temp_name, index=False)
      mlflow.log_artifact(temp_name, "feature-coefficient.csv")
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
  "alpha": 1,
  "normalize": True # so that the coefficients are comparable. 
}

log_ridge(experimentID, "First Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

params = {
  "alpha": 10,
  "normalize": True # so that the coefficients are comparable. 
}

log_ridge(experimentID, "Second Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

params = {
  "alpha": 100,
  "normalize": True # so that the coefficients are comparable. 
}

log_ridge(experimentID, "Third Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

params = {
  "alpha": 1000,
  "normalize": True # so that the coefficients are comparable. 
}

log_ridge(experimentID, "Fourth Run", params, X_train, X_test, y_train, y_test)
