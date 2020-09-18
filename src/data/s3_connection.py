# Databricks notebook source
# MAGIC %md # Connect to AWS S3 buckets

# COMMAND ----------

# MAGIC %md #### 1. Connect to ne-gr5069 for raw data

# COMMAND ----------

ACCESS_KEY = ""
# Encode the Secret Key as that can contain "/"
SECRET_KEY = "".replace("/", "%2F")
AWS_BUCKET_NAME = "ne-gr5069"
MOUNT_NAME = "ne-gr5069"
# the bucket exists on AWS, we should **mount** the bucket in databricks, so that you can use it. 
# your data is on the cloud, you have to reference it on the cloud. Never store your data on github or local folders.

dbutils.fs.mount("s3a://%s:%s@%s" % (ACCESS_KEY, SECRET_KEY, AWS_BUCKET_NAME), "/mnt/%s" % MOUNT_NAME)
display(dbutils.fs.ls("/mnt/%s" % MOUNT_NAME))

# COMMAND ----------

# MAGIC %md #### 2. Connect to cw3210-gr5069 for writing csv file to "processed" folder

# COMMAND ----------

AWS_BUCKET_NAME = "cw3210-gr5069"
MOUNT_NAME = "cw3210-gr5069"

dbutils.fs.mount("s3a://%s:%s@%s" % (ACCESS_KEY, SECRET_KEY, AWS_BUCKET_NAME), "/mnt/%s" % MOUNT_NAME)
display(dbutils.fs.ls("/mnt/%s" % MOUNT_NAME))
