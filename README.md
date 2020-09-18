# What Makes A Better F1 Driver
Chengwei Wang

This projects predicts the F1 drivers' race outcomes by including variables like drivers' age, nationality, their in race performance, and their constructors, etc. 

There are 3 major folders in this repository. 

- **src** folder contains all the scripts, including:
  - *data* folder - s3_connection.py: code to mount databricks folder to AWS s3 buckets. The raw data is stored in ne-gr5069 bucket, while the processed files are stored in cw3210-gr5069 bucket as csv files. 
  - *feature* folder - clean_data_1950_2010.py: code to clean and transform data from 1950-01-01 to 2010-12-31, to prepare data for later modelling, mainly for inferential model. 
  - *feature* folder - clean_data_2011_2017.py: almost the same script as the above one, with the difference in filter condition for years. This generated data is for prediction model, as test set. 
  - *models* folder - inferential_model.py: this script adopt Ridge regression model to test the theoretical relationship between drivers' age, nationality and lap times. 
  - *models* folder - prediction_model.py: this script employs Random Forest model to use variables like constructors' brand and nationality and the country the races were held, and also the three variables included in inferential model, to predict the race outcome.
- **reports** folder contains
  - *documents* folder with md file called *What_makes_a_second_good_F1_driver*, which explains my intention of choosing models, variables, and way to transform data. Also, it has explanations of these model results and the possible reasons behind the differences between these two models. 
  - *figures* folder has png files that I screenshot for modelling comparison. These results are records of parameter use and model's performance that is recorded by *mlflow*. 
- **references** folder contains one markdown file served as data distionary, refering variables' name to its explanation. 

Please feel free to explore my project, and [email](mailto:cw3210@columbia.edu) me if you have any thoughts, doubts and feelings. 

