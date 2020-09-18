# Data Dictionary

This dictionary is for **f1_1950_2010.csv** and **f1_2011_2017.csv** in **cw3210-gr5069/processed** bucket. 

The first 5 columns with names ended with "Id", like circuitId, constructorId, raceId, driverId, resultId, are reference number for data wrangling and data join, therefore are not included in modelling. They can be ignored in terms of modelling and model explanation. 	

*target*: the target variable indicating races' results. Integer variable, starts from 1. The smaller the number, the better the result. 1 indicting the winner. 	

*age*: Integer variable indicating drivers' age at the time when the races were holding. 	

*avg_laptime*: the average time spent in laps in each race by each driver. 	

*nationality_*: column names like this indicting drivers' nationality, with the specific nationality recorded as 1, others recorded 0. For example, if driver1 was born in America, then for driver1, nationality_american is 1, and all the other "nationality_" like columns in the same observation are 0. 

*constructorRef_*: columns indicating every observation's constructor's brand. 

*constructor_nationality_*: columns indicating constructors' nationality. 

*country_*: columns representing the country where the race was held. 