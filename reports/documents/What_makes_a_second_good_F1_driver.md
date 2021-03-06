# What makes a second good F1 driver

## Q1: Inferential Model

### 1. describe your model, and explain why each feature was selected.

I explore the relationship between the  position and two parts of variables:

1) drivers' personal experience and expertise:  their age, and nationality;

2) their performance at the specific race: average time to complete one lap in one race.

I chose these variables because 1) people's age will affect their experience and time of training for F1 race, therefore affecting their outcomes. Besides, some of the countries may pay more attention to F1 race and invest more on related facilities and training teams with good atmosphere, resulting better drivers of that nationality. 2) drivers may feel different every day, therefore, their personal skills are just part of the features defining their race results. Their performance in that specific day is also very importance. Thus, I use the average lap times to represent how good and lucky the drivers were at that time. 

To be more specific, I am expecting that age will negatively affect the race result, meaning older drivers with more experience will win a high rank (smaller position order). The shorter time they spent in laps, the smaller position they would get. 

I chose to use Ridge regression instead of classic OLS because the sparse matrix generated by the categorical variable (nationality) can cause trouble if I used classic OLS model. Besides, I can always fine-tune the alpha in Ridge to see the result from the tranditional OLS model. Therefore, I chose Ridge regression. 

The reason why I chose Ridge over Lasso is that I experimented before the existing script, and the result returned by Lasso in the basic run is worse than the one for Ridge regression. Therefore, I jump a little bit to the choice of Ridge. 

Another issue here is that I tried to use binary outcome variables (whether 2nd position or not) and employed logistic regression (and random forest classifier), but the model results were terrible, with AUC exactly 0.5, meaning no correct prediction. Therefore, I choose to use position order from 1 to approximately 16 as a seemly continous target variable in this project, in order to get a better predicting and explanatory model. 

### 2. provide statistics that show how well the model fits the data

According to the records of my 5 trails of the alpha in the Ridge regression model, when alpha equals 1, the r-squared for this ridge model is the best to fit the data. R-squared being 0.084 means that 8.4% of the variance in the target variable can be explained by the three variabls that I chose (age, averge lap time, and nationality). It is not too big, but since the number of variables is limited and there are so many things to define a second good driver. I would temporarily satisfied with this model. 

![inferential_model](https://github.com/QMSS-GR5069-Spring-2020/final-project-ChengweiWang3210/blob/master/reports/figures/inferential_model.png)

### 3. what is the most important variable in your model? How did you determine that?

First of all, I normalized the variables, therefore, their coefficients can compare with each other. 

According to the coefficients for features in my best fit Ridge model, being an Indian, Hungarian or Danish driver can less likely to get a smaller position (This is not a discrimination aganist anyone from any of these counties). Meaning they would be less likely to get second position. In contrast, being a driver from Columbia or Poland would be more likely to get the 2nd position. Also, (it may sound weird -- )  being too columbian or polish would also make the driver be too good to get the 2nd position (they would then be the winner.) To my surprise, the effect of age and average lap time is so minimal, meaning they may have little impact on the races's result.

![best_ridge_coef](https://github.com/QMSS-GR5069-Spring-2020/final-project-ChengweiWang3210/blob/master/reports/figures/best_ridge_coef.png)

### 4. provide some marginal effects for the variable that you identified as the most important in the model, and interpret it in the context of F1 races: in other words, give us the story that the data is providing you about drivers that come in second place

In my model, like I mentioned above, nationality has the most impactful relationship between the races' outcome. Being born in Poland, Columbia or Finland may result in being a good (winning the 2nd position) F1 driver. 

### 5. does it make sense to think of it as an "explanation" for drivers arriving in second place? or is it simply an association we observe in the data?

I do not think one's nationality can naturally make one a good or bad driver, even though the model said so. 

It is more correlation than causality  to me. Being born in some countries may entail a greater exposure to F1 culture and a richer related resources to them, and with these benefits, they gain more interest in F1 race and better training team.

Therefore, the relationship the model identified is only an association to me. 



## Q2. Prediction Model

### 1. describe your model, and explain how you selected the features that were selected

I use all the variables I could generate from the list of csv files related to F1 race in this model, including not only the three variables in the inferential model, but also the name of the constructors, the nationality of the constructors and the country where the circuits were held. 

About the country variable, even though in every race there is destined to have a 2nd driver, the correlation between certain drivers and counties is not destined. For example, some drivers may perform better in their hometown. Therefore, I also include this variable in my prediction model. 

Two variables of constructors are also important, as their car and speed of construction can largely affect drivers' outcome. Some brands (I assume that is what the constructorRef means) outperform than the others, and some countries (I guess Japan, Germany) have better skills and experience related to cars, and thus do a better construction job. 

I still include age and average lap times in this model, because I am expecting some interactive relationship between them and other variabls that could define the position for these drivers. 

### 2. provide statistics that show how good your model is at predicting, and how well it performed predicting second places in races between 2011 and 2017



I tried 6 groups parameters in random forest model. It turns out that with max_depth = 15 and n_estimators = 1000, the model can have a relatively smaller mean sqaured error and larger R-squared. This means that about 18.7% of variance in the position variable during 2011 to 2017 is explained by the features I chose above. 

![prediction_model](https://github.com/QMSS-GR5069-Spring-2020/final-project-ChengweiWang3210/blob/master/reports/figures/prediction_model.png)

### 3. the most important variable in (1) is bound to also be included in your predictive model. Provide marginal effects or some metric of importance for this variable and make an explicit comparison of this value with the values that you obtained in (1). How different are they? Why are they different?

This may be odd, from the list of the top 22 important features, we can see that average lap times and age, which are of low coefficients in the inferential model, are 2 most important features to predict drivers' position. Following them are several brands of constructors like Ferrari, Mclaren, etc. However, we do not know the direction of the relationship between these features and the outcome (like negatively or positively).

Nationality is also important, but the countries of this variable is changed. Being German or British can help predict the outcome better now, while the once important variables like being Columbian, Hungarian, Indian, or Polish is of rather small importance now (see the following 4 figures for the highlighted results.)

One of the reasons for this huge change is, I guess, the interaction between or even among variables. For example, maybe being relatively older and being a German while have ferrari as his or her constructor at the same time is a strong predictor for this driver's final result than merely being a Columbian (of course at the same time being columbian does not have strong interation with other features to affect the result.)	

![best_rf_importance](https://github.com/QMSS-GR5069-Spring-2020/final-project-ChengweiWang3210/blob/master/reports/figures/best_rf_importance.png)



![best_rf_importance(2)](https://github.com/QMSS-GR5069-Spring-2020/final-project-ChengweiWang3210/blob/master/reports/figures/best_rf_importance(2).png)

![best_rf_importance(3)](https://github.com/QMSS-GR5069-Spring-2020/final-project-ChengweiWang3210/blob/master/reports/figures/best_rf_importance(3).png)

![best_rf_importance(4)](https://github.com/QMSS-GR5069-Spring-2020/final-project-ChengweiWang3210/blob/master/reports/figures/best_rf_importance(4).png)

![best_rf_importance(5)](https://github.com/QMSS-GR5069-Spring-2020/final-project-ChengweiWang3210/blob/master/reports/figures/best_rf_importance(5).png)





