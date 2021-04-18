# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset used in this excerisce contains marketing campaign data from Bank . More specific, data are based on phone calls. Dataset has 32950 client with 20 data columns(e.g. age, job, loan, marital status, education, campaign, month, contact and other economic information). Moreover, it contains a target column with information if the client subscribed or not subscribed to a term deposit. We want to predict target column by applying different machine learning pipelines in azure.  
  
The first pipeline consists of a custom model scikit-learn logistic regression, using HyperDrive for hyperparameter tuning.

The second pileline consists of a model that was created with AutoML where several algorithms are trained and evaluated.

Comparing both pipelines we found that the model with best accuracy was VotingEnsemble with 0.9174 of accuracy from AutoML. 

## Scikit-learn Pipeline

**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
As first step a compute instance is created in azure. An azure jupyter notebook created and two files are added: train.py and the project notebook. A Tabular Dataset created with the Bank Marketing URL. This dataset preprocessed by a data cleaning and then splitted in training and testing datasets. The algorithm used for model training is Logistic Regression.

A sklearn estimator created, that used the train.py and the compute cluster we created. Hyperdrive is an azure package that contains modules for hyperparameter tuning. A random parameter sampling selected to avoid bias due its advantages in the speed/accuracy tradeoff. Also, I defined a discrete search space for the model hyperparameters. Model hyperparameters are Inverse of Regularization Strength and Maximum number of iterations to converge. Random sampling was selected in this project as this supports early termination of low perfomance runs and can be completed faster with a compute cluster. Grid Sampling and Bayesian Sampling also support early termination but it is used for exhaustive or very longer runs. Both of them require higher budget and time to explore the hyperparameter space, also the Bayesian sampling requires a small number of concurrent runs in order to have a better convergence. 

A very important Hyperdrive parameter is the early stopping policy. This helps to avoid overtraining and it is very helpful when the run has slow perfomance. The policy selected for this project is Badit Policy which is based on slack factor/slack amount and evaluation interval. The Bandit policy terminates the runs when the primary metric is not within the specified slack factor/slack amount compared to the best performing run. The Bandit policy was selected in this project because it monitors the best model and kills all the runs that perfoms poorly compate to the best run. This allow us to have a better control on the resources and the time of each run. Two other policies were considered, Median Stopping policy and Truncation Selection Policy. Median Stopping policy can take extend run time because it uses an average of the last measurements keeping some unnecesary runs for more time. The Truncation Selection Policy only kills a percentage of the low perfomance runs, but keeping all the others. 

## AutoML
Azure AutoML automates the process of iteratives steps in the machine learning. The AutoMLConfig class has been configured with the following parameters: 

* experiment_timeout_minutes - 30 minutes, 
* task - classification,
* primary_metric - accuracy, 
* training_data - training_data (Tabular Dataset preprocessed), 
* compute_target - cpu_cluster,
* label_column_name - y, 
* n_cross_validations - 10. 

From the best accuracy model, which is Voting Emsemble, some of the hyperparameters selected are:  
* max_iter=100
* multi_class='ovr'
* solver='saga'
* n_jobs=1
* tol=0.0001,

## Pipeline comparison
The best sklearn model created with hyperdrive has a accuracy value of 0.9124. The AutoML best model was the VotingEnsemble with 0.9174. Even if the difference between them is small, AutoML was better. The main difference in the both models are the structure, the hyperdrive tried one model while the AutoMl train and evaluates different models like LightGBM, XGBoost or RandomForest and also applies emsemble models that combines multiple learning algorithms to obtain the best predictive performance.   

## Future work
* We could increase the discrete search space for the parameter sampler in the hyperdrive configuration. This will enhance the sklearn model with a wider range of option values for the "Inverse of Regularization Strength and Maximum number of iterations to converge of the logistic regression model. This may lead us to a better accuracy model.
* Add more parameters to the hyperdrive for Logistic Regression model like: penalty, learning rate, tolerance of stopping criteria, solver intercept_scaling,. Adding more hyperparameters will allow us to have more tests in the training model and we would be able to get better parameters and eventuyally increace the perfomance of the model.
* For the autoML, we could change the configuration parameter: 'experiment_timeout_minutes'. This will allow us toevaluate of more algorithms like ANNs, and the final ensemble model will maybe have better perfomance.Also, we could include other parameters like 'enable_stack_ensemble and 'enable_dnn'. The first parameter can combine the predictions from multiple well-performing machine learning models. The second parameter'enable_dnn' will allows us to include deep neural network during selection. Adding those parameters will include new capabilities to the AutoML run, which can improve the final emsemble model accuracy.       
