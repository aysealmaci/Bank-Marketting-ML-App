The objective of this project is to build a machine learning model to predict
whether a client of a bank will subscribe to a term deposit or not.


Dataset can be reached from: https://archive.ics.uci.edu/dataset/222/bank+marketing

<b>Steps:</b>

1. Data cleaning
2. Data preprocessing
3. Feature selection
4. Model selection
5. Hyperparameter tuning 
6. Evaluation
7. Deployment

<b>Models we used:</b>

- Logistic Regression
- Random Forest
- Neural Network

The Streamlit cloud address of the deployed model is: https://bankmarketingmlapp-jdqxda7zpddexfnt8xazkf.streamlit.app/

<b>Procedure</b>

First of all, missing values in both categorical and numerical columns were addressed using appropriate techniques. Numerical values were imputed with the mode, while categorical values were imputed with One-Hot Encoding and Label Encoding methods. 

In the Data preprocessing section, One-Hot Encoding, categorical columns were one-hot encoded to represent them as binary values (0 or 1). This step was crucial for transforming categorical variables into a format suitable for analysis with the machine learning models. Label Encoding, categorical columns were label encoded to represent them as numerical values (such as 0, 1 or 2) while preserving their categorical nature. Mode imputation, the mode was used for imputing missing values in numerical columns, maintaining the integrity of numerical data. 

As the next step, Grid Search were used for Feature selection. The machine automatically selected the best features for model training and optimized feature selection based on evaluation metrics. Grid Search was used to automatically select the best hyperparameters for each model, providing an optimized model for term deposit subscription prediction. This technique considered various combinations of features to determine the most effective ones. Moreover, L1 & L2 regularization was used as a feature selection method, emphasizing sparsity and feature elimination for Logistic Regression model. 

In the end, calculating Mean AUC values for models, the best model which is Logistic Regression was selected. In addition, the pipeline of our model is created by using the selected model. Finally, our model is ready to be used in the deployment.
