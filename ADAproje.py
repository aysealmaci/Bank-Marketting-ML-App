#!/usr/bin/env python
# coding: utf-8

# # 

# ### Import Modules

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('bank-additional.csv', sep=';')

# Prints the number of samples (rows) in the DataFrame
print ('Number of samples: ',len(df))


# In[3]:


# Display the first few rows of the DataFrame
df.head()


# ### Box Plot

# The purpose is to visualize the distribution of numerical columns in the DataFrame using boxplots. Boxplots provide insights into the central tendency, spread, and potential outliers of numerical data.

# In[4]:


df['age'] = df['age'].astype('int64')
col_list = list(df.columns)
for col in col_list:
    if ((df[col].dtype=='int64')or(df[col].dtype=='float64')):
        plt.figure(figsize=(5,3))
        df[col].to_frame().boxplot()
        plt.xlabel(col)
        plt.ylabel('count')
        plt.show()


# In[5]:


# for each column
for a in list(df.columns):

    # get a list of unique values
    n = df[a].unique()

    # if number of unique values is less than 30, print the values. Otherwise print the number of unique values
    if len(n)<30:
        print(a)
        print(n)
    else:
        print(a + ': ' +str(len(n)) + ' unique values')


# In[6]:


# List of numerical columns to be transformed
numerical_cols = ['duration', 'pdays','age',]
# Apply square root transformation to the selected numerical columns 
# to mitigate the impact of extreme values.
df[numerical_cols]=np.sqrt(df[numerical_cols])
# Apply natural logarithm transformation to the selected numerical columns 
# after Adding 1 before taking the logarithm is a common practice to handle 
# zero values and prevent undefined results.
df[numerical_cols]=np.log(df[numerical_cols] + 1)


# In[7]:


# replaces all occurrences of the string 'unknown' with NaN (Not a Number) in 
# the entire DataFrame.
df = df.replace('unknown', np.nan)
df = df.replace('nonexistent', np.nan)

# Print the count of missing values for each column
print("\nMissing values:")
print(df.isnull().sum())

# Check for duplicate rows
duplicate_rows = df[df.duplicated()]
print("\nNumber of duplicate rows:", len(duplicate_rows))


# In[8]:


#handling missing values by filling them with the mode (most frequent value) of each respective column. 
df['job'].fillna(df['job'].mode()[0], inplace=True)
df['education'].fillna(df['education'].mode()[0], inplace=True)
df['marital'].fillna(df['marital'].mode()[0], inplace=True)
df['default'].fillna(df['default'].mode()[0], inplace=True)
df['housing'].fillna(df['housing'].mode()[0], inplace=True)
df['loan'].fillna(df['loan'].mode()[0], inplace=True)
df['poutcome'].fillna(df['poutcome'].mode()[0], inplace=True)


# In[9]:


#control -> print count of missing values for each column
missing = df.isnull().sum()
print(missing)


# In[10]:


df.head()


# In[11]:


# Print the data types of each column, the number of non-null values, and 
# memory usage.
print("Dataset information:")
print(df.info())


# In[12]:


df = df.drop("day_of_week", axis='columns')
df = df.drop("month", axis='columns')
df = df.drop("default", axis='columns')
df = df.drop("euribor3m", axis='columns')


# In[13]:


# Define lists of categorical and numerical columns
categorical_cols = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'poutcome']
numerical = ['age','campaign', 'emp.var.rate', 'cons.price.idx','cons.conf.idx', 'nr.employed']
df[numerical].head()


# In[14]:


df[categorical_cols].head()


# #### One-Hot Encoding and Label Encoding

# These encoding methods are used to convert categorical variables into a format that can be fed into machine learning models. One-hot encoding creates binary columns for each category, while label encoding assigns a unique integer to each category. 

# In[15]:


# Iterate through each categorical column and print unique values
for column in categorical_cols:
    unique_values = df[column].unique()
    print(f"Unique values in {column}: {unique_values}")


# In[16]:


#Encode categorical variables (One-Hot Encoding)
data_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df.head()


# In[17]:


from sklearn.preprocessing import LabelEncoder
# Perform label encoding on categorical columns
label_encoder = LabelEncoder()
for column in categorical_cols:
    df[column] = label_encoder.fit_transform(df[column])
df.head()


# In[18]:


from sklearn.preprocessing import LabelEncoder
# Encode the 'y' column (target variable)
label_encoder = LabelEncoder()
df['y'] = label_encoder.fit_transform(df['y'])


# Standardization (or z-score normalization) transforms the numerical features to have a mean of 0 and a standard deviation of 1, making them more amenable to certain algorithms, especially those that rely on distance measures or gradients.

# In[19]:


from sklearn.preprocessing import StandardScaler
# Perform standardization on numerical columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# In[20]:


df.head()


# Each unique category in the original categorical columns has been transformed into a binary column, and the names of these new columns are printed.

# In[21]:


# Assuming 'data_encoded' is your DataFrame
print(data_encoded.columns)


# In[22]:


df.head()


# ### Heatmap

# The heatmap is a useful visualization to understand the relationships between numerical features. Correlation coefficients range from -1 to 1, where 1 indicates a perfect positive correlation, -1 indicates a perfect negative correlation, and 0 indicates no correlation. The color intensity in the heatmap represents the strength and direction of the correlation between pairs of numerical columns.

# In[23]:


# Creates a heatmap to visualize the correlation matrix of numerical columns in the DataFrame using Seaborn.
plt.figure(figsize=(21, 25), dpi=256)
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(df.corr(), annot=True)



# Visualizations helps understand the balance or imbalance of classes, which is essential information for classification tasks.

# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the class distribution for selected columns
selected_columns = ['job', 'marital', 'education', 'housing', 'loan', 'poutcome']

# Create subplots
fig, axes = plt.subplots(nrows=len(selected_columns), figsize=(20,30))

# Plot the class distribution for each selected column
for i, column in enumerate(selected_columns):
    sns.countplot(x=column, data=df, ax=axes[i])
    axes[i].set_title(f'Class Distribution for {column}')

# Adjust layout
plt.tight_layout()
plt.show()


# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


# In[26]:


# Separate features X and target variable y
X = df.drop('y', axis=1)
y = df['y']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# This split is a common practice in machine learning to evaluate the performance of a model on unseen data. The training set is used to train the model, and the testing set is used to evaluate its performance on data it has not seen during training.

# In[27]:


# Prints the count of each unique value in the target variable y
print(y.value_counts())


# ### Logistic Regression

# Logistic regression is used as the base model, and hyperparameter tuning is performed using cross-validated grid search (GridSearchCV). The hyperparameter grid includes choices for penalty, regularization strength (C), and solver.

# In[28]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Standardize the features to ensure that they have zero mean and unit variance.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with Cross-validation and Hyperparameter Tuning
logreg = LogisticRegression()

# Define hyperparameter grid for tuning
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear']
}

# Create StratifiedKFold for cross-validation to ensure that each fold has a balanced class distribution.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create GridSearchCV object perform a search over the specified hyperparameter grid. The scoring metric used is accuracy.
grid_search = GridSearchCV(logreg, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Precision: The number of true positive predictions divided by the total number of positive predictions. It focuses on the accuracy of the positive predictions.
# 
# Recall (Sensitivity): The number of true positive predictions divided by the total number of actual positives. It measures the model's ability to capture all the positives.
# 
# F1 Score: The harmonic mean of precision and recall. It provides a balance between precision and recall.
# 
# Area Under the ROC Curve (AUC-ROC): It measures the ability of the model to distinguish between the classes. A higher AUC-ROC indicates a better model.

# ### Log-Scale Plot

# Log-Scale plot helps visualize how the AUC performance changes with different regularization strengths for both L1 and L2 regularization. It assists in identifying the optimal regularization parameter for the logistic regression model.

# In[29]:


import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Create a range of regularization strengths for logistic regression
C_values = np.logspace(-4, 4, 20)

# Lists to store mean AUC values
mean_auc_l1 = []
mean_auc_l2 = []

# Loop through different values of C
# For each value of C, it uses cross-validated logistic regression models with L1 and L2 regularization
for C_value in C_values:

    # Logistic regression with L2 regularization
    model_l2 = LogisticRegression(penalty='l2', C=C_value, solver='liblinear', random_state=42)
    auc_scores_l2 = cross_val_score(model_l2, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    mean_auc_l2.append(np.mean(auc_scores_l2))

    # Logistic regression with L1 regularization
    model_l1 = LogisticRegression(penalty='l1', C=C_value, solver='liblinear', random_state=42)
    auc_scores_l1 = cross_val_score(model_l1, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    mean_auc_l1.append(np.mean(auc_scores_l1))
    # Computes the mean area under the ROC curve (AUC) for each regularization type.


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(C_values, mean_auc_l2, label='L2 Regularization')
plt.plot(C_values, mean_auc_l1, label='L1 Regularization')

plt.xscale('log')
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Mean AUC')
plt.title('Logistic Regression Mean AUC vs. Regularization Parameter')
plt.legend()
plt.show()


# In[30]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])

plt.figure(figsize=(8, 8))
plt.plot(recall, precision, color='darkorange', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


# ### Random Forest

# Random Forest is an ensemble learning algorithm that belongs to the family of decision tree-based methods. It is used for both classification and regression tasks. The key idea behind Random Forest is to build multiple decision trees during the training phase and combine their predictions to achieve a more accurate and robust model.

# In[31]:


from sklearn.ensemble import RandomForestClassifier

# Random Forest with Cross-validation and Hyperparameter Tuning
rf = RandomForestClassifier()

# Define hyperparameter grid for tuning
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

# Create GridSearchCV object for Random Forest
rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)

# Fit the model
rf_grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_rf_params = rf_grid_search.best_params_
print(f"Best Random Forest Hyperparameters: {best_rf_params}")

# Get the best model
best_rf_model = rf_grid_search.best_estimator_

# Make predictions on the test set
y_rf_pred = best_rf_model.predict(X_test_scaled)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_rf_pred)
print(f"Random Forest Accuracy: {accuracy_rf}")

# Display classification report
print("Random Forest Classification Report:")
print(classification_report(y_test, y_rf_pred))


# In[32]:


# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_rf_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

# ROC Curve for Random Forest
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, best_rf_model.predict_proba(X_test_scaled)[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 8))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'AUC = {roc_auc_rf:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve for Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, best_rf_model.predict_proba(X_test_scaled)[:, 1])

plt.figure(figsize=(8, 8))
plt.plot(recall_rf, precision_rf, color='darkorange', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Random Forest Precision-Recall Curve')
plt.show()


# The mean AUC (Area Under the Curve) is the average value of the AUC scores calculated across multiple folds in a cross-validation process. AUC is a commonly used metric for evaluating the performance of a binary classification model.

# In[33]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score


# Define a custom scorer for cross_val_score
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# Perform cross-validation and get AUC scores for each fold
auc_scores_rf = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=cv, scoring=auc_scorer)

# Calculate the mean AUC
mean_auc_rf = np.mean(auc_scores_rf)

print(f'Mean AUC for Random Forest: {mean_auc_rf:.2f}')


# ### Neural Network

# A neural network is a computational model inspired by the structure and functioning of the human brain.A neural network consists of interconnected nodes, called neurons, organized into layers.The three main types of layers are the input layer, hidden layers, and output layer.The neural network's architecture (hidden layer sizes), activation function, and regularization strength (alpha) are being tuned. The model is then evaluated on the test set, and a classification report is displayed. The classification report provides metrics such as precision, recall, and F1-score for each class in the classification problem.

# In[34]:


from sklearn.neural_network import MLPClassifier

# Neural Network with Cross-validation and Hyperparameter Tuning
nn = MLPClassifier()

# Define hyperparameter grid for tuning
nn_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['logistic', 'tanh', 'relu'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Create GridSearchCV object for Neural Network
nn_grid_search = GridSearchCV(nn, nn_param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)

# Fit the model
nn_grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_nn_params = nn_grid_search.best_params_
print(f"Best Neural Network Hyperparameters: {best_nn_params}")

# Get the best model
best_nn_model = nn_grid_search.best_estimator_

# Make predictions on the test set
y_nn_pred = best_nn_model.predict(X_test_scaled)

# Evaluate the Neural Network model
accuracy_nn = accuracy_score(y_test, y_nn_pred)
print(f"Neural Network Accuracy: {accuracy_nn}")

# Display classification report
print("Neural Network Classification Report:")
print(classification_report(y_test, y_nn_pred))


# In[35]:


from sklearn.neural_network import MLPClassifier

nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=42)

# Cross-validation for Neural Network
# Computes the mean AUC-ROC score across all folds.
auc_scores_nn = cross_val_score(nn_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
mean_auc_nn = np.mean(auc_scores_nn)


# This approach provides an estimate of the model's performance in terms of AUC-ROC across different subsets of the training data. The mean AUC-ROC is a summary statistic indicating the average discriminatory power of the model.

# ### Evaluating

# The mean AUC values for each model are used to generate the plot.

# In[36]:


model_names = ['Logistic Regression (L1)', 'Logistic Regression (L2)', 'Random Forest', 'Neural Network']
mean_auc_values = [mean_auc_l1[-1], mean_auc_l2[-1], mean_auc_rf, mean_auc_nn]

auc_df = pd.DataFrame({'Model': model_names, 'Mean AUC': mean_auc_values})

# Plot the results
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Mean AUC', data=auc_df, palette='viridis')
plt.ylim(0.8, 1.0)  
plt.title('Comparison of Mean AUC for Different Models')
plt.show()


# In[37]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')


# In[38]:


from sklearn.feature_selection import SelectKBest, f_classif
feature_selector = SelectKBest(score_func=f_classif, k=10)


# In[39]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical),
        ('cat', categorical_transformer, categorical_cols),
    ])


# In[40]:


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', feature_selector),
    ('model', LogisticRegression())  
])


# In[41]:


pipeline.fit(X_train, y_train)


# In[42]:


y_pred = pipeline.predict(X_test)


# In[43]:


print(classification_report(y_test, y_pred))


# In[44]:


import joblib
from sklearn.linear_model import LogisticRegression

def train_and_save_model(X_train_scaled, y_train, C_value):
    final_model = LogisticRegression(penalty='l2', C=C_value, solver='liblinear')  
    final_model.fit(X_train_scaled, y_train)
    
    filename = 'bank-additional.sav'
    
    with open(filename, 'wb') as file:
        joblib.dump(final_model, file)
    
    return filename

# Example usage:
# Assuming you have X_train_scaled, y_train, and C_value defined elsewhere
resulting_filename = train_and_save_model(X_train_scaled, y_train, C_value)
print("Model saved to:", resulting_filename)

