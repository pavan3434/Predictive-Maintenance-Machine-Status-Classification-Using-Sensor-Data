#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


## 1) Data Pre-Processing


# In[2]:


# loading data
data = pd.read_csv('sensor.csv')


# In[3]:


# take a quick look at dataframe
data.head()


# In[4]:


status_counts = data['machine_status'].value_counts()
print(status_counts)


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


## I really want to remove some rows which are empty so that i get proper output
data = data.drop('Unnamed: 0', axis =1)
data = data.drop('sensor_00', axis =1)
data = data.drop('sensor_51', axis =1)
data = data.drop('sensor_15', axis =1)
data = data.drop('sensor_50', axis =1)


# In[9]:


data.head()


# In[10]:


data = data.drop('timestamp', axis =1)


# In[11]:


data.head()


# In[ ]:


## 2) Exploratory Data Analysis (EDA)


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the relationships between features and the target variable
for feature in data.columns[:-1]:
    plt.figure()
    sns.scatterplot(x=feature, y='machine_status', data=data)
    plt.title(f'Relationship between {feature} and machine status')
    plt.show()

# Investigate correlations between numeric features
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = data[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[20]:


import pandas as pd
import seaborn as sns


# Investigate correlations between numeric features
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = data[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')  # Set annot to False to hide values
plt.title('Correlation Matrix')
plt.show()


# In[23]:


print(data.columns)


# In[ ]:


## 3) Feature Engineering


# In[26]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardize or normalize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop('machine_status', axis=1))
data_scaled = pd.DataFrame(data_scaled, columns=data.columns[:-1])
data_scaled['machine_status'] = data['machine_status']


# In[28]:


# Create interaction features (if applicable)
data_scaled['sensor1_x_sensor2'] = data_scaled['sensor_01'] * data_scaled['sensor_02']


# In[55]:


# Engineer additional features
data['sensor1_rolling_mean'] = data['sensor_01'].rolling(window=10).mean()
data['sensor2_trend'] = data['sensor_02'].diff()
data['sensor3_lag'] = data['sensor_03'].shift(1)

print(data.head(12))


# In[ ]:


## 4) Predictive Modeling


# In[30]:


from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split the data into training and testing sets
X = data_scaled.drop('machine_status', axis=1)
y = data_scaled['machine_status']

# Impute missing values in X_train and X_test
imputer = SimpleImputer(strategy='mean')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train different models
models = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    HistGradientBoostingClassifier()
]

for model in models:
    model.fit(X_train_imputed, y_train)
    y_pred = model.predict(X_test_imputed)
    print(f'Model: {type(model).__name__}')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('------------------------------')


# In[ ]:


## 5) Model Evaluation and Selection 


# In[34]:


from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import numpy as np

# Impute missing values in X and y
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
y_imputed = y

# Perform cross-validation
cross_val_scores = []
for model in models:
    scores = cross_val_score(model, X_imputed, y_imputed, cv=5)
    cross_val_scores.append(scores)
    print(f'Model: {type(model).__name__}')
    print('Cross-validation scores:', scores)
    print('Mean score:', scores.mean())
    print('------------------------------')

# Select the best model based on cross-validation scores
best_model_idx = np.argmax([scores.mean() for scores in cross_val_scores])
best_model = models[best_model_idx]
print('Best model:', type(best_model).__name__)

