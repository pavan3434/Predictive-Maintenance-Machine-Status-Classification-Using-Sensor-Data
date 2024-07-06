# Predictive-Maintenance-Machine-Status-Classification-Using-Sensor-Data
Description:
This project aims to predict the operational status of machinery using sensor data from a manufacturing process. The dataset, containing various sensor readings and machine status labels, undergoes thorough preprocessing, exploratory data analysis (EDA), feature engineering, and predictive modeling to build robust classification models.

1. Data Preprocessing:
The project starts with loading and examining the dataset to understand its structure and content. Initial steps include removing irrelevant columns (Unnamed: 0, sensor_00, sensor_15, sensor_50, sensor_51, and timestamp) to clean the data. Empty rows are also removed to ensure the dataset's integrity.

2. Exploratory Data Analysis (EDA):
EDA is performed to visualize relationships between sensor readings and machine status. Scatter plots are created for each feature against the target variable to identify any visible patterns. A correlation matrix of numeric features is generated to examine the interrelationships between different sensor readings.

3. Feature Engineering:
To improve the model's predictive power, features are standardized using StandardScaler. Additional features are engineered, such as rolling means, trends, and lagged values for specific sensors. Interaction features, such as the product of sensor readings, are also created to capture more complex relationships in the data.

4. Predictive Modeling:
The dataset is split into training and testing sets. Missing values are imputed using the mean strategy. Several classification models are trained, including Logistic Regression, Decision Tree Classifier, RandomForest Classifier, and HistGradientBoosting Classifier. Each model's performance is evaluated based on accuracy and detailed classification reports.

5. Model Evaluation and Selection:
Cross-validation is employed to assess the robustness of each model. Mean cross-validation scores are computed for each model to ensure consistent performance across different data splits. The best-performing model is selected based on the highest average cross-validation score.

The final chosen model demonstrates the potential for predictive maintenance by accurately classifying machine status based on sensor data. This project not only highlights the importance of thorough data preprocessing and feature engineering but also showcases the effectiveness of various machine learning models in predictive maintenance applications. The methodology and insights gained can be extended to similar predictive maintenance problems in other industrial domains.
