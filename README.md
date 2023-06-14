# Titanic-Classification

This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. The dataset used for this project is the Titanic dataset, which contains information about the passengers such as their age, gender, socio-economic status, and more.

## Dataset

The dataset used for this project can be obtained from Kaggle: https://www.kaggle.com/c/titanic/data. The dataset consists of two files: train.csv and test.csv. The train.csv file is used for training the model, while the test.csv file is used for evaluating the model's performance.

## Project Structure

Library Preparation: Importing the necessary libraries and loading the dataset.
Data Preprocessing: Handling missing data, performing data analysis, and data visualization.
Feature Encoding: Encoding categorical columns such as 'Sex' and 'Embarked'.
Train-Test Split: Splitting the data into training and testing sets.
Feature Scaling: Standardizing the features using StandardScaler.
Model Training: Training various machine learning models such as Logistic Regression, K Nearest Neighbors, Support Vector Machines, Naïve Bayes, Decision Tree, and Random Forest.
Feature Importance: Determining the importance of features using a Random Forest Classifier.
Model Evaluation: Evaluating the performance of each trained model using the test data.

## Results

The accuracy scores obtained for each model on the training set are as follows:
Logistic Regression: 0.80
K Nearest Neighbor: 0.85
Support Vector Machine (Linear Classifier): 0.79
Support Vector Machine (RBF Classifier): 0.84
Gaussian Naive Bayes: 0.76
Decision Tree Classifier: 0.98
Random Forest Classifier: 0.97
Conclusion
Based on the evaluation results, the Decision Tree Classifier and Random Forest Classifier achieved the highest training accuracy scores. These models can be further evaluated and fine-tuned to improve their performance.

For more details, please refer to the Jupyter notebook containing the code implementation.
