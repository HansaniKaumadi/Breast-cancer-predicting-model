# Breast Cancer Prediction Using Logistic Regression

## Overview
This project is a machine learning model built to predict whether breast cancer is malignant or benign using logistic regression. The model uses a dataset from the [Kaggle Breast Cancer Wisconsin (Diagnostic) Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data). Early detection of breast cancer can significantly improve the prognosis and survival rates. This project aims to develop a reliable model that helps tp  predicting the malignancy of breast cancer using basic machine learning techniques.

## Libraries Used
The following Python libraries were utilized to implement the model:

- **Pandas**: For data manipulation and analysis.
- **Seaborn**: For data visualization and visual exploration of missing values.
- **Matplotlib**: For plotting graphs and visualizing data distributions.
- **Scikit-Learn**: For machine learning algorithms (Logistic Regression) and evaluation metrics.

## Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset from Kaggle. It consists of several features describing the characteristics of the cell nuclei present in digitized images of breast masses. The target variable, `diagnosis`, indicates whether the cancer is malignant ('M') or benign ('B').

**Dataset URL**: [Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)

### Key Features:
- **Mean Radius, Mean Texture, Mean Perimeter**, etc.: Various characteristics of the cell nuclei.
- **Diagnosis**: Target variable indicating 'M' (Malignant) or 'B' (Benign).

## Project Workflow

### Data Loading and Exploration:
- Load the dataset using `pandas`.
- Explore the data using `.head()`, `.info()`, and `.describe()` to understand the structure and summary statistics.

### Data Cleaning:
- Visualize missing values using a heatmap with `seaborn`.
- Drop unnecessary columns (`Unnamed: 32` and `id`) to simplify the dataset.

### Data Preprocessing:
- Convert the `diagnosis` column from categorical ('M' or 'B') to numerical (1 for malignant, 0 for benign).
- Visualize the distribution of the target variable using a bar plot.

### Feature Scaling:
- Normalize the features using `StandardScaler` to bring them to a similar scale, which is essential for logistic regression.

### Model Training:
- Split the dataset into training and testing sets using `train_test_split`.
- Train a logistic regression model using `LogisticRegression` on the training data.

### Model Evaluation:
- Predict the target values on the test set using the trained model.
- Evaluate the model performance using `accuracy_score` and `classification_report` to understand precision, recall, and F1-score.

## Results
- The logistic regression model achieved an accuracy of approximately **0.98** on the test set.
- The `classification_report` provides detailed metrics, including precision, recall, and F1-score, for both malignant and benign classifications, indicating the model's ability to recognize between the two classes.

## Conclusion
This project shows how logistic regression can be applied to medical datasets for binary classification tasks. The steps include data cleaning, preprocessing, model training, and evaluation, which are fundamental to building any machine-learning model. This project can be further extended by trying more advanced machine-learning algorithms or by performing additional feature engineering.
