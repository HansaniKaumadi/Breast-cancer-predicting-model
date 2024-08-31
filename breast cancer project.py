#Date: 31/08/2024
#Coded by: Hansani Kaumadi
#A basic ML project to predict a breast cancer is malignant or not using logistic regression 
#Used Libraries: pandas, seaborn, skikit learn, matplotlib
#Dataset taken from kaggle URL: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data

import pandas as pd;
import  seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
data=pd.read_csv(r'D:\ML projects\breast cancer\dataset.csv')

#understanding the data 
print(data.head())
print(data.info())
print(data.describe())

#Cleaning the data
sns.heatmap(data.isnull())
#plt.show()
data.drop(['Unnamed: 32', 'id'],axis=1,inplace=True)
print(data.head())

#Converting diagnosed values to 0 and 1 
data.diagnosis=[1 if value=='M' else 0 for value in data.diagnosis]
print(data.head())

data.diagnosis=data['diagnosis'].astype('category',copy=False)
data.diagnosis.value_counts().plot(kind='bar')
#plt.show()

#Seperating predictors and targets
y=data.diagnosis #target variable
x=data.drop(['diagnosis'],axis=1)

##Normalizing data 

#Creating a scaler object
scaler=StandardScaler()

#Fitting the scaler to the data and then transform
x_scaled=scaler.fit_transform(x)
#print(x_scaled)

#Splitting the data
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.30,random_state=42)

##Training the model

#Creating the lienar regression model
lr=LogisticRegression()

#Training the model on the training data
lr.fit(x_train,y_train)

#Predicting the target variable on test data
y_pred=lr.predict(x_test)

#print(y_pred)
#print(y_test)

##Evaluation of the model
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy: {accuracy: .2f}')
print(classification_report(y_test,y_pred))
