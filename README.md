# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the required packages.

2.Read the data set.

3.Apply label encoder to the non-numerical column inoreder to convert into numerical values.

4.Determine training and test data set.

5.Apply decision tree Classifier and get the values of accuracy and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NAVEENKUMAR V
RegisterNumber: 212221230068
*/

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![decision tree classifier model](sam.png)
![image](https://user-images.githubusercontent.com/94165322/202893038-a2b24373-9269-46f6-ac9a-5109a7ae9c4c.png)
![image](https://user-images.githubusercontent.com/94165322/202893044-47ff827c-974e-40fd-ab8c-e6d5fb279adf.png)
![image](https://user-images.githubusercontent.com/94165322/202893049-0b3c592e-c500-43a3-8c3c-db23e4102b79.png)
![image](https://user-images.githubusercontent.com/94165322/202893059-c0a083e8-60db-49fa-83fd-c4bafcaba54c.png)
![image](https://user-images.githubusercontent.com/94165322/202893063-a02d96c7-1359-4e55-958c-2bed530190ad.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
