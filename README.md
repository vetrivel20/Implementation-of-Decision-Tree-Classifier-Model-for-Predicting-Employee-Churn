# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 
RegisterNumber:  
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
data=pd.read_csv("Employee.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print(data.head())
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
y=data["left"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
dt=DecisionTreeClassifier(criterion="entropy",random_state=100)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
sample=[[0.5,0.8,9,260,6,0,1,2]]
print("Prediction for sample:",dt.predict(sample))
plt.figure(figsize=(12,8))
plot_tree(dt,feature_names=x.columns,class_names=["stayed","left"],filled=True,rounded=True,fontsize=10)
plt.show()
```
## Output:
![alt text](<Screenshot 2026-02-26 104108.png>)
![alt text](<Screenshot 2026-02-26 105257.png>)
![alt text](<Screenshot 2026-02-26 105317.png>)
![alt text](<Screenshot 2026-02-26 105334.png>)
![alt text](<Screenshot 2026-02-26 105356.png>)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
