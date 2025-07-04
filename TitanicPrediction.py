import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset=pd.read_csv('train.csv')
dataset= dataset[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
dataset.dropna(inplace=True)
dataset['Age'].fillna(dataset['Age'].mean(),inplace=True)
label=LabelEncoder()
dataset['Sex']=label.fit_transform(dataset['Sex'])

model=DecisionTreeClassifier()
x=dataset[['Pclass','Sex','Age','Fare']]
y=dataset['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
accuracy=accuracy_score(y_predict,y_test)
print("The acccuracy score is",accuracy)
