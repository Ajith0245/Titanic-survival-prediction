import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('train.csv')
dataset = dataset[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
label = LabelEncoder()
dataset['Sex'] = label.fit_transform(dataset['Sex'])
dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
dataset.dropna(inplace=True)
x = dataset[['Pclass', 'Sex', 'Age', 'Fare']]
y = dataset['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
# Example: Sex=1 for male, 0 for female (check label.classes_)
new_feature = [[3, 1, 22, 35]]
result = model.predict(new_feature)
print("The result for input feature is", result[0])
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy of the model:", accuracy)