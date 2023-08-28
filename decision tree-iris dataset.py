import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("C:/Users/welcome/Desktop/subjects/deep learning/IRIS.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 8)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 5)
classifier.fit(X_train, y_train)
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(classifier, filled=True, rounded=True, feature_names=dataset.columns[:-1].tolist())
plt.show()
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix is:\n",cm)
ac=accuracy_score(y_test, y_pred)
print("accuracy is:",ac)
