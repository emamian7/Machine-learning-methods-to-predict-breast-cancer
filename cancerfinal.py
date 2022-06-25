#Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#SGD Stochastic Gradient Descent library
from sklearn.linear_model import SGDClassifier
#SVM Library
from sklearn.svm import SVC, NuSVC, LinearSVC
#KNN
from sklearn.neighbors import KNeighborsClassifier
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Decision Trees
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
#GridSearch
from sklearn.model_selection import GridSearchCV

#Read data
data = pd.read_csv('data.csv');
#Drop NULL column
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
#Summary data for  cancer
diagnosis_all = list(data.shape)[0]
#Extract only mean feature
features_mean= list(data.columns[1:])
#Transform M,B to 1,0 respectively
mapping = {'M':1, 'B':0}
data['diagnosis'] = data['diagnosis'].map(mapping)
x = data.loc[:]
y = data.loc[:, 'diagnosis']
#Split data to training and test with prob 0.2
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state = 42)
#Stochastic Gradient Descent (SGD)
#Build model
clf = SGDClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("SGD Classifier Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for SGD')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Support Vector Machine 
#Build Model
clf = SVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("SVC Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for SVC')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#NuSVC
#Build Model
clf = NuSVC()
clf.fit(X_train, y_train)
prediciton = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("NuSVC Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for NuSVC')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#LinearSVC
#Build Model
clf = LinearSVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("LinearSVC Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for LinearSVC')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#KNN
#Build Model
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)
#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 
print("KNN Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for KNN')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Naive Bayes Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Naive Bayes')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Random Forest
#Build Model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Random Forest Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Random Forest')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Extra Trees
clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Extra Trees Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Extra Trees')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()
#Decision Trees
#Build Model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)
#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 
print("Decision Trees Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Decision Trees')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()
y = data.loc[:, 'diagnosis']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Stochastic Gradient Descent (SGD)

#Build model
clf = SGDClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("SGD Classifier Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for SGD')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Support Vector Machine 


#Build Model
clf = SVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("SVC Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for SVC')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#NuSVC
#Build Model
clf = NuSVC()
clf.fit(X_train, y_train)
prediciton = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("NuSVC Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for NuSVC')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#LinearSVC
#Build Model
clf = LinearSVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("LinearSVC Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for LinearSVC')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#KNN
#Build Model
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("KNN Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for KNN')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Naive Bayes Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Naive Bayes')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Random Forest
#Build Model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Random Forest Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Random Forest')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Extra Trees

clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Extra Trees Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Extra Trees')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Decision Trees
#Build Model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Decision Trees Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")

print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Decision Trees')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#GridSearchCV Improve Model
x = data.loc[:]
y = data.loc[:, 'diagnosis']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Naive Bayes

#Set Parameters
parameters = {'priors':[[0.01, 0.99],[0.1, 0.9], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7],[0.35, 0.65], [0.4, 0.6]]}

#Build Model
clf = GridSearchCV(GaussianNB(), parameters, scoring = 'average_precision', n_jobs=-1)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)
#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Naive Bayes Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")

print("Best parameters:",clf.best_params_)
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Naive Bayes')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Random Forest

#Set Parameters
parameters = {'n_estimators':list(range(1,101)), 'criterion':['gini', 'entropy']}

#Build Model
clf = GridSearchCV(RandomForestClassifier(), parameters, scoring = 'average_precision', n_jobs=-1)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Random Forests Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Best parameters:",clf.best_params_)
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Random Forests')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

#Extra Trees

#Build Model
clf = GridSearchCV(ExtraTreesClassifier(), parameters, scoring = 'average_precision', n_jobs=-1)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Extra Trees Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Best parameters:",clf.best_params_)
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Extra Trees')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()
#Decision Trees
#Set Parameter
parameters = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random']}

#Build Model
clf = GridSearchCV(DecisionTreeClassifier(), parameters, scoring = 'average_precision', n_jobs=-1)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, x, y, cv=5)

#Compute Accuracy&CVS score
accuracy = accuracy_score(prediction, y_test)
cross_valid = np.mean(scores)

#Confusion Matrix
confusion_mat = confusion_matrix(prediction,y_test) 

print("Decision Trees Accuracy:",accuracy*100,"%")
print("Cross validation score:",cross_valid*100,"%")
print("Best parameters:",clf.best_params_)
print(confusion_mat)
# Plot Confusion Matrix for Test Data
plt.matshow(confusion_mat)
plt.title('Confusion Matrix for Decision Trees')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()


