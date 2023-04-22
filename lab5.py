import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing my csv files
df_white = pd.read_csv ('winequality-white.csv', sep=';')
df_red = pd.read_csv ('winequality-red.csv', sep=';')


#adding white and red dummy values
df_white['white'] = 1
df_white['red'] = 0
df_red['white'] = 0
df_red['red'] = 1

#concatinating my red and white dataframes
frames = [df_red, df_white]
df = pd.concat(frames)

#removing all data outside 3 standard deviations from mean
from scipy import stats
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

#converting to zscale
cols = list(df.columns)
for col in cols:
    col_zscore = col + '_zscore'
    df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)


#i tried dropping each column one at a time to find the combination with the highest accuracy. This seems to be it
#df.drop('pH', axis=1, inplace=True)
#df.drop('density', axis=1, inplace=True)
#df.drop('chlorides', axis=1, inplace=True)
df.drop('sulphates', axis=1, inplace=True)
#df.drop('alcohol', axis=1, inplace=True)
df.drop('total sulfur dioxide', axis=1, inplace=True)
#df.drop('free sulfur dioxide', axis=1, inplace=True)
df.drop('residual sugar', axis=1, inplace=True)
df.drop('citric acid', axis=1, inplace=True)
#df.drop('volatile acidity', axis=1, inplace=True)
#df.drop('fixed acidity', axis=1, inplace=True)
#df.drop('red', axis=1, inplace=True)
#df.drop('white', axis=1, inplace=True)

#seperating quality from the other columns
y = df['quality'].values
X = df.loc[:, df.columns != 'quality'].values

#splitting my training set from my test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#I tried {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’). rbf was the most accurate
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))


#I tried some other models, but they werent as accurate
#from sklearn.naive_bayes import GaussianNB
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#gnb = GaussianNB()
#y_pred = gnb.fit(X_train, y_train).predict(X_test)
#print("Number of mislabeled points out of a total %d points : %d"
#      % (X_test.shape[0], (y_test != y_pred).sum()))


#from sklearn import linear_model
#reg = linear_model.LinearRegression()
#reg.fit(X_train, y_train)
#y_pred = classifier.predict(X_test)

#from sklearn.metrics import confusion_matrix, accuracy_score
#cm = confusion_matrix(y_test, y_pred)
#print(cm)
#print(accuracy_score(y_test,y_pred))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors=5)
#classifier.fit(X_train, y_train)

#y_pred = classifier.predict(X_test)

#from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
