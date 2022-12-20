import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier # Import Adaboost Classifier
from sklearn.linear_model import LinearRegression, LogisticRegression # Import Linear Regression Classifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import svm # Import SVM classifier
from sklearn.neighbors import KNeighborsClassifier # Import KNN classifier
from sklearn import datasets
from sklearn import metrics # for accuracy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing # for one-hot-encoder
from math import nan
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from tensorflow.keras import layers

# ask user for dataset filepath
training_filepath = str(input("Please input the training dataset filepath. \n"))
test_filepath = str(input("Please input the test dataset filepath. \n"))

# reading data and make them into dataframes
df_train = pd.read_csv(training_filepath) 
df_test = pd.read_csv(test_filepath)
# my training data filepath:/content/drive/MyDrive/Colab Notebooks/train_final.csv
# my test data filepath:/content/drive/MyDrive/Colab Notebooks/test_final.csv

# to show the dataframe info and first 5 entries
df_train.head()
df_train.info()
df_test.head()
df_test.info()

# check for null values
df_train.isnull().sum().sort_values(ascending = False)
df_test.isnull().sum().sort_values(ascending = False)
# this shows there is no null values because the missing values are marked with '?'. Hence, to identify them, replace '?' with NaN

# replace '?' to nan
df_train = df_train.replace('?', np.nan)
df_test = df_test.replace('?', np.nan)

# check again for null values
df_train.isnull().sum().sort_values(ascending = False)
df_test.isnull().sum().sort_values(ascending = False)
# Now it shows that there are missing values on occupation, workclass and native country columns for both datasets.

# handling missing values for training data
# replace the nan to the most common value of the workclass
mostcommon_w = df_train['workclass'].value_counts().idxmax()
df_train['workclass'].fillna(mostcommon_w, inplace = True)
# replace the nan to the most common value of the occupation
mostcommon_occ = df_train['occupation'].value_counts().idxmax()
df_train['occupation'].fillna(mostcommon_occ, inplace = True)
# replace the nan to the most common value of the country
mostcommon_c = df_train['native.country'].value_counts().idxmax()
df_train['native.country'].fillna(mostcommon_c, inplace = True)

# handling missing data for test data
# replace the nan to the most common value of the workclass
mostcommon_w = df_test['workclass'].value_counts().idxmax()
df_test['workclass'].fillna(mostcommon_w, inplace = True)
# replace the nan to the most common value of the occupation
mostcommon_occ = df_test['occupation'].value_counts().idxmax()
df_test['occupation'].fillna(mostcommon_occ, inplace = True)
# replace the nan to the most common value of the country
mostcommon_c = df_test['native.country'].value_counts().idxmax()
df_test['native.country'].fillna(mostcommon_c, inplace = True)

# confirm there is no more missing values
df_train.isnull().sum().sort_values(ascending = False)
df_test.isnull().sum().sort_values(ascending = False)

# drop unnecessary columns
df_test= df_test.drop(columns = 'ID')
df_train = df_train.drop(columns = 'race')
df_train = df_train.drop(columns = 'sex')
df_test = df_test.drop(columns = 'race')
df_test = df_test.drop(columns = 'sex')

# multiply the data for better training model
df_t = pd.concat([df_train,df_train]) # 50k rows

# dataset for training the model
X = df_train.loc[:, :"native.country"]
y = df_train["income>50K"] # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# encode the data for converting categorical values to numerical values
ohe = preprocessing.OneHotEncoder(handle_unknown = 'ignore')
ohe.fit(X_train)
transformed = ohe.transform(X_train)
transformed_test = ohe.transform(df_test)

# experiments with learning algorithms for finding the best training model

# Decision Tree
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf.fit(transformed,y_train)
# Predict the response for test dataset
# y_pred = clf.predict(transformed_test)
# x_pred = clf.predict(transformed)
test_pred = clf.predict(transformed_test) 
train_x_pred = clf.predict(transformed) 
print(test_pred)
print("Accuracy:",metrics.accuracy_score(y_train, train_x_pred))

# Result
# [0 0 0 ... 1 0 1]
# Accuracy: 1.0 # overfit

# Adaboost
# Create Adaboost classifer object
clf2 = AdaBoostClassifier()
# Train Adaboost Classifer
clf2.fit(transformed,y_train)
# Predict the response for test dataset
# y_pred = clf.predict(transformed_test)
# x_pred = clf.predict(transformed)
test_pred2 = clf2.predict(transformed_test) 
train_x_pred = clf2.predict(transformed) 
print(test_pred2)
print("Accuracy:",metrics.accuracy_score(y_train, train_x_pred))

# Result
# [0 0 0 ... 1 0 1]
# Accuracy: 0.8554285714285714

# Linear Regression
# Create Linear Regression classifer object
clf3 = LinearRegression()
# Train Linear Regression Classifer
clf3.fit(transformed,y_train)
# Predict the response for test dataset
# y_pred = clf.predict(transformed_test)
# x_pred = clf.predict(transformed)
test_pred3 = clf3.predict(transformed_test) 
train_x_pred = clf3.predict(transformed) 
print(test_pred3)
print("Accuracy:",metrics.accuracy_score(y_train,train_x_pred))

# Result
# [ 0.0086515  -0.3193968   0.05251087 ...  1.32329494  0.19297811
#   0.23181217] # error in measuring accuracy using the prediction labels from the training data

# SVM
# SVM with linear kernel
# Create SVM classifer object
clf4 = svm.SVC(kernel='linear')
# Train SVM Classifer
clf4.fit(transformed,y_train)
# Predict the response for test dataset
# y_pred = clf.predict(transformed_test)
# x_pred = clf.predict(transformed)
test_pred4 = clf4.predict(transformed_test) 
train_x_pred = clf4.predict(transformed) 
print(test_pred4)
print("Accuracy:",metrics.accuracy_score(y_train, train_x_pred))

# Result
# [0 0 0 ... 1 0 0]
# Accuracy: 0.9653142857142857

# SVM with sigmoid kernel
# Create SVM classifer object
clf5 = svm.SVC(kernel='sigmoid')
# Train SVM Classifer
clf5.fit(transformed,y_train)
# Predict the response for test dataset
# y_pred = clf.predict(transformed_test)
# x_pred = clf.predict(transformed)
test_pred5 = clf5.predict(transformed_test) 
train_x_pred = clf5.predict(transformed) 
print(test_pred5)
print("Accuracy:",metrics.accuracy_score(y_train, train_x_pred))

# Result
# [0 0 0 ... 1 0 1]
# Accuracy: 0.7930285714285714

# SVM with rbf kernel
# Create SVM classifer object
clf6 = svm.SVC(kernel='rbf')
# Train SVM Classifer
clf6.fit(transformed,y_train)
# Predict the response for test dataset
# y_pred = clf.predict(transformed_test)
# x_pred = clf.predict(transformed)
test_pred6 = clf6.predict(transformed_test) 
train_x_pred = clf6.predict(transformed) 
print(test_pred6)
print("Accuracy:",metrics.accuracy_score(y_train, train_x_pred))

# Result
# [0 0 0 ... 1 0 0]
# Accuracy: 0.8873142857142857

# SVM with polynomial kernel
# Create SVM classifer object
clf7 = svm.SVC(kernel='poly')
# Train SVM Classifer
clf7.fit(transformed,y_train)
# Predict the response for test dataset
# y_pred = clf.predict(transformed_test)
# x_pred = clf.predict(transformed)
test_pred7 = clf7.predict(transformed_test) 
train_x_pred = clf7.predict(transformed) 
print(test_pred7)
print("Accuracy:",metrics.accuracy_score(y_train, train_x_pred))


# Result
# [0 0 0 ... 1 0 0]
# Accuracy: 0.9050857142857143

# KNN 
# Create KNN classifer object
clf8 = KNeighborsClassifier(n_neighbors=3)
# Train SVM Classifer
clf8.fit(transformed,y_train)
#Predict the response for test dataset
# y_pred = clf.predict(transformed_test)
# x_pred = clf.predict(transformed)
test_pred8 = clf8.predict(transformed_test) 
train_x_pred = clf8.predict(transformed) 
print(test_pred8)
print("Accuracy:",metrics.accuracy_score(y_train, train_x_pred))

# Result
# [0 0 0 ... 1 0 0]
# Accuracy: 0.8924571428571428

# Multi Layer Perceptron
# Create MLP classifer object
clf9 = MLPClassifier(activation = 'relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# Train MLP Classifer
clf9.fit(transformed,y_train) 
# Predict the response for test dataset
# y_pred = clf.predict(transformed_test)
# x_pred = clf.predict(transformed)
test_pred9 = clf9.predict(transformed_test) 
train_x_pred = clf9.predict(transformed) 
print(test_pred9)
print("Accuracy:",metrics.accuracy_score(y_train, train_x_pred))

# Result
# [0 0 0 ... 1 0 0]
# Accuracy: 0.8914285714285715

# Perceptron
# Create perceptron classifer object
clf10 = Perceptron(random_state=42)
# Train perceptron Classifer
clf10.fit(transformed,y_train)
# Predict the response for test dataset
# y_pred = clf.predict(transformed_test)
# x_pred = clf.predict(transformed)
test_pred10 = clf10.predict(transformed_test) 
train_x_pred = clf10.predict(transformed) 
print(test_pred10)
print("Accuracy:",metrics.accuracy_score(y_train, train_x_pred))

# Result
# [0 0 0 ... 1 0 0]
# Accuracy: 0.9702857142857143

# Logistic Regression
# Create logistic regression classifer object
clf11 = LogisticRegression(max_iter = 10000)
# Train logistic regression Classifer
clf11.fit(transformed,y_train)
# Predict the response for test dataset
# y_pred = clf.predict(transformed_test)
# x_pred = clf.predict(transformed)
test_pred11 = clf11.predict_proba(transformed_test) 
test_pred11 = test_pred11[:,1]
train_x_pred = clf11.predict(transformed) 
print(test_pred11)
print("Accuracy:",metrics.accuracy_score(y_train, train_x_pred))

# Result
# [0.11970233 0.04908168 0.10964066 ... 0.80532659 0.1191211  0.45363492]
# Accuracy: 0.9171428571428571

# Create random forrest classifer object
clf12 = RandomForestClassifier(n_estimators=100)
# Train random forrest Classifer
clf12.fit(transformed,y_train)
#Predict the response for test dataset
# y_pred = clf.predict(transformed_test)
# x_pred = clf.predict(transformed)
test_pred12 = clf12.predict_proba(transformed_test) 
test_pred12 = test_pred12[:,1]
train_x_pred = clf12.predict(transformed) 
print(test_pred12)
print("Accuracy:",metrics.accuracy_score(y_train, train_x_pred))

# export the results in the form of a table in a CSV file

# test_pred.savetxt('decision_tree_pred.csv',test_pred, fmt = '%d', delimiter=",")    
# test_pred.tofile('/content/drive/MyDrive/Colab Notebooks/dec_tree_pred.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/decision_tree_pred.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred,1))

# test_pred2.tofile('adaboost_pred.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/adaboost_pred.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred2,1))

# test_pred3.tofile('lin_reg_pred.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/lin_reg_pred.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred3,1))

# test_pred4.tofile('SVM_lin.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/SVM_lin.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred4,1))

# test_pred5.tofile('SVM_sigmoid.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/SVM_sigmoid.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred5,1))

# test_pred6.tofile('SVM_rbf.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/SVM_rbf.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred6,1))

# test_pred7.tofile('SVM_rbf.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/SVM_poly.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred7,1))

# test_pred8.tofile('SVM_rbf.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/KNN.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred7,1))

# test_pred9.tofile('SVM_rbf.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/MLP.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred9,1))

# test_pred10.tofile('Perceptron.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/Perceptron.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred10,1))

# test_pred11.tofile('Logistic_Regression.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/Logistic_Regression.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred11,1))
# obtained the best accuracy result, BEST TRAINING MODEL


# test_pred12.tofile('random_forrest.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/random_forrest.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred12,1))

# experimental evaluation
# test_pred13.tofile('Logistic_Regression_onlyracedropped.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/Logistic_Regression_onlyracedropped.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred11,1))

# test_pred14.tofile('Logistic_Regression_racesexhpwmaritalstatus_dropped.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/Logistic_Regression_racesexhpwmaritalstatus_dropped.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred11,1))

# test_pred15.tofile('Logistic_Regression_onlyracesex_dropped.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/Logistic_Regression_onlyracesex_dropped.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred11,1))

# test_pred16.tofile('Logistic_Regression_racesexcapitalgainlossmaritalstatus_dropped.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/Logistic_Regression_racesexcapitalgainlossmaritalstatus_dropped.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred11,1))

# test_pred17.tofile('Logistic_Regression_race_dropped_splitdata.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/Logistic_Regression_race_dropped_splitdata.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred11,1))
# produces the best combination for the best accuracy for logistic regression training model

# test_pred18.tofile('Logistic_Regression_racesexfnlwgt_dropped_splitdata_maxiter_10000.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/Logistic_Regression_racesexfnlwgt_dropped_splitdata_maxiter_10000.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred11,1))

# test_pred19.tofile('Logistic_Regression_racesexfnlwgt_dropped_splitdata_concat.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/Logistic_Regression_racesexfnlwgt_dropped_splitdata_concat.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred11,1))

# test_pred20.tofile('Logistic_Regression_racesex_dropped_splitdata_concat.csv', sep = ',')
with open('/content/drive/MyDrive/Colab Notebooks/Logistic_Regression_racesex_dropped_splitdata_concat.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerow(['ID','Prediction']) 
    mywriter.writerows(enumerate(test_pred11,1))
