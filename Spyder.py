# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:06:20 2019

@author: Zaoudre
"""

import pandas as pd
from sklearn import metrics

dataset = pd.read_csv('Tweets.csv')
X = dataset["text"]
y = dataset["airline_sentiment"]

##CLEANING TEXTS

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 6918):
    review = re.sub('[^a-zA-Z]', ' ', X[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#CREATING BAG OF WORDS MODEL

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df = .85, max_features = 1500)
X = cv.fit_transform(X).toarray()

#SPLITTING THE DATASET

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
#model_selection are used inplace of cross_validation nowadays xd

#FITTING MULTINOMIAL NAIVEBAIYES TO TRAINING SET

from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X_train, y_train)
y_pred = NB.predict(X_test)
print('\nNaive Bayes')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

# KFOLD CROSS VALIDATION FOR NAIVE BAYES

from sklearn.model_selection import cross_val_score
accuracies_NB = cross_val_score(estimator = NB, X = X_train, y = y_train, cv = 10)
mean_NB = accuracies_NB.mean()
std_NB = accuracies_NB.std()

#FITTING SVM CLASSIFIER TO THE TRAINING SET

from sklearn.svm import LinearSVC
SVM = LinearSVC()
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)
print('\nSupport Vector Machine')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

#KFOLD CROSS VALIDATION FOR SVM

from sklearn.model_selection import cross_val_score
accuracies_SVM = cross_val_score(estimator = SVM, X = X_train, y = y_train, cv = 10)
mean_SVM = accuracies_SVM.mean()
std_SVM = accuracies_SVM.std()

#FITTING LINEAR REGRESSION MODEL TO THE TRAINING SET

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
print('\nLogistic Regression')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

#KFOLD CROSS VALIDATION FOR LOGISTIC REGRESSION 

from sklearn.model_selection import cross_val_score
accuracies_LR = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = 10)
mean_LR = accuracies_LR.mean()
std_LR = accuracies_LR.std()

#FITTING K NEAREST NEIGHBOUR TO THE TRAINING SET

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 3)
KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_test)
print('\nK Nearest Neighbors')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

#KFOLD CROSS VALIDATION FOR KNN

from sklearn.model_selection import cross_val_score
accuracies_KNN = cross_val_score(estimator = KNN, X = X_train, y = y_train, cv = 10)
mean_KNN = accuracies_KNN.mean()
std_KNN = accuracies_KNN.std()

#FITTING DECISION TREE CLASSIFIER TO THE TRAINING SET

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)
print('\nDecision Tree')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

#KFOLD CROSS VALIDATION FOR DECISION TREE

from sklearn.model_selection import cross_val_score
accuracies_DT = cross_val_score(estimator = DT, X = X_train, y = y_train, cv = 10)
mean_DT = accuracies_DT.mean()
std_DT = accuracies_DT.std()

#FITTING RANDOM FOREST CLASSIFIER TO THE TRAINING SET

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 0)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
print('\nRandom Forest')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

#KFOLD CROSS VALIDATION FOR THE RANDOM FOREST

from sklearn.model_selection import cross_val_score
accuracies_RF = cross_val_score(estimator = RF, X = X_train, y = y_train, cv = 10)
mean_RF = accuracies_RF.mean()
std_RF = accuracies_RF.std()

#ANALYSING MODELS

token_words = cv.get_feature_names()
print('\n Analysis')
print('Number of tokens: ',len(token_words))
counts = NB.feature_count_
df_table = {'Token':token_words,'Negative': counts[0,:],'Positive': counts[1,:]}
tokens = pd.DataFrame(df_table, columns= ['Token','Positive','Negative'])
positives = len(tokens[tokens['Positive']>tokens['Negative']])
print('No. of positive tokens: ',positives)
print('No. of negative tokens: ',len(token_words)-positives)

#CHECK POSITIVITY NEGATIVITY OF SPECIFIC TOKENS

token_search = ['awesome']
print('\nSearch Results for token/s:',token_search)
print(tokens.loc[tokens['Token'].isin(token_search)])

#ANALYSE FALSE NEGATIVES

print(X_test[ y_pred < y_test ])

#ANALYSE FALSE POSITIVES

print(X_test[ y_pred > y_test ])





