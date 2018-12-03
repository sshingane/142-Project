#from __future__ import print_function 
#import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords
import collections
import csv
import math
import sklearn.metrics
from sklearn.metrics import confusion_matrix 
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

# Opens file, places contents of file to sentence list
def open_file(sentence):
    size = 0
    with open('train.csv') as my_file:    
        sentence = my_file.read().splitlines()

    return sentence

# Opens file, extracts the instances and labels 
def open_file_csv():
    instances = []
    labels = []
    data = csv.reader(open('train.csv'))
    next(data)  # Skip header row
    for line in data:
        instances.append(line[0:3])
        labels.append(int(line[3]))
    return (instances, labels)

# Finds the tfidf vectorization of the data
def vectorize(sentences, stop):
    vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = stop, max_features = None) 
    vectorizer.fit(sentences)
    vectorizer.transform(sentences).toarray()
    vector = vectorizer.transform(sentences)
    print (len(vectorizer.vocabulary_))
    #print (vectorizer.vocabulary_, file=open("output_14.txt", "a"))   

    return vector

# Calculates the Linear Regression of the data
def regression(instance, labels):
    rounded = []
    reg = LinearRegression(n_jobs = 200, fit_intercept=True, normalize=True).fit(instance, labels)
    prediction = reg.predict(instance)  
    for i in prediction:
        rounded.append(math.trunc(i))
    return rounded  

# Finds the Linear SVM of the data, and returns the prediction
def svm(instance, labels):
    rounded = []
    svm = LinearSVC(random_state=0, tol=1e-5, C = 5)
    svm_fit = svm.fit(instance, labels)
    prediction = svm_fit.predict(instance)
   # print (prediction, file=open("output_15.txt", "a"))
       
    return prediction

# Implements the naive bayes model on the features
def naive_bayes(feature_vector_matrix, actual_labels):
    nb = MultinomialNB()
    nb.fit(feature_vector_matrix, actual_labels)
    predicted_labels = nb.predict(feature_vector_matrix)
    return predicted_labels

def gradient_boost(feature_vector_matrix, actual_labels):
    lf = GradientBoostingClassifier(n_estimators=600, learning_rate=1.0,max_depth = 1, random_state = 0).fit(feature_vector_matrix, actual_labels)
    return lf.predict(feature_vector_matrix)


def ada_boost(feature_vector_matrix, actual_labels):
    labels = []

    #boost real values
    bdt_real = AdaBoostRegressor(
        LinearSVC(random_state=0, tol=1e-5, C = 1),
        n_estimators=15104,
        learning_rate=1)

    #boost discrete values
    bdt_discrete = AdaBoostRegressor(
        LinearSVC(random_state=0, tol=1e-5, C = 1),
        n_estimators=15104,
        learning_rate=1.5)

    for i in actual_labels:
        labels.append(int(i))

    bdt_real.fit(feature_vector_matrix, labels)
   # bdt_discrete.fit(feature_vector_matrix, actual_labels)

    real_test_errors = []
    discrete_test_errors = []

    # for real_test_predict, discrete_train_predict in zip(
    #         bdt_real.staged_predict(feature_vector_matrix), bdt_discrete.staged_predict(feature_vector_matrix)):
    #     real_test_errors.append(
    #         1. - accuracy_score(real_test_predict, actual_labels))
    #     discrete_test_errors.append(
    #         1. - accuracy_score(discrete_train_predict, actual_labels))

    # kfold = model_selection.KFold(n_splits=10)
    #
    # results = model_selection.cross_val_score(bdt_real, feature_vector_matrix, actual_labels, cv=kfold)
    #scores = StratifiedKFold(bdt_real, feature_vector_matrix, actual_labels, n_splits=5)

    return bdt_real.predict(feature_vector_matrix)

# Calculates the metrics of the learning model, which are the accuracy, precision, F-measure, and recall. 
def printPerformance(model_name, actual_labels, predicted_labels):
    labels = [0, 1, 2, 3, 4]

    conf_matrix = sklearn.metrics.confusion_matrix(actual_labels, predicted_labels, labels = labels)
    accuracy = sklearn.metrics.accuracy_score(actual_labels, predicted_labels)
    precision = sklearn.metrics.precision_score(actual_labels, predicted_labels, labels=labels, average=None)
    f_measure = sklearn.metrics.f1_score(actual_labels, predicted_labels, labels=labels, average=None)
    recall = sklearn.metrics.recall_score(actual_labels, predicted_labels, labels=labels, average=None)

    print('='*60)
    print(model_name)
    print('='*60)
    print('CONFUSION MATRIX:')
    print(conf_matrix)
    print('ACCURACY: ' + str(accuracy))
    print('PRECISION: ' + str(precision))
    print('F-MEASURE: ', str(f_measure))
    print ('RECALL: ', str(recall))
        

# Implements 5-fold cross validation on the data, returns the test and train scores
def cross_validation(instance, labels):
    clf = LinearSVC(random_state=0, tol=1e-5, C = 5)
    cv_result = cross_validate(clf, instance, labels, cv=5, return_train_score=True)   
    test_score = cv_result.get('test_score')
    train_score = cv_result.get('train_score')
    print ('5 folds, test score: ', test_score, "train score: ", train_score)

if __name__ == '__main__':
    nums = []
    sentences = []
    corpus = []
    vals = open_file_csv()
    val = vals[0]
    labels = vals[1]
    for i in val:
        corpus.append(i[2])

    stop = set(stopwords.words('english'))

    v_array = vectorize(corpus, stop)
    #v_array.sort_indices()
    v_merge = []
    count = 0
    end = 0

 #   prediction = regression(v_array, labels)
    svm_prediction = svm(v_array, labels)
   # print(svm_prediction)

 #   cross_validation(v_array, labels)

  #  printPerformance('Vanilla Linear Regression', labels, prediction)
    printPerformance('SVM', labels, svm_prediction)    
    #printPerformance('Ada Boost SVM', labels, ada_boost(v_array, labels))

#    with open('./output_6.txt', 'w+') as file_out:
 #       for item in v_merge:
  #          file_out.write("%s\n" % item)
