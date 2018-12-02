import collections
import csv
import math
import re

import numpy as np
import sklearn.metrics
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_validate


"""
Returns tuple of instances, labels of the data from the file 
"""
def open_file():
    instances = []
    labels = []
    data = csv.reader(open('train.csv'))
    next(data)  # Skip header row
    for line in data:
        instances.append(line[0:3])
        labels.append(int(line[3]))
    return (instances, labels)

def trim(data):
    v_trim = []
    for line in data: 
        if line > .5:
            v_trim.append(line)
    return v_trim  

def count_vectorize(instances):
    vectorizer = CountVectorizer(analyzer = 'word', stop_words = stopwords.words('english'))
    corpus = []
    for i in instances:
        # appends the phrase of each instance into the corpus
        corpus.append(i[2])
    corpus_fitted = vectorizer.fit(corpus)
    corpus_transformed = corpus_fitted.transform(corpus)

    print('Count Vectorizer Vocab Length: ' + str(len(vectorizer.vocabulary_)))
    return corpus_transformed

def tfid_vectorize(instances):
    vectorizer = TfidfVectorizer(analyzer='word', preprocessor=None, stop_words=stopwords.words('english'))
    corpus = []
    for i in instances:         
        corpus.append(i[2]) # appends the phrase of each instance into the corpus
    corpus_fitted = vectorizer.fit(corpus)
    corpus_transformed = corpus_fitted.transform(corpus)

    print('tfid Vectorizer Vocab Length: ' + str(len(vectorizer.vocabulary_)))
    return corpus_transformed

def linear_regression(feature_vector_matrix, actual_labels):
    # Vanilla Regression Model 
    # 200 iterations (matches asg 4)
    print('training model')
    vanilla_linear_regression = LinearRegression(fit_intercept=True, normalize=True).fit(feature_vector_matrix, actual_labels)
    print('finished training model')
    predicted_labels = vanilla_linear_regression.predict(feature_vector_matrix)
    rounded = []
    for i in predicted_labels: 
        rounded.append(math.trunc(i))
    return rounded 

def naive_bayes(feature_vector_matrix, actual_labels):
    nb = MultinomialNB()
    nb.fit(feature_vector_matrix, actual_labels)
    predicted_labels = nb.predict(feature_vector_matrix)
    return predicted_labels

def svm(feature_vector_matrix, actual_labels):
    svm = LinearSVC(random_state=0, tol=1e-5, max_iter=4000)
    svm_fit = svm.fit(feature_vector_matrix, actual_labels)
    prediction = svm_fit.predict(feature_vector_matrix)
    return prediction

def perceptron(feature_vector_matrix, actual_labels):
    percep = Perceptron(tol=1e-3, random_state=0)
    percep.fit(feature_vector_matrix, actual_labels)
    prediction = percep.predict(feature_vector_matrix)

    return prediction
    
def printPerformance(model_name, actual_labels, predicted_labels):
    labels = [0, 1, 2, 3, 4]
    conf_matrix = sklearn.metrics.confusion_matrix(actual_labels, predicted_labels, labels = labels)
    accuracy = sklearn.metrics.accuracy_score(actual_labels, predicted_labels)
    precision = sklearn.metrics.precision_score(actual_labels, predicted_labels, labels=labels, average=None)
    recall = sklearn.metrics.recall_score(actual_labels, predicted_labels, labels=labels, average=None)
    print('='*60)
    print(model_name)
    print('='*60)
    print('CONFUSION MATRIX:')
    print(conf_matrix)
    print('ACCURACY: ' + str(accuracy) + ' Ava')
    print('RECALL: ' + str(recall))
    print('PRECISION: ' + str(precision) + '\n')
    

def cross_validation(instance, labels):
    clf = MultinomialNB()
    cv_result = cross_validate(
        clf, instance, labels, cv=7, return_train_score=True)
    test_score = cv_result.get('test_score')
    train_score = cv_result.get('train_score')
    print('5 folds, test score: ', test_score, "train score: ", train_score)


if __name__ == '__main__':
    # Open file 
    file_data = open_file()
    instances = file_data[0]
    actual_labels = file_data[1]
    
    
    # Number of training instances
    print(len(instances))    

    # TFID Vectorize phrases
    tfid_vector_matrix = tfid_vectorize(instances)

    # Count Vectorize phrases
    count_vector_matrix = count_vectorize(instances)
    
    # Call vanilla regression
    lin_reg_predicted_labels = linear_regression(tfid_vector_matrix, actual_labels)
    lin_reg_predicted_labels_count = linear_regression(count_vector_matrix, actual_labels)

    printPerformance('Vanilla Linear Regression + tfid', actual_labels, lin_reg_predicted_labels)
    printPerformance('Vanilla Linear Regression + count', actual_labels, lin_reg_predicted_labels_count)

    # Call Naive Bayes classifier
    naive_bayes_predicted_labels = naive_bayes(tfid_vector_matrix, actual_labels)
    naive_bayes_predicted_labels_count = naive_bayes(count_vector_matrix, actual_labels)
    printPerformance('Naive Bayes + tfid', actual_labels, naive_bayes_predicted_labels)
    printPerformance('Naive Bayes + count', actual_labels, naive_bayes_predicted_labels_count)

    # Call SVM classifier 
    svm_predicted_labels = svm(tfid_vector_matrix, actual_labels)
    svm_predicted_labels_count = svm(count_vector_matrix, actual_labels)

    printPerformance('SVM + tfid', actual_labels, svm_predicted_labels)
    printPerformance('SVM + count', actual_labels, svm_predicted_labels_count)

    # Call Perceptron classifier
    percep_predicted_labels = perceptron(tfid_vector_matrix, actual_labels)
    percep_predicted_labels_count = perceptron(count_vector_matrix, actual_labels)

    printPerformance('Perceptron + tfid', actual_labels, percep_predicted_labels)
    printPerformance('Perceptron + count', actual_labels, percep_predicted_labels_count)

    cross_validation(tfid_vector_matrix, actual_labels)