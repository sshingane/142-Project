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

# Opens file, places contents of file to sentence list
def open_file(sentence):
    size = 0

    with open('train.csv') as my_file:    
        sentence = my_file.read().splitlines()

    return sentence

def open_file_csv():
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

# Parses the file for the labels
def num_parse(sentence):
    num_data = []
    count = 0
    counter = 0

    for line in sentences: 
        liner = line.split(",")
       # print liner
        counter += 1
        #print counter
        
        if counter > 1:
            temp = liner[-1]
            temp = int(temp)
            num_data.append(temp)

    return num_data

# Finds the tfidf vectorization of the data
def vectorize(sentences, stop):
    vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = stop, max_features = None) 
    vectorizer.fit(sentences)
    vectorizer.transform(sentences).toarray()
    vector = vectorizer.transform(sentences)
    print (len(vectorizer.vocabulary_))
    #print (vectorizer.vocabulary_, file=open("output_14.txt", "a"))   

    return vector

def regression(instance, labels):
    rounded = []
    reg = LinearRegression(n_jobs = 200, fit_intercept=True, normalize=True).fit(instance, labels)
    prediction = reg.predict(instance)  
    for i in prediction:
        rounded.append(math.trunc(i))
    return rounded  

def svm(instance, labels):
    rounded = []
    svm = LinearSVC(random_state=0, tol=1e-5, C = 1)
    svm_fit = svm.fit(instance, labels)
    prediction = svm_fit.predict(instance)
   # print (prediction, file=open("output_15.txt", "a"))
       
    return prediction

def printPerformance(model_name, actual_labels, predicted_labels):
    labels = [0, 1, 2, 3, 4]
    conf_matrix = sklearn.metrics.confusion_matrix(actual_labels, predicted_labels, labels = labels)
    accuracy = sklearn.metrics.accuracy_score(actual_labels, predicted_labels)
    precision = sklearn.metrics.precision_score(actual_labels, predicted_labels, labels=labels, average=None)
    f_measure = sklearn.metrics.f1_score(actual_labels, predicted_labels, labels=labels, average=None)
    print('='*60)
    print(model_name)
    print('='*60)
    print('CONFUSION MATRIX:')
    print(conf_matrix)
    print('ACCURACY: ' + str(accuracy))
    print('PRECISION: ' + str(precision))
    print('F-MEASURE: ', str(f_measure))

if __name__ == '__main__':
    nums = []
    sentences = []
    corpus = []
    vals = open_file_csv()
    val = vals[0]
    labels = vals[1]
    for i in val:
        corpus.append(i[2])
   # sentences = open_file(sentences)

   # print (len(sentences))
    #print (len(corpus))
    #nums = num_parse(sentences)
    stop = set(stopwords.words('english'))

    #vector.todok().keys()
    #vector.todok().items()

    v_array = vectorize(corpus, stop)
    #v_array.sort_indices()
    v_data = [] 
    v_data = v_array.data
    #v_data.sort()
    v_partial = []
    #v_partial = trim(v_data)
    v_merge = []
    rounded = []
    test_score = []
    train_score = []
    count = 0
    end = 0

 #   prediction = regression(v_array, labels)
    svm_prediction = svm(v_array, labels)
    print(svm_prediction)
    clf = LinearSVC(random_state=0, tol=1e-5, C = 1)
    cv_result = cross_validate(clf, v_array, labels, cv=5, return_train_score=True)   
    test_score = cv_result.get('test_score')
    train_score = cv_result.get('train_score')
    print ('5 folds, test score: ', test_score, "train score: ", train_score)


  #  printPerformance('Vanilla Linear Regression', labels, prediction)
    printPerformance('SVM', labels, svm_prediction)    
    #printPerformance('SVM Validation', labels, cv_result)

    #while end != 109242:
     #   if (v_partial[end]-(v_array[end].data)).any():
      #      v_merge.append(v_array[end].indices)
       # end += 1


    #print (vectorizer.vocabulary_)
    #print vectorizer.idf_

    #print ((v_array.indices))

    #with open('./output_13.txt', 'w+') as file_out:
     #   for item in v_array:
      #      file_out.write("%s\n" % item)

    #print len(nums)
    #print nums

    #with open('./output_9.txt', 'w+') as file_out:
     #   for item in nums:
      #      file_out.write("%s\n" % item)


#    with open('./output_6.txt', 'w+') as file_out:
 #       for item in v_merge:
  #          file_out.write("%s\n" % item)
