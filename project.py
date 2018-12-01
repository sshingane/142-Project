import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import collections
import csv
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
import math

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

def tokenize_sentences(sentences):
    words = []
    for sentence in sentences:
        w = extract_words(sentence)
        words.extend(w)
        
    words = sorted(list(set(words)))
    return words

def extract_words(sentence):
    ignore_words = ['a', 'the', 'of', 'and', 'to', 'is']
    words = re.sub("[^\w]", " ",  sentence).split() #nltk.word_tokenize(sentence)
    words_cleaned = [w.lower() for w in words if w not in ignore_words]
    return words_cleaned    
    
# def bagofwords(sentence, words):
#     for i in sentence: 
#         sentence_words[i] = extract_words(sentence[i])
#     # frequency word count
#     bag = np.zeros(len(words))
#     for sw in sentence_words:
#         for i,word in enumerate(words):
#             if word == sw: 
#                 bag[i] += 1
                
#     return np.array(bag)

def trim(data):
    v_trim = []
    for line in data: 
        if line > .5:
            v_trim.append(line)
    return v_trim  

# def label_parse(sentence):    
#     labels = []
#     for i in sentences: 
#         labels.append(i[3])

#     return labels

def vectorize(instances):
    vectorizer = TfidfVectorizer(analyzer='word', preprocessor=None, stop_words=stopwords.words('english'))
    corpus = []
    for i in instances:         
        corpus.append(i[2]) # appends the phrase of each instance into the corpus
    corpus_fitted = vectorizer.fit(corpus)
    corpus_transformed = corpus_fitted.transform(corpus)

    print(len(vectorizer.vocabulary_))
    return corpus_transformed

def linear_regression(feature_vector_matrix, actual_labels):
    # Vanilla Regression Model 
    # 200 iterations (matches asg 4)
    print('training model')
    vanilla_linear_regression = LinearRegression(
        n_jobs=1, fit_intercept=True, normalize=True).fit(feature_vector_matrix, actual_labels)
    print('finished training model')
    predicted_labels = vanilla_linear_regression.predict(feature_vector_matrix)
    rounded = []
    for i in predicted_labels: 
        rounded.append(math.trunc(i))
    return rounded 
    
    
def printPerformance(model_name, actual_labels, predicted_labels):
    labels = [0, 1, 2, 3, 4]
    # conf_matrix = sklearn.metrics.confusion_matrix(actual_labels, predicted_labels, labels = labels)
    accuracy = sklearn.metrics.accuracy_score(actual_labels, predicted_labels)
    precision = sklearn.metrics.precision_score(actual_labels, predicted_labels, labels=labels, average=None)
    print('='*60)
    print(model_name)
    print('='*60)
    # print('CONFUSION MATRIX:')
    # print(conf_matrix)
    print('ACCURACY: ' + str(accuracy))
    print('PRECISION: ' + str(precision))

if __name__ == '__main__':
    # Open file 
    file_data = open_file()
    instances = file_data[0]
    actual_labels = file_data[1]
    # print(actual_labels)
    # print(len(actual_labels))
    
    # Number of training instances
    print(len(instances))    

    # Vectorize phrases
    feature_vector_matrix = vectorize(instances)
    
    # Call vanilla regression
    lin_reg_predicted_labels = linear_regression(feature_vector_matrix, actual_labels)

    printPerformance('Vanilla Linear Regression', actual_labels, lin_reg_predicted_labels)

    # feature_vector.sort_indices()
    # v_data = [] 
    # v_data = vectorized_instances.data
    # #v_data.sort()
    # v_partial = []
    # v_partial = trim(v_data)
    # v_merge = []
    # i = 0
    # end = 0

    # while end != 109234:
    #     if (v_partial[end]-(vectorized_instances[end].data)).any():
    #         v_merge.append(vectorized_instances[end].indices)
    #     end += 1

    # with open('./output_6.txt', 'w+') as file_out:
    #     for item in v_merge:
    #         file_out.write("%s\n" % item)
