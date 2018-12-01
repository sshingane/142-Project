#from __future__ import print_function 
import numpy as np
import re
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import csv
from sklearn.


def open_file():
    instances = []
    labels = []
    data = csv.reader(open('train.csv'))
    next(data)  # Skip header row
    for line in data:
        instances.append(line[0:3])
        labels.append(line[3])
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
    vectorizer = TfidfVectorizer(analyzer='word', preprocessor=None)
    corpus = []
    for i in instances:         
        corpus.append(i[2]) # appends the phrase of each instance into the corpus
    corpus_fitted = vectorizer.fit(corpus)
    corpus_transformed = corpus_fitted.transform(corpus)
    vector_matrx = corpus_transformed.toarray()

    vectorized_phrases = []
    for row in vector_matrx: 
        vectorized_phrases.append(row.tolist())
    return vectorized_phrases


if __name__ == '__main__':
    # Open file 
    file_data = open_file()
    instances = file_data[0]
    labels = file_data[1]
    
    # Number of training instances
    print(len(instances))    

    # Parse labels

    stop = ['a', 'the', 'of', 'and', 'to', 'is']

    vectorized_instances = vectorize(instances)
    
    
    # vectorized_instances.sort_indices()
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

    with open('./output_6.txt', 'w+') as file_out:
        for item in v_merge:
            file_out.write("%s\n" % item)

