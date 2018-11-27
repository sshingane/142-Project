#from __future__ import print_function 
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
 

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
    
def bagofwords(sentence, words):
    for i in sentence: 
        sentence_words[i] = extract_words(sentence[i])
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i,word in enumerate(words):
            if word == sw: 
                bag[i] += 1
                
    return np.array(bag)

def open_file(sentence):
    size = 0

    with open('train.csv') as my_file:
        for line in my_file:
            size += 1
            sentence.append(line)

    return sentence

def trim(data):
    v_trim = []

    for line in data: 
        if line > .5:
            v_trim.append(line)

    return v_trim  

def vectorize(sentences, stop):
    vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = stop, max_features = 5000) 
    vectorizer.fit(sentences)
    vectorizer.transform(sentences).toarray()
    vector = vectorizer.transform(sentences)

    return vector


sentences = []
sentences = open_file(sentences)
stop = ['a', 'the', 'of', 'and', 'to', 'is']

#vocabulary = tokenize_sentences(sentences)
#bagofwords("the only thing Avary seems to care about are mean giggles and pulchritude", vocabulary)
#bags = [ collections.Counter(re.findall(r'\w+', txt)) for txt in sentences]
#print "got bags"
#sumbags = sum(bags, collections.Counter())
#print "got sumbags"


#vector.todok().keys()
#vector.todok().items()
v_array = vectorize(sentences, stop)
v_array.sort_indices()
v_data = [] 
v_data = v_array.data
v_data.sort()
v_partial = []
v_partial = trim(v_data)


#print vectorizer.vocabulary_
#print vectorizer.idf_

'''
print vector.shape
print len(vector.toarray())
print 'data:'  
print v_array.data
print 'row:' 
print v_array.row
print 'col:' 
print v_array.col
print size
'''

with open('./output_5.txt', 'w+') as file_out:
    for item in v_partial:
        file_out.write("%s\n" % item)


#print '\n'.join(str(line) for line in vocabulary) 
#print sumbags