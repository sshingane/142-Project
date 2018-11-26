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

sentences = []
size = 0
stop = ['a', 'the', 'of', 'and', 'to', 'is']
with open('train.csv') as my_file:
    for line in my_file:
        size += 1
        sentences.append(line)

#vocabulary = tokenize_sentences(sentences)
#bagofwords("the only thing Avary seems to care about are mean giggles and pulchritude", vocabulary)
#bags = [ collections.Counter(re.findall(r'\w+', txt)) for txt in sentences]
#print "got bags"
#sumbags = sum(bags, collections.Counter())
#print "got sumbags"

vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = stop, max_features = 5000) 
vectorizer.fit(sentences)
vectorizer.transform(sentences).toarray()
vector = vectorizer.transform(sentences)
vector.todok().keys()
vector.todok().items()
#v_array = vector.toarray()

#print vectorizer.vocabulary_
#print vectorizer.idf_\
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

print vector
#print '\n'.join(str(line) for line in vocabulary) 
#print sumbags