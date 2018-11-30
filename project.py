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
        sentence = my_file.read().splitlines()

    return sentence

def trim(data):
    v_trim = []

    for line in data: 
        if line > .5:
            v_trim.append(line)

    return v_trim  

def num_parse(sentence):
    num_data = []
    count = 0
    counter = 0

    for line in sentences: 
        liner = line.split(",")
       # print liner
        #counter += 1
        #print counter
        
        if counter > 1:
            temp = liner[-1]
            temp = int(temp)
            num_data.append(temp)
        '''
        for i in range(len(liner)):
            #print len(liner)
           # count += 1
            #print ("count: ", count, " counter: ", counter)
            
            if liner[i].isdigit():
              #  count += 1
             #   print ("count: ", count, " counter: ", counter) 
                temp = int(liner[i])
                #print temp
                if temp < 5 and i == len(liner)-1:
                    #print temp
                    num_data.append(temp)
                    '''

    return num_data

def vectorize(sentences, stop):
    vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = stop, max_features = 5000) 
    vectorizer.fit(sentences)
    vectorizer.transform(sentences).toarray()
    vector = vectorizer.transform(sentences)

    return vector

nums = []
sentences = []
sentences = open_file(sentences)
print len(sentences)
nums = num_parse(sentences)
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
#v_data.sort()
v_partial = []
v_partial = trim(v_data)
v_merge = []
i = 0
end = 0

while end != 109234:
    if (v_partial[end]-(v_array[end].data)).any():
        v_merge.append(v_array[end].indices)
    end += 1


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
#print v_array[4535].indices

#print len(nums)
#print nums

#with open('./output_9.txt', 'w+') as file_out:
 #   for item in nums:
  #      file_out.write("%s\n" % item)


with open('./output_6.txt', 'w+') as file_out:
    for item in v_merge:
        file_out.write("%s\n" % item)


#print '\n'.join(str(line) for line in vocabulary) 
#print sumbags