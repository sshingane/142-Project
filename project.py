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
        labels.append(line[3])
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

# Finds the tfidf vectorization of the data
def vectorize(sentences, stop):
    vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = stop, max_features = None) 
    vectorizer.fit(sentences)
    vectorizer.transform(sentences).toarray()
    vector = vectorizer.transform(sentences)
    print (len(vectorizer.vocabulary_))
   # print (vectorizer.vocabulary_, file=open("output_11.txt", "a"))   

    return vector

def regression(instance, labels):
    reg = LinearRegression().fit(instance, labels)
    prediction = reg.predict(instance)  
    return prediction  

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

    #vocabulary = tokenize_sentences(sentences)
    #bagofwords("the only thing Avary seems to care about are mean giggles and pulchritude", vocabulary)
    #bags = [ collections.Counter(re.findall(r'\w+', txt)) for txt in sentences]
    #print "got bags"
    #sumbags = sum(bags, collections.Counter())
    #print "got sumbags"


    #vector.todok().keys()
    #vector.todok().items()

    v_array = vectorize(corpus, stop)
    #v_array.sort_indices()
    v_data = [] 
    v_data = v_array.data
    #v_data.sort()
    v_partial = []
    v_partial = trim(v_data)
    v_merge = []
    rounded = []
    count = 0
    end = 0

    prediction = regression(v_array, labels)
    for i in prediction:
        rounded.append(math.trunc(i))
        if i < 5 and i > 4:
            count += 1
    for i in rounded:
        if i == 4:
            end += 1

    print (count, ", ", end)
    print (len(prediction))
    print (len(rounded))

    #while end != 109242:
     #   if (v_partial[end]-(v_array[end].data)).any():
      #      v_merge.append(v_array[end].indices)
       # end += 1


    #print (vectorizer.vocabulary_)
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
    #print ((v_array.indices))

    #with open('./output_13.txt', 'w+') as file_out:
     #   for item in v_array:
      #      file_out.write("%s\n" % item)

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