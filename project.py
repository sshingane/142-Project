import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer

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
    sentence_words = extract_words(sentence)
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i,word in enumerate(words):
            if word == sw: 
                bag[i] += 1
                
    return np.array(bag)

sentences = []
with open('train.csv') as my_file:
    for line in my_file:
        sentences.append(line)

vocabulary = tokenize_sentences(sentences)
bagofwords("Machine learning is great", vocabulary)

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
train_data_features = vectorizer.fit_transform(sentences)
vectorizer.transform(["Machine learning is great"]).toarray()

print '\n'.join(str(line) for line in vocabulary) 