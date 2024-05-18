'''
preprocessing pipline
1. tokenization
2. lower+stemming
3. remove punctuation
4. BOW
'''

# importing the libraries
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords # Stopwords
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import numpy as np


#nltk.download('stopwords')
en_stop=stopwords.words('english')
en_stop.remove('not')
def remove_stop_words(text):
    return " ".join(word for word in text.split() if word not in en_stop )


def tokenization(sentence):
    return word_tokenize(sentence)



def stemming(word):
    
    return stemmer.stem(word.lower()) 


def Bag_of_words(tokenized_sent, all_words):
    tokenized_sent=[stemming(w) for w in tokenized_sent]
    bag = np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sent:
            bag[idx]=1.0
    return bag


a= 'what is the largest animal in the world'
print(a)
s=remove_stop_words(a)
print(s)
t=tokenization(s)
print(t)
stemmed_words = [stemming(word) for word in t]
print(stemmed_words)

'''
import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

'''


