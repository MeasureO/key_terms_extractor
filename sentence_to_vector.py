import nltk
from nltk import word_tokenize, sent_tokenize
import numpy as np

def tokenize(texts):
    # tokenize all 3 sentences; do not lemmatize/stem them
    # make a list of tokens
    # append this list to a bigger list of all 3 sentences
    tokenized = []
    for text in texts:
        tokenized += [word_tokenize(text.lower())]
    return tokenized

def make_voc(tokens):
    # make your own vocabulary here
    # tokens in your vocabulary should have only lowercase
    voc = []
    for sent in tokens:
        voc += sent
    return list(sorted(set(voc)))


def make_matrix(voc, texts):
    matrix = np.zeros((len(texts), len(voc)))
    # make a binary matrix
    for i in range(len(texts)):
        for j in range(len(voc)):
            if voc[j] in texts[i]:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
            
    return matrix

corpus = str(input()) # you get a string of 3 sentences
texts = list(sent_tokenize(corpus)) # list of 3 strings-sentences
tokens = tokenize(texts) # word_tokenize your sentences
voc = make_voc(tokens) # use these tokens to make a vocabulary
print(make_matrix(voc, tokens)) # make a matrix
