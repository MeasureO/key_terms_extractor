from nltk import word_tokenize, sent_tokenize

import numpy as np

def tokenize(_texts):
    # tokenize all 3 sentences; do not lemmatize/stem them
    # make a list of tokens
    # append this list to a bigger list of all 3 sentences
    # tokenized = []
    # for text in _texts:
    #     tokenized += [word_tokenize(text.lower())]
    # return tokenized
    return [word_tokenize(text.lower()) for text in _texts]


def make_vocabulary(_tokens):
    # make your own vocabulary here
    # tokens in your vocabulary should have only lowercase
    return sorted({word for sent in _tokens for word in sent})


def make_matrix(_vocabulary, _texts):
    matrix = np.zeros((len(_texts), len(_vocabulary)))
    # make a binary matrix
    for i, _text in enumerate(_texts):
        for j, _word in enumerate(_vocabulary):
            if _vocabulary[j] in _texts[i]:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
            
    return matrix


corpus = str(input())  # you get a string of 3 sentences
texts = list(sent_tokenize(corpus))  # list of 3 strings-sentences
tokens = tokenize(texts)  # word_tokenize your sentences
vocabulary = make_vocabulary(tokens)  # use these tokens to make a vocabulary
print(make_matrix(vocabulary, tokens))  # make a matrix
