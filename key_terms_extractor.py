from nltk import word_tokenize
from lxml import etree
from collections import Counter
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

import string
import re
import nltk

# nltk.download('averaged_perceptron_tagger')
# print(stopwords.words('english'))
# nltk.download('wordnet')
# nltk.download('omw-1.4')
#nltk.download('punkt')

vectorizer = TfidfVectorizer()
lemmatizer = WordNetLemmatizer()

xml_file = "news.xml"
tree = etree.parse(xml_file)
root = tree.getroot()
# etree.dump(root[0])
articles = {}
for new in root[0]:
    articles[new[0].text] = new[1].text
    #print(new[0].text, new[1].text)
# print(articles)
answer = {}
dataset = []
for item in articles.keys():
    # tokens = word_tokenize(item.lower())
    dirty_tokens = word_tokenize(articles[item].lower())
    # dirty_tokens = [re.sub(r'[^\w\s]', '', i) for i in dirty_tokens]
    dirty_tokens = [lemmatizer.lemmatize(i) for i in dirty_tokens]
    tokens = []
    for word in dirty_tokens:
        # word = re.sub(r'[^\w\s]', '', word)
        if word in stopwords.words("english") or word in string.punctuation:
            continue
        tokens.append(word)

    # print(tokens)
    tokens = [i for i in tokens if i != '']
    tokens = [i for i in tokens if pos_tag([i])[0][1] == "NN"]
    dataset.append(" ".join(tokens))
tfidf_matrix = vectorizer.fit_transform(dataset)
tfidf_matrix = tfidf_matrix.toarray()
terms = vectorizer.get_feature_names()
answer = []
for elem in tfidf_matrix:
    answer.append(sorted(list(zip(terms, elem)), key=lambda x: (x[1], x[0]), reverse=True))
answer = dict(zip(articles.keys(), answer))
# with open("test.txt", 'w') as output:
#     print(dataset, file=output)
#     print(tfidf_matrix, file=output)
#     print(terms, file=output)
#     print(answer, file=output)
    # counter = dict(Counter(tokens))
    # print(counter)
    # counter = {lemmatizer.lemmatize(i): counter[i] for i in counter.keys()}

#     answer[item] = {i: counter[i] for i in sorted(counter, key=lambda x: (counter[x], x), reverse=True)}
# # print(answer)
for item in answer:
    print(item + ':')
    i = 0
    for word in answer[item]:
        print(word[0], end=' ')
        i += 1
        if i > 4:
            break
    print()
