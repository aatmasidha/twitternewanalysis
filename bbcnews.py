from collections import defaultdict
import os
import pickle
import random
import string

# from _cffi_backend import typeof
import nltk
from nltk  import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
nltk.download('punkt')

import numpy as np

# from nltk.sentiment.util import naive_bayes
# Remove common words those are repetative and do not impact analysis

# news dataset downloaded from http://mlg.ucd.ie/datasets/bbc.html

stopWords = set(stopwords.words('english'))
stopWords.add('said')
stopWords.add('mr')

BASE_DIR = './newsdataset'
LABELS = ['business', 'entertainment', 'politics', 'sport', 'tech']


# The function makes the list of all the files present in
# different folders
def createDataSet():
    # baseFolder = './newsdataset'
    with open('data.txt', 'w', encoding='utf8') as outFile:
        for label in LABELS:
            datasetDir = '%s/%s' % (BASE_DIR, label)
            for fileName in os.listdir(datasetDir): 
                fullFileName = '%s/%s' % (datasetDir, fileName)
                print(fullFileName)
                with open(fullFileName, 'rb') as file:
                    text = file.read().decode(errors='replace').replace('\n', '')
                    outFile.write('%s\t%s\t%s\n' % (label, fileName, text))

                    
# create tuples ((label, text) (label, text))
# this will form the tuples with the category and word for that category
def setupDocs():
    docs = []  # (label, text)
    with open('data.txt', 'r', encoding='utf8') as dataFile:
        for row in dataFile:
            parts = row.split('\t')
            doc = (parts[0], parts[2].strip())
            
            docs.append(doc)
    
    return docs        


# Remove punctuation marks
def cleanText(text):
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text


# do not add word if it is present in the stop word
# these words are like this, that, we which are common to
# all categories and do not help in categorization
def getTokens(text):
    # get individual words
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if not t in stopWords]
    return tokens


def printFrequncyDist(docs):
    tokens = defaultdict(list)
    for doc in docs:
        docLabel = doc[0]
        docText = cleanText(doc[1])
        docTokens = getTokens(docText)
        tokens[docLabel].extend(docTokens)
    
    for categoryLabel, categoryTokens in tokens.items():
        print(categoryLabel) 
        fd = FreqDist(categoryTokens)
        print(fd.most_common(20))

# def build_ngrams(text, n=2):
#     tokens = text.lower().split()
#     return list(nltk.ngrams(tokens, n))


def getSplits(docs):
    random.shuffle(docs)
    
    x_train = []  # training documents
    y_train = []  # corresponding training labels          
    
    x_test = []  # test documents
    y_test = []  # corresponding test labels
    
    pivot = int(0.80 * len(docs))
    for i in range (0, pivot):
        x_train.append(docs[i][1])
        y_train.append(docs[i][0])
        
    for i in range (pivot, len(docs)):
        x_test.append(docs[i][1])
        y_test.append(docs[i][0])
    
    return x_train, x_test, y_train, y_test


def evaluate_classifier(title, classifer, vectorizer, x_test, y_test):
    x_test_tfidf = vectorizer.transform(x_test)
    y_pred = classifer.predict(x_test_tfidf)
    
    precision = metrics.precision_score(y_test, y_pred, average=None)
    # print(type(precision))
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1 = metrics.f1_score(y_test, y_pred, average=None)
    
    # print("%s\t%f\t%f\t%f\n" % (title, precision, recall, f1))
    print("%s %f %f %f" % (title, precision[0], recall[0], f1[0]))
    

def trainClassifier(docs):
    x_train, x_test, y_train, y_test = getSplits(docs)
    
    # object contains text converted into vectors
    # vectorizer = CountVectorizer(stopwords = 'english',  ngram_range=(1, 3), min_df = 3, analyzer='word' )
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=3, analyzer='word')
    # vectorizer =  CountVectorizer(analyzer='word', ngram_range=(2, 2)) 

    dtm = vectorizer.fit_transform(x_train).toarray()
    # naive_bayes_classifer = MultinomialNB.fit(dtm, y_train)
    
    naive_bayes_classifer = MultinomialNB()
    naive_bayes_classifer.fit(dtm, y_train)
    
    evaluate_classifier("Naive Bayes\tTRAIN\t", naive_bayes_classifer, vectorizer, x_train, y_train)
    evaluate_classifier("Naive Bayes\tTRAIN\t", naive_bayes_classifer, vectorizer, x_test, y_test)
    
    # store classifier
    classifierFileName = 'naive_bayes_classifier.pkl'
    pickle.dump(naive_bayes_classifer, open(classifierFileName, 'wb'))
    
    # also store the vectorizer so we can transform new data
    vecFileName = 'count_vectorizer.pkl'
    pickle.dump(vectorizer, open(vecFileName, 'wb'))


def classifier(text):
    # load classifier
    classifierFileName = './classification/naive_bayes_classifier.pkl'
    nbl_classifier = pickle.load(open(classifierFileName, 'rb'))

    vectorFileName = './classification/count_vectorizer.pkl'
    vectorizer = pickle.load(open(vectorFileName, 'rb'))
    pred = nbl_classifier.predict(vectorizer.transform([text]))
    
    return pred[0]


def main():
    print("Inside main()")
    createDataSet()
    docs = setupDocs()
    printFrequncyDist(docs)
    trainClassifier(docs)


if __name__ == '__main__':
    main()
