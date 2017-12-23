#file to store most frequent words in words.txt with frequency>=2

import re
import nltk
from gensim import models
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from random import shuffle
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import sys
import os
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from keras.layers import Dense
from keras.layers import Dropout
from numpy import array
import numpy

dir1 = sys.argv[1]
dir2 = sys.argv[2]

files1 = os.listdir(dir1)
files2 = os.listdir(dir2)
sentences=[]
trainfiles = []
testfiles = []
train_sentences = []
test_sentences = []

train_data = numpy.zeros((1800,100))
test_data = numpy.zeros((200,100))
# 90% train 10%test data split
def split_data(f1,f2):
    for f in f1:
        if f.startswith('cv9'):
            testfiles.append(dir1+'/'+f)
        else:
            trainfiles.append(dir1+'/'+f)
    for f in f2:
        if f.startswith('cv9'):
            testfiles.append(dir2+'/'+f)
        else:
            trainfiles.append(dir2+'/'+f)

#load files and update wordcount
def loadfiles(trainfiles,testfiles):
    for it,f in enumerate(trainfiles):
        file1 = open(f,'r')
        readfile = file1.read()
        file1.close()
        words = regtokenize.tokenize(readfile)
        words = [w for w in words if not w in stopwords and len(w)>1 and w.isalpha()]
        train_sentences.append(LabeledSentence(words,[str(it)]))
        sentences.append(LabeledSentence(words,[str(it)]))
        #print (LabeledSentence(words,str(it)))

    for it,f in enumerate(testfiles):
        file1 = open(f,'r')
        readfile = file1.read()
        file1.close()
        words = regtokenize.tokenize(readfile)
        words = [w for w in words if not w in stopwords and len(w)>1 and w.isalpha()]
        test_sentences.append(LabeledSentence(words,[str(it+1800)]))
        sentences.append(LabeledSentence(words,[str(it+1800)]))
    return


def convert_to_doc2vec():
    model = Doc2Vec(alpha=.025, min_alpha=.025,min_count=1, size=100)
    model.build_vocab(sentences)
    model.train(sentences,total_examples=model.corpus_count,epochs=10)
    model.save('./imdb.d2v')
    print (model.most_similar('good'))
    print (model.docvecs['9'])
    #print (model.docvecs['test_1'])


def loadvectors():
    get_model = models.Doc2Vec.load('./imdb.d2v')
    for i in range(1800):
        train_data[i] = get_model.docvecs[str(i)]
    for i in range(200):
        test_data[i] = get_model.docvecs[str(i+1800)]
    return

def neuralnetwork():
    model = Sequential()
    model.add(Dense(64,input_shape=(train_data.shape[1],),activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data, y_train, epochs=100,verbose=2)
    loss,accuracy = model.evaluate(test_data, y_test)
    print (accuracy)
    return

def logisticRegression():
    classifier = LogisticRegression()
    classifier.fit(train_data, y_train)
    print(classifier.score(test_data, y_test))


if __name__ == '__main__':
    wordcount = Counter()
    split_data(files1,files2)
    file2 = open('stopwords.txt','r')
    stopwords = file2.read().split('\n')
    file2.close()
    regtokenize = RegexpTokenizer(r'\w+')
    loadfiles(trainfiles,testfiles)
    convert_to_doc2vec()
    y_train = array([0 for i in range(900)] + [1 for i in range(900)])
    y_test = array([0 for i in range(100)] + [1 for i in range(100)])
    loadvectors()
    train_data = numpy.array(train_data)
    test_data = numpy.array(test_data)
    print (train_data.shape)
    #neuralnetwork()
    logisticRegression()
