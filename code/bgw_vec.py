#file to store most frequent words in words.txt with frequency>=2

import re
import nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import sys
import os
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from numpy import array

dir1 = sys.argv[1]
dir2 = sys.argv[2]

files1 = os.listdir(dir1)
files2 = os.listdir(dir2)

trainfiles = []
testfiles = []

trainvector = []
testvector  = []
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
def loadfiles(trainfiles):
    for f in trainfiles:
        file1 = open(f,'r')
        readfile = file1.read()
        file1.close()
        words = regtokenize.tokenize(readfile)
        words = [w for w in words if not w in stopwords and len(w)>1 and w.isalpha()]
        wordcount.update(words)
    return

def words_to_vectors():
    for f in trainfiles:
        file1 = open(f,'r')
        readfile = file1.read()
        file1.close()
        words = regtokenize.tokenize(readfile)
        words = [w for w in words if not w in stopwords and len(w)>1 and w.isalpha() and w in frequentwords]
        trainvector.append(' '.join(words))
    for f in testfiles:
        file1 = open(f,'r')
        readfile = file1.read()
        file1.close()
        words = regtokenize.tokenize(readfile)
        words = [w for w in words if not w in stopwords and len(w)>1 and w.isalpha() and w in frequentwords]
        testvector.append(' '.join(words))

def neuralnetwork():
    model = Sequential()
    model.add(Dense(64,input_shape=(train_data.shape[1],),activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data, y_train, epochs=50,verbose=2)
    score = model.evaluate(test_data, y_test)
    print (score)


if __name__ == '__main__':
    wordcount = Counter()
    split_data(files1,files2)
    file2 = open('stopwords.txt','r')
    stopwords = file2.read().split('\n')
    file2.close()
    regtokenize = RegexpTokenizer(r'\w+')
    loadfiles(trainfiles)
    frequentwords = [t for t,c in wordcount.items() if c>=2]
    words_to_vectors()
    tk = Tokenizer()
    tk.fit_on_texts(trainvector)
    tk.fit_on_texts(testvector)
    train_data = tk.texts_to_matrix(trainvector,mode='freq')
    test_data = tk.texts_to_matrix(testvector,mode='freq')
    y_train = array([0 for i in range(900)] + [1 for i in range(900)])
    y_test = array([0 for i in range(100)] + [1 for i in range(100)])
    neuralnetwork()
