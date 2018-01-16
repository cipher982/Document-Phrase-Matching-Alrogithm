import numpy as np
import pandas as pd
from math import sqrt, log
from itertools import chain, product
from collections import defaultdict

df1 = pd.read_excel('corpus.xlsx', sheetname='shh',index_col=0)
df2 = pd.read_excel('corpus.xlsx', sheetname='loo',index_col=0)


# Load the main 2 tables I will be comparing
df1 = pd.read_excel('corpus.xlsx', sheetname='shh',index_col=0)
df2 = pd.read_excel('corpus.xlsx', sheetname='loo',index_col=0)

# method of calculating similarity of the one-hot encoding
def cosine_sim(u,v):
    return np.dot(u,v) / (sqrt(np.dot(u,u)) * sqrt(np.dot(v,v)))

# Vectorize to one-hot binary encoding as numbers work better than strings!
def corpus2vectors(corpus):
    def vectorize(sentence, vocab):
        return [sentence.split().count(i.lower()) for i in vocab]
    vectorized_corpus = []
    # Remobe dupes and sort
    vocab = sorted(set(chain(*[i.lower().split() for i in corpus])))
    for i in corpus:
        vectorized_corpus.append((i, vectorize(i, vocab)))
    return vectorized_corpus, vocab

# Basically just combine these two datasets to get a single corpus
def create_test_corpus(data1,data2):
    all_sents = list(data1) + list(data2)
    corpus, vocab = corpus2vectors(all_sents)
    return corpus, vocab      

# Workhouse function, cycle through all combinations and compute scores
def test_cosine(data1,data2):
    score,highestScore = 0,0
    counter = 0
    newMatch = []
    #allMatches = pd.DataFrame(columns=['UID','index','indexOfMatch','score'])
    matches = np.array([0,0,0.])
    
    # Cycle through each row in data1
    for dat1_ix, dat1 in enumerate(data1):
        print("___________________________________________________")
        print("dat1:",dat1)

        # For each iteration of data1, cycle through all of data2 below
        for dat2_ix, dat2 in enumerate(data2):
            print("_______________________")

            print("Phrase 1:",dat1)
            print("Phrase 2:",dat2)
            print('length df1:',len(df1),'   length df2:',len(df2))
            print("counter:",counter)
            
            # Calculate cosime similarity from each line to the other
            vec1 = corpus[dat1_ix][1]
            print("vec2:",(len(dat1) - 1) + (dat2_ix - 2))
            vec2 = corpus[(len(dat1) - 1) + (dat2_ix - 2)][1]
            score  = cosine_sim(vec1, vec2)
            print("cosine = ", score)
            
            #uid = df2.index.values[index1]
            newMatch = np.array([dat1_ix, dat2_ix, score])
            matches = np.vstack((matches, newMatch))
        
        counter = counter + 1
    
    return matches

# Lower
data1 = df1['DESCRIPTION'].str.lower()
data2 = df2['DESCRIPTION'].str.lower()                   
corpus,vocab = create_test_corpus(data1,data2)

test_cosine(data1,data2)