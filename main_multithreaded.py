import numpy as np
import pandas as pd
from math import sqrt, log
from itertools import chain, product
from collections import defaultdict

df1 = pd.read_excel('corpus.xlsx', sheetname='sh',index_col=0)
df2 = pd.read_excel('corpus.xlsx', sheetname='lo',index_col=0)
df2 = df2.iloc[:7,:]


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




##############################################################################################
import multiprocessing
import timeit


# Run this with a pool of 5 agents having a chunksize of 3 until finished
df1_1 = df1.copy().iloc[:1000,:]
df1_2 = df1.copy().iloc[1000:2000,:]
df1_3 = df1.copy().iloc[2000:3000,:]
df1_4 = df1.copy().iloc[3000:4000,:]
df1_5 = df1.copy().iloc[4000:5000,:]
df1_6 = df1.copy().iloc[5000:6000,:]
df1_7 = df1.copy().iloc[6000:,:]

# Make smaller for development purposes
#f2 = df2.iloc[:10,:]
df1_chunks = [df1_1,df1_2,df1_3,df1_4,df1_5,df1_6,df1_7]

'''
if __name__ == '__main__':
    start_time = timeit.default_timer()
    threads = 1
    pool = Pool(processes=threads)
    results = [pool.apply(test_cosine, args=(x,)) for x in df1_chunks]
    pool.close()
    pool.join()

    elapsed = timeit.default_timer() - start_time
    print("Finised in:{0} seconds with {1} threads".format(elapsed,threads))
'''

if __name__ == "__main__":

    manager = multiprocessing.Manager()
    return_list = manager.list()

    def test_cosine(data1, data2=df2):
        score,highestScore = 0,0
        counter = 0
        newMatch = []
        #allMatches = pd.DataFrame(columns=['UID','index','indexOfMatch','score'])
        matches = []
        
        def create_test_corpus(corp1,corp2):
            
            data1_lwr = corp1['DESCRIPTION'].str.lower()
            data2_lwr = corp2['DESCRIPTION'].str.lower() 
            
            all_sents = list(data1_lwr) + list(data2_lwr)
            corpus, vocab = corpus2vectors(all_sents)
            return corpus, vocab, data1_lwr, data2_lwr
                        
        corpus,vocab,corp_1,corp_2 = create_test_corpus(corp1=data1,corp2=data2)
        
        corpus_dict = {}
        test = list(zip(*corpus))
        for counter,i in enumerate(test[0]):
            corpus_dict.update({test[0][counter]:test[1][counter]})
        
        #print("CORPUS")
        #print(corpus)
        # Cycle through each row in data1
        for dat1_ix, dat1 in enumerate(corp_1):
            #print("##############################################################################")
            #print("dat1:",dat1)

            # For each iteration of data1, cycle through all of data2 below
            for dat2_ix, dat2 in enumerate(corp_2):
                #print("_______________________")

                #print("Phrase_1:[{0}] {1}".format(dat1_ix,dat1))
                #print("Phrase_2:[{0}] {1}".format(dat2_ix,dat2))
                #print('length: phrase_1:',len(dat1),' phrase_2:',len(dat2))
                

                # Calculate cosime similarity from each line to the other
                score = cosine_sim(corpus_dict[dat1],corpus_dict[dat2])
                #score=0
                
                #score = matutils.cossim(corpus_dict[dat1],corpus_dict[dat2])
                
                
                #temp = corpus_dict[dat1]
                #temp2= corpus_dict[dat2]
                #score = 0 
                #score  = cosine_sim(vec1, vec2)
                #print("cosine = ", score)
                
                #uid = df2.index.values[index1]
                newMatch = [dat1_ix, dat2_ix, score]
                matches.append(newMatch)
            
            counter = counter + 1
        

        return_list.append(matches)

    # start timer
    start_time = timeit.default_timer()

    # creating processes
    p1 = multiprocessing.Process(target=test_cosine, args=(df1_1, ))
    p2 = multiprocessing.Process(target=test_cosine, args=(df1_2, ))    
    p3 = multiprocessing.Process(target=test_cosine, args=(df1_3, ))
    p4 = multiprocessing.Process(target=test_cosine, args=(df1_4, ))   
    p5 = multiprocessing.Process(target=test_cosine, args=(df1_5, ))
    p6 = multiprocessing.Process(target=test_cosine, args=(df1_6, ))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()

    #print(type(return_list))
    import pickle
    pickle.dump(return_list, open("return_list.pickle", "wb"))


    # end timer
    elapsed = timeit.default_timer() - start_time
    print("Finished in {0} seconds! Dataset is shape: {1}".format(elapsed,np.shape(return_list)))
    #print("Index 0:",np.shape(return_list[0]))
    #print(return_list)



