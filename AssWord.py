#coding=utf-8
from __future__ import division
import numpy as np


class AssWord(object):
    def __init__(self):

        self.__loadwordsVec()
        self.__loadAllDic()

    def __loadwordsVec(self):
        print("loading word vectors...")
        wordsvec = np.load('data/wordsvector.npy')
        self.wordsvec = np.transpose(wordsvec)
        N = self.wordsvec.shape[1]
        print(N)
        self.Norm = np.zeros([1,N])
        for i in range(0, N):
            p = self.wordsvec[:,i]
            self.Norm[0, i] = np.sqrt(p.dot(p))

        self.Norm += 1e-10
        print(np.shape(self.wordsvec))

    def __loadAllDic(self):
        #loading All dic
        print "loading all dictionary..."
        self.idx2char = {}
        self.char2idx = {}

        fin = open("data/allvecwords.txt", 'r')
        word = fin.readline()
        count = 0
        while word:
            word = word.strip()
            self.char2idx[word] = count
            self.idx2char[count] = word
            count += 1
            word = fin.readline()

        fin.close()
        print len(self.char2idx)
        print len(self.idx2char)


    def getA0WordsByVec(self, vec, candiNum):
        seqvec = vec
        C = np.dot(seqvec, self.A0vecs)
        #print(np.shape(C))
        seqvec = np.transpose(seqvec)
        nork = np.sqrt(np.sum(seqvec * seqvec))
        #print(nork)
        #print(np.shape(self.Norm))

        distance = C / self.A0Norm / nork        
        sortedDistIndices = np.argsort(distance)
        sortedDistIndices = sortedDistIndices[0]
        sortedDistIndices = list(sortedDistIndices)
        sortedDistIndices.reverse() 

        distance = distance[0]
        distance = list(distance)

        classCount = []
        N = len(sortedDistIndices)
        assNum = min(candiNum, N)

        for i in range(0, assNum):  
            char = self.A0idx2word[sortedDistIndices[i]]
            dis = distance[sortedDistIndices[i]]

            classCount.append( (char,  dis)   )
            

        return classCount # 0 is the ori word



    def getWordsByVec(self, vec):
        seqvec = vec
        C = np.dot(seqvec, self.wordsvec)
        #print(np.shape(C))
        seqvec = np.transpose(seqvec)
        nork = np.sqrt(np.sum(seqvec * seqvec))
        #print(nork)
        #print(np.shape(self.Norm))

        distance = C / self.Norm / nork        
        sortedDistIndices = np.argsort(distance)
        sortedDistIndices = sortedDistIndices[0]
        sortedDistIndices = list(sortedDistIndices)
        sortedDistIndices.reverse() 

        distance = distance[0]
        distance = list(distance)

        classCount = []
        N = len(sortedDistIndices)
        assNum = min(50, N)

        for i in range(0, assNum):  
            char = self.idx2char[sortedDistIndices[i]]
            dis = distance[sortedDistIndices[i]]
            
            if len(char) == 3:
                continue
            
            classCount.append( (char,  dis)   )
            

        return classCount # 0 is the ori word


    def getKNN(self, word, candinum):
        idx = self.char2idx[word]
        emb = self.wordsvec[:, idx]
        ans = self.getWordsByVec(emb)
        num = min(len(ans), candinum)
        ans = ans[0:num]
        '''
        for i, item in enumerate(ans):
            word = item[0]
            dis = item[1]
            print ("%s %f" % (word, dis))
        '''

        asswords = [item[0] for item in ans]
        return asswords


def main():

    Ass = AssWord()
    while True:
        string = raw_input("input a word> \n")
        #Ass.getCosDis4Chars2Order(string)
        words = Ass.getKNN(string, 20)
        for word in words:
            print (word)

if __name__ == "__main__":
    main()