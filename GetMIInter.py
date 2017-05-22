#coding=utf-8
import numpy as np

class GetMIInter(object):
    def __init__(self):
        # loading inner mi
        print "loading inter mi..."
        self.__interN = 0
        self.__interDicS = {}
        self.__interMIDic = {}
        
        f = open("data/miinter.txt",'r')
        line = f.readline()
        while line:
            para = line.split(" ");
            self.__interMIDic[para[0]] = int(para[1])
            line = f.readline()
        f.close()
        
        f = open("data/miinter_words.txt",'r')
        self.__interN = int(f.readline(),10)
        print "interN: %d " % (self.__interN)
        #line = f.readline().decode('utf-8')
        line = f.readline()
        while line:
            para = line.split(" ");
            self.__interDicS[para[0]] = int(para[1])
            line = f.readline()
        f.close()

    def getMI(self, c1, c2):
        pa = ""
        if (c1 < c2):
            pa = c1 + c2
        else:
            pa = c2 + c1
        #print(pa)
                
        nsent = 0    
        if (self.__interMIDic.has_key(pa)):
            nsent = self.__interMIDic[pa]
                
        na = 0
        nb = 0
        #print(nsent)
                
        if (self.__interDicS.has_key(c1)):
            na = self.__interDicS[c1]
                
        if (self.__interDicS.has_key(c2)):
            nb = self.__interDicS[c2]
                
        innmis = np.log2(float(nsent+1) * float(self.__interN) / (float(na+1) * float(nb+1)))
        return innmis

    def lineSplit(self, line):
        chars = []
        for i in range(0, len(line), 3):
            c = line[i:i+3]
            #print(c)
            chars.append(c)
        return chars

    def getMIScore(self, sline, oline):
        swords = self.lineSplit(sline)
        owords = self.lineSplit(oline)
        

        swordNum = len(swords)
        owordNum = len(owords)

        intmis = 0.0
        
        for i in range(0, owordNum):
            for j in range(0,swordNum):
                pa = ""
                if (owords[i] < swords[j]):
                    pa = owords[i] + swords[j]
                else:
                    pa = swords[j] + owords[i]
                    
                nsent = 0    
                if (self.__interMIDic.has_key(pa)):
                    nsent = self.__interMIDic[pa]
                    
                na = 0
                nb = 0
                    
                if (self.__interDicS.has_key(owords[i])):
                    na = self.__interDicS[owords[i]]
                    
                if (self.__interDicS.has_key(swords[j])):
                    nb = self.__interDicS[swords[j]]
                    
                intmis += np.log2(float(nsent+1) * float(self.__interN) / (float(na+1) * float(nb+1)))
            
        interMIScore = intmis / (owordNum * swordNum)
        return interMIScore

def main():
    myMI = GetMIinter()
    while True:
        string1 = raw_input("input word1> \n")
        string2 = raw_input("input word2> \n")
        string1 = string1.strip()
        string2 = string2.strip()
        mi= myMI.getMI(string1[0:3], string2[0:3])  + myMI.getMI(string1[0:3], string2[3:]) + \
                myMI.getMI(string1[3:], string2[0:3])  + myMI.getMI(string1[3:3], string2[3:])
        print (np.exp(mi/4.0))

if __name__ == "__main__":
    main()
