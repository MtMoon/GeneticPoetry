# coding=utf-8
from __future__ import division
import numpy as np
from AssWord import AssWord
import random
import copy
import lm
from GetMIInter import GetMIInter
from GL2 import GLJudge2 as GLJudge
import time
import cPickle

'''
Genetic algorithm for chinese poetry generation
Here we just consider 7-char poems
'''

class GeneticPoetry(object):
    def __init__(self):
        # load AssVec
        self.Ass = AssWord()
        self.ze = {}
        self.ping = {}
        self.__getpingze(self.ze, "data/newze.txt")
        self.__getpingze(self.ping, "data/newping.txt")
        self.__getyun()
        self.__loadwords()

        print("loading language model...")
        self.mylm = lm.CRnnLM()
        self.mylm.setLambda(0.75);
        self.mylm.setRegularization(0.0000001);
        self.mylm.setRnnLMFile("data/modelforward");
        self.mylm.setRandSeed(315);
        self.mylm.setDebugMode(2);
        self.mylm.restoreNet()

        self.getMiInter = GetMIInter()

        self.__GL = GLJudge()
        # gl data
        self.GLTYPE = [ [0,1,3,2], [1,2,0,1], [2,1,3,2], [3,2,0,1] ]
        self.SENGL = [[-1,1,-1,0,0,1,1], [-1,0,-1,1,1,0,0], [-1,1,0,0,1,1,0], [-1,0,-1,1,0,0,1]]

        # gl to pos and type
        self.gl2pos = [[(1,0),(0,1),(3,1),(1,2),(0,3),(2,3)], [(0,0),(1,1),(0,2),(1,3),(3,3)], [(1,0),(0,1),(2,1),(3,2),(0,3)], [(0,0),(3,0),(1,1),(0,2),(2,2),(1,3)]]

    def __getpingze(self, dic, filepath):
        fin = open(filepath, 'r')
        line = fin.readline()
        while line:
            line = line.strip()
            dic[line] = 1
            line = fin.readline()
        fin.close()

    def __getyun(self):
        print("loading yun dict...")
        f = open("data/YunList.txt", 'r')
        line = f.readline()
        self.yundic = {}
        self.iyundic = {}
        while line:
            para = line.split(" ")

            val = para[0]
            key = int(para[1])
            
            if not self.iyundic.has_key(val):
                self.iyundic[val] = key

            if self.yundic.has_key(key):
                self.yundic[key].append(val)
            else:
                temp = []
                temp.append(val)
                self.yundic[key] = temp
            line = f.readline()

        f.close()


    def __loadwords(self):
        print ("loading words...")
        self.wlist = [[], [], [], []]
        self.clist = [[], []]

        self.wyunlist = {}
        self.cyunlist = {}

        fin = open("data/allwordsfreq.txt", 'r')
        line = fin.readline()
        while line:
            line = line.strip()
            para = line.split(" ")
            word = para[0]
            gl = self.getwordgl(word)
            if gl == -1:
                line = fin.readline()
                continue

            if len(word) == 3:
                rank = len(self.clist[gl])
                self.clist[gl].append( (word, rank) )
                # add to cyunlist
                if word in self.iyundic:
                    yun = self.iyundic[word]
                    if yun in self.cyunlist:
                        rank = len(self.cyunlist[yun])
                        self.cyunlist[yun].append((word, rank))
                    else:
                        tmp = []
                        tmp.append((word, 0))
                        self.cyunlist[yun] = tmp
            else:
                rank = len(self.wlist[gl])
                self.wlist[gl].append( (word,  rank) )
                # add to wyunlist
                tail = word[len(word)-3:]
                if tail in self.iyundic:
                    yun = self.iyundic[tail]
                    if yun in self.wyunlist:
                        rank = len(self.wyunlist[yun])
                        self.wyunlist[yun].append((word, rank))
                    else:
                        tmp = []
                        tmp.append((word, 0))
                        self.wyunlist[yun] = tmp

            line = fin.readline()

        fin.close()
        '''
        for i in xrange(0, 4):
            random.shuffle(self.wlist[i])
        random.shuffle(self.clist[0])
        random.shuffle(self.clist[1])
        '''

        
        self.wlist_size = []
        self.clist_size = []
        
        for i in xrange(0, 4):
            N = len(self.wlist[i])
            self.wlist_size.append(N*(N+1)/2.0)

        N = len(self.clist[0])
        self.clist_size.append(N*(N+1)/2.0)
        N = len(self.clist[1])
        self.clist_size.append(N*(N+1)/2.0)




    '''
    get thr word or the character gl type
    for 2-char word, return 0: 00, 1:01, 2:10, 3:11
    for char, return 0:ping, 1:ze
    return -1, word not in list
    '''
    def getwordgl(self, word):
        ans = 0
        if len(word) == 3:
            if word in self.ze:
                ans = 1
            elif word in self.ping:
                ans = 0
            else:
                ans = -1
        elif len(word) == 6:
            c1 = word[0:3]
            c2 = word[3:]
            if c1 in self.ping and c2 in self.ping:
                ans = 0
            elif c1 in self.ping and c2 in self.ze:
                ans = 1
            elif c1 in self.ze and c2 in self.ping:
                ans = 2
            elif c1 in self.ze and c2 in self.ze:
                ans = 3
            else:
                ans = -1
        else:
            ans = -1

        return ans


    def gls2glidx(self, gls):
        if len(gls) == 1:
            return gls[0]
        gl1 = gls[1]
        gl0 = gls[0]
        if (gl0) == -1:
            if np.random.rand(1)[0] > 0.5:
                gl0 = 0
            else:
                gl0 = 1

        if gl0 == 0 and gl1 == 0:
            return 0
        elif gl0 == 0 and gl1 == 1:
            return 1
        elif gl0 == 1 and gl1 == 0:
            return 2
        elif gl0 == 1 and gl1 == 1:
            return 3
        return 0

    def get_rand_word_byg(self, g, typ, dic, yun=-1):
        if typ == 0:
            if yun >= 0:
                candis=self.cyunlist[yun]
            else:
                candis = self.clist[g]
        else:
            if yun >= 0:
                candis = self.wyunlist[yun]
            else:
                candis = self.wlist[g]


        size = len(candis)
        N = 0.5*size-0.5
        r = random.random()
        s = 0.0
        for i in xrange(0, size):
            f = 1.0 - (candis[i][1]+1) / float(size)
            p = f / N
            s += p

            if s >= r:
                word = candis[i][0]
                break
        #print ("s %f r %f p %f" % (s, r, p))

        while word in dic:
            r = random.random()
            s = 0.0
            for i in xrange(0, size):
                f = 1.0 - (candis[i][1]+1) / float(size)
                p = f / N
                s += p
                if s >= r:
                    word = candis[i][0]
                    break
        
        dic[word] = 1
        return word


    def get_rand_word(self, gls, typ, pos1, pos2, dic, yun=-1):
        g = self.gls2glidx(gls[pos1:pos2])
        return self.get_rand_word_byg(g, typ, dic, yun)

    def select_add_2sen(self, word, pos, gls, sen, wordDic, yun=-1):
        if pos == 0:
            sen.append(word)
            # get a 2-char word
            w = self.get_rand_word(gls, 1, 2, 4, wordDic, yun)
            sen.append(w)
            wordDic[w] = 1
            # xy|z
            if np.random.rand(1)[0] > 0.5:
                w = self.get_rand_word(gls, 1, 4, 6, wordDic, yun)
                sen.append(w)
                w = self.get_rand_word(gls, 0, 6, 7, wordDic, yun)
                sen.append(w)
            else:
                #x|yz
                w = self.get_rand_word(gls, 0, 4, 5, wordDic, yun)
                sen.append(w)
                w = self.get_rand_word(gls, 1, 5, 7, wordDic, yun)
                sen.append(w)
        elif pos == 1:
            # get a 2-char word
            w = self.get_rand_word(gls, 1, 0, 2, wordDic, yun)
            sen.append(w)
            sen.append(word)
            # xy|z
            if np.random.rand(1)[0] > 0.5:
                w = self.get_rand_word(gls, 1, 4, 6, wordDic, yun)
                sen.append(w)
                w = self.get_rand_word(gls, 0, 6, 7, wordDic, yun)
                sen.append(w)
            else:
                #x|yz
                w = self.get_rand_word(gls, 0, 4, 5, wordDic, yun)
                sen.append(w)
                w = self.get_rand_word(gls, 1, 5, 7, wordDic, yun)
                sen.append(w)

        elif pos == 2:
                w = self.get_rand_word(gls, 1, 0, 2, wordDic, yun)
                sen.append(w)
                w = self.get_rand_word(gls, 1, 2, 4, wordDic, yun)
                sen.append(w)
                sen.append(word)
                w = self.get_rand_word(gls, 0, 6, 7, wordDic, yun)
                sen.append(w)
        elif pos == 3:
                w = self.get_rand_word(gls, 1, 0, 2, wordDic, yun)
                sen.append(w)
                w = self.get_rand_word(gls, 1, 2, 4, wordDic, yun)
                sen.append(w)
                w = self.get_rand_word(gls, 0, 4, 5, wordDic, yun)
                sen.append(w)
                sen.append(word)
        
    def fill_sen(self, word, pos_select_num = 10):
        assert len(word) == 6
        gl = self.getwordgl(word)
        if gl == -1:
            return []

        sen_list = []
        idxes = range(0, 4)
        random.shuffle(idxes)
        postyp = self.gl2pos[gl]

        for idx in idxes:

            wordDic = {}
            wordDic[word] = 1
            for j in xrange(pos_select_num):
                sen = []
                #print idx
                pos, typ = postyp[idx]
                #print (pos, typ)
                gls = self.SENGL[typ]
                #print (gls)
                self.select_add_2sen(word, pos, gls, sen, wordDic)
                sen_list.append(sen)



        return sen_list

    def fill_sen2(self, gl, yun, word, pos_select_num = 10):
        gls = self.SENGL[gl]
        assert len(word) == 6
        wgl = self.getwordgl(word)
        if wgl == -1:
            return []
        #print ("_________")
        #print (wgl)
        postyp = self.gl2pos[wgl]
        #print (postyp)
        pos_list = []
        for pos, typ in postyp:
            if typ == gl:
                pos_list.append(pos)
        #print (pos_list)
        sen_list = []
        for pos in pos_list:
            wordDic = {}
            wordDic[word] = 1
            for j in xrange(pos_select_num):
                sen = []
                self.select_add_2sen(word, pos, gls, sen, wordDic, yun)
                sen_list.append(sen)
                '''
                print (word)
                print (pos)
                print (yun)
                print (gl)
                print (gls)
                print (" ".join(sen))
                if yun != -1:
                    tt = input(">")
                '''

        return sen_list

    # initialize the population of first line
    def init_population(self, asswords, pos_select_num, gl=-1, yun=-1, step=0):

        sens = []
        for w in asswords:
            if step == 0:
                sen = self.fill_sen(w, pos_select_num)
            else:
                sen = self.fill_sen2(gl, yun, w, pos_select_num)
            if len(sen) == 0:
                continue
            sens.extend(sen)

        '''
        print (len(sens))
        for sen in sens:
            print (" ".join(sen))
        '''

        return sens

    def fitness_value(self, sen, history):
        line = "".join(sen)
        line = line.strip()
        sline = ""
        for i in range(0, len(line), 3):
            w = line[i:i+3]
            sline += w + " "
        sline = sline.strip()
        #lm_cost = np.exp(self.mylm.testNet_prob(line) /  len(sen))
        lm_cost = np.power(10, self.mylm.testNet_prob(sline) /  len(sen) / 2)
        lm_cost *= 100
        mi_score = 0.0
        if len(history) != 0:
            for line in history:
                line = line.strip()
                mi = max(self.getMiInter.getMIScore(line, sline), 0)
                mi_score += mi
            mi_score /= 2

        #print ()
        if len(history) == 0:
            final_score = lm_cost
        else:
            final_score = np.sqrt(lm_cost * (mi_score/10))

        return final_score, lm_cost, mi_score/10

    def roulette(self, fitness):
        fitness_sum = np.sum(fitness)
        r = random.random()
        s = 0.0
        for i in xrange(0, len(fitness)):
            p = fitness[i] / fitness_sum
            s += p
            if s >= r:
                return i


    def add2newgroup(self, new_group, new_group_dic, sen):
        line = "".join(sen)
        if line in new_group_dic:
            return

        new_group.append(sen)
        new_group_dic[line] = 1

    def isSameYun(self, seq1, seq2):
        yun1 = self.getYun(seq1)
        yun2 = self.getYun(seq2)
        if yun1 != yun2:
            return False
        return True


    def do_mating(self, group, Pc):
        
        size = len(group)
        new_group = []
        new_group_dic = {}
        step = 0

        while len(new_group) < int(1.5* size):
            random.shuffle(group)
            #print ("size of new group %d" % (len(new_group)))
            step += 1
            if step > 5:
                break

            for i in xrange(0, size-1, 2):
                sen1 = group[i]
                sen2 = group[i+1]

                if " ".join(sen1) == " ".join(sen2) or random.random() > Pc:
                    self.add2newgroup(new_group, new_group_dic, sen1)
                    self.add2newgroup(new_group, new_group_dic, sen2)
                # mating
                #print (" ".join(sen1))
                #print (" ".join(sen2))
                glp1 = [(self.getwordgl(w), len(w)) for w in sen1]
                glp2 = [(self.getwordgl(w), len(w)) for w in sen2]

                pairs = []
                for i, tp1 in enumerate(glp1):
                    for j, tp2 in enumerate(glp2):
                        if tp1 == tp2 and  sen1[i] != sen2[j]:
                            pairs.append((i, j))

                if len(pairs) == 0:
                    self.add2newgroup(new_group, new_group_dic, sen1)
                    self.add2newgroup(new_group, new_group_dic, sen2)
                    continue

                #print (pairs)
                child1 = copy.deepcopy(sen1)
                child2 = copy.deepcopy(sen2)
                #print (pairs)
                pairs = random.sample(pairs, random.randint(1, len(pairs)))
                random.shuffle(pairs)
                for idx1, idx2 in pairs:
                    if sen1[idx1] not in child2: 
                        if (idx1 == 3 or idx2 == 3) and (not self.isSameYun(child2[idx2], sen1[idx1])):
                            continue
                        child2[idx2] = sen1[idx1]
                    if sen2[idx2] not in child1:
                        if (idx1 == 3 or idx2 == 3) and (not self.isSameYun(child1[idx1], sen2[idx2])):
                            continue
                        child1[idx1] = sen2[idx2]

                #print (" ".join(child1))
                #print (" ".join(child2))
                new_group.append(child1)
                new_group.append(child2)
                self.add2newgroup(new_group, new_group_dic, child1)
                self.add2newgroup(new_group, new_group_dic, child2)

        return new_group

    def do_genovariation(self, group, Pm, yun=-1):
        
        for i, sen in enumerate(group):
            if Pm < random.random():
                continue

            # genovariation
            #print (" ".join(sen))
            pos = random.randint(0, len(sen)-1)
            gl = self.getwordgl(sen[pos])
            dic = {}
            dic[sen[pos]] = 1
            w = self.get_rand_word_byg(gl, 1-int(len(sen[pos])==3), dic, yun)
            sen[pos] = w
            group[i] = sen
            #print (" ".join(sen))

        return group

    def getSenGL(self, sen):
        sen = "".join(sen)
        sen2 = sen.decode("utf-8")
        gl = self.__GL.gelvJudge(sen2)
        return gl

    def getYun(self, sentence):
        if type(sentence) is list:
            sentence = "".join(sentence)
        length = len(sentence)
        tail = sentence[length-3:length]
        if self.iyundic.has_key(tail) :
            return self.iyundic[tail]
        else:
            return -1

    def evolution(self, group, iternum, shownum = 10, groupsize=400, Pc=0.6, Pm=0.01, required_gl=-1, yun=-1, history=[], automatic=False):
        fitness_save = []
        lm_save = []
        mi_save = []

        for ite in xrange(0, iternum):
            #print (len(group))
            fitness = []
            tmp1 = []
            tmp2 = []
            for sen in group:
                #print (" ".join(sen))
                val, lms, mis = self.fitness_value(sen, history)
                fitness.append(val)
                tmp1.append(lms)
                tmp2.append(mis)

            if not automatic:
                print ("fitness: %.4f" % (np.sum(fitness)/groupsize))
            #print (len(group))
            new_group = []

            # roulette  selection
            group_dic = {}
            while len(new_group) < groupsize:
                idx = self.roulette(fitness)
                line = "".join(group[idx])
                if line in group_dic:
                    continue
                new_group.append(group[idx])
                group_dic[line] = 1

            #print (len(new_group))
            # mating
            new_group = self.do_mating(new_group, Pc)
            #print (len(new_group))
            new_group = self.do_genovariation(new_group, Pm, yun)
            #print (len(new_group))
            group = new_group
            #assert len(group) == groupsize

            lm_save.append(np.mean(tmp1))
            mi_save.append(np.mean(tmp2))
            fitness_save.append(np.mean(fitness))


        fitness = []
        for sen in group:
            val, _, _ = self.fitness_value(sen, history)
            fitness.append(val)
        print ("final fitness: %.4f" % (np.sum(fitness)/groupsize))

        final_group = []
        final_fitness = []
        idxes = np.argsort(-np.array(fitness))
        for i in range(0, len(idxes)):
            idx = idxes[i]
            gl = self.getSenGL(group[idx])
            if required_gl != -1 and gl != required_gl:
                continue

            final_group.append(group[idx])
            final_fitness.append(fitness[idx])
        
        which = 0
        if not automatic:
            for i in range(0, min(len(final_fitness), shownum)):
                print ("%d, %f, %s" % (i, final_fitness[i], "".join(final_group[i])))
            which = input("please select a sentence>")

        output = open("fitness.pkl", 'wb')
        cPickle.dump(fitness_save, output, -1)
        output.close()

        output = open("lm.pkl", 'wb')
        cPickle.dump(lm_save, output, -1)
        output.close()

        output = open("mi.pkl", 'wb')
        cPickle.dump(mi_save, output, -1)
        output.close()


        return final_group[which]

    def get_next_ass_words(self, sen, num, assnum):
        line = "".join(sen)
        line = line.strip()
        num = int(num/4)
        asswords = []
        mivalues = []
        for i in xrange(0, len(self.wlist)):
            asswords.extend(self.wlist[i][0:num])

        for pair in asswords:
            word = pair[0]
            mi = np.exp(self.getMiInter.getMIScore(line, word))
            mivalues.append(mi)

        idxes = np.argsort(-np.array(mivalues))
        final_asswords = []
        for i in xrange(0, min(assnum, len(idxes))):
            idx = idxes[i]
            if line.find(asswords[idx][0]) != -1:
                continue
            #print ("%d %s %f" % (i, asswords[idx][0], mivalues[idx]))
            final_asswords.append(asswords[idx][0])
        return final_asswords


    def generate(self, word):
        history = []
        print ("begin...")
        assnum = 30
        spand_num = 8
        groupsize = spand_num*4*assnum
        groupsize = int(groupsize*0.8)

        # generate first line
        asswords = self.Ass.getKNN(word, assnum)
        group = self.init_population(asswords, spand_num)
        sen1 = self.evolution(group, shownum = 10, iternum=40, groupsize=len(group), Pc=0.8, Pm=0.05)


        print (" ".join(sen1))
        history.append("".join(sen1))


        # generate second line
        asswords2 = self.get_next_ass_words(sen1, num=200, assnum=assnum)
        gl1 = self.getSenGL(sen1)
        assert gl1 >=0 
        glType = self.GLTYPE[gl1]
        print (glType)
        yun = self.getYun(sen1)
        print (yun)
        group = self.init_population(asswords2, spand_num, gl=glType[1], yun=yun, step=1)
        sen2 = self.evolution(group, shownum = 10, iternum=30, groupsize=len(group), Pc=0.8, Pm=0.05,  required_gl=glType[1], yun=yun, history=history)
        history.append("".join(sen2))

 
        if yun < 0:
            yun = self.getYun(sen2)

        # generate third line
        asswords3 = self.get_next_ass_words(sen2, num=200, assnum=assnum)
        print (yun)
        group = self.init_population(asswords3, spand_num, gl=glType[2], yun=-1, step=2)
        sen3 = self.evolution(group, shownum = 10, iternum=30, groupsize=len(group), Pc=0.8, Pm=0.05,  required_gl=glType[2], yun=-1, history=history)
        history.append("".join(sen3))

        # generate line4
        asswords4 = self.get_next_ass_words(sen3, num=200, assnum=assnum)
        print (yun)
        group = self.init_population(asswords4, spand_num, gl=glType[3], yun=yun, step=3)
        sen4 = self.evolution(group, shownum = 10, iternum=30, groupsize=len(group), Pc=0.8, Pm=0.05,  required_gl=glType[3], yun=yun, history=history)
        history.append("".join(sen4))

        for line in history:
            print (line)

    def do_test1(self, sen):
        history = []
        assnum = 30
        spand_num = 8
        groupsize = spand_num*4*assnum
        groupsize = int(groupsize*0.8)

        sen1 = sen.split(" ")
        print (" ".join(sen1))
        history.append("".join(sen1))

        # generate second line
        asswords2 = self.get_next_ass_words(sen1, num=200, assnum=assnum)
        gl1 = self.getSenGL(sen1)
        if gl1 < 0:
            gl1 = 0
        glType = self.GLTYPE[gl1]
        yun = self.getYun(sen1)
        group = self.init_population(asswords2, spand_num, gl=glType[1], yun=yun, step=1)
        '''
        sen2 = self.evolution(group, shownum = 10, iternum=30, groupsize=len(group), Pc=0.8, Pm=0.05,  required_gl=glType[1], yun=yun, history=history, automatic=True)
        history.append("".join(sen2))
        return history[-1]
        '''
        return "".join(group[random.randint(0, len(group)-1)])


    def lineSplit(self, line):
        sen = ""
        for i in xrange(0, len(line), 3):
            sen += line[i:i+3] + " "
        sen = sen.strip()
        return sen

    def do_test_file(self, infile, outfile, step):
        fin = open(infile, 'r')
        lines = fin.readlines()
        fin.close()

        fout = open(outfile, 'w')

        for i, line in enumerate(lines):
            line = line.strip()
            t1 = time.time()
            sen = self.do_test1(line)
            t2 = time.time()
            print ("%d second per sentence" % (t2-t1))
            sen = self.lineSplit(sen)
            print ("%d %s" % (i, sen))
            fout.write(sen + "\n")
            fout.flush()

        fout.close()


def main():
    genetic = GeneticPoetry()
    #genetic.do_test_file('srcfile_split.txt', 'ans_random_3.txt', 1)
    genetic.generate("杨柳")

if __name__ == "__main__":
    main()
