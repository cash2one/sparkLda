#coding=utf-8
#__author__=jianfeng.chen@baifendian.com

import numpy as np
import sys
import jieba
from scipy.special import gammaln, psi
reload(sys)
sys.setdefaultencoding('utf-8')


def dirichlet_expectation(alpha):
    """
    For a vector `theta~Dir(alpha)`, compute `E[log(theta)]`.
    """
    if (len(alpha.shape) == 1):
        result = psi(alpha) - psi(np.sum(alpha))
    else:
        result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
    return result.astype(alpha.dtype)  # keep the same precision as input

class lda():
    def __init__(self, path):
        self.modelPath = path
        self.index = {}          #key:word value:index(int) 
        self.indexList = []      #list[list[]],self.index sorted by value
        self.vocabSize = 0       
        self.topicK = 0
        
        self.iterations = 50
        self.alpha = None
        self.gamma_threshold = 0.001
        self.stats = None
        self.eta = None
        self.expElogbeta = None
        self.minimum_probability = 0.01
        #initialize word*topic matrix
        self.topicMatrix = None
        self.wordMatrix = None
        self.loadModel()

    def loadModel(self):
        f = open(self.modelPath)
        list = f.readlines()
        self.vocabSize = int(list[0].strip())
        self.topicK = int(list[1].strip())
        #self.alpha = np.asarray( [float(list[2].strip())] * self.topicK )
        #self.eta = float(list[3].strip())
        self.topicMatrix = np.zeros((self.topicK, self.vocabSize))
        self.wordMatrix = np.zeros((self.vocabSize, self.topicK))
        j = -1
        location = 0
        for i in range(len(list)):
            if(i < 2):
                continue
            if("#TOPIC#" in list[i]):
                j = j + 1
                continue
            l = list[i].split("\t")
            if(self.index.has_key(l[0].strip())):
                self.topicMatrix[j][self.index[l[0].strip()]] = np.float64(l[1].strip())
            else:
                self.index[l[0].strip()] = location
                location = location + 1
                self.topicMatrix[j][self.index[l[0].strip()]] = np.float64(l[1].strip())
        self.wordMatrix = np.transpose(self.topicMatrix)
        self.indexList = sorted(self.index.iteritems(), key = lambda d:d[1])
        
        self.eta = 1.0 / 200
        self.alpha = np.asarray([1.0 / self.topicK for i in xrange(self.topicK)])
        self.minimum_probability = 2.0 / self.topicK
        self.stats = self.topicMatrix
        self.expElogbeta = np.exp(dirichlet_expectation(self.eta + self.stats))

    def preprocess(self, doc):
        s = doc.replace(" ", "")
        s = doc.replace("\r\n", " ")
        s = doc.replace("\n", " ")
        s = doc.replace("\r", " ")
        #s = " ".join(jieba.cut(s))
        s = " ".join(jieba.cut(s))
        return s

    def doc2bow(self, document):
    #convert document(list of words) into bag-of-words format list of (id, word)
        bowDict = {}
        bow = []
        for i in document.split(' '):
            if(self.index.has_key(i.strip().encode('utf-8'))):
                if(bowDict.has_key(i.strip().encode('utf-8'))):
                    bowDict[i] = bowDict[i] + 1
                else:
                    bowDict[i] = 1
        for i in bowDict.keys():
            bow.append([self.index[i.encode('utf-8')], bowDict[i]])
        return bow
            
    def get_document_topics(self, bow, minimum_probability=None):
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        gamma, _ = self.inference([bow])
        topic_dist = gamma[0] / sum(gamma[0])  # normalize distribution
        return [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)
                if topicvalue >= minimum_probability]


    def inference(self, chunk):
        try:
            _ = len(chunk)
        except:
            # convert iterators/generators to plain list, so we have len() etc.
            chunk = list(chunk)
        if len(chunk) > 1:
            logger.debug("performing inference on a chunk of %i documents" % len(chunk))

        # Initialize the variational distribution q(theta|gamma) for the chunk
        gamma = np.random.gamma(100., 1. / 100., (len(chunk), self.topicK))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        converged = 0

        # Now, for each document d update that document's gamma and phi
        # Inference code copied from Hoffman's `onlineldavb.py` (esp. the
        # Lee&Seung trick which speeds things up by an order of magnitude, compared
        # to Blei's original LDA-C code, cool!).
        for d, doc in enumerate(chunk):
            ids = [id for id, _ in doc]
            cts = np.array([cnt for _, cnt in doc])
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self.expElogbeta[:, ids]

            # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w.
            # phinorm is the normalizer.
            # TODO treat zeros explicitly, instead of adding 1e-100?
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

            # Iterate between gamma and phi until convergence
            for _ in xrange(self.iterations):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self.alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = np.mean(abs(gammad - lastgamma))
                if (meanchange < self.gamma_threshold):
                    converged += 1
                    break
            gamma[d, :] = gammad

        if len(chunk) > 1:
            logger.debug("%i/%i documents converged within %i iterations" %
                         (converged, len(chunk), self.iterations))
        return gamma

    def doc2topic(self, document):
        s = self.preprocess(document)
        bow = self.doc2bow(s)
        gamma = self.inference([bow])
        topic_dist = gamma[0] / sum(gamma[0])
        print topic_dist
        return [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist) if topicvalue >= self.minimum_probability]
    '''    
    def doc2word(self, document):
        topic = self.doc2topic(document)
        result = []
        result = [0] * self.vocabSize
        for i in topic:
            result += self.topicMatrix[i[0]] * i[1]
        enu = enumerate(result)
        sortedResult = sorted(enu, key = lambda k:k[1], reverse = True)
        result_10 = []
        for i in range(10):
            result_10.append([self.indexList[sortedResult[i][0]], sortedResult[i][1]])
        #for i in result_10:
        #    print i[0][0],i[0][1],i[1]
        return result_10
    '''
    def doc2word(self, document):
        topic = self.doc2topic(document)
        topicDict = {}
        for i in topic:
            topicDict['TOPIC' + str(i[0])] = i[1]
        result = {}
        result['topic'] = topicDict
        word = []
        word = [0] * self.vocabSize
        for i in topic:
            word += self.topicMatrix[i[0]] * i[1]
        enu = enumerate(word)
        sortedResult = sorted(enu, key = lambda k:k[1], reverse = True)
        tagDict = {}
        for i in range(10):
            tagDict[self.indexList[sortedResult[i][0]][0]] = sortedResult[i][1]
        result['word'] = tagDict
        return result
        

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print "Usage:"
        print "python lda.py modelPath"
        exit(0)
    ldaModel = lda(sys.argv[1])
    print "vocabSize:",ldaModel.vocabSize
    print "topicK:",ldaModel.topicK
    #print len(ldaModel.index.keys())
    print "topicMatrix:",ldaModel.topicMatrix.shape
    print "wordMatrix:",ldaModel.wordMatrix.shape
    #print ldaModel.wordMatrix[0][0]
    #print ldaModel.wordMatrix[1][2]
    #ldaModel.doc2vector("菲律宾特么不要脸")
    #ldaModel.doc2word(u"菲律宾真特么不要脸", 1)
    for i in range(1000):
        print "请输入文本："
        s = sys.stdin.readline()
    #    print ldaModel.doc2word(s)
        print ldaModel.doc2word(s)
