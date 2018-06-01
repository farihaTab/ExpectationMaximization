import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import random
import math
from sklearn import datasets
from scipy.stats import multivariate_normal
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def plotData(actualMu,actualSigma,predictedMu,predictedSigma,dataMu=None,dataSigma=None):
    N = 100
    X = np.linspace(-4, 6, N)
    Y = np.linspace(-4, 6, N)
    X, Y = np.meshgrid(X, Y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    for i in range(len(actualMu)):
        print(i)
        f1 = multivariate_gaussian(pos, actualMu[i], actualSigma[i])
        f2 = multivariate_gaussian(pos, predictedMu[i],predictedSigma[i])
        plt.contour(X, Y, f1, colors='red')
        plt.contour(X, Y, f2, colors='blue')
        if dataMu is not None:
            f3 = multivariate_gaussian(pos, dataMu[i], dataSigma[i])
            plt.contour(X, Y, f3, colors='green')
    plt.show()

def checkSingularity(cov):
    covDet = np.linalg.det(cov)
    while covDet < 0.0000000001:
        print('singular ')
        for i in range(len(cov)):
            cov[i,i] += np.random.uniform(1,10)*0.0000001
        covDet = np.linalg.det(cov)

def calcGMM(x,mean,cov):
    # print(cov)
    D = len(mean)
    covDet = np.linalg.det(cov)
    while covDet == 0:
        print('singular ')
        for i in range(len(cov)):
            cov[i,i] += random.random()*0.000000001
        # print(cov)
        covDet = np.linalg.det(cov)
    # print('covDet', covDet)
    covInv = np.linalg.inv(cov)
    # print('covInv', covInv)
    x_mean = x-mean
    temp = x_mean.T
    temp = np.dot(temp,covInv)
    temp = np.sum(temp*x_mean)
    # print(temp)
    return 1/math.sqrt( ((2*math.pi)**D)*covDet ) * math.exp(-0.5*temp)

def getCovmatrix(x):
    y = pandas.DataFrame({'%d'%i:x[:,i] for i in range(len(x[0]))})#,'G':x[:,1],'B':px2[:,2]})
    y = np.array(y.cov())
    # print(y)
    return y

def generateData(mean,covariance,w,N):
    datamu = np.zeros(shape=mean.shape)
    datacov = np.zeros(shape=covariance.shape)
    k = len(mean)
    X = None
    for i in range(k):
        if i==k-1:
            n = N-len(X)
        else:
            n = int(N*w[i])
        # print('i', i, 'n', n)
        mu = mean[i]
        cov = covariance[i]
        x = np.random.multivariate_normal(mu, cov, n)
        # print('mean', x.mean(0)) #column wise mean
        # print('std', x.std(0)) #column wise std
        datamu[i] = x.mean(0)
        datacov[i] = getCovmatrix(x)

        if X is None:
            X = x
        else:
            X = np.vstack([X, x])
        # print(x)
    np.random.shuffle(X)
    return X,datamu,datacov

class ExpectationMaximization:
    def __init__(self,X,Y,k):
        self.X = X
        self.Y = Y
        self.N = len(X)
        self.K = k
        self.D = len(X[0])
        self.mu = None
        self.cov = None
        self.w = None
        self.P = None
        self.initialization()
        # self.train()
        self.train_vector()

    def initialization(self):
        print('start init')
        random.seed(100)
        np.random.seed(100)
        self.mu = np.random.uniform(0,1, size=(self.K,self.D))
        self.cov = np.zeros(shape=[self.K,self.D,self.D])
        for i in range(self.K):
            self.cov[i] = sklearn.datasets.make_spd_matrix(self.D)*50
        w = np.random.uniform(0,1,size=(self.K))
        self.w = w/np.sum(w)
        # self.test_init()
        print('mu', self.mu)
        print('cov', self.cov)
        print('w', self.w)
        print('initialized\n')
        # input('enter to begin')

    def test_init(self):
        self.mu = np.array([[0.5,0],[0,0.5]],dtype=float)
        self.w = np.array([0.5,0.5],dtype=float)
        self.cov = np.array([[[1,0],[0,1]],[[10,0],[0,10]]],dtype=float)

    def logLikelihood(self):
        print('calculating log-likelihood')
        sum_out = 0
        for j in range(self.N):
            # print('j'+repr(j))
            sum_in = 0.000000001
            for i in range(self.K):
                # print('i'+repr(i))
                sum_in += self.w[i]*calcGMM(self.X[j],self.mu[i],self.cov[i])
                print('i', i, sum_in)
            sum_out += math.log(sum_in,2)
            print('j', j, 'sum_out', sum_out)
        print('calculated log-likehood ', sum_out)
        return sum_out

    def logLikelihood_vector(self):
        print('calculating log-likelihood')
        N = np.zeros(shape=(self.N,self.K))
        # print('N', N)
        for i in range(self.K):
            checkSingularity(self.cov[i])
            print('covDet ',np.linalg.det(self.cov[i]))
            N[:,i] = multivariate_normal(self.mu[i],self.cov[i],False).pdf(self.X)
        # print('N', N, 'w', self.w)
        t = self.w.reshape(1,self.K)
        N = N*t
        # print('N', N)
        # input()
        N = np.sum(N,1) #row-wise sum
        # print('N', N)
        N = N+0.00000001 #noise to avoid log0
        N = np.log(N)/np.log(2) #2 base log
        # print('N', N)
        N = np.sum(N)
        print('calculated log-likehood ', N)
        return N

    def expectation(self):
        self.P = np.zeros([self.K,self.N])
        for j in range(self.N):
            print('j'+repr(j))
            sum_in = 0.0
            for i in range(self.K):
                print('i'+repr(i))
                self.P[i,j] = self.w[i]*calcGMM(self.X[j],self.mu[i],self.cov[i])
                sum_in += self.P[i,j]
            print(self.P, sum_in)
            self.P[:,j] = self.P[:,j]/sum_in
            print('updated p ', self.P)
        print('expected P');print(self.P)

    def expectation_vector(self):
        print('calculating expectation')
        N = np.zeros(shape=(self.N,self.K))
        # print('N', N)
        for i in range(self.K):
            N[:,i] = multivariate_normal(self.mu[i],self.cov[i]).pdf(self.X)
        print('N', N, 'w', self.w)
        N = self.w.reshape(1,self.K)*N
        print('N', N)
        m = np.sum(N,1).reshape(self.N,1) #row-wise sum
        print('m', m)
        N = N/m
        print('calculated expectation\n', N)
        self.P = N
        return N

    def maximization(self):
        for i in range(self.K):
            print('i '+repr(i))
            # input('enter')
            sum_in = np.squeeze(np.zeros(shape=[1,self.D],dtype=float))
            sum_p_ij = 0.0
            sum_in_2 = np.zeros(shape=[self.D,self.D],dtype=float)
            print('sum')
            print(sum_in, sum_in_2, sum_p_ij)
            for j in range(self.N):
                print('j '+repr(j))
                # input('enter')
                print(self.P[i,j], self.X[j], self.mu[i], (self.X[j]-self.mu[i]))
                sum_in += np.multiply(self.P[i,j],self.X[j])
                temp = (self.X[j]-self.mu[i])
                temp  = np.matmul(temp.reshape(self.D,1),temp.reshape(1,self.D))
                print('temp ');print(temp)
                temp = self.P[i,j]*temp
                print('temp ');
                print(temp)
                sum_in_2 += temp
                sum_p_ij += self.P[i,j]
                print('sum j', j)
                print(sum_in, sum_in_2, sum_p_ij)
            if sum_p_ij < 0.00000001: sum_p_ij = 0.00000001
            print('old i', i)
            print(self.mu[i])
            print(self.cov[i])
            print(self.w[i])
            self.mu[i] = sum_in/sum_p_ij
            self.cov[i] = sum_in_2/sum_p_ij
            # noise = np.identity(self.D)
            # for x in range(self.D):
            #     noise[x,x] = random.random()*0.00000001
            # self.cov[i] += noise
            self.w[i] = sum_p_ij/self.N
            print('mu[', i,']');print(self.mu[i])
            print('cov', i);print(self.cov[i])
            print('w', i);print(self.w[i])
        print('maximized mu')
        print(self.mu)
        print('maximized cov')
        print(self.cov)
        print('maximized w')
        print(self.w)

    def maximization_vector(self):
        temp = np.matmul(np.transpose(self.P),self.X)
        s = np.sum(self.P,0) #column wise sum
        s = s.reshape(self.K,1)
        self.mu = temp/s
        print('mu\n', self.mu)
        w = s/self.N
        print('w\n', w)
        # temp = self.X - self.mu
        T = np.zeros(shape=(self.K,self.D,self.D))
        for i in range(self.K):
            temp = self.X -self.mu[i]
            temp_t = np.transpose(temp)
            print('temp\n',temp)
            print('P\n',self.P[:,i])
            temp = self.P[:,i].reshape(self.N,1)*temp
            print('temp\n',temp)
            temp = np.matmul(temp_t,temp)
            print('temp\n',temp)
            temp = temp/s[i]
            print('tempf\n',temp)
            T[i] = temp
        print('cov\n',T)
        # self.mu = mu
        self.w = w
        self.cov = T


    def train(self):
        change = 100
        sensitivity = 0.001
        curr = 0
        prev = self.logLikelihood()
        itr = 0
        self.inspect()
        print('current log-likelihood ', prev, 'change ', change)
        while change > sensitivity:
            print('\nloop', itr)
            itr += 1
            blockPrint()
            print('expectation')
            self.expectation()
            # input('expectation calculated\n')
            print('maximization')
            self.maximization()
            # input('maximized\n')
            print('calc log-likelihood')
            curr = self.logLikelihood()
            change = abs(curr-prev)
            prev = curr
            enablePrint()
            print('current log-likelihood ', curr, 'change ', change)
            # input('next\n')
        print('\nresult')
        print('mean ',self.mu)
        print('cov ',self.cov)
        print('weight ',self.w)

    def train_vector(self):
        curr = self.logLikelihood_vector()
        prev = 0
        change = abs(curr-prev)
        curr = prev
        sensitivity = 0.0000001
        itr = 0
        self.inspect()
        print('current log-likelihood ', prev, 'change ', change)
        while change > sensitivity:
            print('\nloop', itr)
            itr += 1
            blockPrint()
            print('expectation')
            self.expectation_vector()
            # input('expectation calculated\n')
            print('maximization')
            self.maximization_vector()
            # input('maximized\n')
            # enablePrint()
            print('calc log-likelihood')
            curr = self.logLikelihood_vector()
            change = abs(curr-prev)
            prev = curr
            enablePrint()
            print('current log-likelihood ', curr, 'change ', change)
            # input('next\n')
        print('\nresult')
        print('mean ',self.mu)
        print('cov ',self.cov)
        print('weight ',self.w)

    def inspect(self):
        print('\n\nw ')
        print(self.w)
        print('mu')
        print(self.mu)
        print('cov')
        print(self.cov)
        print('P')
        print(self.P)
        print('\n')

def experimentTrivial():
    X = np.array([[1, 1], [1, 2], [5, 5], [5, 6]], dtype=float)
    ExpectationMaximization(X, None, 2)
    # print(em.logLikelihood())
    # print(em.logLikelihood_vector())
    # em.expectation_vector()
    # em.maximization_vector()

def experiment():
    mu = np.array([[-2,-2,-2],[0,3,3],[2,2,2]])
    cov = np.array([[[1.0,0.0,0.0],[0.0,.7,0.0],[0.0,0.0,0.2]],[[.3,0.0,0.0],[0.0,.5,0.0],[0.0,0.0,.1]],[[0.5,0.0,0.0],[0.0,0.5,0.0],[0.0,0.0,0.5]]])
    weight = np.array([0.5,0.3,0.2])
    np.random.seed(100)
    X,datamu,datacov = generateData(mu,cov,weight,1000)
    em = ExpectationMaximization(X,None,3)
    # plotData(mu,cov,em.mu,em.cov)

def experiment1():
    mu = np.array([[-2,-2],[0,3],[2,2]])
    cov = np.array([[[1.0,0.0],[0.0,.7]],[[.3,0.0],[0.0,.5]],[[0.5,0.0],[0.0,0.5]]])
    weight = [0.5,0.3,0.2]
    X, datamu, datacov = generateData(mu, cov, weight, 1000)
    em = ExpectationMaximization(X, None, 3)
    plotData(mu, cov, em.mu, em.cov)

def experiment2():
    mu = np.array([[0,0],[0,4],[3,3]])
    cov = np.array([[[.5,0.0],[0.0,.2]],[[.3,0.0],[0.0,0.01]],[[0.5,0.0],[0.0,0.5]]])
    weight = np.array([0.5,0.3,0.2])
    np.random.seed(100)
    X,datamu,datacov = generateData(mu,cov,weight,1000)
    em = ExpectationMaximization(X,None,3)
    plotData(mu,cov,em.mu,em.cov)
experiment1()