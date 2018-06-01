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
    print(temp)
    return 1/math.sqrt( ((2*math.pi)**D)*covDet ) * math.exp(-0.5*temp)

def generateData(mean,covariance,w,N):
    k = len(mean)
    X = None
    for i in range(k):
        if i==k-1:
            n = N-len(X)
        else:
            n = int(N*w[i])
        print('i', i, 'n', n)
        mu = mean[i]
        cov = covariance[i]
        x = np.random.multivariate_normal(mu, cov, n)
        if X is None:
            X = x
        else:
            X = np.vstack([X, x])
        print(x)
    return X

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
        self.train()

    def initialization(self):
        print('start init')
        random.seed(100)
        np.random.seed(100)
        # Initialize the means μ_i, covariances Σ_i and mixing coefficients w_i, and evaluate the initial value of the log likelihood.
        self.mu = np.random.uniform(0,1, size=(self.K,self.D))
        self.cov = np.zeros(shape=[self.K,self.D,self.D])
        for i in range(self.K):
            self.cov[i] = sklearn.datasets.make_spd_matrix(self.D)#*50
        w = np.random.uniform(0,1,size=(self.K))
        self.w = w/np.sum(w)
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
    mu = [[1,0],[0,1],[1,1]]
    cov = [[[1.0,0.0],[0.0,2.0]],[[2.0,0.0],[0.0,1.0]],[[0.5,0.0],[0.0,0.5]]]
    weight = [0.5,0.3,0.2]
    np.random.seed(1000)
    X = generateData(mu,cov,weight,10)
    ExpectationMaximization(X,None,3)

def experiment2():
    mu = [[10,0],[0,10],[10,10]]
    cov = [[[1.0,0.0],[0.0,2.0]],[[2.0,0.0],[0.0,1.0]],[[0.5,0.0],[0.0,0.5]]]
    weight = [0.5,0.3,0.2]
    np.random.seed(100)
    X = generateData(mu,cov,weight,1000)
    ExpectationMaximization(X,None,3)
experiment2()
