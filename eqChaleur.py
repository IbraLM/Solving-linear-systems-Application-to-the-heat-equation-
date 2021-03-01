import numpy as np
import matplotlib.pyplot as pl


import time
import math
from random import *
def cholesky(A):
    n=len(A)
    L=np.zeros((n,n))
    for i in range(n):
        for j in range(i + 1):  
            sum1 = 0; 
            if (j == i):  
                for k in range(j): 
                    sum1 += pow(L[j,k], 2); 
                L[j,j] = math.sqrt(A[j,j] - sum1);  
            else:   
                for k in range(j): 
                    sum1 += (L[i,k]*L[j,k]);     
                
                L[i,j] = (A[i,j] - sum1) /L[j,j];
    return L   

def icholesky(A):
    n=len(A)
    L=np.zeros((n,n))
    for i in range(n):
        for j in range(i + 1):  
            sum1 = 0; 
            sum1 = sum(L[i][k] * L[j][k] for k in range(j))
            if(A[i][j]!=0):
                
                if (j == i):  
                    L[j][j] = np.sqrt(A[j][j] - sum1);  
                else:   
                    #if(L[j,j] > 0): 
                    L[i][j] = (A[i][j] - sum1)/L[j][j];
    return L   


def conjgrad(A, b,x=None):
    if not x:
        x=np.ones(len(A))
    r =-np.dot(A,x)+b
    
    p =r
    rsold = np.dot(np.transpose(r), r)
    for i in range(1,pow(10,6)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        p = r + (rsnew/rsold)*p
        rsold = rsnew
        if np.sqrt(rsnew) < 1e-10:
           break
    return x
    
    

def descente(T,b):
    Y = np.zeros(len(b))
    Y[0]= b[0]/T[0,0]
    for i in range(1,len(b)):
        s = 0
        for j in range(i):
            s += T[i,j]*Y[j]
        Y[i] = (b[i] - s)/T[i,i]
    return Y

def remonte(T,y):
    n=len(Y)
    x = np.zeros(len(y))
    x[n-1] = y[n-1]/T[n-1,n-1]
    for i in range(n-2,-1,-1):
        s = 0
        for j in range(i+1,n):
            s += T[i,j] * x[j]
        x[i] = (y[i] -s)/T[i,i]
    return x

def sol_cholesky(A,b):
    T = np.linalg.cholesky(-A)
    Y = descente(T,-b)
    X = remonte(np.transpose(T),Y)
    return X
def createA(n):
    A=np.zeros((n**2,n**2))
    for i in range(n**2-n):
        A[i,i]=-4
        A[i,i+1]=1
        A[i+1,i]=1
        A[i+n,i]=1
        A[i,i+n]=1
    for i in range(n**2-n,n**2-1):
        A[i,i]=-4
        A[i,i+1]=1
        A[i+1,i]=1
    A[n**2-1,n**2-1]=-4    
    return A*((n+1)**2)
        
    
def createb(f,N):
    b = np.zeros(N**2)
    for i in range(N):
        for j in range(N):
            b[i*N+j] = f(i,j,N)
    return b

def rad_centre(i,j,N):
    if (i > N/2-2 and i < N/2+2 and j > N/2-2 and j < N/2+2):
        return -10
    return 0

def mur_nord(i,j,N):
    if (i > N-2):
        return -10
    return 0


def solution(N, fonction, met):
    A = createA(N)
    b = createb(fonction,N)
    if (met == "numpy"):
        x = np.linalg.solve(A,b)
    if (met == "gradient"):
        x = conjgrad(A,b)
    if (met == "cholesky"):
        x = sol_cholesky(A,b)
    x=np.reshape(x,(N,N))
    return x
N=40
def main():
    dim=[0,1,0,1]
    pl.style.use('classic')
    pl.imshow(solution(N,rad_centre,"gradient"),cmap=pl.cm.hot,origin='lower',interpolation='bilinear',extent=dim)
    #pl.imshow(solution(N,rad_centre,"numpy"),cmap=pl.cm.hot,origin='lower',interpolation='bilinear',extent=dim)
    pl.colorbar()
    pl.title("Heat diffusion for a radiator placed in the center")
    pl.show()
   
if __name__== "__main__":
  main()