import numpy as np
import matplotlib.pyplot as pl

import time
import math
from random import *
liste1=[]
def conjgrad(A, b,x=None):
    if not x:
        x=np.ones(len(A))
    r =-np.dot(A,x)+b
    bn=np.linalg.norm(b)
    
    p =r
    rsold = np.dot(np.transpose(r), r)
    liste1.append(np.log(np.sqrt(rsold)/bn))
    for i in range(1,pow(10,2)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        liste1.append(np.log(np.sqrt(rsnew)/bn))
        p = r + (rsnew/rsold)*p
        rsold = rsnew
        if np.sqrt(rsnew) < 1e-8:
           break
    return x
    

liste2=[]
def conjgrad2(A,b,x=None):
    if not x:
        x=np.ones(len(A))
    E=icholesky(A)
    r =-np.dot(A,x)+b
    bn=np.linalg.norm(b)
    Minv=np.dot(np.linalg.inv(np.transpose(E)),np.linalg.inv(E))
    p =np.dot(Minv,r)
    rsold = np.dot(r, np.dot(Minv,r))
    liste2.append(np.log(np.sqrt(rsold)/bn))
    for i in range(1,pow(10,2)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p, Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(r, np.dot(Minv,r))
        liste2.append(np.log(np.sqrt(rsnew)/bn))
        p = np.dot(Minv,r) + (rsnew/rsold)*p
        rsold = rsnew
        if np.sqrt(rsnew) < 1e-8:
           break
    return x

def sdp(n,k):
    M=np.zeros((n,n))
    c=0
    while(c<k):
        i=randint(0,n-1)
        j=randint(0,n-1)
        while(j==i or M[i,j]!=0):
            j=randint(0,n-1)
        while(M[i,j]==0):
            M[i,j]=uniform(-10,10)
            M[j,i]=M[i,j]
        c=c+1
    for i in range (0,n): 
        L1=[abs(M[i,j]) for j in range(0,n) if j !=i] 
        L2=[abs(M[j,i]) for j in range(0,n) if j !=i] 
        s= sum(L1)+sum(L2) 
        M[i,i]=s+randint(5,10)
    return M

def icholesky(A):
    n=len(A)
    L=np.zeros((n,n))
    for i in range(n):
        for j in range(i + 1):  
            sum1 = 0; 
            if (j == i and A[i,j]!=0):  
                for k in range(j): 
                    sum1 += pow(L[j,k], 2); 
                L[j,j] = math.sqrt(A[j,j] - sum1);  
            if(i>j and A[i,j]!=0):   
                for k in range(j): 
                    sum1 += (L[i,k]*L[j,k]);     
                if(L[j,j] > 0): 
                    L[i,j] = (A[i,j] - sum1)/L[j,j];
    return L    
    
def main():
    n=10
    B=sdp(n,n/2)
    b=np.array([1,2,5,5,0,1,0,4,1547,-9])
    y=conjgrad2(B,b)
    u=conjgrad(B,b)
    print("/-/-/-/-/Solution avec GC/-/-/-/-/:")
    print(u)
    print("\n")
    print("/-/-/-/-/Solution avec GCprecontionee/-/-/-/-/:")
    print(y)
    w=np.arange(0,len(liste1),1)
    r=np.arange(0,len(liste2),1)
    pl.plot(w,liste1,label="conjgrad")
    pl.plot(r,liste2,label="conjgrad preconditionne")
    pl.title("CG vs CG-preconditionne")
    pl.xlabel("iterations")
    pl.ylabel("log||Ax-b||/log||b||, log(relative residuel)")    
    pl.legend() 
    pl.show()  
if __name__== "__main__":
  main()