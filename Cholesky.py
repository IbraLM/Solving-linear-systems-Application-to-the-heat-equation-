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
                #if(L[j,j] > 0): 
                L[i,j] = (A[i,j] - sum1) /L[j,j];
    return L   

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
                #if(L[j,j] > 0): 
                L[i,j] = (A[i,j] - sum1)/L[j,j];
    return L   


def condt(M,A):
    if(np.linalg.cond(np.dot(np.linalg.inv(M),A))<np.linalg.cond(A)):
        print("Cond It's OK")
        return np.linalg.cond(np.dot(np.linalg.inv(M),A))
    else:
        print("Cond It's NOT OK")
        return np.linalg.cond(np.dot(np.linalg.inv(M),A))  
    

def main():
    n=10
    A=np.array([[-4, 1,0 , 0,1,0,0,0,0,0], [1, -4, 1, 0,0,1,0,0,0,0], [0,1,-4,1,0,0,1,0,0,0], [0, 0,1,-4,0,0,0,1,0,0], [1,0,0,0,-4,1,0,0,1,0], [0,1,0,0,1,-4,1,0,0,1], [0,0,1,0,0,1,-4,1,0,0], [0,0,0,1,0,0,1,-4,0,0], [0,0,0,0,1,0,0,0,-4,1], [0,0,0,0,0,1,0,0,1,-4]])
    print("/-/-/-/-/Generer une matrice symétrique définie positive creuse A:/-/-/-/-/")
    A=sdp(n,3); #generer matrice symetrique définie positive creuse
    print(A)
    print("\n")
    L =cholesky(A) #comparaison entre l'algo écrit pour cholesky et cholesky incomplete et np.linal.cholesky
    T=np.linalg.cholesky(A)    
    C=icholesky(A)
    print("/-/-/-/-/Solution de choleskey dense/-/-/-/-/:")
    print(T)
    print("\n")
    print ("/-/-/-/-/Solution avec np.linalg.cholesky/-/-/-/-/:")
    print(L)
    print("\n")
    print ("/-/-/-/-/Solution avec choleskey incomplete/-/-/-/-/:")
    print(C)
    max=0
    s=0
    for i in range(n):
        for j in range(n):
            s=abs(C[i,j]-T[i,j])
            if s>max:
                max=s
    print("la precision de l'algorithme ecrit est :",max)
    M=np.dot(T,np.transpose(T))
    N=np.dot(C,np.transpose(C))
    print("conditionnement avec cholesky dense")
    condt(M,A)                          
    
    print("conditionnement avec cholesky incomplete")
    condt(N,A)
if __name__== "__main__":
  main()
