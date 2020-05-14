'''
Created on Jul 1, 2019

@author: alexlin
'''
import numpy as np
import os
from pathlib import Path

def dfa_fast(vdata, istart, iend, L_all):
#function takes in your time series, the start and end time points, and the
#different L values you want to use in the implementation of DFA
    
    #number of columns of vdata
    lenY=len(vdata[1,:])
    #number of rows of vdata
    lenX=len(vdata)
    
    #takes the cumulative sum
    for y in range(lenY):
        for x in range(1,lenX):
            vdata[x][y]=vdata[x][y]+vdata[x-1][y]

    FL_all=np.zeros((len(L_all),lenY))
    
    #iterating through the L values you want to use
    for il in range(len(L_all)):
        L=L_all[il]
        y=[]
        for x in range(L):
            y.append(x)
        X=np.ones((L,2))
        for i in range(L):
            X[i][0]=y[i]
        
        #nice thing about this approach is if your data isn't an integer
        #multiple of the length L, it will just average as many windows as can fit
        c=0
        FL=np.zeros((1,len(vdata[1,:])))
        for i in range(istart,min(iend,lenX)-L+1,L):
            vtmp=np.zeros((L,lenY))
            for m in range(L):
                for n in range(lenY):
                    vtmp[m][n]=vdata[m+i][n]
                     
                
            #b=X\vtmp;
            #y=X*b;
            #r=vtmp-X*(X\vtmp);
            #calculates rms for that window
            
            #note: linalg.lstsq returns a list
            #d[0] is what we want
            d=np.linalg.lstsq(X, vtmp)
            
            m=np.matmul(X,d[0])
            s=np.subtract(vtmp,m)
            p=np.power(s,2)
            a=np.mean(p,axis=0)

            rms=np.sqrt(a)
            
            FL=np.add(FL,rms)
            c+=1
        
        FL=np.divide(FL,c)
        FL_all[il,:]=FL
    
    logFL=np.log(FL_all)
    X=np.reshape(L_all,(len(L_all),1))
    
    lx=np.log(X)
    logX=np.ones((len(X),2))
    for i in range(len(lx)):
            logX[i][0]=lx[i]
             
    b=np.linalg.lstsq(logX,logFL)
    res=b[0]
    return res[0,:]

def prep(file, row, col):
#inputs the data from a txt file to array
    data = file.readlines()
    allNums=np.zeros((row,col))
    x=0
    y=0
    for line in data:
        line.strip()
        for num in line.split(', '):
            float(num)
            allNums[x][y]=num
            y+=1
        x+=1
        y=0
    return allNums

import os

folder=os.listdir('/Users/alexlin/Desktop/Lab/MS/Run')
#folder = os.listdir('/Users/alexlin/Desktop/Lab/MS/Test'⁩)
ideal = [10,16,20,32]
allData = np.zeros((95,86))
n=0
for file in folder:
    if file.endswith('.txt') and file != 'Names.txt':
        f=open(file,'r')
        p=prep(f,180,86)
        d=dfa_fast(p,0,180,ideal)
        for i in range(len(d)):
            allData[n][i]=d[i]
        n+=1
        f.close
        continue
    else: 
        continue

import csv
n=open('Names.txt','r')
fn=n.readlines()
fields=[]
for line in fn:
    name=line[line.find('=')+1:len(line)-1]
    fields.append(name)
#print(fields)

import pandas as pd
from pandas import DataFrame

from matplotlib import pyplot as plt
plt.imshow(allData)

# writing to csv file
with open('HurstExp.csv', 'w') as csvfile: 
    # creating a csv writer object 
    writer = csv.writer(csvfile) 
    # writing the fields 
    writer.writerow(fields) 
    # writing the data rows 
    writer.writerows(allData)