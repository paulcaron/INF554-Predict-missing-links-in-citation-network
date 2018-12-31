#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 23:07:06 2018

@author: bnoyt
"""

import nltk
import csv
import pandas as pd
import keras
from sklearn.utils import shuffle
from math import sqrt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import seaborn as sns



def getJournalMatrix(node_info,training_set):
    a = pd.merge(left = training_set,right=node_info[["Id","Journal"]],left_on="Source",right_on="Id")
    a = pd.merge(left = a,right=node_info[["Id","Journal"]],left_on="Target",right_on="Id")
    
    nv = a.values
    jindices = {}
    journaux = []
    
    l = node_info["Journal"].dropna().unique()
    journaux = list(l)
    
    journaux.append("NO_JOURNAL")
    for i in range(len(journaux)):
        jindices[journaux[i]] = i
        
    Moui = np.zeros((len(journaux),len(journaux)))
    Mtout = np.zeros((len(journaux),len(journaux)))
            
    for e in nv:
        journal_1 = e[4]
        journal_2 = e[6]
        if type(journal_1)==float:
            journal_1 = "NO_JOURNAL"
        if type(journal_2)==float:
            journal_2 = "NO_JOURNAL"
        i1 = jindices[journal_1]
        i2 = jindices[journal_2]
        Mtout[i1, i2]+=1
        Mtout[i2, i1]+=1
        if e[2]==1:
            Moui[i1,i2]+=1
            Moui[i2,i1]+=1
    for i in range(Mtout.shape[0]):
        for j in range(Mtout.shape[0]):
            if Mtout[i, j] !=0:
                Moui[i, j] /= Mtout[i, j]
                Moui[j, i] = Moui[i, j] 
                
    return Moui,jindices
        