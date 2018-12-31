#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 23:57:41 2018

@author: bnoyt
"""
import re
import numpy as np
import nltk
import csv
import pandas as pd
import keras
from sklearn.utils import shuffle
from math import sqrt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf


def removeParenthesis(string):
    count=0
    finalString = ""
    for c in string:
        if c=="(":
            count+=1
        elif c==")":
            count = max(0, count-1)
        else:
            if count==0:
                finalString+=c
    return finalString

def cleanList(l):
    while '' in l:
        l.remove('')
    while ' ' in l:
        l.remove(' ')
    while '.' in l:
        l.remove('.')
    while ',' in l:
        l.remove(',')


def getAuthorsFromList(l):
    if type(l) == float:
        return ["NO_AUTHOR"]
    l = removeParenthesis(l)
    listOfAuthors = l.split(",")
    cleanList(listOfAuthors)
    for i in range(len(listOfAuthors)):
        a = listOfAuthors[i].split(" ")
        cleanList(a)
        b = a[-1].split(".")
        cleanList(b)
        listOfAuthors[i] = b[-1]
    return listOfAuthors

def getAuthorMatrix(node_info,training_set):
    a = pd.merge(left = training_set,right=node_info[["Id","Authors"]],left_on="Source",right_on="Id")
    a = pd.merge(left = a,right=node_info[["Id","Authors"]],left_on="Target",right_on="Id")
    
    nv = a.values
    aindices = {}
    authors = []

    l = node_info["Authors"].dropna().unique()
    listOfAuthors = []
    for i in range(len(l)):
        listOfAuthors += getAuthorsFromList(l[i])
    listOfAuthors = pd.Series(listOfAuthors).unique()
    
    authors = list(listOfAuthors)
    authors += ["NO_AUTHOR"]
    for i in range(len(authors)):
        aindices[authors[i]] = i
        
    Moui = np.zeros((len(authors),len(authors)))
    Mtout = np.zeros((len(authors),len(authors)))
            
    
    for e in nv:
        authors_1 = e[4]
        authors_2 = e[6]
        if type(authors_1)==float:
            authors_1 = ["NO_AUTHOR"]
        else:
            authors_1 = getAuthorsFromList(authors_1)
        if type(authors_2)==float:
            authors_2 = ["NO_AUTHOR"]
        else:
            authors_2 = getAuthorsFromList(authors_2)
            
        for author_1 in authors_1:
            for author_2 in authors_2:
                i1 = aindices[author_1]
                i2 = aindices[author_2]
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
                
    return Moui,aindices
