from objets import *
from journals import *
from authors import *
import nltk
import csv
import pandas as pd
import keras
from sklearn.utils import shuffle
from math import sqrt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from nlp import *
import seaborn as sns

node_info = pd.read_csv("data/node_information.csv", names = ["Id","Pubyear","Title","Authors","Journal","Abstract"])
node_edges = pd.read_csv("data/training_set.csv", delimiter=" ", names = ["Source","Target","Type"])

node_edges = shuffle(node_edges)


testing_set = node_edges[:10000]
training_set = node_edges[10000:]

n_topics_title = 20
n_topics_abstract = 20




#creation du modèle LSI

nv = node_info[["Id","Abstract"]].values

texts = []
hach = {}

for i in range(len(nv)):
    element = nv[i]
    title = element[1]
    Id = element[0]
    hach[Id] = i
    texts.append(title)
    
lsimodel_abstract, corpus_abstract = LSI_topicExtraction(texts, n_topics_abstract)


nv = node_info[["Id","Title"]].values

texts = []
hach = {}

for i in range(len(nv)):
    element = nv[i]
    title = element[1]
    Id = element[0]
    hach[Id] = i
    texts.append(title)
    
lsimodel_title, corpus_title = LSI_topicExtraction(texts, n_topics_title)
    


#Matrice des journeaux

journalMatrix,jindices = getJournalMatrix(node_info,training_set)

#Matrice es auteurs

authorMatrix,aindices = getAuthorMatrix(node_info,training_set)


def getTitleCosine(k_x,k_y):
    id1 = hach[k_x]
    id2 = hach[k_y]
    q1 = np.zeros(lsimodel_title.projection.u.shape[0])
    for t in corpus_title[id1]:
        q1[t[0]] = t[1]
    q2 = np.zeros(lsimodel_title.projection.u.shape[0])
    for t in corpus_title[id2]:
        q2[t[0]] = t[1]
        
    Lk = np.diag(lsimodel_title.projection.s)
    
    q1n = np.dot(np.dot(q1, lsimodel_title.projection.u), Lk)
    q2n = np.dot(np.dot(q2, lsimodel_title.projection.u), Lk)
    q = np.concatenate((q1n, q2n), axis = None)
    q = cosine(q1n,q2n)
    
    
    return q      

def getAbstractCosine(k_x,k_y):
    id1 = hach[k_x]
    id2 = hach[k_y]
    q1 = np.zeros(lsimodel_abstract.projection.u.shape[0])
    for t in corpus_abstract[id1]:
        q1[t[0]] = t[1]
    q2 = np.zeros(lsimodel_abstract.projection.u.shape[0])
    for t in corpus_abstract[id2]:
        q2[t[0]] = t[1]
        
    Lk = np.diag(lsimodel_abstract.projection.s)
    
    q1n = np.dot(np.dot(q1, lsimodel_abstract.projection.u), Lk)
    q2n = np.dot(np.dot(q2, lsimodel_abstract.projection.u), Lk)
    q = np.concatenate((q1n, q2n), axis = None)
    q = cosine(q1n,q2n)
    
    
    return q       

def getAuthorSimilarity(k_x,k_y):
    a_x = getAuthorsFromList(node_info["Authors"].values[hach[k_x]])
    a_y = getAuthorsFromList(node_info["Authors"].values[hach[k_y]])
    
    res = 0
    
    for x in a_x:
        for y in a_y:
            res += authorMatrix[aindices[x]][aindices[y]]
            
    return res / (len(a_x)*len(a_y))


def getAuthorSimilarityBis(k_x,k_y):
    a_x = getAuthorsFromList(node_info["Authors"].values[hach[k_x]])
    a_y = getAuthorsFromList(node_info["Authors"].values[hach[k_y]])
    
    res = 0
    
    for x in a_x:
        for y in a_y:
            res += authorMatrix[aindices[x]][aindices[y]]
            
    return res

def getJournalSimilarity(k_x,k_y):
    j_x = node_info["Journal"].values[hach[k_x]]
    j_y = node_info["Journal"].values[hach[k_y]]
    if type(j_x)==float:
        j_x = "NO_JOURNAL"
    if type(j_y)==float:
        j_y = "NO_JOURNAL"
    
    return journalMatrix[jindices[j_x]][jindices[j_y]]
    


def get_features(k_x,k_y):
    authorSimilarity = getAuthorSimilarityBis(k_x, k_y)
    journalSimilarity = getJournalSimilarity(k_x, k_y)
    abstractCosine = getAbstractCosine(k_x,k_y)
    titleCosine = getTitleCosine(k_x,k_y)  
    return [authorSimilarity, journalSimilarity, abstractCosine, titleCosine]

ev = training_set.values

X_train=[]
Y_train=[]
n = len(ev)
for element in ev:
    X_train.append(get_features(element[0],element[1]))
    Y_train.append([element[2]])
    p = len(X_train)
    if(p % (n//100) == 0):
        print(p)

    
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC

classifier = LinearSVC()

print("Fitting...")
classifier.fit(X_train, Y_train)
print("...Done")

ev_test = testing_set.values

print("Compute X_test, Y_true...")
X_test=[]
Y_true=[]
n_test = len(ev_test)
for element in ev_test:
    X_test.append(get_features(element[0],element[1]))
    Y_true.append([element[2]])
    p = len(X_train)
    if(p % (n_test//100) == 0):
        print(p)
print("...Done")


Y_pred = classifier.predict(X_test)

scores = precision_recall_fscore_support(Y_true, Y_pred)








node_edges_to_predict = pd.read_csv("data/testing_set.csv", delimiter=" ", names = ["Source","Target"])
ev_to_predict = node_edges_to_predict.values

print("Compute X_to_predict...")
X_to_predict=[]
n_to_predict = len(ev_to_predict)
for element in ev_to_predict:
    X_to_predict.append(get_features(element[0],element[1]))
    p = len(X_to_predict)
    if(p % (n_to_predict//100) == 0):
        print(p)
print("...Done")

y_final_predict = classifier.predict(X_to_predict)
prediction = list(y_final_predict)
predictions_SVM = zip(range(len(y_final_predict)), prediction)

with open("data/improved_predictions.csv","w") as pred1:
    csv_out = csv.writer(pred1)
    for row in predictions_SVM:
        csv_out.writerow(row)
































#e = Estimator(node_info,training_set,hach,lsimodel_title,corpus_title,lsimodel_abstract,corpus_abstract)

"""
X_train,Y_train=e.train(10)
X_train = pd.DataFrame(X_train)
Y_train = pd.DataFrame(Y_train)
Y_train = Y_train.rename(columns={0:1})
df = pd.concat((X_train, Y_train), axis=1)
df = df.dropna()
df0 = df[df[1]==0]
df0 = df0.reindex()
df1 = df[df[1]==1]
df1 = df1.reindex()
sns.distplot(df0[0])
sns.distplot(df1[0])
"""
